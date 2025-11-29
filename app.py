import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------
# BASIC CONSTANTS
# -------------------------------------------------------

PRODUCTS = ["Product 1", "Product 2", "Product 3"]
AREAS = ["South", "West", "North", "Export"]

# Table 3 – simplified
MIN_MACHINING_TIME = {
    "Product 1": 60.0,   # minutes
    "Product 2": 75.0,
    "Product 3": 120.0,
}
MIN_ASSEMBLY_TIME = {
    "Product 1": 100.0,
    "Product 2": 150.0,
    "Product 3": 300.0,
}
MATERIAL_PER_UNIT = {
    "Product 1": 1.0,
    "Product 2": 2.0,
    "Product 3": 3.0,
}

# Table 5 – hours per machine per quarter by shift
MACHINE_HOURS_PER_SHIFT = {
    1: 576,
    2: 1068,
    3: 1602,
}

# Simple economic base values
BASE_DEMAND_INDEX = 1.0
BASE_GDP = 100.0
BASE_UNEMPLOYMENT = 6.0
BASE_CB_RATE = 3.0  # %

BASE_MATERIAL_PRICE = 100.0  # per 1000 units

# Additional constants from manual Tables
MIN_SALES_SALARY_PER_QUARTER = 2000
ASSEMBLY_MIN_WAGE_RATE = 8.50
MIN_MANAGEMENT_BUDGET = 40_000
CONTRACTED_MAINTENANCE_RATE = 60.0
UNCONTRACTED_MAINTENANCE_RATE = 120.0
SALESPERSON_EXPENSES = 3000
VEHICLE_CAPACITY = {"Product 1": 40, "Product 2": 40, "Product 3": 20}
JOURNEY_TIME_DAYS = {"South": 1, "West": 2, "North": 4, "Export": 6}
FLEET_FIXED_COST_PER_VEHICLE = 7000
OWN_VEHICLE_RUNNING_COST_PER_DAY = 50
HIRED_VEHICLE_COST_PER_DAY = 200
SCRAP_VALUE = {"Product 1": 20.0, "Product 2": 40.0, "Product 3": 60.0}
SERVICING_CHARGE = {"Product 1": 60.0, "Product 2": 120.0, "Product 3": 200.0}
PRODUCT_STOCK_VALUATION = {"Product 1": 80.0, "Product 2": 120.0, "Product 3": 200.0}
MACHINE_COST = 200_000
VEHICLE_COST = 15_000
TAX_RATE = 0.30


# -------------------------------------------------------
# DATA CLASSES
# -------------------------------------------------------

@dataclass
class Economy:
    quarter: int = 1
    year: int = 1
    gdp: float = BASE_GDP
    unemployment: float = BASE_UNEMPLOYMENT
    cb_rate: float = BASE_CB_RATE
    material_price: float = BASE_MATERIAL_PRICE

    def advance(self):
        """Very simple stochastic economy with seasonality."""
        self.quarter += 1
        if self.quarter > 4:
            self.quarter = 1
            self.year += 1

        # AR(1)-style small random drift
        shock = np.random.normal(0, 1.5)
        self.gdp = max(80, self.gdp * (1 + shock / 100))

        # unemployment moves inversely to GDP
        u_shock = np.random.normal(0, 0.3)
        self.unemployment = min(
            15,
            max(2, self.unemployment + u_shock - shock / 40),
        )

        # central bank responds (roughly) to GDP
        rate_target = 2.5 + (self.gdp - BASE_GDP) / 40
        self.cb_rate = max(0.25, 0.75 * self.cb_rate + 0.25 * rate_target)

        # materials follow GDP & interest
        self.material_price = max(
            60,
            self.material_price * (1 + (self.cb_rate - 2.5) / 200 + np.random.normal(0, 0.01)),
        )


@dataclass
class Decisions:
    prices_home: Dict[str, float]
    prices_export: Dict[str, float]
    credit_days: int
    assembly_time: Dict[str, float]
    advertising: Dict[Tuple[str, str], float]  # (product, area) -> spend
    product_dev: Dict[str, float]
    sales_allocation: Dict[str, int]  # area -> #salespeople
    recruit_sales: int
    dismiss_sales: int
    train_assembly: int
    recruit_assembly: int
    dismiss_assembly: int
    shift_level: int
    maintenance_hours_per_machine: float
    deliveries: Dict[Tuple[str, str], int]  # (product, area) -> units scheduled
    dividend_per_share: float
    management_budget: float
    materials_order_qty: float  # simplistic single order quantity


@dataclass
class CompanyState:
    name: str
    shares_outstanding: float = 1_000_000.0

    machines: int = 10
    machine_efficiency: float = 0.9  # 0–1
    vehicles: int = 5

    salespeople: int = 10
    assembly_workers: int = 40

    backlog: Dict[Tuple[str, str], int] = field(default_factory=dict)
    stocks: Dict[Tuple[str, str], int] = field(default_factory=dict)

    material_stock: float = 5_000.0  # units
    material_on_order: float = 5_000.0

    share_price: float = 1.0
    reserves: float = 0.0  # retained earnings
    cash: float = 200_000.0
    overdraft: float = 0.0
    unsecured_loan: float = 0.0
    fixed_assets_machines: float = 10 * 200_000.0
    fixed_assets_vehicles: float = 5 * 15_000.0
    tax_liability: float = 0.0

    # For reporting
    last_report: Dict = field(default_factory=dict)

    def net_worth(self) -> float:
        assets = self.cash + self.material_stock * (self.fixed_material_valuation()) + \
                 self.fixed_assets_machines + self.fixed_assets_vehicles
        liabilities = self.overdraft + self.unsecured_loan + self.tax_liability
        return assets - liabilities

    def fixed_material_valuation(self) -> float:
        # simple constant for now, can be overridden by economy
        return 0.05  # £ per unit (approx 50% of 100/1000)

    def total_employees(self) -> int:
        # ignore machinists (derived from machines & shift)
        machinists = self.machines * 4 * self.current_shift_level
        return self.salespeople + self.assembly_workers + machinists

    @property
    def current_shift_level(self) -> int:
        # crude mapping from efficiency to shift level if not stored per-state
        if self.last_report.get("shift_level") is not None:
            return self.last_report["shift_level"]
        return 1


# -------------------------------------------------------
# SIMULATION ENGINE
# -------------------------------------------------------

class Simulation:
    def __init__(self, n_companies: int = 4, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.economy = Economy()
        self.companies: List[CompanyState] = [
            CompanyState(name=f"Company {i+1}") for i in range(n_companies)
        ]
        self.history: List[Dict] = []

    # --------------- Demand & Marketing ------------------

    def demand_for_product(
        self,
        company: CompanyState,
        decisions: Decisions,
        product: str,
        area: str,
    ) -> float:
        """Expected orders for product/area next quarter (units)."""

        # Base demand scaled by economy & area
        seasonal_factor = 1.0 + (0.10 if self.economy.quarter == 4 else 0.0)
        gdp_factor = self.economy.gdp / BASE_GDP
        base_area = {"South": 1.0, "West": 0.7, "North": 1.3, "Export": 1.5}[area]

        base_demand = 1000 * base_area * seasonal_factor * gdp_factor

        # Price sensitivity – assume home vs export
        if area == "Export":
            price = decisions.prices_export[product]
        else:
            price = decisions.prices_home[product]

        # Relative to a "reference" price
        ref_price = 100 + 20 * PRODUCTS.index(product)
        price_factor = math.exp(-0.015 * (price - ref_price))

        # Advertising effect (diminishing returns)
        adv = decisions.advertising[(product, area)]
        adv_factor = 1 + 0.0003 * math.sqrt(max(0, adv))

        # Quality / product image proxy: assembly time vs min
        q = decisions.assembly_time[product] / MIN_ASSEMBLY_TIME[product]
        quality_factor = min(1.4, 0.7 + 0.7 * q)

        # Product development cumulative effect approximated by spend
        dev_spend = decisions.product_dev[product]
        dev_factor = 1 + 0.0002 * math.log1p(max(0, dev_spend))

        # Credit terms – longer credit slightly boosts orders
        credit_factor = 1 + (decisions.credit_days - 30) / 200.0

        demand = base_demand * price_factor * adv_factor * quality_factor * dev_factor * credit_factor

        # Backlog impact – if large backlog already, retailers get frustrated
        backlog = company.backlog.get((product, area), 0)
        if backlog > 0:
            demand *= max(0.6, 1 - backlog / 4000.0)

        return max(0, demand)

    # --------------- Operations & Production -----------

    def production_capacity(
        self,
        company: CompanyState,
        decisions: Decisions,
    ) -> Tuple[float, float]:
        """Return (max_units_by_machining, max_units_by_assembly) overall."""
        shift = decisions.shift_level
        hours_per_machine = MACHINE_HOURS_PER_SHIFT[shift]
        total_machine_hours = company.machines * hours_per_machine

        # Effective hours adjusted by efficiency and maintenance
        maint = decisions.maintenance_hours_per_machine
        maintenance_factor = min(1.1, 0.9 + maint / 200.0)
        eff = min(1.0, company.machine_efficiency * maintenance_factor)
        effective_machine_hours = total_machine_hours * eff

        # Assume "average" machining time across products
        avg_mach_time = np.mean(list(MIN_MACHINING_TIME.values())) / 60.0  # hours
        max_units_machining = effective_machine_hours / avg_mach_time

        # Assembly capacity: workers * hours / time per unit (weighted)
        max_hours_per_worker = 420 + 84 + 72  # allow for overtime upper bound
        total_assembly_hours = company.assembly_workers * max_hours_per_worker

        avg_assembly_time = np.mean(list(decisions.assembly_time.values())) / 60.0
        max_units_assembly = total_assembly_hours / max_assembly_time

        return max_units_machining, max_units_assembly

    def simulate_quarter_for_company(
        self,
        company: CompanyState,
        decisions: Decisions,
        is_player: bool = False,
    ) -> Dict:
        """Core quarterly simulation."""
        econ = self.economy

        # --- Production planning from deliveries decision ---
        planned_deliveries = decisions.deliveries

        # Total required production = deliveries + replacements for rejects (approx later)
        total_planned_units = sum(planned_deliveries.values())

        # --- Capacity ---
        cap_mach, cap_assy = self.production_capacity(company, decisions)
        max_units = min(cap_mach, cap_assy)

        capacity_ratio = min(1.0, max_units / max(total_planned_units, 1))

        # Produced units per (product, area) proportionally if constrained
        produced: Dict[Tuple[str, str], int] = {}
        for key, qty in planned_deliveries.items():
            produced[key] = int(qty * capacity_ratio)

        # --- Materials consumption & ordering ---
        avg_material_per_unit = np.mean(list(MATERIAL_PER_UNIT.values()))
        material_required = total_planned_units * avg_material_per_unit

        # Use existing stock then new material arrives
        material_opening = company.material_stock
        # Delivery of last quarter's order:
        material_delivered = company.material_on_order
        material_available = material_opening + material_delivered

        material_used = min(material_available, material_required)
        material_closing = material_available - material_used

        # Place new order for quarter+2
        company.material_on_order = decisions.materials_order_qty

        # --- Quality & rejects ---
        rejects: Dict[Tuple[str, str], int] = {}
        quality_costs = 0.0
        for (prod, area), qty in produced.items():
            q_factor = decisions.assembly_time[prod] / MIN_ASSEMBLY_TIME[prod]
            reject_rate = max(0.01, 0.10 / max(0.8, q_factor))
            r = int(qty * reject_rate)
            rejects[(prod, area)] = r
            produced[(prod, area)] = qty - r
            # guarantee servicing in future – approximate cost this quarter
            quality_costs += r * 20.0  # placeholder (Table 7–like)

        # --- Products delivered to area warehouses ---
        deliveries = produced

        # --- Demand & orders ---
        new_orders: Dict[Tuple[str, str], int] = {}
        sales: Dict[Tuple[str, str], int] = {}
        backlog_new: Dict[Tuple[str, str], int] = {}
        stocks_new: Dict[Tuple[str, str], int] = {}

        revenue = 0.0

        for prod in PRODUCTS:
            for area in AREAS:
                key = (prod, area)
                # Opening stocks & backlog from previous quarter
                opening_stock = company.stocks.get(key, 0)
                opening_backlog = company.backlog.get(key, 0)

                demand_units = int(self.demand_for_product(company, decisions, prod, area))
                new_orders[key] = demand_units

                available_units = opening_stock + deliveries.get(key, 0)
                potential_sales = opening_backlog + demand_units

                sold = min(available_units, potential_sales)
                sales[key] = sold

                # closing stock & backlog
                stocks_new[key] = available_units - sold

                unsatisfied_orders = max(0, potential_sales - sold)
                # roughly half cancel
                remaining_backlog = int(unsatisfied_orders * 0.5)
                backlog_new[key] = remaining_backlog

                # price
                price = decisions.prices_export[prod] if area == "Export" else decisions.prices_home[prod]
                revenue += sold * price

        company.stocks = stocks_new
        company.backlog = backlog_new
        company.material_stock = material_closing

        # --- Marketing & personnel costs ---
        ads_cost = sum(decisions.advertising.values())
        prod_dev_cost = sum(decisions.product_dev.values())

        # Sales force: salary + commission
        avg_salary = max(2000.0, decisions.management_budget * 0.02 / max(company.salespeople, 1))
        commission_rate = 0.03
        salespeople_salary_cost = company.salespeople * avg_salary
        commission_cost = revenue * commission_rate

        # Personnel changes
        recruit_cost_sales = decisions.recruit_sales * 1500
        dismiss_cost_sales = decisions.dismiss_sales * 5000
        train_cost_assembly = decisions.train_assembly * 4500
        recruit_cost_assembly = decisions.recruit_assembly * 1200
        dismiss_cost_assembly = decisions.dismiss_assembly * 3000

        # Update headcount
        company.salespeople += decisions.recruit_sales + decisions.train_assembly - decisions.dismiss_sales
        company.assembly_workers += decisions.recruit_assembly + decisions.train_assembly - decisions.dismiss_assembly
        company.salespeople = max(0, company.salespeople)
        company.assembly_workers = max(0, company.assembly_workers)

        personnel_costs = (
            recruit_cost_sales
            + dismiss_cost_sales
            + train_cost_assembly
            + recruit_cost_assembly
            + dismiss_cost_assembly
        )

        # Maintenance cost
        contracted_hours_total = company.machines * decisions.maintenance_hours_per_machine
        maint_cost = contracted_hours_total * 60.0  # table 4 approx

        # Warehousing – simple cost per stock unit + fixed
        total_stock_units = sum(stocks_new.values())
        warehousing_cost = 3750 + 2.0 * total_stock_units

        management_cost = max(40_000.0, decisions.management_budget)

        # Assembly & machining wages (simplified)
        # Assume hourly wage 8.5, premium for machinists
        max_hours_per_worker = 420 + 84 + 72
        assy_hours = company.assembly_workers * max_hours_per_worker * min(1.0, capacity_ratio)
        assy_wage_rate = 8.5
        assy_wages = assy_hours * assy_wage_rate

        machinist_count = company.machines * 4 * decisions.shift_level
        mach_hours = cap_mach  # total effective hours
        machinist_wage_rate = assy_wage_rate * 0.65 * (1 + 0.3 * (decisions.shift_level - 1))
        mach_wages = mach_hours * machinist_wage_rate

        # Materials cost (material_delivered priced at econ.material_price)
        materials_bought_cost = material_delivered * (econ.material_price / 1000.0)

        # Machine running overhead (Table 8-ish)
        supervision_cost = 10_000 * decisions.shift_level
        overhead_per_machine = 2000
        running_cost_per_hour = 7
        planning_cost_per_unit = 1

        machine_overhead_cost = (
            supervision_cost
            + overhead_per_machine * company.machines
            + running_cost_per_hour * cap_mach
            + planning_cost_per_unit * total_planned_units
        )

        # Transport
        # approximate vehicles needed from deliveries
        total_units_per_area = {area: 0 for area in AREAS}
        for (prod, area), qty in deliveries.items():
            total_units_per_area[area] += qty

        # capacity per vehicle (rough mix)
        def vehicle_days_for_area(units: int, area: str) -> int:
            if units == 0:
                return 0
            capacity = 40  # rough across products
            trips = math.ceil(units / capacity)
            days = {"South": 1, "West": 2, "North": 4, "Export": 6}[area]
            return trips * days

        total_vehicle_days_required = sum(
            vehicle_days_for_area(units, area) for area, units in total_units_per_area.items()
        )

        own_vehicle_capacity_days = company.vehicles * 60
        own_days = min(own_vehicle_capacity_days, total_vehicle_days_required)
        hired_days = max(0, total_vehicle_days_required - own_days)

        fleet_fixed = company.vehicles * 7000
        own_running = own_days * 50
        hired_running = hired_days * 200
        transport_cost = fleet_fixed + own_running + hired_running

        # Overheads bucket
        overheads = (
            ads_cost
            + prod_dev_cost
            + salespeople_salary_cost
            + commission_cost
            + personnel_costs
            + maint_cost
            + warehousing_cost
            + management_cost
            + transport_cost
            + quality_costs
        )

        # Cost of sales
        cost_of_sales = (
            materials_bought_cost
            + assy_wages
            + mach_wages
            + machine_overhead_cost
        )

        gross_profit = revenue - cost_of_sales
        ebitda = gross_profit - overheads

        # Interest & depreciation
        # Simple banking: if cash<0 we use overdraft / loan
        total_cash_like = company.cash - company.overdraft - company.unsecured_loan
        if total_cash_like < 0:
            # need borrowing
            borrow_needed = -total_cash_like
            # overdraft first
            overdraft_limit = max(0, 0.8 * company.net_worth())
            use_overdraft = min(borrow_needed, max(0, overdraft_limit - company.overdraft))
            company.overdraft += use_overdraft
            remaining = borrow_needed - use_overdraft
            if remaining > 0:
                company.unsecured_loan += remaining
            company.cash = 0.0
        else:
            company.cash = total_cash_like
            company.overdraft = max(0.0, company.overdraft)
            company.unsecured_loan = max(0.0, company.unsecured_loan)

        # Interest calculation on average balances
        deposit_rate = max(0.0, econ.cb_rate - 2.0) / 100.0
        overdraft_rate = (econ.cb_rate + 4.0) / 100.0
        loan_rate = (econ.cb_rate + 10.0) / 100.0

        interest_received = max(0.0, company.cash) * deposit_rate / 4.0
        interest_paid = (
            company.overdraft * overdraft_rate / 4.0
            + company.unsecured_loan * loan_rate / 4.0
        )

        # Depreciation
        dep_machines = 0.025 * company.fixed_assets_machines
        dep_vehicles = 0.0625 * company.fixed_assets_vehicles
        depreciation = dep_machines + dep_vehicles
        company.fixed_assets_machines -= dep_machines
        company.fixed_assets_vehicles -= dep_vehicles

        profit_before_tax = ebitda + interest_received - interest_paid - depreciation

        # Tax (once per year on cumulative profit, simplified to per-quarter)
        tax_rate = 0.30
        tax = max(0.0, profit_before_tax * tax_rate)
        net_profit = profit_before_tax - tax

        # Update reserves & cash
        dividends = decisions.dividend_per_share * company.shares_outstanding
        dividends = min(dividends, max(0.0, net_profit + company.reserves + company.cash))

        retained = net_profit - dividends
        company.reserves += retained
        company.cash += net_profit - dividends + interest_received - interest_paid

        # Update tax liability (simplified)
        company.tax_liability = tax

        # Update share price as a function of net worth, profit, dividend, and expectations
        nw = company.net_worth()
        eps = net_profit / company.shares_outstanding
        dps = dividends / company.shares_outstanding if company.shares_outstanding else 0
        share_price = max(
            0.1,
            0.5 * company.share_price
            + 0.3 * (nw / company.shares_outstanding)
            + 5 * eps
            + 3 * dps,
        )
        company.share_price = share_price

        report = {
            "revenue": revenue,
            "cost_of_sales": cost_of_sales,
            "gross_profit": gross_profit,
            "overheads": overheads,
            "ebitda": ebitda,
            "interest_received": interest_received,
            "interest_paid": interest_paid,
            "depreciation": depreciation,
            "tax": tax,
            "net_profit": net_profit,
            "dividends": dividends,
            "retained": retained,
            "cash": company.cash,
            "overdraft": company.overdraft,
            "loan": company.unsecured_loan,
            "net_worth": nw,
            "share_price": company.share_price,
            "shift_level": decisions.shift_level,
            "machine_efficiency": company.machine_efficiency,
            "materials_used": material_used,
            "material_opening": material_opening,
            "material_closing": material_closing,
            "material_delivered": material_delivered,
            "stocks": stocks_new,
            "backlog": backlog_new,
            "sales": sales,
            "new_orders": new_orders,
            "deliveries": deliveries,
            "rejects": rejects,
        }

        company.last_report = report
        return report

    # --------------- Competitor AI ----------------------

    def auto_decisions(self, company: CompanyState) -> Decisions:
        """Very simple heuristic decisions for AI companies."""
        base_price = 100
        prices_home = {p: base_price + 15 * i + random.randint(-10, 10) for i, p in enumerate(PRODUCTS)}
        prices_export = {p: ph * 1.1 for p, ph in prices_home.items()}

        assembly_time = {
            p: MIN_ASSEMBLY_TIME[p] * random.uniform(1.0, 1.4) for p in PRODUCTS
        }

        advertising = {}
        for p in PRODUCTS:
            for a in AREAS:
                advertising[(p, a)] = random.choice([0, 5000, 10000, 20000])

        product_dev = {p: random.choice([0, 5000, 10000]) for p in PRODUCTS}

        # allocate salespeople proportionally to area base
        total_sales = company.salespeople
        base = {"South": 1.0, "West": 0.7, "North": 1.3, "Export": 1.2}
        total_base = sum(base.values())
        sales_alloc = {a: int(total_sales * base[a] / total_base) for a in AREAS}
        # fix rounding
        while sum(sales_alloc.values()) < total_sales:
            area = random.choice(AREAS)
            sales_alloc[area] += 1

        deliveries = {}
        for p in PRODUCTS:
            for a in AREAS:
                deliveries[(p, a)] = random.randint(200, 1500)

        d = Decisions(
            prices_home=prices_home,
            prices_export=prices_export,
            credit_days=random.choice([30, 45, 60]),
            assembly_time=assembly_time,
            advertising=advertising,
            product_dev=product_dev,
            sales_allocation=sales_alloc,
            recruit_sales=random.choice([0, 1, 2]),
            dismiss_sales=0,
            train_assembly=random.choice([0, 2, 4]),
            recruit_assembly=random.choice([0, 2, 4]),
            dismiss_assembly=0,
            shift_level=random.choice([1, 2, 3]),
            maintenance_hours_per_machine=random.choice([20, 40, 60]),
            deliveries=deliveries,
            dividend_per_share=random.choice([0.0, 0.02, 0.04]),
            management_budget=random.choice([40_000, 50_000, 60_000]),
            materials_order_qty=random.choice([4000, 6000, 8000]),
        )
        return d

    # --------------- Public API -------------------------

    def step(self, player_decisions: Decisions):
        """Run one quarter for all companies."""
        reports = []
        for i, c in enumerate(self.companies):
            if i == 0:
                dec = player_decisions
            else:
                dec = self.auto_decisions(c)
            rep = self.simulate_quarter_for_company(c, dec, is_player=(i == 0))
            rep_with_meta = {
                **rep,
                "company": c.name,
                "quarter": self.economy.quarter,
                "year": self.economy.year,
            }
            reports.append(rep_with_meta)

        self.history.extend(reports)
        # advance economy for next round
        self.economy.advance()
        return reports


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

st.set_page_config(page_title="Topaz-style Management Simulation", layout="wide")

if "sim" not in st.session_state:
    st.session_state.sim = Simulation(n_companies=4, seed=42)

sim: Simulation = st.session_state.sim

st.title("Topaz-Style Business Management Simulation")

st.markdown(
    """
This app implements a **quarterly management simulation** with 3 products, 4 market areas,  
and interacting decisions across **Marketing, Operations, Personnel and Finance**.

You control **Company 1**; the other companies are automated competitors.  
Your performance is judged primarily by **share price**.
"""
)

# --- Economy panel ---
st.sidebar.header("Economy (last quarter)")
econ = sim.economy
st.sidebar.metric("Year / Quarter", f"Y{econ.year} Q{econ.quarter}")
st.sidebar.metric("GDP index", f"{econ.gdp:0.1f}")
st.sidebar.metric("Unemployment %", f"{econ.unemployment:0.1f}%")
st.sidebar.metric("Central Bank Rate (next qtr)", f"{econ.cb_rate:0.2f}%")
st.sidebar.metric("Material price (per 1000 units)", f"£{econ.material_price:0.1f}")

if st.sidebar.button("Reset simulation", type="primary"):
    st.session_state.sim = Simulation(n_companies=4, seed=42)
    st.experimental_rerun()

player_company: CompanyState = sim.companies[0]

st.subheader(f"Your company: {player_company.name}")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Share price", f"£{player_company.share_price:0.2f}")
col_b.metric("Net worth", f"£{player_company.net_worth():,.0f}")
col_c.metric("Cash", f"£{player_company.cash:,.0f}")
col_d.metric("Employees (approx)", f"{player_company.total_employees():,}")

st.markdown("### 1. Marketing decisions")

with st.expander("Prices, credit, quality & product development", expanded=True):
    price_cols = st.columns(len(PRODUCTS))
    prices_home = {}
    prices_export = {}
    assembly_time = {}
    product_dev = {}

    for i, p in enumerate(PRODUCTS):
        with price_cols[i]:
            st.markdown(f"**{p}**")
            base_home = 100 + 20 * i
            prices_home[p] = st.number_input(
                f"Home price {p} (£/unit)",
                min_value=10.0,
                max_value=400.0,
                value=float(base_home),
                step=5.0,
                key=f"ph_{p}",
            )
            prices_export[p] = st.number_input(
                f"Export price {p} (£/unit)",
                min_value=10.0,
                max_value=400.0,
                value=float(base_home * 1.1),
                step=5.0,
                key=f"pe_{p}",
            )

            assembly_time[p] = st.number_input(
                f"Assembly time {p} (mins/unit)",
                min_value=MIN_ASSEMBLY_TIME[p],
                max_value=MIN_ASSEMBLY_TIME[p] * 2.0,
                value=float(MIN_ASSEMBLY_TIME[p] * 1.2),
                step=10.0,
                key=f"assy_{p}",
            )

            product_dev[p] = st.number_input(
                f"Product dev spend {p} (£000)",
                min_value=0.0,
                max_value=200.0,
                value=20.0,
                step=5.0,
                key=f"dev_{p}",
            ) * 1000.0

    credit_days = st.slider("Credit days offered to retailers", 15, 90, 30, 5)

with st.expander("Advertising & sales force allocation"):
    adv = {}
    st.markdown("#### Advertising spend (£000 per quarter)")
    adv_table = []
    for p in PRODUCTS:
        row = {"Product": p}
        for a in AREAS:
            val = st.number_input(
                f"Ads {p} in {a} (£000)",
                min_value=0.0,
                max_value=200.0,
                value=20.0 if a != "West" else 10.0,
                step=5.0,
                key=f"adv_{p}_{a}",
            )
            row[a] = val
            adv[(p, a)] = val * 1000.0
        adv_table.append(row)
    st.table(pd.DataFrame(adv_table))

    st.markdown("#### Salespeople allocation by area (existing sales force)")
    total_salespeople = player_company.salespeople
    st.info(f"You currently have **{total_salespeople}** salespeople.")
    sales_alloc = {}
    remaining = total_salespeople
    for i, a in enumerate(AREAS):
        if i == len(AREAS) - 1:
            val = remaining
        else:
            val = st.number_input(
                f"Salespeople in {a}",
                min_value=0,
                max_value=remaining,
                value=remaining // (len(AREAS) - i),
                step=1,
                key=f"sales_{a}",
            )
        sales_alloc[a] = val
        remaining -= val
    st.write("Final allocation:", sales_alloc)

st.markdown("### 2. Operations & production decisions")

with st.expander("Shift level, maintenance & materials", expanded=True):
    shift_level = st.radio("Shift level", [1, 2, 3], index=0, horizontal=True)
    maint_hours = st.number_input(
        "Contracted maintenance hours per machine",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=5.0,
    )
    materials_order_qty = st.number_input(
        "Materials order quantity (units) – for quarter after next",
        min_value=0.0,
        max_value=50_000.0,
        value=6_000.0,
        step=500.0,
    )

with st.expander("Delivery schedule (units to deliver next quarter)"):
    deliveries = {}
    for p in PRODUCTS:
        st.markdown(f"**{p}**")
        area_cols = st.columns(len(AREAS))
        for i, area in enumerate(AREAS):
            with area_cols[i]:
                val = st.number_input(
                    f"{area}",
                    min_value=0,
                    max_value=10_000,
                    value=0,
                    step=50,
                    key=f"del_{p}_{area}",
                )
                deliveries[(p, area)] = val
    
    st.markdown("**Total units to deliver:** " + str(sum(deliveries.values())))

st.markdown("### 3. Personnel decisions")

with st.expander("Salespeople", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        recruit_sales = st.number_input("Recruit salespeople", min_value=0, max_value=20, value=0, step=1)
    with col2:
        dismiss_sales = st.number_input("Dismiss salespeople", min_value=0, max_value=player_company.salespeople, value=0, step=1)
    with col3:
        train_sales = st.number_input("Train salespeople from unemployed", min_value=0, max_value=9, value=0, step=1)
    
    sales_salary = st.number_input(
        "Sales salary per quarter (£)",
        min_value=MIN_SALES_SALARY_PER_QUARTER,
        max_value=50_000.0,
        value=float(player_company.sales_salary if hasattr(player_company, 'sales_salary') else MIN_SALES_SALARY_PER_QUARTER),
        step=500.0,
    )
    sales_commission = st.number_input(
        "Sales commission (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.5,
    )

with st.expander("Assembly workers", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        recruit_assembly = st.number_input("Recruit assembly workers", min_value=0, max_value=50, value=0, step=1)
    with col2:
        dismiss_assembly = st.number_input("Dismiss assembly workers", min_value=0, max_value=player_company.assembly_workers, value=0, step=1)
    with col3:
        train_assembly = st.number_input("Train assembly workers from unemployed", min_value=0, max_value=9, value=0, step=1)
    
    assembly_wage_rate = st.number_input(
        "Assembly worker hourly wage rate (£)",
        min_value=ASSEMBLY_MIN_WAGE_RATE,
        max_value=50.0,
        value=float(ASSEMBLY_MIN_WAGE_RATE),
        step=0.50,
    )

st.markdown("### 4. Finance decisions")

col1, col2 = st.columns(2)
with col1:
    dividend_per_share = st.number_input(
        "Dividend per share (pence) - Q1 and Q3 only",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        disabled=(econ.quarter not in [1, 3]),
    )
    if econ.quarter not in [1, 3]:
        st.caption("Dividends can only be paid in Q1 and Q3")

with col2:
    management_budget = st.number_input(
        "Management budget (£)",
        min_value=MIN_MANAGEMENT_BUDGET,
        max_value=200_000.0,
        value=float(MIN_MANAGEMENT_BUDGET),
        step=5000.0,
    )

# Add missing constants
MIN_SALES_SALARY_PER_QUARTER = 2000
ASSEMBLY_MIN_WAGE_RATE = 8.50
MIN_MANAGEMENT_BUDGET = 40_000

# Submit decisions
if st.button("Submit Decisions and Run Quarter", type="primary", use_container_width=True):
    # Create Decisions object
    decisions = Decisions(
        prices_home=prices_home,
        prices_export=prices_export,
        credit_days=credit_days,
        assembly_time=assembly_time,
        advertising=adv,
        product_dev=product_dev,
        sales_allocation=sales_alloc,
        recruit_sales=recruit_sales,
        dismiss_sales=dismiss_sales,
        train_assembly=train_assembly,
        recruit_assembly=recruit_assembly,
        dismiss_assembly=dismiss_assembly,
        shift_level=shift_level,
        maintenance_hours_per_machine=maint_hours,
        deliveries=deliveries,
        dividend_per_share=dividend_per_share / 100.0,  # convert pence to pounds
        management_budget=management_budget,
        materials_order_qty=materials_order_qty,
    )
    
    # Run simulation
    reports = sim.step(decisions)
    
    st.success("Quarter completed! View results below.")
    st.balloons()

# Display last report if available
if player_company.last_report:
    st.markdown("### Latest Results")
    
    report = player_company.last_report
    
    # Key metrics
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Revenue", f"£{report.get('revenue', 0):,.0f}")
    with metrics_cols[1]:
        st.metric("Net Profit", f"£{report.get('net_profit', 0):,.0f}")
    with metrics_cols[2]:
        st.metric("EBITDA", f"£{report.get('ebitda', 0):,.0f}")
    with metrics_cols[3]:
        st.metric("Cash", f"£{report.get('cash', 0):,.0f}")
    
    # Detailed report expander
    with st.expander("Detailed Report", expanded=False):
        st.json(report)

# Display competitor information
st.markdown("### Competitor Information")
competitor_df = []
for i, comp in enumerate(sim.companies[1:], 1):
    competitor_df.append({
        "Company": comp.name,
        "Share Price": f"£{comp.share_price:.2f}",
        "Net Worth": f"£{comp.net_worth():,.0f}",
        "Employees": comp.total_employees(),
    })
if competitor_df:
    st.dataframe(pd.DataFrame(competitor_df), use_container_width=True)
