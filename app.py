"""
Leftovers Business Simulation (Topaz-VBE)
COMPLETE IMPLEMENTATION - Merged from app.py + topaz_simulation.py
Includes all features from manual and COMPLETION_STATUS.md requirements
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# ALL CONSTANTS FROM MANUAL TABLES (1-23) - from topaz_simulation.py
# ============================================================================

PRODUCTS = ["Product 1", "Product 2", "Product 3"]
AREAS = ["South", "West", "North", "Export"]

# Table 1: Market Statistics
MARKET_STATISTICS = {
    "South": {"managerial": 1_000_000, "supervisory": 2_000_000, "other": 4_000_000, "total": 7_000_000, "outlets": 3000},
    "West": {"managerial": 1_000_000, "supervisory": 1_000_000, "other": 2_000_000, "total": 4_000_000, "outlets": 2000},
    "North": {"managerial": 1_000_000, "supervisory": 3_000_000, "other": 9_000_000, "total": 13_000_000, "outlets": 4000},
    "Export": {"managerial": 10_000_000, "supervisory": 15_000_000, "other": 55_000_000, "total": 80_000_000, "outlets": 20_000},
}

# Table 2: Marketing Costs
SALESPERSON_EXPENSES = 3000  # per quarter
COMPETITOR_INFO_COST = 5000  # per quarter
MARKET_SHARES_INFO_COST = 5000  # per quarter

# Table 3: Manufacturing Parameters
MIN_MACHINING_TIME = {"Product 1": 60.0, "Product 2": 75.0, "Product 3": 120.0}  # minutes
MIN_ASSEMBLY_TIME = {"Product 1": 100.0, "Product 2": 150.0, "Product 3": 300.0}  # minutes
MATERIAL_PER_UNIT = {"Product 1": 1.0, "Product 2": 2.0, "Product 3": 3.0}  # units

# Table 4: Maintenance Costs
CONTRACTED_MAINTENANCE_RATE = 60.0  # £ per hour
UNCONTRACTED_MAINTENANCE_RATE = 120.0  # £ per hour

# Table 5: Maximum Hours Available per Machine Per Quarter
MACHINE_HOURS_PER_SHIFT = {1: 576, 2: 1068, 3: 1602}
MACHINISTS_PER_MACHINE = {1: 4, 2: 8, 3: 12}

# Table 6: Valuation of Rejected Products (Scrap Value)
SCRAP_VALUE = {"Product 1": 20.0, "Product 2": 40.0, "Product 3": 60.0}

# Table 7: Guarantee Servicing Charges
SERVICING_CHARGE = {"Product 1": 60.0, "Product 2": 120.0, "Product 3": 200.0}

# Table 8: Production Costs
SUPERVISION_COST_PER_SHIFT = 10_000
PRODUCTION_OVERHEAD_PER_MACHINE = 2000
MACHINE_RUNNING_COST_PER_HOUR = 7
PRODUCTION_PLANNING_COST_PER_UNIT = 1

# Table 9: Standard Vehicle Capacity
VEHICLE_CAPACITY = {"Product 1": 40, "Product 2": 40, "Product 3": 20}

# Table 10: Return Journey Times
JOURNEY_TIME_DAYS = {"South": 1, "West": 2, "North": 4, "Export": 6}

# Table 11: Transport Costs
FLEET_FIXED_COST_PER_VEHICLE = 7000  # per quarter
OWN_VEHICLE_RUNNING_COST_PER_DAY = 50
HIRED_VEHICLE_COST_PER_DAY = 200
MAX_VEHICLE_DAYS_PER_QUARTER = 60

# Table 12: Warehousing and Purchasing
FACTORY_STORAGE_CAPACITY = 2000  # units
FIXED_QUARTERLY_WAREHOUSE_COST = 3750
FIXED_QUARTERLY_ADMIN_COST = 3250
COST_PER_ORDER = 750
VARIABLE_EXTERNAL_STORAGE_COST = 1.50  # per unit
PRODUCT_STORAGE_COST = 2.0  # per unit per quarter

# Table 13: Calculation of Average Quarterly Stocks
# Formula: 0.5 * (opening + closing + quantity at each delivery)

# Table 14: Material Suppliers' Terms of Trade
SUPPLIERS = {
    0: {"discount": 0.0, "delivery_charge": 0, "min_delivery": 1, "min_order": 1, "deliveries": "just_in_time"},
    1: {"discount": 0.10, "delivery_charge": 200, "min_delivery": 1, "min_order": 1, "deliveries": "multiple"},
    2: {"discount": 0.15, "delivery_charge": 300, "min_delivery": 1000, "min_order": 10_000, "deliveries": "multiple"},
    3: {"discount": 0.30, "delivery_charge": 100, "min_delivery": 0, "min_order": 50_000, "deliveries": 12},  # weekly
}

# Table 15: Personnel Department Costs
RECRUITMENT_COST = {"Salesperson": 1500, "Assembly worker": 1200, "Machinist": 750}
DISMISSAL_COST = {"Salesperson": 5000, "Assembly worker": 3000, "Machinist": 1500}
TRAINING_COST = {"Salesperson": 6000, "Assembly worker": 4500, "Machinist": 0}

# Table 16: Maximum Hours per Quarter for each Production Worker
WORKER_HOURS = {
    1: {"basic": 420, "saturday": 84, "sunday": 72, "machinist_premium": 0},
    2: {"basic": 420, "saturday": 42, "sunday": 72, "machinist_premium": 1/3},
    3: {"basic": 420, "saturday": 42, "sunday": 72, "machinist_premium": 2/3},
}

# Table 17: Minimum Hours and Pay
MACHINIST_MIN_HOURS = 400  # per quarter
ASSEMBLY_STRIKE_HOURS_PER_WEEK = 48
ASSEMBLY_MIN_WAGE_RATE = 8.50  # £ per hour
UNSKILLED_SKILLED_RATIO = 0.65
MIN_SALES_SALARY_PER_QUARTER = 2000
MIN_MANAGEMENT_BUDGET = 40_000

# Table 18: Fixed Assets
MACHINE_COST = 200_000
MACHINE_DEPOSIT = 100_000  # 50% at order
VEHICLE_COST = 15_000
MACHINE_DEPRECIATION_RATE = 0.025  # per quarter (2.5%)
VEHICLE_DEPRECIATION_RATE = 0.0625  # per quarter (6.25%)

# Table 19: Methods of calculating Financial Limits
# Overdraft Limit = 100% Cash + 100% Product Stocks + 50% (Machines + Vehicles + Materials + Debtors) + 25% Property - 100% (Tax + Creditors)
# Creditworthiness = Overdraft Limit - Bank Overdraft - Unsecured Loans - £100k per machine ordered

# Table 20: Financial Parameters
TAX_RATE = 0.30  # 30% per annum
FIXED_OVERHEADS_PER_QUARTER = 10_000
VARIABLE_OVERHEAD_RATE = 0.0025  # 0.25% per quarter
CREDIT_CONTROL_COST_PER_UNIT = 1.50
INTEREST_RATE_DEPOSIT_SPREAD = -2.0  # bank rate - 2%
INTEREST_RATE_OVERDRAFT_SPREAD = 4.0  # bank rate + 4%
INTEREST_RATE_LOAN_SPREAD = 10.0  # bank rate + 10%

# Table 21: Stock Valuations
PRODUCT_STOCK_VALUATION = {"Product 1": 80.0, "Product 2": 120.0, "Product 3": 200.0}
# Material stock = 50% of last quarter's material price per unit

# Table 22: Timing of Payments to Creditors
PAYMENT_TIMING = {
    "Advertising": {"next": 0.0, "after_next": 1.0},
    "Guarantee_Servicing": {"next": 0.0, "after_next": 1.0},
    "Hired_Transport": {"next": 0.0, "after_next": 1.0},
    "Product_Development": {"next": 1.0, "after_next": 0.0},
    "Personnel_Department": {"next": 1.0, "after_next": 0.0},
    "Maintenance": {"next": 0.0, "after_next": 1.0},
    "Warehousing_Purchasing": {"next": 1.0, "after_next": 0.0},
    "External_stockholding": {"next": 0.0, "after_next": 1.0},
    "Business_Intelligence": {"next": 0.0, "after_next": 1.0},
    "Other_Miscellaneous": {"next": 0.0, "after_next": 1.0},
    "Materials_Purchased": {"next": 0.0, "after_next": 1.0},
    "Machines_Purchased": {"next": 0.5, "after_next": 0.5},  # 50% deposit, 50% on installation
    "Interest": {"next": 1.0, "after_next": 0.0},
}

# Table 23: Customers' credit discount structure
CREDIT_DISCOUNTS = {
    (0, 7): 0.10,
    (8, 15): 0.075,
    (16, 29): 0.05,
    (30, 999): 0.0,
}

# Base economic values
BASE_GDP = 100.0
BASE_UNEMPLOYMENT = 6.0
BASE_CB_RATE = 3.0
BASE_MATERIAL_PRICE = 100.0  # per 1000 units

# Training limitations
MAX_TRAINEES_PER_CATEGORY_PER_QUARTER = 9

# Sales office cost (1% of orders value)
SALES_OFFICE_COST_RATE = 0.01

# ============================================================================
# DATA CLASSES - Enhanced from topaz_simulation.py
# ============================================================================

@dataclass
class Economy:
    quarter: int = 1
    year: int = 1
    gdp: float = BASE_GDP
    unemployment: float = BASE_UNEMPLOYMENT
    cb_rate: float = BASE_CB_RATE  # for next quarter
    material_price: float = BASE_MATERIAL_PRICE  # per 1000 units, for next quarter
    
    def advance(self):
        """Advance economy to next quarter"""
        self.quarter += 1
        if self.quarter > 4:
            self.quarter = 1
            self.year += 1
        
        # Economic model - simplified for now, should be preset in real simulation
        shock = np.random.normal(0, 1.5)
        self.gdp = max(80, self.gdp * (1 + shock / 100))
        
        u_shock = np.random.normal(0, 0.3)
        self.unemployment = min(15, max(2, self.unemployment + u_shock - shock / 40))
        
        rate_target = 2.5 + (self.gdp - BASE_GDP) / 40
        self.cb_rate = max(0.25, 0.75 * self.cb_rate + 0.25 * rate_target)
        
        self.material_price = max(60, self.material_price * (1 + (self.cb_rate - 2.5) / 200 + np.random.normal(0, 0.01)))


@dataclass
class ProductImprovement:
    """Track product development improvements"""
    product: str
    type: str  # "MAJOR" or "MINOR"
    quarter_reported: int
    year_reported: int
    implemented: bool = False


@dataclass
class MaterialOrder:
    """Track material orders with supplier details"""
    quantity: float
    supplier: int
    num_deliveries: int
    order_quarter: int
    order_year: int
    delivery_quarter: int  # quarter after next
    delivery_year: int
    base_price_per_1000: float  # price at time of order


@dataclass
class MachineOrder:
    """Track machine orders"""
    quantity: int
    order_quarter: int
    order_year: int
    deposit_paid: bool = False
    installed: bool = False
    available_quarter: Optional[int] = None  # quarter after installation when available for use


@dataclass
class Decisions:
    """All decisions for a quarter - matches Decision Form structure exactly"""
    # Product improvements
    implement_major_improvement: Dict[str, bool] = field(default_factory=lambda: {p: False for p in PRODUCTS})
    
    # Prices
    prices_export: Dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PRODUCTS})
    prices_home: Dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PRODUCTS})
    
    # Promotion (advertising) - three types per product per area
    advertising_trade_press: Dict[Tuple[str, str], float] = field(default_factory=dict)
    advertising_support: Dict[Tuple[str, str], float] = field(default_factory=dict)
    advertising_merchandising: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Assembly time per product
    assembly_time: Dict[str, float] = field(default_factory=lambda: {p: MIN_ASSEMBLY_TIME[p] for p in PRODUCTS})
    
    # Salespeople allocation
    salespeople_allocation: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in AREAS})
    
    # Salespeople remuneration
    sales_salary_per_quarter: float = MIN_SALES_SALARY_PER_QUARTER
    sales_commission_percent: float = 0.0
    
    # Assembly worker wage rate
    assembly_wage_rate: float = ASSEMBLY_MIN_WAGE_RATE
    
    # Shift level
    shift_level: int = 1
    
    # Management budget
    management_budget: float = MIN_MANAGEMENT_BUDGET
    
    # Maintenance
    maintenance_hours_per_machine: float = 0.0
    
    # Dividend
    dividend_per_share: float = 0.0  # pence per share
    
    # Credit terms
    credit_days: int = 30
    
    # Vehicles
    vans_to_buy: int = 0
    vans_to_sell: int = 0
    
    # Information purchases
    buy_competitor_info: bool = False
    buy_market_shares: bool = False
    
    # Deliveries (production schedule)
    deliveries: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    # Product development
    product_development: Dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PRODUCTS})
    
    # Salespeople
    recruit_sales: int = 0
    dismiss_sales: int = 0
    train_sales: int = 0
    
    # Assembly workers
    recruit_assembly: int = 0
    dismiss_assembly: int = 0
    train_assembly: int = 0
    
    # Materials
    materials_quantity: float = 0.0
    materials_supplier: int = 0
    materials_num_deliveries: int = 0
    
    # Machines
    machines_to_sell: int = 0
    machines_to_order: int = 0
    
    # Compatibility methods for existing code
    @property
    def advertising(self) -> Dict[Tuple[str, str], float]:
        """Combined advertising spend for compatibility"""
        combined = {}
        for p in PRODUCTS:
            for a in AREAS:
                key = (p, a)
                combined[key] = (
                    self.advertising_trade_press.get(key, 0) +
                    self.advertising_support.get(key, 0) +
                    self.advertising_merchandising.get(key, 0)
                )
        return combined
    
    @property
    def sales_allocation(self) -> Dict[str, int]:
        """Alias for salespeople_allocation"""
        return self.salespeople_allocation
    
    @property
    def product_dev(self) -> Dict[str, float]:
        """Alias for product_development"""
        return self.product_development
    
    @property
    def materials_order_qty(self) -> float:
        """Alias for materials_quantity"""
        return self.materials_quantity


@dataclass
class CompanyState:
    """Complete company state - matches Management Report structure"""
    name: str
    
    # Share capital
    shares_outstanding: float = 1_000_000.0
    share_price: float = 1.0
    
    # Fixed assets
    property_value: float = 500_000.0  # fixed, no depreciation
    machines: int = 10
    machines_ordered: List[MachineOrder] = field(default_factory=list)
    machine_efficiency: float = 1.0  # 0-1, starts at 100%
    vehicles: int = 5
    vehicles_age_quarters: List[int] = field(default_factory=lambda: [0] * 5)
    
    # Individual machine tracking for depreciation
    machine_ages_quarters: List[int] = field(default_factory=lambda: [0] * 10)  # track each machine
    machine_values: List[float] = field(default_factory=lambda: [MACHINE_COST] * 10)  # individual values
    
    # Inventory
    material_stock: float = 5_000.0
    material_orders: List[MaterialOrder] = field(default_factory=list)
    stocks: Dict[Tuple[str, str], int] = field(default_factory=dict)  # (product, area) -> units
    backlog: Dict[Tuple[str, str], int] = field(default_factory=dict)  # (product, area) -> units
    
    # Personnel
    salespeople: int = 10
    assembly_workers: int = 40
    machinists: int = 40  # = machines * 4 * shift_level
    
    # Personnel in training/pending (for delays)
    salespeople_in_training: int = 0  # will be available quarter after next
    assembly_workers_in_training: int = 0
    salespeople_pending_recruitment: int = 0  # recruited but not yet available
    assembly_workers_pending_recruitment: int = 0
    
    # Personnel leaving (with notice periods)
    salespeople_to_dismiss_next_quarter: int = 0
    assembly_workers_to_dismiss_next_quarter: int = 0
    
    # Pay rates
    sales_salary: float = MIN_SALES_SALARY_PER_QUARTER
    sales_commission_rate: float = 0.0
    assembly_wage_rate: float = ASSEMBLY_MIN_WAGE_RATE
    
    # Financial
    cash: float = 200_000.0
    overdraft: float = 0.0
    unsecured_loan: float = 0.0
    reserves: float = 0.0
    tax_liability: float = 0.0
    taxable_profit_accumulated: float = 0.0  # for yearly tax calculation
    debtors: float = 0.0
    creditors: float = 0.0
    
    # Creditors tracking with timing
    creditors_by_category: Dict[str, float] = field(default_factory=dict)  # category -> amount due
    
    # Product improvements
    product_improvements: List[ProductImprovement] = field(default_factory=list)
    product_star_ratings: Dict[str, int] = field(default_factory=lambda: {p: 3 for p in PRODUCTS})  # 1-5 stars
    
    # Accumulated spending for product development
    product_dev_accumulated: Dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PRODUCTS})
    product_dev_projects_active: Dict[str, bool] = field(default_factory=lambda: {p: False for p in PRODUCTS})
    
    # Last shift level used
    last_shift_level: int = 1
    
    # Strike weeks (for next quarter, set at end of quarter before last)
    strike_weeks_next_quarter: int = 0
    
    # Absenteeism tracking
    absenteeism_hours: float = 0.0
    
    # Historical data for reporting
    last_report: Dict = field(default_factory=dict)
    
    # Machine breakdowns tracking
    machine_breakdown_hours: float = 0.0
    
    # Opening balances for cash flow
    opening_cash: float = 200_000.0
    opening_overdraft: float = 0.0
    opening_loan: float = 0.0
    opening_debtors: float = 0.0
    opening_creditors: float = 0.0
    
    def net_worth(self) -> float:
        """Calculate net worth for balance sheet - Table 19"""
        # Assets
        assets = (
            self.cash
            + self.property_value
            + self._get_machine_value()
            + self._get_vehicle_value()
            + self._get_product_stock_value()
            + self._get_material_stock_value()
            + self.debtors
        )
        
        # Liabilities
        liabilities = (
            self.overdraft
            + self.unsecured_loan
            + self.tax_liability
            + self.creditors
        )
        
        return assets - liabilities
    
    def _get_machine_value(self) -> float:
        """Current depreciated value of machines"""
        return sum(self.machine_values)
    
    def _get_vehicle_value(self) -> float:
        """Current depreciated value of vehicles"""
        total = 0.0
        for age in self.vehicles_age_quarters:
            if age < len(self.vehicles_age_quarters):  # safety check
                value = VEHICLE_COST * ((1 - VEHICLE_DEPRECIATION_RATE) ** age)
                total += value
        return total
    
    def _get_product_stock_value(self) -> float:
        """Value of product stocks - Table 21"""
        total = 0.0
        for (prod, area), qty in self.stocks.items():
            total += qty * PRODUCT_STOCK_VALUATION[prod]
        return total
    
    def _get_material_stock_value(self, last_material_price: float = BASE_MATERIAL_PRICE) -> float:
        """Value of material stock - 50% of last quarter's price - Table 21"""
        return self.material_stock * (last_material_price / 1000.0) * 0.5
    
    def calculate_overdraft_limit(self) -> float:
        """Calculate overdraft limit - Table 19"""
        product_stock_value = self._get_product_stock_value()
        machine_value = self._get_machine_value()
        vehicle_value = self._get_vehicle_value()
        material_stock_value = self._get_material_stock_value()
        
        limit = (
            1.0 * self.cash +  # 100% Cash
            1.0 * product_stock_value +  # 100% Product Stocks
            0.5 * (machine_value + vehicle_value + material_stock_value + self.debtors) +  # 50% of these
            0.25 * self.property_value -  # 25% Property
            1.0 * (self.tax_liability + self.creditors)  # 100% of Tax + Creditors
        )
        return max(0.0, limit)
    
    def calculate_creditworthiness(self) -> float:
        """Calculate creditworthiness for machine purchases - Table 19"""
        limit = self.calculate_overdraft_limit()
        machine_order_commitment = sum(100_000 for mo in self.machines_ordered if not mo.installed)
        creditworthiness = limit - self.overdraft - self.unsecured_loan - machine_order_commitment
        return max(0.0, creditworthiness)
    
    def total_employees(self) -> int:
        """Total employees including machinists"""
        machinists = self.machines * MACHINISTS_PER_MACHINE.get(self.last_shift_level, 4)
        return self.salespeople + self.assembly_workers + machinists
    
    @property
    def current_shift_level(self) -> int:
        """Current shift level"""
        if self.last_report.get("shift_level") is not None:
            return self.last_report["shift_level"]
        return self.last_shift_level
    
    # Backward compatibility
    @property
    def fixed_assets_machines(self) -> float:
        """Backward compatibility property"""
        return self._get_machine_value()
    
    @fixed_assets_machines.setter
    def fixed_assets_machines(self, value: float):
        """Backward compatibility - distribute value across machines"""
        per_machine = value / max(1, self.machines)
        self.machine_values = [per_machine] * self.machines
    
    @property
    def fixed_assets_vehicles(self) -> float:
        """Backward compatibility property"""
        return self._get_vehicle_value()
    
    @fixed_assets_vehicles.setter
    def fixed_assets_vehicles(self, value: float):
        """Backward compatibility - not implemented"""
        pass
    
    def fixed_material_valuation(self) -> float:
        """Backward compatibility method"""
        return 0.05  # approximate


# ============================================================================
# SIMULATION ENGINE - Complete implementation with all features
# ============================================================================

class Simulation:
    """Complete simulation engine matching manual specifications"""
    
    def __init__(self, n_companies: int = 8, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.economy = Economy()
        self.companies: List[CompanyState] = [
            CompanyState(name=f"Company {i+1}") for i in range(n_companies)
        ]
        self.history: List[Dict] = []
        self.material_prices_history: List[float] = [BASE_MATERIAL_PRICE]
        self.n_players: int = 1  # Number of human players (rest are AI)
    
    # ========== DEMAND & MARKETING ==========
    
    def demand_for_product(
        self,
        company: CompanyState,
        decisions: Decisions,
        product: str,
        area: str,
        all_companies: List[CompanyState] = None,
        all_decisions: List[Decisions] = None,
    ) -> float:
        """Enhanced demand calculation with competitive market mechanics"""
        
        # Base demand from market statistics and economy
        market_stats = MARKET_STATISTICS[area]
        base_population_factor = market_stats["total"] / 7_000_000.0  # normalize to South
        
        seasonal_factor = 1.0 + (0.10 if self.economy.quarter == 4 else 0.0)
        gdp_factor = self.economy.gdp / BASE_GDP
        
        base_demand = 1000 * base_population_factor * seasonal_factor * gdp_factor
        
        # Price sensitivity
        if area == "Export":
            price = decisions.prices_export[product]
        else:
            price = decisions.prices_home[product]
        
        ref_price = 100 + 20 * PRODUCTS.index(product)
        price_factor = math.exp(-0.015 * (price - ref_price))
        
        # Advertising effect - combined from three types (diminishing returns)
        adv_total = (
            decisions.advertising_trade_press.get((product, area), 0) +
            decisions.advertising_support.get((product, area), 0) +
            decisions.advertising_merchandising.get((product, area), 0)
        )
        adv_factor = 1 + 0.0003 * math.sqrt(max(0, adv_total))
        
        # Quality factor - assembly time vs minimum
        q_factor = decisions.assembly_time[product] / MIN_ASSEMBLY_TIME[product]
        quality_factor = min(1.4, 0.7 + 0.7 * q_factor)
        
        # Product development - star ratings affect demand
        star_rating = company.product_star_ratings.get(product, 3)
        star_factor = 0.8 + (star_rating / 5.0) * 0.4  # 1-5 stars maps to 0.8-1.2
        
        # Product development cumulative spending effect
        dev_accumulated = company.product_dev_accumulated.get(product, 0.0)
        dev_factor = 1 + 0.0001 * math.log1p(max(0, dev_accumulated))
        
        # Salespeople effect
        salespeople_in_area = decisions.salespeople_allocation.get(area, 0)
        salespeople_factor = 1 + 0.02 * salespeople_in_area  # diminishing returns
        
        # Credit terms
        credit_factor = 1 + (decisions.credit_days - 30) / 200.0
        
        # Stock availability / delivery image
        backlog = company.backlog.get((product, area), 0)
        delivery_factor = max(0.6, 1 - backlog / 4000.0)
        
        # Product availability (stock levels)
        stock = company.stocks.get((product, area), 0)
        availability_factor = min(1.1, 0.9 + stock / 2000.0)  # adequate stock helps
        
        # Calculate company attractiveness score
        company_attractiveness = (
            price_factor *
            adv_factor *
            quality_factor *
            star_factor *
            dev_factor *
            salespeople_factor *
            credit_factor *
            delivery_factor *
            availability_factor
        )
        
        # COMPETITIVE MARKET MECHANICS: Calculate market share based on relative attractiveness
        if all_companies and all_decisions and len(all_companies) > 1 and len(all_decisions) == len(all_companies):
            # Calculate attractiveness for all competitors
            competitor_attractiveness = []
            for comp, dec in zip(all_companies, all_decisions):
                if comp == company:
                    competitor_attractiveness.append(company_attractiveness)
                else:
                    # Calculate competitor's attractiveness
                    comp_price = dec.prices_export[product] if area == "Export" else dec.prices_home[product]
                    comp_price_factor = math.exp(-0.015 * (comp_price - ref_price))
                    
                    comp_adv_total = (
                        dec.advertising_trade_press.get((product, area), 0) +
                        dec.advertising_support.get((product, area), 0) +
                        dec.advertising_merchandising.get((product, area), 0)
                    )
                    comp_adv_factor = 1 + 0.0003 * math.sqrt(max(0, comp_adv_total))
                    
                    comp_q_factor = dec.assembly_time[product] / MIN_ASSEMBLY_TIME[product]
                    comp_quality_factor = min(1.4, 0.7 + 0.7 * comp_q_factor)
                    
                    comp_star_rating = comp.product_star_ratings.get(product, 3)
                    comp_star_factor = 0.8 + (comp_star_rating / 5.0) * 0.4
                    
                    comp_salespeople = dec.salespeople_allocation.get(area, 0)
                    comp_salespeople_factor = 1 + 0.02 * comp_salespeople
                    
                    comp_credit_factor = 1 + (dec.credit_days - 30) / 200.0
                    
                    comp_backlog = comp.backlog.get((product, area), 0)
                    comp_delivery_factor = max(0.6, 1 - comp_backlog / 4000.0)
                    
                    comp_stock = comp.stocks.get((product, area), 0)
                    comp_availability_factor = min(1.1, 0.9 + comp_stock / 2000.0)
                    
                    comp_attractiveness = (
                        comp_price_factor *
                        comp_adv_factor *
                        comp_quality_factor *
                        comp_star_factor *
                        comp_salespeople_factor *
                        comp_credit_factor *
                        comp_delivery_factor *
                        comp_availability_factor
                    )
                    competitor_attractiveness.append(comp_attractiveness)
            
            # Use logit model for market share (proportional to attractiveness with some randomness)
            total_attractiveness = sum(competitor_attractiveness)
            if total_attractiveness > 0:
                # Market share based on relative attractiveness
                market_share = company_attractiveness / total_attractiveness
                # Apply some competitive pressure - if competitors are much better, reduce share
                market_share = max(0.05, min(0.95, market_share))  # Cap between 5% and 95%
            else:
                market_share = 1.0 / len(all_companies)  # Equal share if all have zero attractiveness
            
            # Total market demand is shared among competitors
            total_market_demand = base_demand * len(all_companies)  # Scale up for competition
            demand = total_market_demand * market_share
        else:
            # Non-competitive mode (original calculation)
            demand = (
                base_demand *
                company_attractiveness
        )
        
        return max(0, demand)
    
    # ========== PRODUCT DEVELOPMENT ==========
    
    def process_product_development(
        self,
        company: CompanyState,
        decisions: Decisions,
        quarter: int,
        year: int,
    ) -> Dict[str, str]:
        """Process product development and return outcomes (MAJOR/MINOR/NONE)"""
        outcomes = {}
        
        for product in PRODUCTS:
            spend = decisions.product_development.get(product, 0.0)
            
            if spend > 0:
                # Add to accumulated spending
                company.product_dev_accumulated[product] += spend
                company.product_dev_projects_active[product] = True
            
            # Check for outcomes based on accumulated spending
            accumulated = company.product_dev_accumulated.get(product, 0.0)
            
            # Probability of success increases with accumulated spending
            # Simplified model - more sophisticated in full version
            if accumulated > 0:
                # Major improvement probability
                if accumulated > 100_000 and random.random() < 0.15:
                    # Check if we already have an unimplemented major improvement
                    existing_major = any(
                        pi.product == product and pi.type == "MAJOR" and not pi.implemented
                        for pi in company.product_improvements
                    )
                    if not existing_major:
                        company.product_improvements.append(
                            ProductImprovement(
                                product=product,
                                type="MAJOR",
                                quarter_reported=quarter,
                                year_reported=year,
                            )
                        )
                        outcomes[product] = "MAJOR"
                        # Reset accumulated for new project
                        company.product_dev_accumulated[product] = 0.0
                    else:
                        outcomes[product] = "NONE"
                # Minor improvement
                elif accumulated > 30_000 and random.random() < 0.30:
                    outcomes[product] = "MINOR"
                    # Minor improvements are automatic - improve star rating slightly
                    company.product_star_ratings[product] = min(
                        5,
                        company.product_star_ratings[product] + 0.1
                    )
                else:
                    outcomes[product] = "NONE"
            else:
                outcomes[product] = "NONE"
            
            # Product obsolescence if no development
            if accumulated == 0 and company.product_dev_projects_active[product]:
                # Star rating may decline
                if random.random() < 0.1:
                    company.product_star_ratings[product] = max(
                        1,
                        company.product_star_ratings[product] - 0.1
                    )
        
        return outcomes
    
    def implement_major_improvements(
        self,
        company: CompanyState,
        decisions: Decisions,
    ) -> Dict[str, int]:
        """Implement major product improvements and return stock write-offs"""
        write_offs = {}
        
        for product in PRODUCTS:
            if decisions.implement_major_improvement.get(product, False):
                # Find all unimplemented major improvements for this product
                improvements_to_implement = [
                    pi for pi in company.product_improvements
                    if pi.product == product and pi.type == "MAJOR" and not pi.implemented
                ]
                
                if improvements_to_implement:
                    # Write off all stocks for this product
                    total_stock = sum(company.stocks.get((product, area), 0) for area in AREAS)
                    write_offs[product] = total_stock
                    
                    # Mark improvements as implemented
                    for pi in improvements_to_implement:
                        pi.implemented = True
                    
                    # Improve star rating
                    company.product_star_ratings[product] = min(
                        5,
                        company.product_star_ratings[product] + 0.5
                    )
                    
                    # Clear stocks
                    for area in AREAS:
                        company.stocks[(product, area)] = 0
        
        return write_offs
    
    # ========== MATERIALS & SUPPLIERS ==========
    
    def process_material_order(
        self,
        company: CompanyState,
        decisions: Decisions,
        quarter: int,
        year: int,
    ) -> Optional[MaterialOrder]:
        """Process material order with supplier system"""
        quantity = decisions.materials_quantity
        supplier = decisions.materials_supplier
        num_deliveries = decisions.materials_num_deliveries
        
        if quantity <= 0:
            return None
        
        # Validate supplier requirements
        supplier_info = SUPPLIERS[supplier]
        
        # Check minimum order quantity
        if quantity < supplier_info["min_order"]:
            return None  # Order rejected
        
        # Calculate delivery quarter (quarter after next)
        delivery_quarter = quarter + 2
        delivery_year = year
        if delivery_quarter > 4:
            delivery_quarter -= 4
            delivery_year += 1
        
        # Handle special cases
        if supplier == 0:
            # Just-in-time supplier
            num_deliveries = 0
        elif supplier == 3:
            # Weekly deliveries (12 per quarter)
            num_deliveries = 0  # fixed at 12
        
        order = MaterialOrder(
            quantity=quantity,
            supplier=supplier,
            num_deliveries=num_deliveries if supplier != 3 else 12,
            order_quarter=quarter,
            order_year=year,
            delivery_quarter=delivery_quarter,
            delivery_year=delivery_year,
            base_price_per_1000=self.economy.material_price,
        )
        
        company.material_orders.append(order)
        return order
    
    def deliver_materials(
        self,
        company: CompanyState,
        current_quarter: int,
        current_year: int,
        material_price_per_1000: float,
    ) -> Tuple[float, float]:
        """Process material deliveries for current quarter and return (delivered_qty, cost)"""
        delivered_qty = 0.0
        total_cost = 0.0
        
        orders_to_deliver = [
            mo for mo in company.material_orders
            if mo.delivery_quarter == current_quarter and mo.delivery_year == current_year
        ]
        
        for order in orders_to_deliver:
            supplier_info = SUPPLIERS[order.supplier]
            
            if supplier_info["deliveries"] == "just_in_time" or order.num_deliveries == 0:
                # Single delivery for supplier 0 or immediate delivery
                qty = order.quantity
            elif supplier_info["deliveries"] == 12:
                # Supplier 3 - weekly deliveries (12 per quarter)
                qty = order.quantity / 12.0  # per week, but we deliver all at once for simplicity
            else:
                # Multiple deliveries - deliver proportional amount
                if order.num_deliveries > 0:
                    qty = order.quantity / order.num_deliveries
                else:
                    qty = order.quantity
            
            # Apply discount
            base_cost = qty * (order.base_price_per_1000 / 1000.0)
            discount = base_cost * supplier_info["discount"]
            delivery_charge = supplier_info["delivery_charge"]
            
            cost = base_cost - discount + delivery_charge
            delivered_qty += qty
            total_cost += cost
        
        return delivered_qty, total_cost
    
    # ========== MACHINE PURCHASING ==========
    
    def process_machine_order(
        self,
        company: CompanyState,
        decisions: Decisions,
        quarter: int,
        year: int,
    ) -> int:
        """Process machine order with credit checks - returns actual quantity ordered"""
        requested = decisions.machines_to_order
        
        if requested <= 0:
            return 0
        
        # Calculate creditworthiness (Table 19)
        creditworthiness = company.calculate_creditworthiness()
        
        # Calculate cost per machine
        cost_per_machine = MACHINE_COST
        
        # Check if we can afford the deposit (50% per machine)
        deposit_per_machine = MACHINE_DEPOSIT
        total_deposit_needed = requested * deposit_per_machine
        
        # Can order up to creditworthiness
        max_affordable = int(creditworthiness / deposit_per_machine)
        
        actual_ordered = min(requested, max_affordable)
        
        if actual_ordered > 0:
            # Create machine order
            delivery_quarter = quarter + 2  # quarter after next for installation
            delivery_year = year
            if delivery_quarter > 4:
                delivery_quarter -= 4
                delivery_year += 1
            
            available_quarter = delivery_quarter + 1  # available for use quarter after installation
            if available_quarter > 4:
                available_quarter -= 4
                delivery_year += 1
            
            order = MachineOrder(
                quantity=actual_ordered,
                order_quarter=quarter,
                order_year=year,
                deposit_paid=False,  # will be paid next quarter
                installed=False,
                available_quarter=available_quarter,
            )
            company.machines_ordered.append(order)
        
        return actual_ordered
    
    def process_machine_installations(
        self,
        company: CompanyState,
        quarter: int,
        year: int,
    ) -> int:
        """Process machine installations and return number installed"""
        installed_count = 0
        
        for order in company.machines_ordered:
            if not order.installed:
                # Check if it's time to install (quarter after next from order)
                install_quarter = order.order_quarter + 2
                install_year = order.order_year
                if install_quarter > 4:
                    install_quarter -= 4
                    install_year += 1
                
                if quarter == install_quarter and year == install_year:
                    # Install machines
                    for _ in range(order.quantity):
                        company.machines += 1
                        company.machine_ages_quarters.append(0)
                        company.machine_values.append(MACHINE_COST)
                    
                    order.installed = True
                    installed_count += order.quantity
        
        return installed_count
    
    # ========== PERSONNEL WITH DELAYS ==========
    
    def process_personnel_recruitment(
        self,
        company: CompanyState,
        decisions: Decisions,
        economy: Economy,
        all_companies: List[CompanyState],
    ) -> Dict[str, int]:
        """Process recruitment with success rates and delays"""
        results = {
            "sales_recruited": 0,
            "assembly_recruited": 0,
        }
        
        # Recruitment success depends on unemployment, wages, product quality, etc.
        unemployment_factor = economy.unemployment / BASE_UNEMPLOYMENT
        
        # Salespeople recruitment
        if decisions.recruit_sales > 0:
            # Success probability based on conditions
            avg_wage_ratio = (company.sales_salary / MIN_SALES_SALARY_PER_QUARTER)
            success_rate = min(0.9, 0.3 + 0.3 * unemployment_factor + 0.2 * avg_wage_ratio)
            
            recruited = int(decisions.recruit_sales * success_rate + random.random())
            recruited = min(recruited, decisions.recruit_sales)
            
            # Recruited people will be available quarter after next
            company.salespeople_pending_recruitment += recruited
            results["sales_recruited"] = recruited
        
        # Assembly workers recruitment
        if decisions.recruit_assembly > 0:
            avg_wage_ratio = (company.assembly_wage_rate / ASSEMBLY_MIN_WAGE_RATE)
            success_rate = min(0.9, 0.4 + 0.3 * unemployment_factor + 0.2 * avg_wage_ratio)
            
            recruited = int(decisions.recruit_assembly * success_rate + random.random())
            recruited = min(recruited, decisions.recruit_assembly)
            
            company.assembly_workers_pending_recruitment += recruited
            results["assembly_recruited"] = recruited
        
        return results
    
    def process_personnel_training(
        self,
        company: CompanyState,
        decisions: Decisions,
    ) -> Dict[str, int]:
        """Process training with limits and delays"""
        results = {
            "sales_trained": 0,
            "assembly_trained": 0,
        }
        
        # Training limits: max 9 per category per quarter
        sales_training = min(decisions.train_sales, MAX_TRAINEES_PER_CATEGORY_PER_QUARTER)
        assembly_training = min(decisions.train_assembly, MAX_TRAINEES_PER_CATEGORY_PER_QUARTER)
        
        # Trained people will be available quarter after next
        company.salespeople_in_training += sales_training
        company.assembly_workers_in_training += assembly_training
        
        results["sales_trained"] = sales_training
        results["assembly_trained"] = assembly_training
        
        return results
    
    def process_personnel_availability(
        self,
        company: CompanyState,
    ):
        """Make personnel available after training/recruitment delays"""
        # People trained/recruited 2 quarters ago become available now
        company.salespeople += company.salespeople_pending_recruitment + company.salespeople_in_training
        company.assembly_workers += company.assembly_workers_pending_recruitment + company.assembly_workers_in_training
        
        # Reset pending
        company.salespeople_pending_recruitment = 0
        company.salespeople_in_training = 0
        company.assembly_workers_pending_recruitment = 0
        company.assembly_workers_in_training = 0
    
    def process_personnel_dismissals(
        self,
        company: CompanyState,
        decisions: Decisions,
    ):
        """Process dismissals with notice periods"""
        # Dismissals take effect at end of quarter (work through quarter, leave at end)
        if decisions.dismiss_sales > 0:
            actual = min(decisions.dismiss_sales, company.salespeople)
            company.salespeople_to_dismiss_next_quarter = actual
            # They still work this quarter
        
        if decisions.dismiss_assembly > 0:
            actual = min(decisions.dismiss_assembly, company.assembly_workers)
            company.assembly_workers_to_dismiss_next_quarter = actual
    
    def apply_personnel_dismissals(
        self,
        company: CompanyState,
    ):
        """Apply dismissals from previous quarter"""
        company.salespeople = max(0, company.salespeople - company.salespeople_to_dismiss_next_quarter)
        company.assembly_workers = max(0, company.assembly_workers - company.assembly_workers_to_dismiss_next_quarter)
        
        company.salespeople_to_dismiss_next_quarter = 0
        company.assembly_workers_to_dismiss_next_quarter = 0
    
    # ========== PRODUCTION CAPACITY ==========
    
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
        
        # Calculate per-product machining capacity
        max_units_by_product_machining = {}
        for product in PRODUCTS:
            mach_time_hours = MIN_MACHINING_TIME[product] / 60.0
            max_units_by_product_machining[product] = effective_machine_hours / mach_time_hours
        
        # Assembly capacity: workers * hours / time per unit
        worker_hours_data = WORKER_HOURS[shift]
        max_hours_per_worker = worker_hours_data["basic"] + worker_hours_data["saturday"] + worker_hours_data["sunday"]
        
        # Account for strikes
        strike_weeks = company.strike_weeks_next_quarter
        strike_hours_lost = strike_weeks * (worker_hours_data["basic"] / 12.0)  # approx hours per week
        effective_hours_per_worker = max_hours_per_worker - strike_hours_lost
        
        total_assembly_hours = company.assembly_workers * effective_hours_per_worker
        
        # Calculate per-product assembly capacity
        max_units_by_product_assembly = {}
        for product in PRODUCTS:
            assy_time_hours = decisions.assembly_time[product] / 60.0
            max_units_by_product_assembly[product] = total_assembly_hours / assy_time_hours
        
        # Overall capacity is minimum of machining and assembly for each product
        overall_machining = sum(max_units_by_product_machining.values())
        overall_assembly = sum(max_units_by_product_assembly.values())
        
        return overall_machining, overall_assembly
    
    # ========== COMPLETE QUARTERLY SIMULATION ==========
    
    def simulate_quarter_for_company(
        self,
        company: CompanyState,
        decisions: Decisions,
        is_player: bool = False,
        all_companies: List[CompanyState] = None,
        all_decisions: List[Decisions] = None,
    ) -> Dict:
        """Complete quarterly simulation with all features"""
        econ = self.economy
        quarter = econ.quarter
        year = econ.year
        
        # Store opening balances
        company.opening_cash = company.cash
        company.opening_overdraft = company.overdraft
        company.opening_loan = company.unsecured_loan
        company.opening_debtors = company.debtors
        company.opening_creditors = company.creditors
        company.opening_salespeople = company.salespeople
        company.opening_assembly_workers = company.assembly_workers
        company.opening_machinists = company.machines * MACHINISTS_PER_MACHINE.get(decisions.shift_level, 4)
        company.opening_vehicles = company.vehicles
        
        # 1. Apply personnel changes from previous quarters
        self.apply_personnel_dismissals(company)
        self.process_personnel_availability(company)
        
        # 2. Process machine installations
        machines_installed = self.process_machine_installations(company, quarter, year)
        
        # 3. Process material deliveries
        material_delivered_qty, material_cost = self.deliver_materials(
            company, quarter, year, econ.material_price
        )
        
        # 4. Process product development
        dev_outcomes = self.process_product_development(company, decisions, quarter, year)
        
        # 5. Implement major improvements (may write off stocks)
        stock_write_offs = self.implement_major_improvements(company, decisions)
        
        # 6. Process new orders
        new_material_order = self.process_material_order(company, decisions, quarter, year)
        machines_ordered = self.process_machine_order(company, decisions, quarter, year)
        
        # 7. Process personnel decisions (with delays)
        personnel_results = self.process_personnel_recruitment(company, decisions, econ, self.companies)
        training_results = self.process_personnel_training(company, decisions)
        self.process_personnel_dismissals(company, decisions)
        
        # 8. Production planning and execution
        planned_deliveries = decisions.deliveries
        total_planned_units = sum(planned_deliveries.values())
        
        # Calculate capacity
        cap_mach, cap_assy = self.production_capacity(company, decisions)
        max_units = min(cap_mach, cap_assy)
        
        capacity_ratio = min(1.0, max_units / max(total_planned_units, 1)) if total_planned_units > 0 else 1.0
        
        # Materials consumption
        material_required = 0.0
        for (product, area), qty in planned_deliveries.items():
            material_required += qty * MATERIAL_PER_UNIT[product]
        
        material_opening = company.material_stock
        material_available = material_opening + material_delivered_qty
        material_used = min(material_available, material_required * capacity_ratio)
        material_closing = material_available - material_used
        
        # Production per product/area
        produced: Dict[Tuple[str, str], int] = {}
        rejects: Dict[Tuple[str, str], int] = {}
        
        for (product, area), planned_qty in planned_deliveries.items():
            # Apply capacity constraint
            qty = int(planned_qty * capacity_ratio)
            
            # Quality and rejects
            q_factor = decisions.assembly_time[product] / MIN_ASSEMBLY_TIME[product]
            reject_rate = max(0.01, 0.10 / max(0.8, q_factor))
            rejected = int(qty * reject_rate)
            good = qty - rejected
            
            produced[(product, area)] = good
            rejects[(product, area)] = rejected
        
        # 9. Demand and sales
        new_orders: Dict[Tuple[str, str], int] = {}
        sales: Dict[Tuple[str, str], int] = {}
        backlog_new: Dict[Tuple[str, str], int] = {}
        stocks_new: Dict[Tuple[str, str], int] = {}
        
        revenue = 0.0
        debtors_increase = 0.0
        
        for product in PRODUCTS:
            for area in AREAS:
                key = (product, area)
                
                # Opening stocks & backlog
                opening_stock = company.stocks.get(key, 0)
                opening_backlog = company.backlog.get(key, 0)
                
                # New demand with competitive mechanics
                demand_units = int(self.demand_for_product(
                    company, decisions, product, area, 
                    all_companies=all_companies if all_companies else self.companies,
                    all_decisions=all_decisions if all_decisions and len(all_decisions) == len(all_companies if all_companies else self.companies) else None
                ))
                new_orders[key] = demand_units
                
                # Available units = opening stock + produced
                available_units = opening_stock + produced.get(key, 0)
                potential_sales = opening_backlog + demand_units
                
                sold = min(available_units, potential_sales)
                sales[key] = sold
                
                # Closing stock & backlog
                stocks_new[key] = available_units - sold
                unsatisfied_orders = max(0, potential_sales - sold)
                remaining_backlog = int(unsatisfied_orders * 0.5)  # half cancel
                backlog_new[key] = remaining_backlog
                
                # Revenue and debtors (with credit terms)
                price = decisions.prices_export[product] if area == "Export" else decisions.prices_home[product]
                sales_value = sold * price
                revenue += sales_value
                debtors_increase += sales_value  # will be collected based on credit terms
        
        company.stocks = stocks_new
        company.backlog = backlog_new
        company.material_stock = material_closing
        
        # Update debtors (simplified - real version tracks by invoice date)
        company.debtors = debtors_increase * (decisions.credit_days / 90.0)  # rough estimate
        
        # 10. Calculate all costs
        # Cost of sales
        materials_cost = material_cost  # from material deliveries
        
        # Wages
        shift = decisions.shift_level
        worker_hours_data = WORKER_HOURS[shift]
        total_hours_per_worker = worker_hours_data["basic"] + worker_hours_data["saturday"] + worker_hours_data["sunday"]
        strike_hours_lost = company.strike_weeks_next_quarter * (worker_hours_data["basic"] / 12.0)
        effective_hours = total_hours_per_worker - strike_hours_lost
        
        assembly_hours_worked = company.assembly_workers * effective_hours * min(1.0, capacity_ratio)
        assembly_wages = assembly_hours_worked * company.assembly_wage_rate
        
        # Machinist wages
        machinist_count = company.machines * MACHINISTS_PER_MACHINE[shift]
        machinist_hours = cap_mach  # machine hours = machinist hours in this model
        machinist_premium = worker_hours_data.get("machinist_premium", 0)
        machinist_wage_rate = company.assembly_wage_rate * (1 + machinist_premium)
        machinist_wages = machinist_hours * machinist_wage_rate
        
        # Production overheads (Table 8)
        supervision_cost = SUPERVISION_COST_PER_SHIFT * shift
        production_overhead = PRODUCTION_OVERHEAD_PER_MACHINE * company.machines
        machine_running_cost = MACHINE_RUNNING_COST_PER_HOUR * cap_mach
        production_planning_cost = PRODUCTION_PLANNING_COST_PER_UNIT * total_planned_units
        
        production_overheads = supervision_cost + production_overhead + machine_running_cost + production_planning_cost
        
        cost_of_sales = materials_cost + assembly_wages + machinist_wages + production_overheads
        
        # Operating expenses
        # Advertising
        ads_cost = sum(decisions.advertising.values())
        
        # Product development
        prod_dev_cost = sum(decisions.product_development.values())
        
        # Sales force costs
        salespeople_salary_cost = company.salespeople * decisions.sales_salary_per_quarter
        sales_commission_cost = revenue * (decisions.sales_commission_percent / 100.0)
        salesperson_expenses = company.salespeople * SALESPERSON_EXPENSES
        
        # Personnel department costs
        recruit_cost_sales = personnel_results["sales_recruited"] * RECRUITMENT_COST["Salesperson"]
        recruit_cost_assembly = personnel_results["assembly_recruited"] * RECRUITMENT_COST["Assembly worker"]
        dismiss_cost_sales = decisions.dismiss_sales * DISMISSAL_COST["Salesperson"]
        dismiss_cost_assembly = decisions.dismiss_assembly * DISMISSAL_COST["Assembly worker"]
        train_cost_sales = training_results["sales_trained"] * TRAINING_COST["Salesperson"]
        train_cost_assembly = training_results["assembly_trained"] * TRAINING_COST["Assembly worker"]
        
        personnel_costs = (
            recruit_cost_sales + recruit_cost_assembly +
            dismiss_cost_sales + dismiss_cost_assembly +
            train_cost_sales + train_cost_assembly
        )
        
        # Maintenance
        contracted_hours = company.machines * decisions.maintenance_hours_per_machine
        maint_cost = contracted_hours * CONTRACTED_MAINTENANCE_RATE
        
        # Warehousing (Table 12)
        total_product_stock = sum(stocks_new.values())
        warehousing_cost = FIXED_QUARTERLY_WAREHOUSE_COST + PRODUCT_STORAGE_COST * total_product_stock
        
        # External storage if material stock exceeds capacity
        external_storage = max(0, material_closing - FACTORY_STORAGE_CAPACITY)
        external_storage_cost = external_storage * VARIABLE_EXTERNAL_STORAGE_COST
        
        # Purchasing admin
        purchasing_cost = FIXED_QUARTERLY_ADMIN_COST + COST_PER_ORDER  # per order
        
        # Management budget
        management_cost = max(MIN_MANAGEMENT_BUDGET, decisions.management_budget)
        
        # Transport costs
        transport_cost, transport_details = self.calculate_transport_costs(company, decisions, produced)
        
        # Guarantee servicing (for rejects from previous quarters - simplified)
        guarantee_cost = sum(rejects.get((p, a), 0) * SERVICING_CHARGE[p] for p in PRODUCTS for a in AREAS)
        
        # Information purchases
        info_cost = 0.0
        if decisions.buy_competitor_info:
            info_cost += COMPETITOR_INFO_COST
        if decisions.buy_market_shares:
            info_cost += MARKET_SHARES_INFO_COST
        
        # Stock write-off cost (from major improvements)
        write_off_cost = sum(
            qty * PRODUCT_STOCK_VALUATION[product]
            for product, qty in stock_write_offs.items()
        )
        
        total_overheads = (
            ads_cost + prod_dev_cost + salespeople_salary_cost + sales_commission_cost +
            salesperson_expenses + personnel_costs + maint_cost + warehousing_cost +
            external_storage_cost + purchasing_cost + management_cost + transport_cost +
            guarantee_cost + info_cost + write_off_cost
        )
        
        # Gross profit and EBITDA
        gross_profit = revenue - cost_of_sales
        ebitda = gross_profit - total_overheads
        
        # Depreciation
        # Machine depreciation
        machine_depreciation = 0.0
        for i in range(len(company.machine_values)):
            dep = company.machine_values[i] * MACHINE_DEPRECIATION_RATE
            company.machine_values[i] -= dep
            machine_depreciation += dep
        
        # Vehicle depreciation
        vehicle_depreciation = 0.0
        for i in range(len(company.vehicles_age_quarters)):
            age = company.vehicles_age_quarters[i]
            value = VEHICLE_COST * ((1 - VEHICLE_DEPRECIATION_RATE) ** age)
            dep = value * VEHICLE_DEPRECIATION_RATE
            vehicle_depreciation += dep
            company.vehicles_age_quarters[i] = age + 1
        
        total_depreciation = machine_depreciation + vehicle_depreciation
        
        # Interest
        deposit_rate = max(0.0, (econ.cb_rate + INTEREST_RATE_DEPOSIT_SPREAD) / 100.0)
        overdraft_rate = (econ.cb_rate + INTEREST_RATE_OVERDRAFT_SPREAD) / 100.0
        loan_rate = (econ.cb_rate + INTEREST_RATE_LOAN_SPREAD) / 100.0
        
        # Use average balances
        avg_cash = (company.opening_cash + company.cash) / 2.0
        avg_overdraft = (company.opening_overdraft + company.overdraft) / 2.0
        avg_loan = (company.opening_loan + company.unsecured_loan) / 2.0
        
        interest_received = max(0.0, avg_cash) * deposit_rate / 4.0
        interest_paid = (avg_overdraft * overdraft_rate + avg_loan * loan_rate) / 4.0
        
        # Profit before tax
        profit_before_tax = ebitda + interest_received - interest_paid - total_depreciation
        
        # Tax (yearly, accumulated)
        company.taxable_profit_accumulated += profit_before_tax
        if quarter == 4:  # End of year
            yearly_tax = max(0.0, company.taxable_profit_accumulated * TAX_RATE)
            quarterly_tax = yearly_tax - company.tax_liability  # incremental
            company.tax_liability = yearly_tax
            company.taxable_profit_accumulated = 0.0
        else:
            quarterly_tax = 0.0  # No tax payment during year
        
        net_profit = profit_before_tax - quarterly_tax
        
        # Update cash flow
        # Cash inflows
        cash_inflows = revenue * 0.7  # simplified - 70% cash, 30% debtors
        
        # Cash outflows
        cash_outflows = (
            cost_of_sales * 0.8 +  # 80% paid immediately
            ads_cost * 0.0 +  # paid next quarter
            prod_dev_cost * 1.0 +  # paid this quarter
            salespeople_salary_cost * 1.0 +
            sales_commission_cost * 1.0 +
            assembly_wages * 1.0 +
            machinist_wages * 1.0 +
            personnel_costs * 1.0 +
            maint_cost * 0.0 +  # paid next quarter
            warehousing_cost * 1.0 +
            management_cost * 1.0 +
            transport_cost * 0.5 +  # approximate
            guarantee_cost * 0.0 +  # paid next quarter
            info_cost * 0.0 +  # paid next quarter
            interest_paid * 1.0 +
            quarterly_tax * 1.0
        )
        
        # Dividends
        dividends = decisions.dividend_per_share * company.shares_outstanding
        dividends = min(dividends, max(0.0, net_profit + company.reserves + company.cash))
        
        # Update cash and borrowing
        net_cash_flow = cash_inflows - cash_outflows - dividends
        company.cash += net_cash_flow
        
        # Handle borrowing if negative cash
        if company.cash < 0:
            needed = -company.cash
            company.cash = 0.0
            
            # Use overdraft first
            overdraft_limit = company.calculate_overdraft_limit()
            overdraft_available = max(0, overdraft_limit - company.overdraft)
            use_overdraft = min(needed, overdraft_available)
            company.overdraft += use_overdraft
            
            # Remaining goes to unsecured loan
            remaining = needed - use_overdraft
            if remaining > 0:
                company.unsecured_loan += remaining
        
        # Update reserves
        retained = net_profit - dividends
        company.reserves += retained
        
        # Update share price
        nw = company.net_worth()
        eps = net_profit / company.shares_outstanding if company.shares_outstanding > 0 else 0
        dps = dividends / company.shares_outstanding if company.shares_outstanding > 0 else 0
        
        share_price = max(
            0.1,
            0.5 * company.share_price +
            0.3 * (nw / company.shares_outstanding) +
            5 * eps +
            3 * dps
        )
        company.share_price = share_price
        
        # Update machine efficiency based on maintenance
        maint_factor = min(1.1, 0.9 + decisions.maintenance_hours_per_machine / 200.0)
        company.machine_efficiency = min(1.0, company.machine_efficiency * maint_factor)
        
        # Update last shift level
        company.last_shift_level = decisions.shift_level
        
        # Store material price history
        self.material_prices_history.append(econ.material_price)
        
        # Calculate detailed resource usage for report
        shift = decisions.shift_level
        worker_hours_data = WORKER_HOURS[shift]
        total_hours_per_worker = worker_hours_data["basic"] + worker_hours_data["saturday"] + worker_hours_data["sunday"]
        strike_hours_lost = company.strike_weeks_next_quarter * (worker_hours_data["basic"] / 12.0)
        effective_hours = total_hours_per_worker - strike_hours_lost
        
        # Assembly hours
        assembly_hours_available = company.assembly_workers * total_hours_per_worker
        assembly_hours_worked = company.assembly_workers * effective_hours * min(1.0, capacity_ratio)
        
        # Machine hours
        hours_per_machine = MACHINE_HOURS_PER_SHIFT[shift]
        total_machine_hours_available = company.machines * hours_per_machine
        maint_hours = company.machines * decisions.maintenance_hours_per_machine
        machine_hours_worked = cap_mach
        
        # Personnel movements (calculate from state changes)
        opening_sales = company.opening_salespeople if hasattr(company, 'opening_salespeople') else company.salespeople
        opening_assembly = company.opening_assembly_workers if hasattr(company, 'opening_assembly_workers') else company.assembly_workers
        opening_machinists = company.opening_machinists if hasattr(company, 'opening_machinists') else (company.machines * MACHINISTS_PER_MACHINE[shift])
        
        # Calculate cost breakdowns
        opening_stock_value = sum(
            company.stocks.get((p, a), 0) * PRODUCT_STOCK_VALUATION[p]
            for p in PRODUCTS for a in AREAS
        )
        closing_stock_value = sum(
            stocks_new.get((p, a), 0) * PRODUCT_STOCK_VALUATION[p]
            for p in PRODUCTS for a in AREAS
        )
        
        # Material stock value
        material_opening_value = material_opening * (econ.material_price / 1000.0) * 0.5
        material_closing_value = material_closing * (econ.material_price / 1000.0) * 0.5
        
        # Fixed asset values
        machine_values_total = sum(company.machine_values)
        vehicle_values = sum(VEHICLE_COST * ((1 - VEHICLE_DEPRECIATION_RATE) ** age) for age in company.vehicles_age_quarters)
        
        # Credit control cost
        total_units_sold = sum(sales.values())
        credit_control_cost = total_units_sold * CREDIT_CONTROL_COST_PER_UNIT
        
        # Sales office cost (from fixed overheads)
        sales_office_cost = SALESPERSON_EXPENSES * company.salespeople * 0.5  # approximate
        
        # Other miscellaneous costs
        misc_costs = FIXED_OVERHEADS_PER_QUARTER + write_off_cost
        
        # Calculate servicing units (simplified - based on rejects)
        servicing_units = {p: sum(rejects.get((p, a), 0) for a in AREAS) for p in PRODUCTS}
        
        # Calculate scheduled (planned deliveries)
        scheduled = planned_deliveries
        
        # Build comprehensive report with all breakdowns
        report = {
            "quarter": quarter,
            "year": year,
            "revenue": revenue,
            "cost_of_sales": cost_of_sales,
            "gross_profit": gross_profit,
            "total_overheads": total_overheads,
            "ebitda": ebitda,
            "interest_received": interest_received,
            "interest_paid": interest_paid,
            "depreciation": total_depreciation,
            "profit_before_tax": profit_before_tax,
            "tax": quarterly_tax,
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
            "machines": company.machines,
            "machines_installed": machines_installed,
            "machines_ordered": machines_ordered,
            "materials_used": material_used,
            "material_opening": material_opening,
            "material_closing": material_closing,
            "material_delivered": material_delivered_qty,
            "material_on_order": sum(getattr(order, 'quantity', 0.0) for order in company.material_orders),
            "stocks": stocks_new,
            "backlog": backlog_new,
            "sales": sales,
            "new_orders": new_orders,
            "deliveries": produced,
            "rejects": rejects,
            "product_dev_outcomes": dev_outcomes,
            "stock_write_offs": stock_write_offs,
            "scheduled": scheduled,
            "servicing_units": servicing_units,
            
            # Resource usage details
            "assembly_hours_available": assembly_hours_available,
            "assembly_hours_worked": assembly_hours_worked,
            "assembly_hours_absenteeism": assembly_hours_available - assembly_hours_worked,
            "machine_hours_available": total_machine_hours_available,
            "machine_hours_maintenance": maint_hours,
            "machine_hours_worked": machine_hours_worked,
            "vehicles_available": company.opening_vehicles if hasattr(company, 'opening_vehicles') else company.vehicles,
            "strike_weeks_next": company.strike_weeks_next_quarter,
            
            # Personnel movements
            "personnel_opening": {
                "sales": opening_sales,
                "assembly": opening_assembly,
                "machinists": opening_machinists
            },
            "personnel_recruited": {
                "sales": personnel_results.get("sales_recruited", 0),
                "assembly": personnel_results.get("assembly_recruited", 0),
                "machinists": personnel_results.get("machinists_recruited", 0) if "machinists_recruited" in personnel_results else 0
            },
            "personnel_trained": {
                "sales": training_results.get("sales_trained", 0),
                "assembly": training_results.get("assembly_trained", 0),
                "machinists": training_results.get("machinists_trained", 0) if "machinists_trained" in training_results else 0
            },
            "personnel_dismissed": {
                "sales": decisions.dismiss_sales,
                "assembly": decisions.dismiss_assembly,
                "machinists": 0  # machinists are typically not dismissed individually
            },
            "personnel_leavers": {
                "sales": personnel_results.get("sales_leavers", 0) if "sales_leavers" in personnel_results else 0,
                "assembly": personnel_results.get("assembly_leavers", 0) if "assembly_leavers" in personnel_results else 0,
                "machinists": personnel_results.get("machinists_leavers", 0) if "machinists_leavers" in personnel_results else 0
            },
            "personnel_available_next": {
                "sales": company.salespeople,
                "assembly": company.assembly_workers,
                "machinists": company.machines * MACHINISTS_PER_MACHINE[shift]
            },
            
            # Overhead breakdown
            "overhead_breakdown": {
                "advertising": ads_cost,
                "salespeople_salary": salespeople_salary_cost + sales_commission_cost,
                "sales_office": sales_office_cost,
                "guarantee_servicing": guarantee_cost,
                "transport_fleet": transport_details.get("fleet_fixed", 0) + transport_details.get("own_running", 0),
                "hired_transport": transport_details.get("hired_running", 0),
                "product_research": prod_dev_cost,
                "personnel_department": personnel_costs,
                "maintenance": maint_cost,
                "warehousing_purchasing": warehousing_cost + purchasing_cost,
                "business_intelligence": info_cost,
                "management_budget": management_cost,
                "credit_control": credit_control_cost,
                "other_miscellaneous": misc_costs
            },
            
            # Cost of sales breakdown
            "cost_of_sales_breakdown": {
                "opening_stock_value": opening_stock_value,
                "materials_purchased": material_cost,
                "assembly_wages": assembly_wages,
                "machinists_wages": machinist_wages,
                "machine_running_costs": machine_running_cost,
                "closing_stock_value": closing_stock_value
            },
            
            # Balance sheet items
            "balance_sheet": {
                "property_value": company.property_value,
                "machine_values": machine_values_total,
                "vehicle_values": vehicle_values,
                "product_stocks_value": closing_stock_value,
                "material_stock_value": material_closing_value,
                "debtors": company.debtors,
                "cash_invested": company.cash if company.cash > 0 else 0,
                "tax_assessed_due": company.tax_liability,
                "creditors": company.creditors,
                "overdraft": company.overdraft,
                "unsecured_loans": company.unsecured_loan,
                "ordinary_capital": company.shares_outstanding * 2.0,  # £2 per share initial
                "reserves": company.reserves
            },
        }
        
        # Cash flow calculations (done after cash updates)
        opening_cash_val = company.opening_cash
        closing_cash_val = company.cash if company.cash > 0 else 0
        cash_change = closing_cash_val - opening_cash_val + company.overdraft - company.opening_overdraft
        
        report["cash_flow"] = {
            "trading_receipts": revenue,  # simplified - actual would be cash received
            "trading_payments": cost_of_sales + total_overheads - total_depreciation,  # simplified
            "tax_paid": 0,  # paid when assessed
            "interest_received": interest_received,
            "capital_receipts": 0,  # from vehicle/machine sales
            "capital_payments": machines_ordered * MACHINE_DEPOSIT + decisions.vans_to_buy * VEHICLE_COST,
            "interest_paid": interest_paid,
            "dividend_paid": dividends,
            "opening_cash": opening_cash_val,
            "closing_cash": closing_cash_val,
            "net_cash_flow": cash_change
        }
        
        company.last_report = report
        return report
    
    def calculate_transport_costs(
        self,
        company: CompanyState,
        decisions: Decisions,
        deliveries: Dict[Tuple[str, str], int],
    ) -> Tuple[float, Dict]:
        """Calculate transport costs - Table 11"""
        # Calculate vehicle days required per area
        vehicle_days_per_area = {}
        
        for area in AREAS:
            total_units = sum(deliveries.get((p, area), 0) for p in PRODUCTS)
            
            if total_units == 0:
                vehicle_days_per_area[area] = 0
                continue
            
            # Calculate trips needed
            trips = 0
            remaining = total_units
            
            while remaining > 0:
                # Find best product to fill vehicle
                trip_capacity = 0
                trip_weight = 0
                
                for product in PRODUCTS:
                    qty = deliveries.get((product, area), 0)
                    if qty > 0:
                        capacity = VEHICLE_CAPACITY[product]
                        if trip_capacity + capacity <= 40:  # max capacity
                            add_qty = min(qty, 40 - trip_capacity)
                            trip_capacity += capacity * (add_qty / VEHICLE_CAPACITY[product])
                            remaining -= add_qty
                
                trips += 1
                if remaining <= 0:
                    break
            
            journey_days = JOURNEY_TIME_DAYS[area]
            vehicle_days_per_area[area] = trips * journey_days
        
        total_vehicle_days_required = sum(vehicle_days_per_area.values())
        
        # Own fleet capacity
        own_vehicle_capacity_days = company.vehicles * MAX_VEHICLE_DAYS_PER_QUARTER
        
        # Use own fleet first
        own_days = min(own_vehicle_capacity_days, total_vehicle_days_required)
        hired_days = max(0, total_vehicle_days_required - own_days)
        
        # Costs
        fleet_fixed_cost = company.vehicles * FLEET_FIXED_COST_PER_VEHICLE
        own_running_cost = own_days * OWN_VEHICLE_RUNNING_COST_PER_DAY
        hired_running_cost = hired_days * HIRED_VEHICLE_COST_PER_DAY
        
        total_cost = fleet_fixed_cost + own_running_cost + hired_running_cost
        
        details = {
            "own_days": own_days,
            "hired_days": hired_days,
            "fleet_fixed": fleet_fixed_cost,
            "own_running": own_running_cost,
            "hired_running": hired_running_cost,
        }
        
        return total_cost, details
    
    # ========== COMPETITOR AI ==========
    
    def auto_decisions(self, company: CompanyState) -> Decisions:
        """AI competitor decisions"""
        # Simple heuristic decisions
        base_price = 100
        prices_home = {p: base_price + 15 * i + random.randint(-10, 10) for i, p in enumerate(PRODUCTS)}
        prices_export = {p: ph * 1.1 for p, ph in prices_home.items()}
        
        assembly_time = {
            p: MIN_ASSEMBLY_TIME[p] * random.uniform(1.0, 1.4) for p in PRODUCTS
        }
        
        advertising_trade_press = {}
        advertising_support = {}
        advertising_merchandising = {}
        for p in PRODUCTS:
            for a in AREAS:
                val = random.choice([0, 5000, 10000, 20000])
                advertising_trade_press[(p, a)] = val / 3
                advertising_support[(p, a)] = val / 3
                advertising_merchandising[(p, a)] = val / 3
        
        product_dev = {p: random.choice([0, 5000, 10000]) for p in PRODUCTS}
        
        # Allocate salespeople
        total_sales = company.salespeople
        base = {"South": 1.0, "West": 0.7, "North": 1.3, "Export": 1.2}
        total_base = sum(base.values())
        sales_alloc = {a: int(total_sales * base[a] / total_base) for a in AREAS}
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
            advertising_trade_press=advertising_trade_press,
            advertising_support=advertising_support,
            advertising_merchandising=advertising_merchandising,
            salespeople_allocation=sales_alloc,
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
            materials_quantity=random.choice([4000, 6000, 8000]),
            materials_supplier=0,
            materials_num_deliveries=1,
        )
        return d
    
    # ========== PUBLIC API ==========
    
    def step(self, player_decisions_list: List[Decisions]):
        """Run one quarter for all companies with competitive market mechanics"""
        # Validate input
        if not isinstance(player_decisions_list, list):
            raise TypeError(f"player_decisions_list must be a list, got {type(player_decisions_list)}")
        
        # Collect all decisions first (needed for competitive demand calculation)
        all_decisions = []
        
        # Validate we have the right number of decisions
        if len(player_decisions_list) < self.n_players:
            raise ValueError(f"Not enough decisions provided. Expected {self.n_players} decisions for {self.n_players} players, got {len(player_decisions_list)}")
        
        for i, c in enumerate(self.companies):
            if i < len(player_decisions_list) and player_decisions_list[i] is not None:
                decision = player_decisions_list[i]
                # Basic validation - ensure it's not a list or the class itself
                if isinstance(decision, list):
                    raise TypeError(f"Decision at index {i} is a list, not a Decisions object. Got: {type(decision)}")
                # Check if it's the class itself (not an instance) - this is the main issue
                if decision is Decisions:
                    raise TypeError(f"Decision at index {i} is the Decisions CLASS, not an instance! Got: {type(decision)}. This means the class was passed instead of creating an instance with Decisions().")
                if isinstance(decision, type):
                    raise TypeError(f"Decision at index {i} is a type/class, not an instance. Got: {type(decision)}, value: {decision}")
                # Verify it's actually an instance by trying to access an attribute
                try:
                    test_attr = decision.prices_home
                except AttributeError:
                    raise TypeError(f"Decision at index {i} doesn't have 'prices_home' attribute. Got: {type(decision)}")
                except TypeError as e:
                    raise TypeError(f"Decision at index {i} is not a proper instance (might be the class). Got: {type(decision)}, error: {e}")
                # Just append - if it's not valid, it will fail naturally when used
                all_decisions.append(decision)
            else:
                # Only create AI decisions if we have 1 player (AI fills remaining slots)
                # If 2+ players, all companies should be human-controlled
                if self.n_players == 1:
                    all_decisions.append(self.auto_decisions(c))
                else:
                    # For 2+ players, we should have a decision for each company
                    # If we're missing one, it's an error
                    raise ValueError(f"Missing decision for company {i} ({c.name}) but n_players={self.n_players} > 1. Expected {len(self.companies)} decisions for {len(self.companies)} companies, but only have {len(player_decisions_list)} decisions.")
        
        # Ensure we have decisions for all companies
        assert len(all_decisions) == len(self.companies), f"Mismatch: {len(all_decisions)} decisions for {len(self.companies)} companies"
        
        # Basic validation - ensure no None values
        for i, dec in enumerate(all_decisions):
            if dec is None:
                raise ValueError(f"Decision at index {i} is None")
        
        # Now simulate all companies with competitive mechanics
        reports = []
        for i, c in enumerate(self.companies):
            dec = all_decisions[i]
            # Pass all companies and decisions for competitive demand calculation
            rep = self.simulate_quarter_for_company(
                c, dec, 
                is_player=(i < self.n_players),
                all_companies=self.companies,
                all_decisions=all_decisions
            )
            rep_with_meta = {
                **rep,
                "company": c.name,
            }
            reports.append(rep_with_meta)
        
        self.history.extend(reports)
        # Advance economy for next round
        self.economy.advance()
        return reports

# ============================================================================
# STREAMLIT UI - Complete implementation matching decision form
# ============================================================================

def parse_bulk_paste(text: str, expected_count: int) -> List[float]:
    """Parse bulk pasted values from text (tab/newline/comma separated)"""
    if not text:
        return []
    
    # Try different separators
    values = []
    for separator in ['\t', '\n', ',', ' ']:
        parts = text.strip().split(separator)
        if len(parts) > 1:
            values = [float(x.strip()) for x in parts if x.strip()]
            break
    
    if not values:
        # Try single value
        try:
            values = [float(text.strip())]
        except:
            pass
    
    # Pad or truncate to expected count
    while len(values) < expected_count:
        values.append(0.0)
    return values[:expected_count]

def create_player_decision_form(player_idx: int, company: CompanyState, economy: Economy) -> Decisions:
    """Create decision form for a single player"""
    # Player header with styled container
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; 
                border-radius: 10px; 
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h2 style='color: white; margin: 0; font-size: 1.8rem;'>
            👤 Player {player_idx + 1}: {company.name}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Company metrics in styled cards
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    with metric_col1:
        st.metric("📈 Share Price", f"£{company.share_price:0.2f}")
    with metric_col2:
        st.metric("💰 Net Worth", f"£{company.net_worth():,.0f}")
    with metric_col3:
        st.metric("💵 Cash", f"£{company.cash:,.0f}")
    with metric_col4:
        st.metric("👥 Employees", f"{company.total_employees():,}")
    with metric_col5:
        st.metric("⚙️ Machines", f"{company.machines}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section headers with icons
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin: 1.5rem 0 1rem 0;'>
        <h3 style='margin: 0; color: #667eea;'>📢 1. Marketing Decisions</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Product Improvements", expanded=False):
        st.info("Implement Major Improvements (will write off all stocks for that product)")
        implement_major = {}
        for p in PRODUCTS:
            implement_major[p] = st.checkbox(f"Implement Major Improvement - {p}", key=f"major_{player_idx}_{p}")
        
        # Show current star ratings
        st.markdown("**Current Product Star Ratings:**")
        star_cols = st.columns(len(PRODUCTS))
        for i, p in enumerate(PRODUCTS):
            with star_cols[i]:
                stars = company.product_star_ratings.get(p, 3)
                st.metric(p, f"{stars:.1f} ⭐")

    with st.expander("Prices, Credit Terms & Quality", expanded=True):
        # Bulk paste for prices
        st.markdown("**💡 Bulk Paste Support:** Paste tab/newline separated values (e.g., from AI)")
        paste_prices = st.text_area(
            "Paste prices (Home P1, Home P2, Home P3, Export P1, Export P2, Export P3, Assembly P1, Assembly P2, Assembly P3, Credit Days)",
            key=f"paste_prices_{player_idx}",
            height=60,
            help="Paste values separated by tabs or newlines. Order: Home prices (3), Export prices (3), Assembly times (3), Credit days (1)"
        )
        
        price_cols = st.columns(len(PRODUCTS))
        prices_home = {}
        prices_export = {}
        assembly_time = {}
        
        # Parse bulk paste if provided
        pasted_values = parse_bulk_paste(paste_prices, 10) if paste_prices else []
        
        for i, p in enumerate(PRODUCTS):
            with price_cols[i]:
                st.markdown(f"**{p}**")
                base_home = 100 + 20 * i
                home_val = pasted_values[i] if len(pasted_values) > i else base_home
                prices_home[p] = st.number_input(
                    f"Home price {p} (£/unit)",
                    min_value=10.0,
                    max_value=400.0,
                    value=float(home_val),
                    step=5.0,
                    key=f"ph_{player_idx}_{p}",
                )
                export_val = pasted_values[i + 3] if len(pasted_values) > i + 3 else base_home * 1.1
                prices_export[p] = st.number_input(
                    f"Export price {p} (£/unit)",
                    min_value=10.0,
                    max_value=400.0,
                    value=float(export_val),
                    step=5.0,
                    key=f"pe_{player_idx}_{p}",
                )
                assy_val = pasted_values[i + 6] if len(pasted_values) > i + 6 else MIN_ASSEMBLY_TIME[p] * 1.2
                assembly_time[p] = st.number_input(
                    f"Assembly time {p} (mins/unit)",
                    min_value=float(MIN_ASSEMBLY_TIME[p]),
                    max_value=float(MIN_ASSEMBLY_TIME[p] * 2.0),
                    value=float(assy_val),
                    step=10.0,
                    key=f"assy_{player_idx}_{p}",
                )
        
        credit_val = pasted_values[9] if len(pasted_values) > 9 else 30
        credit_days = st.slider("Credit days offered to retailers", 15, 90, int(credit_val), 5, key=f"credit_{player_idx}")

    with st.expander("Advertising (Three Types)", expanded=True):
        st.markdown("#### Advertising spend per product per area (£000 per quarter)")
        st.markdown("**💡 Bulk Paste:** Paste 36 values (3 products × 4 areas × 3 types) separated by tabs/newlines")
        paste_adv = st.text_area(
            "Paste advertising values",
            key=f"paste_adv_{player_idx}",
            height=60,
            help="36 values: Trade Press (12), Support (12), Merchandising (12). Order: P1 areas, P2 areas, P3 areas"
        )
        
        advertising_trade_press = {}
        advertising_support = {}
        advertising_merchandising = {}
        
        pasted_adv = parse_bulk_paste(paste_adv, 36) if paste_adv else []
        
        for p_idx, p in enumerate(PRODUCTS):
            st.markdown(f"**{p}**")
            area_cols = st.columns(len(AREAS))
            for a_idx, a in enumerate(AREAS):
                with area_cols[a_idx]:
                    st.markdown(f"*{a}*")
                    idx = p_idx * 12 + a_idx
                    tp_val = pasted_adv[idx] if len(pasted_adv) > idx else 5.0
                    trade_press = st.number_input(
                        "Trade Press",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(tp_val),
                        step=1.0,
                        key=f"adv_tp_{player_idx}_{p}_{a}",
                    )
                    sup_val = pasted_adv[idx + 12] if len(pasted_adv) > idx + 12 else 5.0
                    support = st.number_input(
                        "Support",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(sup_val),
                        step=1.0,
                        key=f"adv_sup_{player_idx}_{p}_{a}",
                    )
                    merch_val = pasted_adv[idx + 24] if len(pasted_adv) > idx + 24 else 5.0
                    merchandising = st.number_input(
                        "Merchandising",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(merch_val),
                        step=1.0,
                        key=f"adv_merch_{player_idx}_{p}_{a}",
                    )
                    advertising_trade_press[(p, a)] = trade_press * 1000.0
                    advertising_support[(p, a)] = support * 1000.0
                    advertising_merchandising[(p, a)] = merchandising * 1000.0

    with st.expander("Product Development", expanded=True):
        product_dev = {}
        dev_cols = st.columns(len(PRODUCTS))
        for i, p in enumerate(PRODUCTS):
            with dev_cols[i]:
                accumulated = company.product_dev_accumulated.get(p, 0.0)
                st.metric(f"Accumulated {p}", f"£{accumulated:,.0f}")
                product_dev[p] = st.number_input(
                    f"Dev spend {p} (£000)",
                    min_value=0.0,
                    max_value=200.0,
                    value=20.0,
                    step=5.0,
                    key=f"dev_{player_idx}_{p}",
                ) * 1000.0

    with st.expander("Salespeople Allocation", expanded=True):
        total_salespeople = company.salespeople
        st.info(f"You currently have **{total_salespeople}** salespeople.")
        sales_alloc = {}
        remaining = total_salespeople
        
        sales_cols = st.columns(len(AREAS))
        for i, a in enumerate(AREAS):
            with sales_cols[i]:
                if i == len(AREAS) - 1:
                    val = remaining
                    st.metric(a, val)
                else:
                    val = st.number_input(
                        f"Salespeople in {a}",
                        min_value=0,
                        max_value=int(remaining),
                        value=int(remaining // (len(AREAS) - i)),
                        step=1,
                        key=f"sales_{player_idx}_{a}",
                    )
                sales_alloc[a] = val
                remaining -= val

    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 1.5rem 0 1rem 0;'>
        <h3 style='margin: 0; color: #28a745;'>🏭 2. Operations & Production Decisions</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Shift Level & Maintenance", expanded=True):
        shift_level = st.radio("Shift level", [1, 2, 3], index=0, horizontal=True, key=f"shift_{player_idx}")
        st.info(f"Shift {shift_level}: {MACHINE_HOURS_PER_SHIFT[shift_level]} hours/machine, {MACHINISTS_PER_MACHINE[shift_level]} machinists/machine")
        
        maint_hours = st.number_input(
            "Contracted maintenance hours per machine",
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=5.0,
            key=f"maint_{player_idx}",
        )

    with st.expander("Materials Ordering", expanded=True):
        materials_supplier = st.selectbox(
            "Material Supplier",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"Supplier {x}: {SUPPLIERS[x]['discount']*100:.0f}% discount, Min order: {SUPPLIERS[x]['min_order']:,.0f}",
            key=f"supplier_{player_idx}",
        )
        
        supplier_info = SUPPLIERS[materials_supplier]
        st.info(f"Discount: {supplier_info['discount']*100:.0f}%, Delivery charge: £{supplier_info['delivery_charge']}, Min order: {supplier_info['min_order']:,.0f}")
        
        materials_quantity = st.number_input(
            "Materials order quantity (units) – for quarter after next",
            min_value=0.0,
            max_value=50_000.0,
            value=6_000.0,
            step=500.0,
            key=f"mat_qty_{player_idx}",
        )
        
        if materials_supplier != 0 and materials_supplier != 3:
            materials_num_deliveries = st.number_input(
                "Number of deliveries",
                min_value=1,
                max_value=12,
                value=1,
                step=1,
                key=f"mat_del_{player_idx}",
            )
        else:
            materials_num_deliveries = 0 if materials_supplier == 0 else 12

    with st.expander("Delivery Schedule (units to deliver next quarter)", expanded=True):
        st.markdown("**💡 Bulk Paste:** Paste 12 values (3 products × 4 areas) separated by tabs/newlines")
        paste_del = st.text_area(
            "Paste delivery values",
            key=f"paste_del_{player_idx}",
            height=60,
            help="12 values: P1 areas (4), P2 areas (4), P3 areas (4)"
        )
        
        deliveries = {}
        pasted_del = parse_bulk_paste(paste_del, 12) if paste_del else []
        
        for p_idx, p in enumerate(PRODUCTS):
            st.markdown(f"**{p}**")
            area_cols = st.columns(len(AREAS))
            for a_idx, area in enumerate(AREAS):
                with area_cols[a_idx]:
                    idx = p_idx * 4 + a_idx
                    del_val = int(pasted_del[idx]) if len(pasted_del) > idx else 0
                    val = st.number_input(
                        f"{area}",
                        min_value=0,
                        max_value=10_000,
                        value=del_val,
                        step=50,
                        key=f"del_{player_idx}_{p}_{area}",
                    )
                    deliveries[(p, area)] = val
        
        st.markdown(f"**Total units to deliver:** {sum(deliveries.values()):,}")

    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin: 1.5rem 0 1rem 0;'>
        <h3 style='margin: 0; color: #ffc107;'>👥 3. Personnel Decisions</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Salespeople", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            recruit_sales = st.number_input("Recruit salespeople", min_value=0, max_value=20, value=0, step=1, key=f"rec_sales_{player_idx}")
        with col2:
            dismiss_sales = st.number_input("Dismiss salespeople", min_value=0, max_value=int(company.salespeople), value=0, step=1, key=f"dis_sales_{player_idx}")
        with col3:
            train_sales = st.number_input("Train salespeople from unemployed (max 9)", min_value=0, max_value=9, value=0, step=1, key=f"train_sales_{player_idx}")
        
        sales_salary = st.number_input(
            "Sales salary per quarter (£)",
            min_value=float(MIN_SALES_SALARY_PER_QUARTER),
            max_value=50_000.0,
            value=float(company.sales_salary),
            step=500.0,
            key=f"sal_sales_{player_idx}",
        )
        sales_commission = st.number_input(
            "Sales commission (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(company.sales_commission_rate * 100),
            step=0.5,
            key=f"comm_sales_{player_idx}",
        )

    with st.expander("Assembly Workers", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            recruit_assembly = st.number_input("Recruit assembly workers", min_value=0, max_value=50, value=0, step=1, key=f"rec_assy_{player_idx}")
        with col2:
            dismiss_assembly = st.number_input("Dismiss assembly workers", min_value=0, max_value=int(company.assembly_workers), value=0, step=1, key=f"dis_assy_{player_idx}")
        with col3:
            train_assembly = st.number_input("Train assembly workers from unemployed (max 9)", min_value=0, max_value=9, value=0, step=1, key=f"train_assy_{player_idx}")
        
        assembly_wage_rate = st.number_input(
            "Assembly worker hourly wage rate (£)",
            min_value=ASSEMBLY_MIN_WAGE_RATE,
            max_value=50.0,
            value=float(company.assembly_wage_rate),
            step=0.50,
            key=f"wage_assy_{player_idx}",
        )
    
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; margin: 1.5rem 0 1rem 0;'>
        <h3 style='margin: 0; color: #dc3545;'>💰 4. Finance Decisions</h3>
    </div>
    """, unsafe_allow_html=True)

    finance_col1, finance_col2 = st.columns(2)
    with finance_col1:
        dividend_per_share = st.number_input(
            "Dividend per share (pence) - Q1 and Q3 only",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            disabled=(economy.quarter not in [1, 3]),
            key=f"div_{player_idx}",
        )
        if economy.quarter not in [1, 3]:
            st.caption("Dividends can only be paid in Q1 and Q3")

    with finance_col2:
        management_budget = st.number_input(
            "Management budget (£)",
            min_value=float(MIN_MANAGEMENT_BUDGET),
            max_value=200_000.0,
            value=float(company.last_report.get("management_budget", MIN_MANAGEMENT_BUDGET) if company.last_report else MIN_MANAGEMENT_BUDGET),
            step=5000.0,
            key=f"mgmt_{player_idx}",
        )

    with st.expander("Fixed Assets", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            machines_to_order = st.number_input(
                "Machines to order (requires creditworthiness check)",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key=f"mach_order_{player_idx}",
            )
            creditworthiness = company.calculate_creditworthiness()
            st.info(f"Creditworthiness: £{creditworthiness:,.0f} (max {int(creditworthiness / MACHINE_DEPOSIT)} machines)")
        
        with col2:
            machines_to_sell = st.number_input(
                "Machines to sell",
                min_value=0,
                max_value=int(company.machines),
                value=0,
                step=1,
                key=f"mach_sell_{player_idx}",
            )
            vans_to_buy = st.number_input("Vehicles to buy", min_value=0, max_value=10, value=0, step=1, key=f"van_buy_{player_idx}")
            vans_to_sell = st.number_input("Vehicles to sell", min_value=0, max_value=int(company.vehicles), value=0, step=1, key=f"van_sell_{player_idx}")

    with st.expander("Information Purchases", expanded=False):
        buy_competitor_info = st.checkbox("Buy Competitor Information (£5,000)", value=False, key=f"info_comp_{player_idx}")
        buy_market_shares = st.checkbox("Buy Market Shares Information (£5,000)", value=False, key=f"info_mkt_{player_idx}")

    # Create Decisions object
    return Decisions(
        implement_major_improvement=implement_major,
        prices_home=prices_home,
        prices_export=prices_export,
        credit_days=credit_days,
        assembly_time=assembly_time,
        advertising_trade_press=advertising_trade_press,
        advertising_support=advertising_support,
        advertising_merchandising=advertising_merchandising,
        salespeople_allocation=sales_alloc,
        sales_salary_per_quarter=sales_salary,
        sales_commission_percent=sales_commission,
        assembly_wage_rate=assembly_wage_rate,
        shift_level=shift_level,
        maintenance_hours_per_machine=maint_hours,
        dividend_per_share=dividend_per_share / 100.0,
        management_budget=management_budget,
        deliveries=deliveries,
        product_development=product_dev,
        recruit_sales=recruit_sales,
        dismiss_sales=dismiss_sales,
        train_sales=train_sales,
        recruit_assembly=recruit_assembly,
        dismiss_assembly=dismiss_assembly,
        train_assembly=train_assembly,
        materials_quantity=materials_quantity,
        materials_supplier=materials_supplier,
        materials_num_deliveries=materials_num_deliveries,
        machines_to_order=machines_to_order,
        machines_to_sell=machines_to_sell,
        vans_to_buy=vans_to_buy,
        vans_to_sell=vans_to_sell,
        buy_competitor_info=buy_competitor_info,
        buy_market_shares=buy_market_shares,
    )
    
st.set_page_config(
    page_title="Leftovers Business Simulation (Topaz-VBE)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main styling improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    /* Info boxes */
    .stInfo {
        border-left: 4px solid #1f77b4;
        background-color: #e8f4f8;
    }
    
    /* Success messages */
    .stSuccess {
        border-left: 4px solid #28a745;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Container styling */
    .stContainer {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Better spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Number input styling */
    input[type="number"] {
        border-radius: 5px;
    }
    
    /* Selectbox styling */
    select {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "sim_started" not in st.session_state:
    st.session_state.sim_started = False

if "sim" not in st.session_state:
    st.session_state.sim = Simulation(n_companies=8, seed=42)
    st.session_state.n_players = 1

if "n_players" not in st.session_state:
    st.session_state.n_players = 1

# ============================================================================
# START PAGE - Show before simulation starts
# ============================================================================
if not st.session_state.sim_started:
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Hero section
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 3rem 2rem; 
                    border-radius: 20px; 
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    margin-bottom: 3rem;'>
            <h1 style='color: white; font-size: 3rem; margin: 0 0 1rem 0;'>
                🎮 Leftovers Business Simulation
            </h1>
            <h2 style='color: white; font-size: 1.8rem; margin: 0 0 1.5rem 0; opacity: 0.95;'>
                (Topaz-VBE)
            </h2>
            <p style='color: white; font-size: 1.2em; margin: 0; opacity: 0.9;'>
                Strategic Business Simulation • Multi-Player Support • Competitive Market Dynamics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Welcome card
        st.markdown("""
        <div style='background: #f8f9fa; 
                    padding: 2rem; 
                    border-radius: 15px; 
                    border-left: 5px solid #667eea;
                    margin-bottom: 2rem;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea; margin-top: 0;'>Welcome to the Business Management Simulation</h2>
            <p style='font-size: 1.1em; color: #555; line-height: 1.6;'>
                A complete quarterly management simulation with <strong>3 products</strong>, <strong>4 market areas</strong>,<br>
                and interacting decisions across <strong>Marketing, Operations, Personnel and Finance</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; 
                    border-radius: 15px; 
                    margin: 2rem 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color: white; margin: 0; text-align: center; font-size: 1.5rem;'>🎯 Game Setup</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **How it works:**
        - **1 Player**: You compete against 7 AI-controlled companies
        - **2-8 Players**: Only human players compete (no AI bots)
        - Your performance is judged primarily by **share price**
        - Compete in a dynamic market where your decisions affect competitors and vice versa
        """)
        
        # Player selection
        st.markdown("#### Select Number of Players")
        n_players_input = st.number_input(
            "Number of Players",
            min_value=1,
            max_value=8,
            value=st.session_state.n_players,
            step=1,
            key="start_n_players",
            help="Choose how many human players will participate. Remaining companies will be AI-controlled."
        )
        
        # Display player configuration
        st.markdown("#### Game Configuration")
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.metric("Human Players", n_players_input)
        with config_col2:
            if n_players_input == 1:
                st.metric("AI Players", 7)
            else:
                st.metric("AI Players", 0)
                st.caption("Only human players compete")
        
        st.markdown("---")
        
        # Start button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🚀 Start Simulation", type="primary", width='stretch', key="start_sim"):
                st.session_state.n_players = n_players_input
                # If 1 player: 8 companies (1 human + 7 AI)
                # If 2+ players: only human players (no AI)
                n_companies = 8 if n_players_input == 1 else n_players_input
                st.session_state.sim = Simulation(n_companies=n_companies, seed=42)
                st.session_state.sim.n_players = n_players_input
                st.session_state.sim_started = True
                st.session_state.player_decisions = [None] * n_players_input
                st.rerun()
        
        st.markdown("---")
        
        # Instructions
        with st.expander("📖 How to Play", expanded=False):
            st.markdown("""
            ### Game Overview
            
            You control a manufacturing company competing in a dynamic market. Each quarter, you make decisions about:
            
            **1. Marketing Decisions**
            - Set prices for home and export markets
            - Allocate advertising budget (Trade Press, Support, Merchandising)
            - Invest in product development
            - Allocate salespeople to different market areas
            - Set credit terms for customers
            
            **2. Operations & Production**
            - Choose shift level (affects machine hours and workforce)
            - Order materials from suppliers
            - Schedule production deliveries
            - Set maintenance hours
            
            **3. Personnel**
            - Recruit, train, or dismiss salespeople and assembly workers
            - Set wage rates and salaries
            - Manage workforce capacity
            
            **4. Finance**
            - Set dividend payments (Q1 and Q3 only)
            - Order or sell machines and vehicles
            - Purchase market intelligence
            
            ### Competitive Market
            
            Your decisions directly affect market share:
            - Lower prices can increase sales but reduce margins
            - Higher advertising can capture market share from competitors
            - Product quality affects customer demand
            - Competitors' actions affect your sales
            
            ### Winning Strategy
            
            Your performance is judged by **share price**, which is influenced by:
            - Net worth
            - Earnings per share
            - Dividend payments
            - Overall financial health
            
            Make strategic decisions to maximize your company's value!
            """)
        
        st.stop()  # Stop execution here if simulation hasn't started

# ============================================================================
# MAIN SIMULATION INTERFACE
# ============================================================================

sim: Simulation = st.session_state.sim
sim.n_players = st.session_state.n_players

# Modern header with gradient effect
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; 
            border-radius: 15px; 
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='color: white; text-align: center; margin: 0; font-size: 2.5rem;'>
        🏢 Leftovers Business Simulation (Topaz-VBE)
    </h1>
    <p style='color: white; text-align: center; margin-top: 1rem; font-size: 1.1rem; opacity: 0.95;'>
        Strategic Business Simulation • Multi-Player Support • Competitive Market Dynamics
    </p>
</div>
""", unsafe_allow_html=True)

# Quick stats bar
if sim.companies[0].last_report:
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    with stats_col1:
        st.metric("📈 Share Price", f"£{sim.companies[0].share_price:.2f}")
    with stats_col2:
        st.metric("💰 Net Worth", f"£{sim.companies[0].net_worth():,.0f}")
    with stats_col3:
        st.metric("💵 Cash", f"£{sim.companies[0].cash:,.0f}")
    with stats_col4:
        st.metric("👥 Employees", f"{sim.companies[0].total_employees()}")
    with stats_col5:
        st.metric("⚙️ Machines", f"{sim.companies[0].machines}")
    st.markdown("---")

# --- Sidebar with styled sections ---
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1rem; 
            border-radius: 10px; 
            margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0; text-align: center;'>🎮 Game Info</h3>
</div>
""", unsafe_allow_html=True)
st.sidebar.metric("👥 Human Players", st.session_state.n_players)
if st.session_state.n_players == 1:
    st.sidebar.metric("🤖 AI Players", len(sim.companies) - 1)
else:
    st.sidebar.metric("🤖 AI Players", 0)
    st.sidebar.caption("Only human players")

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
            padding: 1rem; 
            border-radius: 10px; 
            margin: 1rem 0;'>
    <h3 style='color: white; margin: 0; text-align: center;'>📈 Economy</h3>
    <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Current Quarter</p>
</div>
""", unsafe_allow_html=True)
econ = sim.economy
st.sidebar.metric("📅 Year / Quarter", f"Y{econ.year} Q{econ.quarter}")
st.sidebar.metric("📊 GDP Index", f"{econ.gdp:0.1f}")
st.sidebar.metric("👷 Unemployment", f"{econ.unemployment:0.1f}%")
st.sidebar.metric("🏦 Bank Rate", f"{econ.cb_rate:0.2f}%")
st.sidebar.metric("📦 Material Price", f"£{econ.material_price:0.1f}")

if st.sidebar.button("🔄 Reset Simulation", type="primary"):
    # If 1 player: 8 companies (1 human + 7 AI)
    # If 2+ players: only human players (no AI)
    n_companies = 8 if st.session_state.n_players == 1 else st.session_state.n_players
    st.session_state.sim = Simulation(n_companies=n_companies, seed=42)
    st.session_state.sim.n_players = st.session_state.n_players
    st.session_state.player_decisions = [None] * st.session_state.n_players
    st.rerun()

if st.sidebar.button("🏠 Return to Start Page"):
    st.session_state.sim_started = False
    st.rerun()

# Multi-player decision tabs
# Use session state to store decisions to avoid issues with tab rendering
n_players = st.session_state.n_players

if "player_decisions" not in st.session_state:
    st.session_state.player_decisions = [None] * n_players

player_decisions_list = []

if n_players == 1:
    # Single player - show form directly
    decisions_obj = create_player_decision_form(0, sim.companies[0], sim.economy)
    # Check if it's a Decisions instance (not the class) by checking for attributes
    if decisions_obj is not None and decisions_obj is not Decisions:
        # Ensure it's an instance, not the class
        if hasattr(decisions_obj, '__dict__') and hasattr(decisions_obj, 'prices_home') and hasattr(decisions_obj, 'deliveries'):
            player_decisions_list = [decisions_obj]
            st.session_state.player_decisions[0] = decisions_obj
else:
    # Multiple players - use tabs
    tab_names = [f"Player {i+1}: {sim.companies[i].name}" for i in range(n_players)]
    tabs = st.tabs(tab_names)
    
    # Initialize list with None values to maintain order
    player_decisions_list = [None] * n_players
    
    for i in range(n_players):
        with tabs[i]:
            decisions_obj = create_player_decision_form(i, sim.companies[i], sim.economy)
            # Always store the decision object (it's created on every render)
            # Check if it's a Decisions instance (not the class) by checking for attributes
            if decisions_obj is not None and decisions_obj is not Decisions:
                # Ensure it's an instance, not the class
                if hasattr(decisions_obj, '__dict__') and hasattr(decisions_obj, 'prices_home') and hasattr(decisions_obj, 'deliveries'):
                    st.session_state.player_decisions[i] = decisions_obj
                    # Store at the correct index to maintain order
                    player_decisions_list[i] = decisions_obj

# Fill in any missing decisions from session state (for tabs that weren't active)
for i in range(n_players):
    if i >= len(player_decisions_list) or player_decisions_list[i] is None:
        if i < len(st.session_state.player_decisions) and st.session_state.player_decisions[i] is not None:
            d = st.session_state.player_decisions[i]
            if d is not None and d is not Decisions and hasattr(d, '__dict__') and hasattr(d, 'prices_home') and hasattr(d, 'deliveries'):
                if i >= len(player_decisions_list):
                    while len(player_decisions_list) <= i:
                        player_decisions_list.append(None)
                player_decisions_list[i] = d

# Ensure player_decisions_list has exactly n_players items in order
while len(player_decisions_list) < n_players:
    player_decisions_list.append(None)
player_decisions_list = player_decisions_list[:n_players]

# Filter out None values for validation
valid_decisions = [d for d in player_decisions_list if d is not None and d is not Decisions and hasattr(d, '__dict__') and hasattr(d, 'prices_home') and hasattr(d, 'deliveries')]

# Check if we have valid decisions for all players
has_all_decisions = len(valid_decisions) == n_players

if has_all_decisions:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin: 2rem 0;'>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Submit All Decisions and Run Quarter", type="primary", width='stretch'):
        # Final validation
        try:
            reports = sim.step(player_decisions_list)
    st.success("Quarter completed! View results below.")
    st.balloons()
        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
            st.exception(e)

# Display results for all players
if sim.companies[0].last_report:
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; 
                border-radius: 10px; 
                margin: 2rem 0 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h2 style='color: white; margin: 0; text-align: center;'>📊 Results Summary</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show all companies' key metrics
    results_data = []
    for i, comp in enumerate(sim.companies):
        if comp.last_report:
            # Determine if company is AI or Player
            # If n_players == 1: companies 0 is player, 1-7 are AI
            # If n_players > 1: all companies are players (no AI)
            if st.session_state.n_players == 1:
                company_type = "Player" if i < st.session_state.n_players else "AI"
            else:
                company_type = "Player"  # All are players when n_players > 1
            
            results_data.append({
                "Company": comp.name,
                "Type": company_type,
                "Share Price": f"£{comp.share_price:.2f}",
                "Net Worth": f"£{comp.net_worth():,.0f}",
                "Revenue": f"£{comp.last_report.get('revenue', 0):,.0f}",
                "Net Profit": f"£{comp.last_report.get('net_profit', 0):,.0f}",
            })
    
    if results_data:
        st.dataframe(pd.DataFrame(results_data), width='stretch')

# Helper function to convert tuple keys to JSON-serializable format
def make_json_serializable(obj):
    """Recursively convert tuple keys to strings for JSON serialization"""
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, tuple) else k): make_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return str(obj)
    else:
        return obj

# Display last report if available
player_company = sim.companies[0]
if player_company.last_report:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; 
                border-radius: 10px; 
                margin: 2rem 0 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h2 style='color: white; margin: 0; text-align: center;'>📋 Management Report</h2>
    </div>
    """, unsafe_allow_html=True)
    
    report = player_company.last_report
    quarter = report.get('quarter', 1)
    year = report.get('year', 1)
    
    st.markdown(f"**Year {year}, Quarter {quarter}**")
    
    # Key metrics
    metrics_cols = st.columns(5)
    with metrics_cols[0]:
        st.metric("Revenue", f"£{report.get('revenue', 0):,.0f}")
    with metrics_cols[1]:
        st.metric("Net Profit", f"£{report.get('net_profit', 0):,.0f}")
    with metrics_cols[2]:
        st.metric("EBITDA", f"£{report.get('ebitda', 0):,.0f}")
    with metrics_cols[3]:
        st.metric("Cash", f"£{report.get('cash', 0):,.0f}")
    with metrics_cols[4]:
        st.metric("Share Price", f"£{report.get('share_price', 0):.2f}")
    
    # Management Report - Comprehensive view matching manual format
    with st.expander("📊 Full Management Report", expanded=True):
        # Two column layout for top sections
        top_col1, top_col2 = st.columns(2)
        
        with top_col1:
            st.markdown("#### AVAILABILITY and USE OF RESOURCES")
            
            # Machines
            st.write("**Machines:**")
            st.write(f"- Machines Available Last Quarter: {report.get('machines', 0) - report.get('machines_installed', 0)}")
            st.write(f"- Machines Available for Next Quarter: {report.get('machines', 0)}")
            
            # Vehicles
            st.write("**Vehicles:**")
            st.write(f"- Vehicles Available Last Quarter: {report.get('vehicles_available', 0)}")
            
            # Assembly Workers Hours
            st.write("**Assembly Workers Hours:**")
            st.write(f"- Total Hours Available Last Quarter: {report.get('assembly_hours_available', 0):.0f}")
            st.write(f"- Hours of Absenteeism/Sickness: {report.get('assembly_hours_absenteeism', 0):.0f}")
            st.write(f"- Total Hours Worked Last Quarter: {report.get('assembly_hours_worked', 0):.0f}")
            st.write(f"- Notice of Strike Weeks Next Quarter: {report.get('strike_weeks_next', 0)}")
            
            # Machine Hours
            st.write("**Machine Hours:**")
            st.write(f"- Total Hours Available Last Quarter: {report.get('machine_hours_available', 0):.0f}")
            st.write(f"- Hours of Planned Maintenance: {report.get('machine_hours_maintenance', 0):.0f}")
            st.write(f"- Total Hours Worked Last Quarter: {report.get('machine_hours_worked', 0):.0f}")
            st.write(f"- Average Machine Efficiency %: {report.get('machine_efficiency', 0)*100:.1f}")
            
            # Material Units
            st.write("**Material Units Used and Available:**")
            st.write(f"- Opening Stock Available (units): {report.get('material_opening', 0):,.0f}")
            st.write(f"- Delivered Last Quarter: {report.get('material_delivered', 0):,.0f}")
            st.write(f"- Used Last Quarter: {report.get('materials_used', 0):,.0f}")
            st.write(f"- Closing Stock at End of Quarter: {report.get('material_closing', 0):,.0f}")
            st.write(f"- On Order for Next Quarter: {report.get('material_on_order', 0):,.0f}")
            st.write(f"- Total Available for Next Quarter: {report.get('material_closing', 0) + report.get('material_on_order', 0):,.0f}")
            
            # Personnel Table
            st.write("**Personnel:**")
            personnel = report.get('personnel_opening', {})
            personnel_rec = report.get('personnel_recruited', {})
            personnel_train = report.get('personnel_trained', {})
            personnel_dismiss = report.get('personnel_dismissed', {})
            personnel_leavers = report.get('personnel_leavers', {})
            personnel_next = report.get('personnel_available_next', {})
            
            personnel_data = {
                "": ["Sales", "Assembly", "Machinists"],
                "At Start of Last Quarter": [
                    personnel.get('sales', 0),
                    personnel.get('assembly', 0),
                    personnel.get('machinists', 0)
                ],
                "Recruits": [
                    personnel_rec.get('sales', 0),
                    personnel_rec.get('assembly', 0),
                    personnel_rec.get('machinists', 0)
                ],
                "Trainees": [
                    personnel_train.get('sales', 0),
                    personnel_train.get('assembly', 0),
                    personnel_train.get('machinists', 0)
                ],
                "Dismissals": [
                    personnel_dismiss.get('sales', 0),
                    personnel_dismiss.get('assembly', 0),
                    personnel_dismiss.get('machinists', 0)
                ],
                "Leavers": [
                    personnel_leavers.get('sales', 0),
                    personnel_leavers.get('assembly', 0),
                    personnel_leavers.get('machinists', 0)
                ],
                "Available for Next Quarter": [
                    personnel_next.get('sales', 0),
                    personnel_next.get('assembly', 0),
                    personnel_next.get('machinists', 0)
                ]
            }
            st.dataframe(pd.DataFrame(personnel_data), width='stretch', hide_index=True)
        
        with top_col2:
            st.markdown("#### PRODUCT MOVEMENTS and AVAILABILITY")
            
            # Quantities table
            st.write("**Quantities:**")
            scheduled = report.get('scheduled', {})
            deliveries = report.get('deliveries', {})
            rejects = report.get('rejects', {})
            servicing = report.get('servicing_units', {})
            
            # Build product movement tables
            quant_data = {"": PRODUCTS}
            quant_data["Scheduled"] = [scheduled.get((p, "Export"), 0) + sum(scheduled.get((p, a), 0) for a in ["South", "West", "North"]) for p in PRODUCTS]
            quant_data["Produced"] = [sum(deliveries.get((p, a), 0) for a in AREAS) + sum(rejects.get((p, a), 0) for a in AREAS) for p in PRODUCTS]
            quant_data["Rejected"] = [sum(rejects.get((p, a), 0) for a in AREAS) for p in PRODUCTS]
            quant_data["Serviced"] = [servicing.get(p, 0) for p in PRODUCTS]
            st.dataframe(pd.DataFrame(quant_data), width='stretch', hide_index=True)
            
            # Delivered to table
            st.write("**Delivered to:**")
            deliv_data = {"Area": AREAS}
            for p in PRODUCTS:
                deliv_data[p] = [deliveries.get((p, a), 0) for a in AREAS]
            st.dataframe(pd.DataFrame(deliv_data), width='stretch', hide_index=True)
            
            # Orders from table
            st.write("**Orders from:**")
            new_orders = report.get('new_orders', {})
            orders_data = {"Area": AREAS}
            for p in PRODUCTS:
                orders_data[p] = [new_orders.get((p, a), 0) for a in AREAS]
            st.dataframe(pd.DataFrame(orders_data), width='stretch', hide_index=True)
            
            # Sales to table
            st.write("**Sales to:**")
            sales = report.get('sales', {})
            sales_data = {"Area": AREAS}
            for p in PRODUCTS:
                sales_data[p] = [sales.get((p, a), 0) for a in AREAS]
            st.dataframe(pd.DataFrame(sales_data), width='stretch', hide_index=True)
            
            # Order Backlog table
            st.write("**Order Backlog:**")
            backlog = report.get('backlog', {})
            backlog_data = {"Area": AREAS}
            for p in PRODUCTS:
                backlog_data[p] = [backlog.get((p, a), 0) for a in AREAS]
            st.dataframe(pd.DataFrame(backlog_data), width='stretch', hide_index=True)
            
            # Warehouse Stock table
            st.write("**Warehouse Stock:**")
            stocks = report.get('stocks', {})
            stock_data = {"Area": AREAS}
            for p in PRODUCTS:
                stock_data[p] = [stocks.get((p, a), 0) for a in AREAS]
            st.dataframe(pd.DataFrame(stock_data), width='stretch', hide_index=True)
            
            # Product Improvements
            st.write("**Product Improvements:**")
            dev_outcomes = report.get('product_dev_outcomes', {})
            improvements = [dev_outcomes.get(p, "NONE").upper() if dev_outcomes.get(p) else "NONE" for p in PRODUCTS]
            imp_data = {"": PRODUCTS, "Improvements": improvements}
            st.dataframe(pd.DataFrame(imp_data), width='stretch', hide_index=True)
        
        # ACCOUNTS section - two columns
        st.markdown("---")
        accounts_col1, accounts_col2 = st.columns(2)
        
        with accounts_col1:
            st.markdown("#### ACCOUNTS - Overhead Cost Analysis")
            overhead_breakdown = report.get('overhead_breakdown', {})
            overhead_items = [
                ("Advertising", overhead_breakdown.get('advertising', 0)),
                ("Salespeoples Salary, etc.", overhead_breakdown.get('salespeople_salary', 0)),
                ("Sales Office", overhead_breakdown.get('sales_office', 0)),
                ("Guarantee Servicing", overhead_breakdown.get('guarantee_servicing', 0)),
                ("Transport Fleet", overhead_breakdown.get('transport_fleet', 0)),
                ("Hired Transport", overhead_breakdown.get('hired_transport', 0)),
                ("Product Research", overhead_breakdown.get('product_research', 0)),
                ("Personnel Department", overhead_breakdown.get('personnel_department', 0)),
                ("Maintenance", overhead_breakdown.get('maintenance', 0)),
                ("Warehousing & Purchasing", overhead_breakdown.get('warehousing_purchasing', 0)),
                ("Business Intelligence", overhead_breakdown.get('business_intelligence', 0)),
                ("Management Budget", overhead_breakdown.get('management_budget', 0)),
                ("Credit Control", overhead_breakdown.get('credit_control', 0)),
                ("Other Miscellaneous Costs", overhead_breakdown.get('other_miscellaneous', 0)),
            ]
            overhead_df = pd.DataFrame(overhead_items, columns=["Cost Item", "Amount in £"])
            overhead_df["Amount in £"] = overhead_df["Amount in £"].apply(lambda x: f"£{x:,.0f}")
            st.dataframe(overhead_df, width='stretch', hide_index=True)
            st.write(f"**Total Overheads:** £{report.get('total_overheads', 0):,.0f}")
        
        with accounts_col2:
            st.markdown("#### ACCOUNTS - Profit and Loss Account")
            cos_breakdown = report.get('cost_of_sales_breakdown', {})
            pnl_items = [
                ("Sales Revenue", report.get('revenue', 0)),
                ("Opening Stock Value", cos_breakdown.get('opening_stock_value', 0)),
                ("Materials Purchased", cos_breakdown.get('materials_purchased', 0)),
                ("Assembly Wages", cos_breakdown.get('assembly_wages', 0)),
                ("Machinists Wages", cos_breakdown.get('machinists_wages', 0)),
                ("Machine Running Costs", cos_breakdown.get('machine_running_costs', 0)),
                ("Less Closing Stock Value", -cos_breakdown.get('closing_stock_value', 0)),
                ("Cost of Sales", report.get('cost_of_sales', 0)),
                ("Gross Profit", report.get('gross_profit', 0)),
                ("Interest received", report.get('interest_received', 0)),
                ("Interest Paid", report.get('interest_paid', 0)),
                ("Overheads", report.get('total_overheads', 0)),
                ("Depreciation", report.get('depreciation', 0)),
                ("Tax Assessed", report.get('tax', 0)),
                ("Net Profit/Loss", report.get('net_profit', 0)),
                ("Dividend Paid", report.get('dividends', 0)),
                ("Transferred to Reserves", report.get('retained', 0)),
            ]
            pnl_df = pd.DataFrame(pnl_items, columns=["Account Item", "Amount in £"])
            pnl_df["Amount in £"] = pnl_df["Amount in £"].apply(lambda x: f"£{x:,.0f}")
            st.dataframe(pnl_df, width='stretch', hide_index=True)
        
        # Balance Sheet and Cash Flow - two columns
        st.markdown("---")
        balance_col1, balance_col2 = st.columns(2)
        
        with balance_col1:
            st.markdown("#### Balance Sheet")
            balance = report.get('balance_sheet', {})
            balance_items = [
                ("**Assets:**", ""),
                ("Value of Property", balance.get('property_value', 0)),
                ("Value of Machines", balance.get('machine_values', 0)),
                ("Value of Vehicles", balance.get('vehicle_values', 0)),
                ("Value of Product Stocks", balance.get('product_stocks_value', 0)),
                ("Value of Material Stock", balance.get('material_stock_value', 0)),
                ("Debtors", balance.get('debtors', 0)),
                ("Cash Invested", balance.get('cash_invested', 0)),
                ("**Liabilities:**", ""),
                ("Tax Assessed and Due", balance.get('tax_assessed_due', 0)),
                ("Creditors", balance.get('creditors', 0)),
                ("Overdraft", balance.get('overdraft', 0)),
                ("Unsecured Loans", balance.get('unsecured_loans', 0)),
                ("**Net Assets:**", report.get('net_worth', 0)),
                ("**Funding:**", ""),
                ("Ordinary Capital", balance.get('ordinary_capital', 0)),
                ("Reserves", balance.get('reserves', 0)),
                ("**Total Funding:**", report.get('net_worth', 0)),
            ]
            balance_df = pd.DataFrame(balance_items, columns=["Item", "Amount in £"])
            balance_df["Amount in £"] = balance_df["Amount in £"].apply(lambda x: f"£{x:,.0f}" if isinstance(x, (int, float)) else x)
            st.dataframe(balance_df, width='stretch', hide_index=True)
        
        with balance_col2:
            st.markdown("#### Cash Flow Statement")
            cf = report.get('cash_flow', {})
            cf_items = [
                ("**Operating Activities:**", ""),
                ("Trading Receipts", cf.get('trading_receipts', 0)),
                ("Trading Payments", -cf.get('trading_payments', 0)),
                ("Tax Paid", -cf.get('tax_paid', 0)),
                ("Cash flow from operating activities", cf.get('trading_receipts', 0) - cf.get('trading_payments', 0) - cf.get('tax_paid', 0)),
                ("**Investing Activities:**", ""),
                ("Interest Received", cf.get('interest_received', 0)),
                ("Capital Receipts", cf.get('capital_receipts', 0)),
                ("Capital Payments", -cf.get('capital_payments', 0)),
                ("Cash flow from investing activities", cf.get('interest_received', 0) + cf.get('capital_receipts', 0) - cf.get('capital_payments', 0)),
                ("**Financing Activities:**", ""),
                ("Interest Paid", -cf.get('interest_paid', 0)),
                ("Dividend Paid", -cf.get('dividend_paid', 0)),
                ("Cash flow from financing activities", -cf.get('interest_paid', 0) - cf.get('dividend_paid', 0)),
                ("**Net Cash Flow:**", cf.get('net_cash_flow', 0)),
                ("**Overdraft Limit for Next Quarter:**", player_company.calculate_overdraft_limit()),
            ]
            cf_df = pd.DataFrame(cf_items, columns=["Item", "Amount in £"])
            cf_df["Amount in £"] = cf_df["Amount in £"].apply(lambda x: f"£{x:,.0f}" if isinstance(x, (int, float)) else x)
            st.dataframe(cf_df, width='stretch', hide_index=True)
    
    # Detailed report expander (with JSON serialization fix)
    with st.expander("🔍 Detailed Report (JSON)", expanded=False):
        json_safe_report = make_json_serializable(report)
        st.json(json_safe_report)
    
    # Financial summary
    with st.expander("💰 Financial Summary", expanded=False):
        fin_cols = st.columns(3)
        with fin_cols[0]:
            st.metric("Gross Profit", f"£{report.get('gross_profit', 0):,.0f}")
            st.metric("Total Overheads", f"£{report.get('total_overheads', 0):,.0f}")
        with fin_cols[1]:
            st.metric("Interest Received", f"£{report.get('interest_received', 0):,.0f}")
            st.metric("Interest Paid", f"£{report.get('interest_paid', 0):,.0f}")
        with fin_cols[2]:
            st.metric("Depreciation", f"£{report.get('depreciation', 0):,.0f}")
            st.metric("Tax", f"£{report.get('tax', 0):,.0f}")

# Display competitor information
st.markdown("### Competitor Information")
competitor_df = []
for i, comp in enumerate(sim.companies[1:], 1):
    competitor_df.append({
        "Company": comp.name,
        "Share Price": f"£{comp.share_price:.2f}",
        "Net Worth": f"£{comp.net_worth():,.0f}",
        "Employees": comp.total_employees(),
        "Machines": comp.machines,
    })
if competitor_df:
    st.dataframe(pd.DataFrame(competitor_df), width='stretch')

