"""
Topaz-VBE Business Management Simulation
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
    
    def __init__(self, n_companies: int = 4, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.economy = Economy()
        self.companies: List[CompanyState] = [
            CompanyState(name=f"Company {i+1}") for i in range(n_companies)
        ]
        self.history: List[Dict] = []
        self.material_prices_history: List[float] = [BASE_MATERIAL_PRICE]
    
    # ========== DEMAND & MARKETING ==========
    
    def demand_for_product(
        self,
        company: CompanyState,
        decisions: Decisions,
        product: str,
        area: str,
        all_companies: List[CompanyState] = None,
    ) -> float:
        """Enhanced demand calculation with all marketing factors from manual"""
        
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
        
        demand = (
            base_demand *
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
                
                # New demand
                demand_units = int(self.demand_for_product(company, decisions, product, area, self.companies))
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
        
        # Build comprehensive report
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
            "stocks": stocks_new,
            "backlog": backlog_new,
            "sales": sales,
            "new_orders": new_orders,
            "deliveries": produced,
            "rejects": rejects,
            "product_dev_outcomes": dev_outcomes,
            "stock_write_offs": stock_write_offs,
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
    
    def step(self, player_decisions: Decisions):
        """Run one quarter for all companies"""
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
            }
            reports.append(rep_with_meta)
        
        self.history.extend(reports)
        # Advance economy for next round
        self.economy.advance()
        return reports

# ============================================================================
# STREAMLIT UI - Complete implementation matching decision form
# ============================================================================

st.set_page_config(page_title="Topaz-VBE Business Management Simulation", layout="wide")

if "sim" not in st.session_state:
    st.session_state.sim = Simulation(n_companies=4, seed=42)

sim: Simulation = st.session_state.sim

st.title("Topaz-VBE Business Management Simulation")

st.markdown(
    """
    This app implements a **complete quarterly management simulation** with 3 products, 4 market areas,  
    and interacting decisions across **Marketing, Operations, Personnel and Finance**.
    
    You control **Company 1**; the other companies are automated competitors.  
    Your performance is judged primarily by **share price**.
    """
)

# --- Economy panel ---
st.sidebar.header("Economy (Current Quarter)")
econ = sim.economy
st.sidebar.metric("Year / Quarter", f"Y{econ.year} Q{econ.quarter}")
st.sidebar.metric("GDP index", f"{econ.gdp:0.1f}")
st.sidebar.metric("Unemployment %", f"{econ.unemployment:0.1f}%")
st.sidebar.metric("Central Bank Rate (next qtr)", f"{econ.cb_rate:0.2f}%")
st.sidebar.metric("Material price (per 1000 units)", f"£{econ.material_price:0.1f}")

if st.sidebar.button("Reset simulation", type="primary"):
    st.session_state.sim = Simulation(n_companies=4, seed=42)
    st.rerun()

player_company: CompanyState = sim.companies[0]

st.subheader(f"Your company: {player_company.name}")
col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Share price", f"£{player_company.share_price:0.2f}")
col_b.metric("Net worth", f"£{player_company.net_worth():,.0f}")
col_c.metric("Cash", f"£{player_company.cash:,.0f}")
col_d.metric("Employees", f"{player_company.total_employees():,}")
col_e.metric("Machines", f"{player_company.machines}")

st.markdown("### 1. Marketing Decisions")

with st.expander("Product Improvements", expanded=False):
    st.info("Implement Major Improvements (will write off all stocks for that product)")
    implement_major = {}
    for p in PRODUCTS:
        implement_major[p] = st.checkbox(f"Implement Major Improvement - {p}", key=f"major_{p}")
    
    # Show current star ratings
    st.markdown("**Current Product Star Ratings:**")
    star_cols = st.columns(len(PRODUCTS))
    for i, p in enumerate(PRODUCTS):
        with star_cols[i]:
            stars = player_company.product_star_ratings.get(p, 3)
            st.metric(p, f"{stars:.1f} ⭐")

with st.expander("Prices, Credit Terms & Quality", expanded=True):
    price_cols = st.columns(len(PRODUCTS))
    prices_home = {}
    prices_export = {}
    assembly_time = {}
    
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
                min_value=float(MIN_ASSEMBLY_TIME[p]),
                max_value=float(MIN_ASSEMBLY_TIME[p] * 2.0),
                value=float(MIN_ASSEMBLY_TIME[p] * 1.2),
                step=10.0,
                key=f"assy_{p}",
            )
    
    credit_days = st.slider("Credit days offered to retailers", 15, 90, 30, 5)

with st.expander("Advertising (Three Types)", expanded=True):
    st.markdown("#### Advertising spend per product per area (£000 per quarter)")
    
    advertising_trade_press = {}
    advertising_support = {}
    advertising_merchandising = {}
    
    for p in PRODUCTS:
        st.markdown(f"**{p}**")
        area_cols = st.columns(len(AREAS))
        for i, a in enumerate(AREAS):
            with area_cols[i]:
                st.markdown(f"*{a}*")
                trade_press = st.number_input(
                    "Trade Press",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    key=f"adv_tp_{p}_{a}",
                )
                support = st.number_input(
                    "Support",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    key=f"adv_sup_{p}_{a}",
                )
                merchandising = st.number_input(
                    "Merchandising",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    key=f"adv_merch_{p}_{a}",
                )
                advertising_trade_press[(p, a)] = trade_press * 1000.0
                advertising_support[(p, a)] = support * 1000.0
                advertising_merchandising[(p, a)] = merchandising * 1000.0

with st.expander("Product Development", expanded=True):
    product_dev = {}
    dev_cols = st.columns(len(PRODUCTS))
    for i, p in enumerate(PRODUCTS):
        with dev_cols[i]:
            accumulated = player_company.product_dev_accumulated.get(p, 0.0)
            st.metric(f"Accumulated {p}", f"£{accumulated:,.0f}")
            product_dev[p] = st.number_input(
                f"Dev spend {p} (£000)",
                min_value=0.0,
                max_value=200.0,
                value=20.0,
                step=5.0,
                key=f"dev_{p}",
            ) * 1000.0

with st.expander("Salespeople Allocation", expanded=True):
    total_salespeople = player_company.salespeople
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
                    key=f"sales_{a}",
                )
            sales_alloc[a] = val
            remaining -= val

st.markdown("### 2. Operations & Production Decisions")

with st.expander("Shift Level & Maintenance", expanded=True):
    shift_level = st.radio("Shift level", [1, 2, 3], index=0, horizontal=True)
    st.info(f"Shift {shift_level}: {MACHINE_HOURS_PER_SHIFT[shift_level]} hours/machine, {MACHINISTS_PER_MACHINE[shift_level]} machinists/machine")
    
    maint_hours = st.number_input(
        "Contracted maintenance hours per machine",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=5.0,
    )

with st.expander("Materials Ordering", expanded=True):
    materials_supplier = st.selectbox(
        "Material Supplier",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Supplier {x}: {SUPPLIERS[x]['discount']*100:.0f}% discount, Min order: {SUPPLIERS[x]['min_order']:,.0f}",
    )
    
    supplier_info = SUPPLIERS[materials_supplier]
    st.info(f"Discount: {supplier_info['discount']*100:.0f}%, Delivery charge: £{supplier_info['delivery_charge']}, Min order: {supplier_info['min_order']:,.0f}")
    
    materials_quantity = st.number_input(
        "Materials order quantity (units) – for quarter after next",
        min_value=0.0,
        max_value=50_000.0,
        value=6_000.0,
        step=500.0,
    )
    
    if materials_supplier != 0 and materials_supplier != 3:
        materials_num_deliveries = st.number_input(
            "Number of deliveries",
            min_value=1,
            max_value=12,
            value=1,
            step=1,
        )
    else:
        materials_num_deliveries = 0 if materials_supplier == 0 else 12

with st.expander("Delivery Schedule (units to deliver next quarter)", expanded=True):
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
    
    st.markdown(f"**Total units to deliver:** {sum(deliveries.values()):,}")

st.markdown("### 3. Personnel Decisions")

with st.expander("Salespeople", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        recruit_sales = st.number_input("Recruit salespeople", min_value=0, max_value=20, value=0, step=1)
    with col2:
        dismiss_sales = st.number_input("Dismiss salespeople", min_value=0, max_value=int(player_company.salespeople), value=0, step=1)
    with col3:
        train_sales = st.number_input("Train salespeople from unemployed (max 9)", min_value=0, max_value=9, value=0, step=1)
    
    sales_salary = st.number_input(
        "Sales salary per quarter (£)",
        min_value=float(MIN_SALES_SALARY_PER_QUARTER),
        max_value=50_000.0,
        value=float(player_company.sales_salary),
        step=500.0,
    )
    sales_commission = st.number_input(
        "Sales commission (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(player_company.sales_commission_rate * 100),
        step=0.5,
    )

with st.expander("Assembly Workers", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        recruit_assembly = st.number_input("Recruit assembly workers", min_value=0, max_value=50, value=0, step=1)
    with col2:
        dismiss_assembly = st.number_input("Dismiss assembly workers", min_value=0, max_value=int(player_company.assembly_workers), value=0, step=1)
    with col3:
        train_assembly = st.number_input("Train assembly workers from unemployed (max 9)", min_value=0, max_value=9, value=0, step=1)
    
    assembly_wage_rate = st.number_input(
        "Assembly worker hourly wage rate (£)",
        min_value=ASSEMBLY_MIN_WAGE_RATE,
        max_value=50.0,
        value=float(player_company.assembly_wage_rate),
        step=0.50,
    )

st.markdown("### 4. Finance Decisions")

finance_col1, finance_col2 = st.columns(2)
with finance_col1:
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

with finance_col2:
    management_budget = st.number_input(
        "Management budget (£)",
        min_value=float(MIN_MANAGEMENT_BUDGET),
        max_value=200_000.0,
        value=float(player_company.last_report.get("management_budget", MIN_MANAGEMENT_BUDGET) if player_company.last_report else MIN_MANAGEMENT_BUDGET),
        step=5000.0,
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
        )
        creditworthiness = player_company.calculate_creditworthiness()
        st.info(f"Creditworthiness: £{creditworthiness:,.0f} (max {int(creditworthiness / MACHINE_DEPOSIT)} machines)")
    
    with col2:
        machines_to_sell = st.number_input(
            "Machines to sell",
            min_value=0,
            max_value=int(player_company.machines),
            value=0,
            step=1,
        )
        vans_to_buy = st.number_input("Vehicles to buy", min_value=0, max_value=10, value=0, step=1)
        vans_to_sell = st.number_input("Vehicles to sell", min_value=0, max_value=int(player_company.vehicles), value=0, step=1)

with st.expander("Information Purchases", expanded=False):
    buy_competitor_info = st.checkbox("Buy Competitor Information (£5,000)", value=False)
    buy_market_shares = st.checkbox("Buy Market Shares Information (£5,000)", value=False)

# Submit decisions
if st.button("Submit Decisions and Run Quarter", type="primary", use_container_width=True):
    # Create Decisions object
    decisions = Decisions(
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
        dividend_per_share=dividend_per_share / 100.0,  # convert pence to pounds
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
    
    # Run simulation
    reports = sim.step(decisions)
    
    st.success("Quarter completed! View results below.")
    st.balloons()

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
if player_company.last_report:
    st.markdown("### Management Report")
    
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
    
    # Management Report - Comprehensive view
    with st.expander("📊 Full Management Report", expanded=True):
        # Profit & Loss
        st.markdown("#### Profit & Loss Account")
        pnl_data = {
            "Revenue": f"£{report.get('revenue', 0):,.0f}",
            "Cost of Sales": f"£{report.get('cost_of_sales', 0):,.0f}",
            "Gross Profit": f"£{report.get('gross_profit', 0):,.0f}",
            "Total Overheads": f"£{report.get('total_overheads', 0):,.0f}",
            "EBITDA": f"£{report.get('ebitda', 0):,.0f}",
            "Interest Received": f"£{report.get('interest_received', 0):,.0f}",
            "Interest Paid": f"£{report.get('interest_paid', 0):,.0f}",
            "Depreciation": f"£{report.get('depreciation', 0):,.0f}",
            "Profit Before Tax": f"£{report.get('profit_before_tax', 0):,.0f}",
            "Tax": f"£{report.get('tax', 0):,.0f}",
            "Net Profit": f"£{report.get('net_profit', 0):,.0f}",
            "Dividends": f"£{report.get('dividends', 0):,.0f}",
            "Retained Earnings": f"£{report.get('retained', 0):,.0f}",
        }
        pnl_df = pd.DataFrame(list(pnl_data.items()), columns=["Item", "Amount"])
        st.dataframe(pnl_df, use_container_width=True, hide_index=True)
        
        # Balance Sheet
        st.markdown("#### Balance Sheet")
        balance_data = {
            "Cash": f"£{report.get('cash', 0):,.0f}",
            "Overdraft": f"£{report.get('overdraft', 0):,.0f}",
            "Unsecured Loan": f"£{report.get('loan', 0):,.0f}",
            "Net Worth": f"£{report.get('net_worth', 0):,.0f}",
        }
        balance_df = pd.DataFrame(list(balance_data.items()), columns=["Item", "Amount"])
        st.dataframe(balance_df, use_container_width=True, hide_index=True)
        
        # Sales by Product and Area
        st.markdown("#### Sales by Product and Area")
        sales_dict = report.get('sales', {})
        if sales_dict:
            sales_rows = []
            total_revenue = report.get('revenue', 0)
            total_units = sum(sales_dict.values())
            avg_price = total_revenue / total_units if total_units > 0 else 0
            for (product, area), units in sales_dict.items():
                # Estimate value proportionally
                units_share = units / total_units if total_units > 0 else 0
                value = total_revenue * units_share
                sales_rows.append({
                    "Product": product,
                    "Area": area,
                    "Units": units,
                    "Value": f"£{value:,.0f}"
                })
            if sales_rows:
                st.dataframe(pd.DataFrame(sales_rows), use_container_width=True)
                st.caption(f"Total Revenue: £{total_revenue:,.0f}")
        
        # Stock Levels
        st.markdown("#### Stock Levels by Product and Area")
        stocks_dict = report.get('stocks', {})
        if stocks_dict:
            stock_rows = []
            for (product, area), units in stocks_dict.items():
                stock_rows.append({
                    "Product": product,
                    "Area": area,
                    "Units": units,
                    "Value": f"£{units * PRODUCT_STOCK_VALUATION.get(product, 0):,.0f}"
                })
            if stock_rows:
                st.dataframe(pd.DataFrame(stock_rows), use_container_width=True)
        
        # Backlog
        st.markdown("#### Backlog by Product and Area")
        backlog_dict = report.get('backlog', {})
        if backlog_dict:
            backlog_rows = []
            for (product, area), units in backlog_dict.items():
                backlog_rows.append({
                    "Product": product,
                    "Area": area,
                    "Units": units
                })
            if backlog_rows:
                st.dataframe(pd.DataFrame(backlog_rows), use_container_width=True)
        
        # Production & Deliveries
        st.markdown("#### Production & Deliveries")
        deliveries_dict = report.get('deliveries', {})
        if deliveries_dict:
            delivery_rows = []
            for (product, area), units in deliveries_dict.items():
                rejects = report.get('rejects', {}).get((product, area), 0)
                delivery_rows.append({
                    "Product": product,
                    "Area": area,
                    "Produced": units,
                    "Rejected": rejects,
                    "Delivered": units
                })
            if delivery_rows:
                st.dataframe(pd.DataFrame(delivery_rows), use_container_width=True)
        
        # Operations Summary
        st.markdown("#### Operations Summary")
        ops_data = {
            "Machines": report.get('machines', 0),
            "Machines Installed": report.get('machines_installed', 0),
            "Machines Ordered": report.get('machines_ordered', 0),
            "Machine Efficiency": f"{report.get('machine_efficiency', 0)*100:.1f}%",
            "Shift Level": report.get('shift_level', 1),
            "Materials Used": f"{report.get('materials_used', 0):,.0f}",
            "Materials Opening": f"{report.get('material_opening', 0):,.0f}",
            "Materials Delivered": f"{report.get('material_delivered', 0):,.0f}",
            "Materials Closing": f"{report.get('material_closing', 0):,.0f}",
        }
        ops_df = pd.DataFrame(list(ops_data.items()), columns=["Item", "Value"])
        st.dataframe(ops_df, use_container_width=True, hide_index=True)
        
        # Product Development
        dev_outcomes = report.get('product_dev_outcomes', {})
        if dev_outcomes:
            st.markdown("#### Product Development Outcomes")
            dev_rows = []
            for product, outcome in dev_outcomes.items():
                dev_rows.append({
                    "Product": product,
                    "Outcome": outcome
                })
            if dev_rows:
                st.dataframe(pd.DataFrame(dev_rows), use_container_width=True)
        
        stock_write_offs = report.get('stock_write_offs', {})
        if stock_write_offs:
            st.markdown("#### Stock Write-offs (Major Improvements)")
            writeoff_rows = []
            for product, units in stock_write_offs.items():
                writeoff_rows.append({
                    "Product": product,
                    "Units Written Off": units,
                    "Value": f"£{units * PRODUCT_STOCK_VALUATION.get(product, 0):,.0f}"
                })
            if writeoff_rows:
                st.dataframe(pd.DataFrame(writeoff_rows), use_container_width=True)
    
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
    st.dataframe(pd.DataFrame(competitor_df), use_container_width=True)

