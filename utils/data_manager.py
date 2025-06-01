# utils/data_manager.py - Consolidated Data Management

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from sqlalchemy import text

from .db import get_db_engine
from .settings_manager import SettingsManager

logger = logging.getLogger(__name__)

class DataManager:
    """Unified data management singleton - combines functionality from all data modules"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._cache = {}
            self._metadata = {}
            self._last_refresh = {}
            self._insights_cache = {}
            # Initialize settings manager separately to avoid circular dependency
            self._settings_manager = None
            self._cache_ttl = 300  # 5 minutes default
            self._initialized = True
    
    def _get_settings_manager(self):
        """Lazy load settings manager to avoid initialization issues"""
        if self._settings_manager is None:
            from .settings_manager import SettingsManager
            self._settings_manager = SettingsManager()
        return self._settings_manager
    
    # === Core Data Loading Methods ===
    
    @st.cache_data(ttl=1800)
    def load_demand_oc(_self):
        """Load OC (Order Confirmation) pending delivery data"""
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.outbound_oc_pending_delivery_view;"
        df = pd.read_sql(text(query), engine)
        
        # Debug logging
        if st.session_state.get('debug_mode', False):
            logger.info(f"Loaded OC data: {len(df)} rows")
            if 'etd' in df.columns:
                logger.info(f"OC ETD range: {df['etd'].min()} to {df['etd'].max()}")
        
        return df
    
    @st.cache_data(ttl=1800)
    def load_demand_forecast(_self):
        """Load customer demand forecast data"""
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.customer_demand_forecast_full_view;"
        df = pd.read_sql(text(query), engine)
        
        # Debug: Check data format
        if st.session_state.get('debug_mode', False) and 'is_converted_to_oc' in df.columns:
            logger.info(f"Forecast is_converted_to_oc dtype: {df['is_converted_to_oc'].dtype}")
            logger.info(f"Forecast is_converted_to_oc unique values: {df['is_converted_to_oc'].unique()}")
            logger.info(f"Forecast is_converted_to_oc value counts: {df['is_converted_to_oc'].value_counts().to_dict()}")
        
        return df
    
    @st.cache_data(ttl=1800)
    def load_inventory(_self):
        """Load current inventory data"""
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.inventory_detailed_view"
        return pd.read_sql(text(query), engine)
    
    @st.cache_data(ttl=1800)
    def load_pending_can(_self):
        """Load pending CAN (Container Arrival Note) data"""
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.can_pending_stockin_view"
        return pd.read_sql(text(query), engine)
    
    @st.cache_data(ttl=1800)
    def load_pending_po(_self):
        """Load pending Purchase Order data"""
        engine = get_db_engine()
        query = """
        SELECT * FROM prostechvn.purchase_order_full_view
        WHERE pending_standard_arrival_quantity > 0
        """
        return pd.read_sql(text(query), engine)
    
    @st.cache_data(ttl=1800)
    def load_pending_wh_transfer(_self):
        """Load pending Warehouse Transfer data"""
        engine = get_db_engine()
        query = """
        SELECT * FROM prostechvn.warehouse_transfer_details_view wtdv
        WHERE wtdv.is_completed = 0
        """
        return pd.read_sql(text(query), engine)
    
    @st.cache_data(ttl=3600)
    def load_product_master(_self):
        """Load product master data"""
        engine = get_db_engine()
        query = """
        SELECT 
            p.id as product_id,
            p.pt_code,
            p.name as product_name,
            p.description,
            b.brand_name as brand,
            p.package_size,
            p.uom as standard_uom,
            p.shelf_life,
            p.shelf_life_uom,
            p.storage_condition,
            p.hs_code,
            p.legacy_pt_code
        FROM products p
        LEFT JOIN brands b ON p.brand_id = b.id
        WHERE p.delete_flag = 0
        """
        return pd.read_sql(text(query), engine)
    
    @st.cache_data(ttl=3600)
    def load_customer_master(_self):
        """Load customer master data"""
        engine = get_db_engine()
        query = """
        SELECT 
            c.id as customer_id,
            c.english_name as customer_name,
            c.company_code as customer_code,
            c.registration_code,
            c.local_name,
            tnc.limit_credit as credit_limit,
            cur.code as credit_limit_currency,
            pt.name as payment_term_days
        FROM companies c
        JOIN companies_company_types cct ON c.id = cct.companies_id
        JOIN company_types ct ON cct.company_type_id = ct.id
        JOIN term_and_conditions tnc ON c.customer_term_condition_id = tnc.id
        JOIN payment_terms pt ON tnc.payment_term_id = pt.id
        JOIN currencies cur ON tnc.credit_currency_id = cur.id
        WHERE c.delete_flag = 0
        AND ct.name = 'Customer'
        """
        return pd.read_sql(text(query), engine)
    
    @st.cache_data(ttl=1800)
    def load_active_allocations(_self):
        """Load active allocations affecting supply"""
        engine = get_db_engine()
        query = """
        SELECT * FROM active_allocations_view
        WHERE undelivered_qty > 0
        """
        return pd.read_sql(text(query), engine)
    
    # === Unified Data Access Methods ===
    
    def get_demand_data(self, sources: List[str], include_converted: bool = False) -> pd.DataFrame:
        """Get combined demand data with standardization"""
        df_parts = []
        
        if "OC" in sources:
            df_oc = self.load_demand_oc()
            if not df_oc.empty:
                df_oc["source_type"] = "OC"
                df_parts.append(self._standardize_demand_df(df_oc, is_forecast=False))
        
        if "Forecast" in sources:
            df_fc = self.load_demand_forecast()
            if not df_fc.empty:
                df_fc["source_type"] = "Forecast"
                standardized_fc = self._standardize_demand_df(df_fc, is_forecast=True)
                
                if not include_converted and 'is_converted_to_oc' in standardized_fc.columns:
                    # Handle multiple possible formats for converted status
                    # Check actual data format and handle accordingly
                    converted_values = ['Yes', 'yes', 'Y', 'y', '1', 1, True, 'True', 'true']
                    
                    # Debug log
                    if st.session_state.get('debug_mode', False):
                        before_count = len(standardized_fc)
                        converted_count = standardized_fc['is_converted_to_oc'].isin(converted_values).sum()
                        logger.info(f"Filtering converted forecasts: {converted_count} out of {before_count}")
                    
                    standardized_fc = standardized_fc[~standardized_fc["is_converted_to_oc"].isin(converted_values)]
                
                df_parts.append(standardized_fc)
        
        return pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()
    
    def get_supply_data(self, sources: List[str], exclude_expired: bool = True) -> pd.DataFrame:
        """Get combined supply data with standardization"""
        today = pd.to_datetime("today").normalize()
        df_parts = []
        
        if "Inventory" in sources:
            inv_df = self.load_inventory()
            if not inv_df.empty:
                inv_df = self._prepare_inventory_data(inv_df, today, exclude_expired)
                df_parts.append(inv_df)
        
        if "Pending CAN" in sources:
            can_df = self.load_pending_can()
            if not can_df.empty:
                can_df = self._prepare_can_data(can_df)
                df_parts.append(can_df)
        
        if "Pending PO" in sources:
            po_df = self.load_pending_po()
            if not po_df.empty:
                po_df = self._prepare_po_data(po_df)
                df_parts.append(po_df)
        
        if "Pending WH Transfer" in sources:
            wht_df = self.load_pending_wh_transfer()
            if not wht_df.empty:
                wht_df = self._prepare_wh_transfer_data(wht_df, today, exclude_expired)
                df_parts.append(wht_df)
        
        if not df_parts:
            return pd.DataFrame()
        
        standardized_parts = [self._standardize_supply_df(df) for df in df_parts]
        return pd.concat(standardized_parts, ignore_index=True)
    
    def preload_all_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Parallel loading of all data types"""
        # Check cache validity
        if not force_refresh and self._is_bulk_cache_valid():
            return self._cache
        
        loading_tasks = {
            'demand_oc': self.load_demand_oc,
            'demand_forecast': self.load_demand_forecast,
            'supply_inventory': self.load_inventory,
            'supply_can': self.load_pending_can,
            'supply_po': self.load_pending_po,
            'supply_wh_transfer': self.load_pending_wh_transfer,
            'master_products': self.load_product_master,
            'master_customers': self.load_customer_master,
            'active_allocations': self.load_active_allocations
        }
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_key = {
                executor.submit(func): key 
                for key, func in loading_tasks.items()
            }
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    self._cache[key] = future.result()
                    self._last_refresh[key] = datetime.now()
                except Exception as e:
                    logger.error(f"Error loading {key}: {str(e)}")
                    self._cache[key] = pd.DataFrame()
        
        # Calculate insights after loading
        self._calculate_insights()
        
        return self._cache
    
    def get_insights(self) -> Dict[str, Any]:
        """Get calculated insights across all data"""
        if not self._insights_cache:
            self._calculate_insights()
        return self._insights_cache
    
    def get_critical_alerts(self) -> List[Dict[str, Any]]:
        """Get critical alerts requiring immediate action"""
        alerts = []
        insights = self.get_insights()
        
        # Demand-only products alert
        if insights.get('demand_only_products'):
            alerts.append({
                'level': 'critical',
                'icon': '📤',
                'message': f"{len(insights.get('demand_only_products', set()))} Demand-Only products",
                'value': f"${insights.get('demand_only_value', 0):,.0f}",
                'action': 'no supply'
            })
        
        # Overdue orders alert
        if insights.get('demand_overdue_count', 0) > 0:
            alerts.append({
                'level': 'critical',
                'icon': '🕐',
                'message': f"{insights['demand_overdue_count']} Past ETD orders",
                'value': f"${insights['demand_overdue_value']:,.0f}",
                'action': 'overdue delivery'
            })
        
        # Expired items alert
        if insights.get('expired_items_count', 0) > 0:
            alerts.append({
                'level': 'critical',
                'icon': '💀',
                'message': f"{insights['expired_items_count']} Expired items",
                'value': f"${insights['expired_items_value']:,.0f}",
                'action': 'immediate disposal'
            })
        
        return alerts
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get warning level insights"""
        warnings = []
        insights = self.get_insights()
        
        # Supply-only products warning
        if insights.get('supply_only_products'):
            warnings.append({
                'level': 'warning',
                'icon': '📦',
                'message': f"{len(insights.get('supply_only_products', set()))} Supply-Only products",
                'value': f"${insights.get('supply_only_value', 0):,.0f}",
                'action': 'potential dead stock'
            })
        
        # Missing dates warnings
        if insights.get('demand_missing_etd', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '⚠️',
                'message': f"{insights['demand_missing_etd']} records missing ETD",
                'action': 'demand side'
            })
        
        # Near expiry warnings
        if insights.get('near_expiry_7d_count', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '📅',
                'message': f"{insights['near_expiry_7d_count']} items expiring in 7 days",
                'value': f"${insights['near_expiry_7d_value']:,.0f}"
            })
        
        return warnings
    
    # === Private Helper Methods ===
    
    def _is_bulk_cache_valid(self) -> bool:
        """Check if bulk cache is still valid"""
        if not self._cache:
            return False
        
        # Check age of oldest cache entry
        if self._last_refresh:
            oldest = min(self._last_refresh.values())
            elapsed = (datetime.now() - oldest).seconds
            return elapsed < self._cache_ttl
        
        return False
    
    def _standardize_demand_df(self, df: pd.DataFrame, is_forecast: bool) -> pd.DataFrame:
        """Standardize demand dataframe"""
        df = df.copy()
        
        # Date columns
        if 'etd' in df.columns:
            df["etd"] = pd.to_datetime(df["etd"], errors="coerce")
            
            # Apply time adjustments
            etd_offset = self._get_settings_manager().get_setting('time_adjustments.etd_offset_days', 0)
            if etd_offset != 0 and df["etd"].notna().any():
                df["etd"] = df["etd"] + timedelta(days=etd_offset)
        
        if 'oc_date' in df.columns:
            df["oc_date"] = pd.to_datetime(df["oc_date"], errors="coerce")
        
        # Quantity and value columns - check existence first
        if is_forecast:
            if 'standard_quantity' in df.columns:
                df['demand_quantity'] = pd.to_numeric(df['standard_quantity'], errors='coerce').fillna(0)
            else:
                df['demand_quantity'] = 0
                
            if 'total_amount_usd' in df.columns:
                df['value_in_usd'] = pd.to_numeric(df['total_amount_usd'], errors='coerce').fillna(0)
            else:
                df['value_in_usd'] = 0
                
            df['demand_number'] = df.get('forecast_number', '')
            
            # Handle is_converted_to_oc properly - preserve original value
            if 'is_converted_to_oc' in df.columns:
                # Just ensure it's string type for consistent comparison
                df['is_converted_to_oc'] = df['is_converted_to_oc'].astype(str).str.strip()
            else:
                df['is_converted_to_oc'] = 'No'
            
            if 'forecast_line_id' in df.columns:
                df['demand_line_id'] = df['forecast_line_id'].astype(str) + '_FC'
            else:
                df['demand_line_id'] = ''
        else:
            if 'pending_standard_delivery_quantity' in df.columns:
                df['demand_quantity'] = pd.to_numeric(df['pending_standard_delivery_quantity'], errors='coerce').fillna(0)
            else:
                df['demand_quantity'] = 0
                
            if 'outstanding_amount_usd' in df.columns:
                df['value_in_usd'] = pd.to_numeric(df['outstanding_amount_usd'], errors='coerce').fillna(0)
            else:
                df['value_in_usd'] = 0
                
            df['demand_number'] = df.get('oc_number', '')
            df['is_converted_to_oc'] = 'N/A'
            
            if 'ocd_id' in df.columns:
                df['demand_line_id'] = df['ocd_id'].astype(str) + '_OC'
            else:
                df['demand_line_id'] = ''
        
        # Clean string columns - only if they exist
        string_cols = ['product_name', 'pt_code', 'brand', 'legal_entity', 'customer']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            else:
                df[col] = ''  # Create column with empty string if not exists
        
        # UOM and package size
        if 'standard_uom' in df.columns:
            df['standard_uom'] = df['standard_uom'].astype(str).str.strip().str.upper()
        else:
            df['standard_uom'] = ''
            
        if 'package_size' in df.columns:
            df['package_size'] = df['package_size'].astype(str).str.strip()
        else:
            df['package_size'] = ''
        
        return df
    
    def _prepare_inventory_data(self, inv_df: pd.DataFrame, today: pd.Timestamp, exclude_expired: bool) -> pd.DataFrame:
        """Prepare inventory data"""
        inv_df = inv_df.copy()
        inv_df["source_type"] = "Inventory"
        inv_df["date_ref"] = today
        
        # Map columns with existence check
        if 'remaining_quantity' in inv_df.columns:
            inv_df["quantity"] = pd.to_numeric(inv_df["remaining_quantity"], errors="coerce").fillna(0)
        else:
            inv_df["quantity"] = 0
            
        if 'inventory_value_usd' in inv_df.columns:
            inv_df["value_in_usd"] = pd.to_numeric(inv_df["inventory_value_usd"], errors="coerce").fillna(0)
        else:
            inv_df["value_in_usd"] = 0
            
        if 'owning_company_name' in inv_df.columns:
            inv_df["legal_entity"] = inv_df["owning_company_name"]
        else:
            inv_df["legal_entity"] = ''
            
        if 'expiry_date' in inv_df.columns:
            inv_df["expiry_date"] = pd.to_datetime(inv_df["expiry_date"], errors="coerce")
            inv_df["days_until_expiry"] = (inv_df["expiry_date"] - today).dt.days
            
            if exclude_expired:
                inv_df = inv_df[(inv_df["expiry_date"].isna()) | (inv_df["expiry_date"] >= today)]
        
        # Add supply number for tracking
        if 'inventory_history_id' in inv_df.columns:
            inv_df["supply_number"] = inv_df["inventory_history_id"].astype(str)
        else:
            inv_df["supply_number"] = ''
        
        return inv_df
    
    def _prepare_can_data(self, can_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare CAN data"""
        can_df = can_df.copy()
        can_df["source_type"] = "Pending CAN"
        
        if 'arrival_date' in can_df.columns:
            can_df["date_ref"] = pd.to_datetime(can_df["arrival_date"], errors="coerce")
        else:
            can_df["date_ref"] = pd.NaT
            
        if 'pending_quantity' in can_df.columns:
            can_df["quantity"] = pd.to_numeric(can_df["pending_quantity"], errors="coerce").fillna(0)
        else:
            can_df["quantity"] = 0
            
        if 'pending_value_usd' in can_df.columns:
            can_df["value_in_usd"] = pd.to_numeric(can_df["pending_value_usd"], errors="coerce").fillna(0)
        else:
            can_df["value_in_usd"] = 0
            
        if 'consignee' in can_df.columns:
            can_df["legal_entity"] = can_df["consignee"]
        else:
            can_df["legal_entity"] = ''
            
        if 'arrival_note_number' in can_df.columns:
            can_df["supply_number"] = can_df["arrival_note_number"].astype(str)
        else:
            can_df["supply_number"] = ''
            
        return can_df
    
    def _prepare_po_data(self, po_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare PO data"""
        po_df = po_df.copy()
        po_df["source_type"] = "Pending PO"
        
        # Use cargo_ready_date or crd
        date_col = 'cargo_ready_date' if 'cargo_ready_date' in po_df.columns else 'crd'
        if date_col in po_df.columns:
            po_df["date_ref"] = pd.to_datetime(po_df[date_col], errors="coerce")
        else:
            po_df["date_ref"] = pd.NaT
            
        if 'pending_standard_arrival_quantity' in po_df.columns:
            po_df["quantity"] = pd.to_numeric(po_df["pending_standard_arrival_quantity"], errors="coerce").fillna(0)
        else:
            po_df["quantity"] = 0
            
        if 'outstanding_arrival_amount_usd' in po_df.columns:
            po_df["value_in_usd"] = pd.to_numeric(po_df["outstanding_arrival_amount_usd"], errors="coerce").fillna(0)
        else:
            po_df["value_in_usd"] = 0
            
        if 'legal_entity' not in po_df.columns:
            po_df["legal_entity"] = ''
            
        if 'po_number' in po_df.columns:
            po_df["supply_number"] = po_df["po_number"].astype(str)
        else:
            po_df["supply_number"] = ''
            
        # Add vendor info if available
        if 'vendor_name' in po_df.columns:
            po_df["vendor"] = po_df["vendor_name"]
            
        return po_df
    
    def _prepare_wh_transfer_data(self, wht_df: pd.DataFrame, today: pd.Timestamp, exclude_expired: bool) -> pd.DataFrame:
        """Prepare warehouse transfer data"""
        wht_df = wht_df.copy()
        wht_df["source_type"] = "Pending WH Transfer"
        
        if 'transfer_date' in wht_df.columns:
            wht_df["transfer_date"] = pd.to_datetime(wht_df["transfer_date"], errors="coerce")
            # Apply transfer lead time from settings
            transfer_lead_time = self._get_settings_manager().get_setting('time_adjustments.wh_transfer_lead_time', 2)
            wht_df["date_ref"] = wht_df["transfer_date"] + pd.Timedelta(days=transfer_lead_time)
        else:
            wht_df["date_ref"] = pd.NaT
            
        if 'transfer_quantity' in wht_df.columns:
            wht_df["quantity"] = pd.to_numeric(wht_df["transfer_quantity"], errors="coerce").fillna(0)
        else:
            wht_df["quantity"] = 0
            
        if 'warehouse_transfer_value_usd' in wht_df.columns:
            wht_df["value_in_usd"] = pd.to_numeric(wht_df["warehouse_transfer_value_usd"], errors="coerce").fillna(0)
        else:
            wht_df["value_in_usd"] = 0
            
        if 'owning_company_name' in wht_df.columns:
            wht_df["legal_entity"] = wht_df["owning_company_name"]
        else:
            wht_df["legal_entity"] = ''
            
        if 'warehouse_transfer_line_id' in wht_df.columns:
            wht_df["supply_number"] = wht_df["warehouse_transfer_line_id"].astype(str)
        else:
            wht_df["supply_number"] = ''
            
        # Handle expiry date
        if 'expiry_date' in wht_df.columns:
            wht_df["expiry_date"] = pd.to_datetime(wht_df["expiry_date"], errors="coerce")
            
            if exclude_expired:
                wht_df = wht_df[(wht_df["expiry_date"].isna()) | (wht_df["expiry_date"] >= today)]
        
        # Add transfer route info if available
        if 'from_warehouse' in wht_df.columns and 'to_warehouse' in wht_df.columns:
            wht_df["transfer_route"] = wht_df["from_warehouse"] + " → " + wht_df["to_warehouse"]
            
        return wht_df
    
    def _standardize_supply_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize supply dataframe"""
        df = df.copy()
        
        # Clean string columns - only if they exist
        string_cols = ["pt_code", "product_name", "brand"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            else:
                df[col] = ''  # Create column with empty string if not exists
        
        # UOM and package size
        if 'standard_uom' in df.columns:
            df["standard_uom"] = df["standard_uom"].astype(str).str.strip().str.upper()
        else:
            df["standard_uom"] = ''
            
        if 'package_size' in df.columns:
            df["package_size"] = df["package_size"].astype(str).str.strip()
        else:
            df["package_size"] = ''
        
        # Ensure required columns exist
        if "value_in_usd" not in df.columns:
            df["value_in_usd"] = 0
            
        if "quantity" not in df.columns:
            df["quantity"] = 0
            
        if "legal_entity" not in df.columns:
            df["legal_entity"] = ''
            
        if "date_ref" not in df.columns:
            df["date_ref"] = pd.NaT
            
        if "source_type" not in df.columns:
            df["source_type"] = 'Unknown'
        
        # Select standard columns - ensure they all exist
        standard_cols = [
            "source_type", "pt_code", "product_name", "brand", 
            "package_size", "standard_uom", "legal_entity", "date_ref", 
            "quantity", "value_in_usd"
        ]
        
        # Add optional columns if they exist
        optional_cols = ["supply_number", "expiry_date", "days_until_expiry", 
                        "days_since_arrival", "vendor", "transfer_route", 
                        "days_in_transfer", "from_warehouse", "to_warehouse"]
        
        for col in optional_cols:
            if col in df.columns:
                standard_cols.append(col)
        
        return df[standard_cols]
    
    def _calculate_insights(self):
        """Calculate key insights from loaded data"""
        insights = {}
        settings = self._get_settings_manager()
        
        # Get thresholds from settings
        critical_days = settings.get_setting('alert_thresholds.critical_shortage_days', 3)
        warning_days = settings.get_setting('alert_thresholds.warning_shortage_days', 7)
        excess_months = settings.get_setting('alert_thresholds.excess_inventory_months', 6)
        shelf_life_threshold = settings.get_setting('business_rules.shelf_life_threshold_days', 30)
        
        # Demand insights
        if 'demand_oc' in self._cache and not self._cache['demand_oc'].empty:
            oc_df = self._cache['demand_oc'].copy()
            
            # Check if required columns exist
            if 'etd' in oc_df.columns:
                oc_df['etd'] = pd.to_datetime(oc_df['etd'], errors='coerce')
            
            insights['demand_oc_pending_count'] = len(oc_df)
            
            if 'outstanding_amount_usd' in oc_df.columns:
                insights['demand_oc_pending_value'] = oc_df['outstanding_amount_usd'].sum()
            else:
                insights['demand_oc_pending_value'] = 0
            
            if 'etd' in oc_df.columns:
                insights['demand_missing_etd'] = len(oc_df[oc_df['etd'].isna()])
                
                # Overdue analysis - use normalized dates for comparison
                today = pd.Timestamp.now().normalize()
                overdue_mask = oc_df['etd'] < today
                insights['demand_overdue_count'] = len(oc_df[overdue_mask])
                
                if 'outstanding_amount_usd' in oc_df.columns:
                    insights['demand_overdue_value'] = oc_df[overdue_mask]['outstanding_amount_usd'].sum()
                else:
                    insights['demand_overdue_value'] = 0
                
                # Critical shortage analysis
                critical_date = today + timedelta(days=critical_days)
                critical_mask = (oc_df['etd'] <= critical_date) & (~overdue_mask)
                insights['critical_shortage_count'] = len(oc_df[critical_mask])
                
                if 'outstanding_amount_usd' in oc_df.columns:
                    insights['critical_shortage_value'] = oc_df[critical_mask]['outstanding_amount_usd'].sum()
                else:
                    insights['critical_shortage_value'] = 0
            else:
                insights['demand_missing_etd'] = 0
                insights['demand_overdue_count'] = 0
                insights['demand_overdue_value'] = 0
                insights['critical_shortage_count'] = 0
                insights['critical_shortage_value'] = 0
        
        # Supply insights
        if 'supply_inventory' in self._cache and not self._cache['supply_inventory'].empty:
            inv_df = self._cache['supply_inventory'].copy()
            
            if 'expiry_date' in inv_df.columns:
                inv_df['expiry_date'] = pd.to_datetime(inv_df['expiry_date'], errors='coerce')
            
            if 'inventory_value_usd' in inv_df.columns:
                insights['inventory_total_value'] = inv_df['inventory_value_usd'].sum()
            else:
                insights['inventory_total_value'] = 0
            
            # Expiry analysis
            if 'expiry_date' in inv_df.columns:
                today = pd.Timestamp.now().normalize()
                expired_mask = inv_df['expiry_date'] < today
                insights['expired_items_count'] = len(inv_df[expired_mask])
                
                if 'inventory_value_usd' in inv_df.columns:
                    insights['expired_items_value'] = inv_df[expired_mask]['inventory_value_usd'].sum()
                else:
                    insights['expired_items_value'] = 0
                
                # Near expiry analysis
                near_expiry_7d_mask = (
                    (inv_df['expiry_date'] >= today) & 
                    (inv_df['expiry_date'] <= today + timedelta(days=7))
                )
                insights['near_expiry_7d_count'] = len(inv_df[near_expiry_7d_mask])
                
                if 'inventory_value_usd' in inv_df.columns:
                    insights['near_expiry_7d_value'] = inv_df[near_expiry_7d_mask]['inventory_value_usd'].sum()
                else:
                    insights['near_expiry_7d_value'] = 0
            else:
                insights['expired_items_count'] = 0
                insights['expired_items_value'] = 0
                insights['near_expiry_7d_count'] = 0
                insights['near_expiry_7d_value'] = 0
            
            # Excess inventory analysis
            if 'days_in_warehouse' in inv_df.columns:
                excess_days = excess_months * 30
                excess_mask = inv_df['days_in_warehouse'] > excess_days
                insights['excess_inventory_count'] = len(inv_df[excess_mask])
                
                if 'inventory_value_usd' in inv_df.columns:
                    insights['excess_inventory_value'] = inv_df[excess_mask]['inventory_value_usd'].sum()
                else:
                    insights['excess_inventory_value'] = 0
            else:
                insights['excess_inventory_count'] = 0
                insights['excess_inventory_value'] = 0
        
        # Product matching insights - ONLY OC vs Inventory for dashboard
        if ('demand_oc' in self._cache and not self._cache['demand_oc'].empty and
            'supply_inventory' in self._cache and not self._cache['supply_inventory'].empty):
            
            demand_df = self._cache['demand_oc']
            supply_df = self._cache['supply_inventory']
            
            # Check if pt_code exists in both dataframes
            if 'pt_code' in demand_df.columns and 'pt_code' in supply_df.columns:
                demand_products = set(demand_df['pt_code'].dropna().unique())
                supply_products = set(supply_df['pt_code'].dropna().unique())
                
                insights['demand_only_products'] = demand_products - supply_products
                insights['supply_only_products'] = supply_products - demand_products
                insights['matched_products'] = demand_products & supply_products
                
                # Calculate values for unmatched products
                if insights['demand_only_products'] and 'outstanding_amount_usd' in demand_df.columns:
                    insights['demand_only_value'] = demand_df[
                        demand_df['pt_code'].isin(insights['demand_only_products'])
                    ]['outstanding_amount_usd'].sum()
                else:
                    insights['demand_only_value'] = 0
                
                if insights['supply_only_products'] and 'inventory_value_usd' in supply_df.columns:
                    insights['supply_only_value'] = supply_df[
                        supply_df['pt_code'].isin(insights['supply_only_products'])
                    ]['inventory_value_usd'].sum()
                else:
                    insights['supply_only_value'] = 0
            else:
                # If pt_code doesn't exist, set empty values
                insights['demand_only_products'] = set()
                insights['supply_only_products'] = set()
                insights['matched_products'] = set()
                insights['demand_only_value'] = 0
                insights['supply_only_value'] = 0
        else:
            # Set default empty values
            insights['demand_only_products'] = set()
            insights['supply_only_products'] = set()
            insights['matched_products'] = set()
            insights['demand_only_value'] = 0
            insights['supply_only_value'] = 0
        
        self._insights_cache = insights
        return insights
    
    def clear_cache(self, data_type: str = None):
        """Clear cache for specific data type or all"""
        if data_type:
            keys_to_clear = [k for k in self._cache.keys() if k.startswith(data_type)]
            for key in keys_to_clear:
                if key in self._cache:
                    del self._cache[key]
                if key in self._last_refresh:
                    del self._last_refresh[key]
        else:
            self._cache.clear()
            self._last_refresh.clear()
            self._insights_cache.clear()
            st.cache_data.clear()
    
    def set_cache_ttl(self, seconds: int):
        """Set cache time-to-live"""
        self._cache_ttl = seconds