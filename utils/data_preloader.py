import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.settings_manager import SettingsManager
from utils.data_loader import (
    load_outbound_demand_data,
    load_customer_forecast_data,
    load_inventory_data,
    load_pending_can_data,
    load_pending_po_data,
    load_pending_wh_transfer_data,
    load_product_master,
    load_customer_master,
    load_vendor_master,
    load_active_allocations,
    load_allocation_history
)

class DataPreloader:
    """Centralized data management with settings-aware processing"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataPreloader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._data_cache = {}
            self._insights_cache = {}
            self._load_timestamp = None
            self._settings_manager = SettingsManager()
            self._initialized = True
    
    def preload_all_data(self, force_refresh=False):
        """Parallel data loading for performance"""
        if self._load_timestamp and not force_refresh:
            time_diff = (datetime.now() - self._load_timestamp).seconds
            if time_diff < 300:  # 5 minutes
                return self._data_cache
        
        with st.spinner("🔄 Loading all data..."):
            loading_tasks = {
                'demand_oc': load_outbound_demand_data,
                'demand_forecast': load_customer_forecast_data,
                'supply_inventory': load_inventory_data,
                'supply_can': load_pending_can_data,
                'supply_po': load_pending_po_data,
                'supply_wh_transfer': load_pending_wh_transfer_data,
                'master_products': load_product_master,
                'master_customers': load_customer_master,
                'master_vendors': load_vendor_master,
                'active_allocations': load_active_allocations
            }
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_key = {
                    executor.submit(func): key 
                    for key, func in loading_tasks.items()
                }
                
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        self._data_cache[key] = future.result()
                    except Exception as e:
                        st.error(f"Error loading {key}: {str(e)}")
                        self._data_cache[key] = pd.DataFrame()
            
            self._load_timestamp = datetime.now()
            self._calculate_insights()
            
        return self._data_cache
    
    def _apply_time_adjustments(self, df: pd.DataFrame, date_column: str, adjustment_key: str) -> pd.DataFrame:
        """Apply time adjustments from settings"""
        adjustment_days = self._settings_manager.get_setting(f'time_adjustments.{adjustment_key}', 0)
        
        if adjustment_days != 0 and date_column in df.columns:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column]) + timedelta(days=adjustment_days)
        
        return df
    
    def _calculate_insights(self):
        """Calculate key insights from loaded data"""
        insights = {}
        settings = self._settings_manager
        
        critical_days = settings.get_setting('alert_thresholds.critical_shortage_days', 3)
        warning_days = settings.get_setting('alert_thresholds.warning_shortage_days', 7)
        excess_months = settings.get_setting('alert_thresholds.excess_inventory_months', 6)
        shelf_life_threshold = settings.get_setting('business_rules.shelf_life_threshold_days', 30)
        
        # Demand insights
        if not self._data_cache.get('demand_oc', pd.DataFrame()).empty:
            oc_df = self._data_cache['demand_oc'].copy()
            oc_df = self._apply_time_adjustments(oc_df, 'etd', 'etd_offset_days')
            
            insights['demand_oc_pending_count'] = len(oc_df)
            insights['demand_oc_pending_value'] = oc_df['outstanding_amount_usd'].sum()
            insights['demand_missing_etd'] = len(oc_df[oc_df['etd'].isna()])
            
            oc_df['etd'] = pd.to_datetime(oc_df['etd'])
            overdue_mask = oc_df['etd'] < datetime.now()
            insights['demand_overdue_count'] = len(oc_df[overdue_mask])
            insights['demand_overdue_value'] = oc_df[overdue_mask]['outstanding_amount_usd'].sum()
            
            critical_date = datetime.now() + timedelta(days=critical_days)
            critical_mask = (oc_df['etd'] <= critical_date) & (~overdue_mask)
            insights['critical_shortage_count'] = len(oc_df[critical_mask])
            insights['critical_shortage_value'] = oc_df[critical_mask]['outstanding_amount_usd'].sum()
        
        # Supply insights
        if not self._data_cache.get('supply_inventory', pd.DataFrame()).empty:
            inv_df = self._data_cache['supply_inventory'].copy()
            insights['inventory_total_value'] = inv_df['inventory_value_usd'].sum()
            
            inv_df['expiry_date'] = pd.to_datetime(inv_df['expiry_date'])
            
            expired_mask = inv_df['expiry_date'] < datetime.now()
            insights['expired_items_count'] = len(inv_df[expired_mask])
            insights['expired_items_value'] = inv_df[expired_mask]['inventory_value_usd'].sum()
            
            near_expiry_7d_mask = (
                (inv_df['expiry_date'] >= datetime.now()) & 
                (inv_df['expiry_date'] <= datetime.now() + timedelta(days=7))
            )
            insights['near_expiry_7d_count'] = len(inv_df[near_expiry_7d_mask])
            insights['near_expiry_7d_value'] = inv_df[near_expiry_7d_mask]['inventory_value_usd'].sum()
            
            near_expiry_30d_mask = (
                (inv_df['expiry_date'] >= datetime.now()) & 
                (inv_df['expiry_date'] <= datetime.now() + timedelta(days=30))
            )
            insights['near_expiry_30d_count'] = len(inv_df[near_expiry_30d_mask])
            insights['near_expiry_30d_value'] = inv_df[near_expiry_30d_mask]['inventory_value_usd'].sum()
            
            inv_df['days_to_expiry'] = (inv_df['expiry_date'] - datetime.now()).dt.days
            below_threshold_mask = (
                (inv_df['days_to_expiry'] > 0) & 
                (inv_df['days_to_expiry'] < shelf_life_threshold)
            )
            insights['below_shelf_life_threshold'] = len(inv_df[below_threshold_mask])
            insights['below_shelf_life_value'] = inv_df[below_threshold_mask]['inventory_value_usd'].sum()
            
            excess_days = excess_months * 30
            excess_mask = inv_df['days_in_warehouse'] > excess_days
            insights['excess_inventory_count'] = len(inv_df[excess_mask])
            insights['excess_inventory_value'] = inv_df[excess_mask]['inventory_value_usd'].sum()
        
        # Check for missing dates in supply
        missing_dates_count = 0
        
        if not self._data_cache.get('supply_can', pd.DataFrame()).empty:
            can_df = self._data_cache['supply_can']
            missing_dates_count += len(can_df[can_df['arrival_date'].isna()])
        
        if not self._data_cache.get('supply_po', pd.DataFrame()).empty:
            po_df = self._data_cache['supply_po']
            missing_dates_count += len(po_df[po_df['crd'].isna()])
        
        insights['supply_missing_dates'] = missing_dates_count
        
        # Product matching insights
        if ('demand_oc' in self._data_cache and 'supply_inventory' in self._data_cache):
            demand_products = set(self._data_cache['demand_oc']['pt_code'].unique())
            supply_products = set(self._data_cache['supply_inventory']['pt_code'].unique())
            
            insights['demand_only_products'] = demand_products - supply_products
            insights['supply_only_products'] = supply_products - demand_products
            insights['matched_products'] = demand_products & supply_products
            
            if insights['demand_only_products']:
                oc_df = self._data_cache['demand_oc']
                demand_only_value = oc_df[
                    oc_df['pt_code'].isin(insights['demand_only_products'])
                ]['outstanding_amount_usd'].sum()
                insights['demand_only_value'] = demand_only_value
            
            if insights['supply_only_products']:
                inv_df = self._data_cache['supply_inventory']
                supply_only_value = inv_df[
                    inv_df['pt_code'].isin(insights['supply_only_products'])
                ]['inventory_value_usd'].sum()
                insights['supply_only_value'] = supply_only_value
        
        # Allocation insights
        if not self._data_cache.get('active_allocations', pd.DataFrame()).empty:
            alloc_df = self._data_cache['active_allocations']
            insights['active_allocations_count'] = len(alloc_df)
            insights['allocated_undelivered_qty'] = alloc_df['undelivered_qty'].sum()
            insights['avg_fulfillment_rate'] = alloc_df['avg_fulfillment_rate'].mean()
        
        # Products with zero supply in next 7 days
        if 'demand_oc' in self._data_cache:
            oc_df = self._data_cache['demand_oc']
            near_term_mask = pd.to_datetime(oc_df['etd']) <= datetime.now() + timedelta(days=7)
            near_term_products = set(oc_df[near_term_mask]['pt_code'].unique())
            
            supply_products = set()
            if 'supply_inventory' in self._data_cache:
                supply_products.update(self._data_cache['supply_inventory']['pt_code'].unique())
            if 'supply_can' in self._data_cache:
                can_df = self._data_cache['supply_can']
                near_can_mask = pd.to_datetime(can_df['arrival_date']) <= datetime.now() + timedelta(days=7)
                supply_products.update(can_df[near_can_mask]['pt_code'].unique())
            
            zero_supply_products = near_term_products - supply_products
            insights['zero_supply_7d_count'] = len(zero_supply_products)
        
        self._insights_cache = insights
        return insights
    
    def get_data(self, key):
        """Get specific dataset"""
        return self._data_cache.get(key, pd.DataFrame())
    
    def get_insights(self):
        """Get calculated insights"""
        return self._insights_cache
    
    def get_critical_alerts(self):
        """Get critical alerts requiring immediate action"""
        alerts = []
        insights = self._insights_cache
        
        if insights.get('demand_only_products'):
            alerts.append({
                'level': 'critical',
                'icon': '📤',
                'message': f"{len(insights.get('demand_only_products', set()))} Demand-Only products",
                'value': f"${insights.get('demand_only_value', 0):,.0f}",
                'action': 'no supply'
            })
        
        if insights.get('demand_overdue_count', 0) > 0:
            alerts.append({
                'level': 'critical',
                'icon': '🕐',
                'message': f"{insights['demand_overdue_count']} Past ETD orders",
                'value': f"${insights['demand_overdue_value']:,.0f}",
                'action': 'overdue delivery'
            })
        
        if insights.get('zero_supply_7d_count', 0) > 0:
            alerts.append({
                'level': 'critical',
                'icon': '❌',
                'message': f"{insights['zero_supply_7d_count']} products with 0 supply in next 7 days",
                'value': '',
                'action': ''
            })
        
        if insights.get('expired_items_count', 0) > 0:
            alerts.append({
                'level': 'critical',
                'icon': '💀',
                'message': f"{insights['expired_items_count']} Expired items",
                'value': f"${insights['expired_items_value']:,.0f}",
                'action': 'immediate disposal'
            })
        
        return alerts
    
    def get_warnings(self):
        """Get warning level insights"""
        warnings = []
        insights = self._insights_cache
        
        if insights.get('supply_only_products'):
            warnings.append({
                'level': 'warning',
                'icon': '📦',
                'message': f"{len(insights.get('supply_only_products', set()))} Supply-Only products",
                'value': f"${insights.get('supply_only_value', 0):,.0f}",
                'action': 'potential dead stock'
            })
        
        if insights.get('demand_missing_etd', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '⚠️',
                'message': f"{insights['demand_missing_etd']} records missing ETD",
                'value': '',
                'action': 'demand side'
            })
        
        if insights.get('supply_missing_dates', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '⚠️',
                'message': f"{insights['supply_missing_dates']} records missing dates",
                'value': '',
                'action': 'supply side'
            })
        
        if insights.get('near_expiry_7d_count', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '📅',
                'message': f"{insights['near_expiry_7d_count']} items expiring in 7 days",
                'value': f"${insights['near_expiry_7d_value']:,.0f}"
            })
        
        if insights.get('near_expiry_30d_count', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '📅',
                'message': f"{insights['near_expiry_30d_count']} items expiring in 30 days",
                'value': f"${insights['near_expiry_30d_value']:,.0f}"
            })
        
        return warnings
    
    def get_info_metrics(self):
        """Get informational metrics"""
        insights = self._insights_cache
        metrics = []
        
        matched = len(insights.get('matched_products', set()))
        total = len(set().union(
            insights.get('demand_only_products', set()),
            insights.get('supply_only_products', set()),
            insights.get('matched_products', set())
        ))
        
        if total > 0:
            coverage = (matched / total * 100)
            metrics.append({
                'icon': '🔗',
                'message': f"{matched} Matched products",
                'value': f"{coverage:.1f}% coverage"
            })
        
        avg_fulfillment = insights.get('avg_fulfillment_rate', 0)
        if avg_fulfillment > 0:
            metrics.append({
                'icon': '📊',
                'message': "Avg fulfillment rate",
                'value': f"{avg_fulfillment:.1f}%"
            })
        
        risk_value = (
            insights.get('expired_items_value', 0) +
            insights.get('near_expiry_7d_value', 0) +
            insights.get('excess_inventory_value', 0)
        )
        if risk_value > 0:
            metrics.append({
                'icon': '💰',
                'message': "Total inventory at risk",
                'value': f"${risk_value:,.0f}"
            })
        
        return metrics