import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any, Optional

class SettingsManager:
    """Business settings and parameters management (separate from system config)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            if 'business_settings' not in st.session_state:
                st.session_state.business_settings = self.get_default_settings()
                st.session_state.settings_last_updated = None
            self._initialized = True
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Default business settings for SCM operations"""
        return {
            'time_adjustments': {
                'etd_offset_days': 0,
                'supply_arrival_offset': 0,
                'wh_transfer_lead_time': 2,
                'transportation_time': 3,
                'buffer_days': 7,
                'working_days_per_week': 5
            },
            'planning_parameters': {
                'safety_stock_days': 14,
                'reorder_point_days': 21,
                'forecast_confidence': 0.8,
                'planning_horizon_days': 90,
                'min_order_coverage_days': 30,
                'max_order_coverage_days': 180
            },
            'order_constraints': {
                'default_moq': 100,
                'default_spq': 50,
                'max_order_limit': 10000,
                'order_rounding': 'up',
                'enforce_spq': True
            },
            'business_rules': {
                'allocation_method': 'FIFO',
                'customer_priority_enabled': True,
                'product_priority_enabled': False,
                'shelf_life_threshold_days': 30,
                'shelf_life_allocation_percent': 0.75,
                'seasonality_enabled': True,
                'auto_approve_allocation': False
            },
            'analysis_options': {
                'include_forecast_in_gap': True,
                'forecast_probability_threshold': 0.7,
                'consolidate_by_week': False,
                'exclude_expired_inventory': True,
                'exclude_blocked_inventory': True,
                'consider_in_transit': True
            },
            'alert_thresholds': {
                'critical_shortage_days': 3,
                'warning_shortage_days': 7,
                'excess_inventory_months': 6,
                'slow_moving_months': 3,
                'min_fulfillment_rate': 0.85,
                'max_allocation_variance': 0.15
            }
        }
    
    def get_setting(self, path: str, default=None) -> Any:
        """Get setting by dot notation path"""
        keys = path.split('.')
        value = st.session_state.business_settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_setting(self, path: str, value: Any):
        """Set individual setting by path"""
        keys = path.split('.')
        settings = st.session_state.business_settings
        
        for key in keys[:-1]:
            if key not in settings:
                settings[key] = {}
            settings = settings[key]
        
        settings[keys[-1]] = value
        st.session_state.settings_last_updated = datetime.now()
    
    def save_settings(self, settings_dict: Dict[str, Any]):
        """Save multiple settings at once"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(st.session_state.business_settings, settings_dict)
        st.session_state.settings_last_updated = datetime.now()
    
    def reset_to_defaults(self, category: Optional[str] = None):
        """Reset settings to defaults"""
        if category:
            defaults = self.get_default_settings()
            if category in defaults:
                st.session_state.business_settings[category] = defaults[category]
        else:
            st.session_state.business_settings = self.get_default_settings()
        
        st.session_state.settings_last_updated = datetime.now()
    
    def export_settings(self) -> str:
        """Export settings as JSON string"""
        return json.dumps(st.session_state.business_settings, indent=2)
    
    def import_settings(self, json_string: str) -> bool:
        """Import settings from JSON string"""
        try:
            imported_settings = json.loads(json_string)
            default_keys = set(self.get_default_settings().keys())
            imported_keys = set(imported_settings.keys())
            
            if not imported_keys.issubset(default_keys):
                raise ValueError("Invalid settings structure")
            
            self.save_settings(imported_settings)
            return True
        except Exception as e:
            st.error(f"Failed to import settings: {str(e)}")
            return False
    
    def get_applied_adjustments(self) -> list:
        """Get all active adjustments for display"""
        adjustments = []
        
        time_adj = st.session_state.business_settings.get('time_adjustments', {})
        for key, value in time_adj.items():
            if value != 0 and key.endswith('_days'):
                adjustments.append({
                    'category': 'Time',
                    'setting': key.replace('_', ' ').title(),
                    'value': f"{value} days"
                })
        
        planning = st.session_state.business_settings.get('planning_parameters', {})
        if planning.get('forecast_confidence', 1.0) < 1.0:
            adjustments.append({
                'category': 'Planning',
                'setting': 'Forecast Confidence',
                'value': f"{planning['forecast_confidence']*100:.0f}%"
            })
        
        rules = st.session_state.business_settings.get('business_rules', {})
        if rules.get('allocation_method') != 'FIFO':
            adjustments.append({
                'category': 'Business Rule',
                'setting': 'Allocation Method',
                'value': rules['allocation_method']
            })
        
        return adjustments