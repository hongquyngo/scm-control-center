# utils/session_state.py - Session State Initialization

import streamlit as st
from typing import Any, Dict

def initialize_session_state():
    """Initialize all required session state variables"""
    
    # Default business settings
    # default_settings = {
    #     'time_adjustments': {
    #         'etd_offset_days': 0,
    #         'supply_arrival_offset': 0,
    #         'wh_transfer_lead_time': 2,
    #         'transportation_time': 3,
    #         'buffer_days': 7,
    #         'working_days_per_week': 5
    #     },
    #     'planning_parameters': {
    #         'safety_stock_days': 14,
    #         'reorder_point_days': 21,
    #         'forecast_confidence': 0.8,
    #         'planning_horizon_days': 90,
    #         'min_order_coverage_days': 30,
    #         'max_order_coverage_days': 180
    #     },
    #     'order_constraints': {
    #         'default_moq': 100,
    #         'default_spq': 50,
    #         'max_order_limit': 10000,
    #         'order_rounding': 'up',
    #         'enforce_spq': True
    #     },
    #     'business_rules': {
    #         'allocation_method': 'FIFO',
    #         'customer_priority_enabled': True,
    #         'product_priority_enabled': False,
    #         'shelf_life_threshold_days': 30,
    #         'shelf_life_allocation_percent': 0.75,
    #         'seasonality_enabled': True,
    #         'auto_approve_allocation': False
    #     },
    #     'analysis_options': {
    #         'include_forecast_in_gap': True,
    #         'forecast_probability_threshold': 0.7,
    #         'consolidate_by_week': False,
    #         'exclude_expired_inventory': True,
    #         'exclude_blocked_inventory': True,
    #         'consider_in_transit': True
    #     },
    #     'alert_thresholds': {
    #         'critical_shortage_days': 3,
    #         'warning_shortage_days': 7,
    #         'excess_inventory_months': 6,
    #         'slow_moving_months': 3,
    #         'min_fulfillment_rate': 0.85,
    #         'max_allocation_variance': 0.15
    #     }
    # }
    


# Default business settings
    default_settings = {
        'time_adjustments': {
            'oc_etd_offset': 0,
            'forecast_etd_offset': 0,
            'can_arrival_offset': 0,
            'po_crd_offset': 0
        },
        'allocation_rules': {
            'method': 'FIFO',
            'min_shelf_life_percent': 70,
            'customer_priorities': {}
        },
        'po_rules': {
            'reorder_point_days': 30,
            'min_coverage_days': 30,
            'max_coverage_days': 90,
            'default_moq': 1,
            'default_spq': 1,
            'round_to_spq': True
        }
    }




    # Initialize business_settings if not exists
    if 'business_settings' not in st.session_state:
        st.session_state.business_settings = default_settings
        st.session_state.settings_last_updated = None
    
    # Initialize other common session state variables
    defaults = {
        'all_data_loaded': False,
        'data_load_time': None,
        'debug_mode': False,
        'gap_analysis_ran': False,
        'gap_analysis_data': None,
        'filter_entities': [],
        'filter_products': [],
        'filter_brands': [],
        'filter_customers': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_session_value(key: str, default: Any = None) -> Any:
    """Safely get value from session state"""
    return st.session_state.get(key, default)

def set_session_value(key: str, value: Any):
    """Safely set value in session state"""
    st.session_state[key] = value

def clear_session_cache():
    """Clear cache-related session state"""
    cache_keys = [
        'all_data_loaded',
        'data_load_time',
        'gap_analysis_ran',
        'gap_analysis_data',
        'gap_analysis_result',
        'demand_filtered',
        'supply_filtered',
        'gap_df_cached',
        'gap_period_type_cache'
    ]
    
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]