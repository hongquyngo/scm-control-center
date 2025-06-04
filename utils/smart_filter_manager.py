# utils/smart_filter_manager.py

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

class SmartFilterManager:
    """
    Smart Multilevel Interactive Cascading Filter Manager
    Provides real-time filter updates based on selections
    """
    
    def __init__(self, key_prefix: str = ""):
        self.key_prefix = key_prefix
        self._filter_cache = {}
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for filter selections"""
        filter_keys = [
            'entity_selection',
            'customer_selection', 
            'product_selection',
            'brand_selection',
            'status_selection'
        ]
        
        for key in filter_keys:
            full_key = f"{self.key_prefix}{key}"
            if full_key not in st.session_state:
                st.session_state[full_key] = []
    
    def _create_cache_key(self, data_hash: str, selections: Dict[str, List[str]]) -> str:
        """Create unique cache key for filter combinations"""
        cache_data = {
            'data_hash': data_hash,
            'selections': {k: sorted(v) for k, v in selections.items()}
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Create hash of dataframe for cache validation"""
        # Use shape and sample of data for hash
        hash_str = f"{df.shape}_{df.columns.tolist()}_{len(df)}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def get_filtered_options(self, df: pd.DataFrame, 
                           current_selections: Dict[str, List[str]],
                           filter_config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Get available options for each filter based on current selections
        Uses cascading logic - each filter affects the options of filters below it
        """
        if df.empty:
            return {key: [] for key in filter_config.keys()}
        
        # Create cache key
        data_hash = self._get_data_hash(df)
        cache_key = self._create_cache_key(data_hash, current_selections)
        
        # Check cache
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]
        
        # Apply filters in cascade order
        filtered_df = df.copy()
        
        # Apply each filter to narrow down the dataframe
        for filter_key, filter_values in current_selections.items():
            if filter_values and filter_key in filter_config:
                column = filter_config[filter_key]['column']
                if column in filtered_df.columns:
                    if filter_key == 'product_selection':
                        # Special handling for product (PT Code - Name format)
                        pt_codes = [v.split(' - ')[0] for v in filter_values if ' - ' in v]
                        if pt_codes:
                            filtered_df = filtered_df[filtered_df['pt_code'].isin(pt_codes)]
                    else:
                        filtered_df = filtered_df[filtered_df[column].isin(filter_values)]
        
        # Extract available options from filtered dataframe
        available_options = {}
        
        for filter_key, config in filter_config.items():
            column = config['column']
            
            if filter_key == 'product_selection':
                # Special handling for products
                if 'pt_code' in filtered_df.columns and 'product_name' in filtered_df.columns:
                    products_df = filtered_df[['pt_code', 'product_name']].drop_duplicates()
                    products_df = products_df[
                        products_df['pt_code'].notna() & 
                        (products_df['pt_code'] != '') & 
                        (products_df['pt_code'] != 'nan')
                    ]
                    
                    options = []
                    for _, row in products_df.iterrows():
                        pt_code = str(row['pt_code'])
                        product_name = str(row['product_name'])[:50] if pd.notna(row['product_name']) else ""
                        options.append(f"{pt_code} - {product_name}")
                    
                    available_options[filter_key] = sorted(options)
                else:
                    available_options[filter_key] = []
            
            elif column in filtered_df.columns:
                # Regular columns
                unique_values = filtered_df[column].dropna().unique()
                options = sorted([str(v) for v in unique_values if str(v) != 'nan'])
                available_options[filter_key] = options
            else:
                available_options[filter_key] = []
        
        # Cache the result
        self._filter_cache[cache_key] = available_options
        
        # Limit cache size
        if len(self._filter_cache) > 100:
            # Remove oldest entries
            keys_to_remove = list(self._filter_cache.keys())[:50]
            for key in keys_to_remove:
                del self._filter_cache[key]
        
        return available_options
    
    def render_smart_filters(self, df: pd.DataFrame, 
                           filter_config: Dict[str, Dict[str, Any]],
                           show_date_filters: bool = True,
                           date_column: str = "etd") -> Dict[str, Any]:
        """
        Render interactive cascading filters with real-time updates
        
        Args:
            df: DataFrame to filter
            filter_config: Configuration for each filter
                {
                    'entity_selection': {
                        'column': 'legal_entity',
                        'label': 'Legal Entity',
                        'help': 'Select entities'
                    },
                    ...
                }
            show_date_filters: Whether to show date range filters
            date_column: Column name for date filtering
            
        Returns:
            Dict with all filter selections
        """
        
        with st.expander("📎 Smart Filters", expanded=True):
            # Get current selections from session state
            current_selections = {}
            for filter_key in filter_config.keys():
                session_key = f"{self.key_prefix}{filter_key}"
                current_selections[filter_key] = st.session_state.get(session_key, [])
            
            # Get filtered options based on current selections
            with st.spinner("Updating filter options..."):
                available_options = self.get_filtered_options(df, current_selections, filter_config)
            
            # Render filters based on configuration order
            # Group filters into rows for better layout
            filter_keys = list(filter_config.keys())
            
            # Row 1: Entity, Customer, Product
            if len(filter_keys) >= 3:
                col1, col2, col3 = st.columns(3)
                
                # Entity filter
                if 'entity_selection' in filter_config:
                    with col1:
                        self._render_single_filter(
                            'entity_selection',
                            filter_config['entity_selection'],
                            available_options.get('entity_selection', []),
                            current_selections.get('entity_selection', [])
                        )
                
                # Customer filter
                if 'customer_selection' in filter_config:
                    with col2:
                        self._render_single_filter(
                            'customer_selection',
                            filter_config['customer_selection'],
                            available_options.get('customer_selection', []),
                            current_selections.get('customer_selection', [])
                        )
                
                # Product filter
                if 'product_selection' in filter_config:
                    with col3:
                        self._render_single_filter(
                            'product_selection',
                            filter_config['product_selection'],
                            available_options.get('product_selection', []),
                            current_selections.get('product_selection', [])
                        )
            
            # Row 2: Brand and other filters
            remaining_filters = [k for k in filter_keys if k not in ['entity_selection', 'customer_selection', 'product_selection']]
            
            if remaining_filters:
                cols = st.columns(min(3, len(remaining_filters)))
                for idx, filter_key in enumerate(remaining_filters):
                    with cols[idx % len(cols)]:
                        self._render_single_filter(
                            filter_key,
                            filter_config[filter_key],
                            available_options.get(filter_key, []),
                            current_selections.get(filter_key, [])
                        )
            
            # Date filters (if enabled)
            date_filters = {}
            if show_date_filters and date_column in df.columns:
                col_date1, col_date2 = st.columns(2)
                
                # Get date range from data
                dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
                
                with col_date1:
                    default_start = dates.min().date() if len(dates) > 0 else date.today()
                    date_filters['start_date'] = st.date_input(
                        f"From Date ({date_column.replace('_', ' ').title()})",
                        value=default_start,
                        key=f"{self.key_prefix}start_date"
                    )
                
                with col_date2:
                    default_end = dates.max().date() if len(dates) > 0 else date.today()
                    date_filters['end_date'] = st.date_input(
                        f"To Date ({date_column.replace('_', ' ').title()})",
                        value=default_end,
                        key=f"{self.key_prefix}end_date"
                    )
            
            # Show filter summary
            self._show_filter_summary(current_selections, available_options)
            
            # Combine all filters
            all_filters = {
                'selections': current_selections,
                'date_filters': date_filters,
                'filter_config': filter_config
            }
            
            return all_filters
    
    def _render_single_filter(self, filter_key: str, config: Dict[str, Any], 
                            options: List[str], current_selection: List[str]):
        """Render a single multiselect filter with smart updates"""
        session_key = f"{self.key_prefix}{filter_key}"
        
        # Handle case where current selection has items not in available options
        # This can happen when filters are narrowed down
        valid_selection = [v for v in current_selection if v in options]
        
        # Update session state if selection was filtered
        if len(valid_selection) != len(current_selection):
            st.session_state[session_key] = valid_selection
            current_selection = valid_selection
        
        # Render multiselect
        selection = st.multiselect(
            config['label'],
            options=options,
            default=current_selection,
            key=f"{session_key}_widget",
            help=config.get('help', ''),
            placeholder=config.get('placeholder', f"Select {config['label'].lower()}")
        )
        
        # Check if selection changed
        if sorted(selection) != sorted(current_selection):
            st.session_state[session_key] = selection
            st.rerun()
    
    def _show_filter_summary(self, selections: Dict[str, List[str]], 
                           available_options: Dict[str, List[str]]):
        """Show summary of active filters"""
        active_filters = []
        for key, values in selections.items():
            if values:
                active_filters.append(f"{len(values)} {key.replace('_selection', '')}")
        
        if active_filters:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.caption(f"🔍 Active filters: {', '.join(active_filters)}")
            
            with col2:
                total_options = sum(len(opts) for opts in available_options.values())
                st.caption(f"📊 {total_options} options available")
            
            with col3:
                if st.button("🔄 Clear All", key=f"{self.key_prefix}clear_filters"):
                    for key in selections.keys():
                        session_key = f"{self.key_prefix}{key}"
                        st.session_state[session_key] = []
                    st.rerun()
    
    def apply_filters_to_dataframe(self, df: pd.DataFrame, 
                                 filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters to dataframe"""
        filtered_df = df.copy()
        
        selections = filters.get('selections', {})
        date_filters = filters.get('date_filters', {})
        filter_config = filters.get('filter_config', {})
        
        # Apply selection filters
        for filter_key, filter_values in selections.items():
            if filter_values and filter_key in filter_config:
                column = filter_config[filter_key]['column']
                
                if column in filtered_df.columns:
                    if filter_key == 'product_selection':
                        # Special handling for products
                        pt_codes = [v.split(' - ')[0] for v in filter_values if ' - ' in v]
                        if pt_codes:
                            filtered_df = filtered_df[filtered_df['pt_code'].isin(pt_codes)]
                    else:
                        filtered_df = filtered_df[filtered_df[column].isin(filter_values)]
        
        # Apply date filters
        if date_filters:
            date_column = filter_config.get('date_column', 'etd')
            if date_column in filtered_df.columns:
                filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors='coerce')
                
                if 'start_date' in date_filters:
                    start_date = pd.to_datetime(date_filters['start_date'])
                    filtered_df = filtered_df[
                        filtered_df[date_column].isna() | 
                        (filtered_df[date_column] >= start_date)
                    ]
                
                if 'end_date' in date_filters:
                    end_date = pd.to_datetime(date_filters['end_date'])
                    filtered_df = filtered_df[
                        filtered_df[date_column].isna() | 
                        (filtered_df[date_column] <= end_date)
                    ]
        
        return filtered_df
    
    def render_filter_toggle(self) -> bool:
        """Render toggle button for filter mode selection"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Initialize session state for filter mode
            mode_key = f"{self.key_prefix}filter_mode"
            if mode_key not in st.session_state:
                st.session_state[mode_key] = "Standard"  # Default to Standard
            
            # Toggle between modes
            filter_mode = st.radio(
                "🔧 Filter Mode",
                ["Standard Filters", "Smart Filters (Cascading)"],
                index=0 if st.session_state[mode_key] == "Standard" else 1,
                horizontal=True,
                key=f"{self.key_prefix}filter_mode_radio",
                help="""
                **Standard Filters**: Independent filters - select any combination
                **Smart Filters**: Cascading filters - options update based on selections
                """
            )
            
            # Update session state
            st.session_state[mode_key] = "Standard" if "Standard" in filter_mode else "Smart"
            
        with col2:
            if st.session_state[mode_key] == "Smart":
                st.info("🔗 Options will cascade")
            else:
                st.success("✅ All combinations allowed")
                
        with col3:
            # Quick stats about current mode
            st.caption(f"Mode: {st.session_state[mode_key]}")
            
        return st.session_state[mode_key] == "Smart"


    def clear_cache(self):
        """Clear filter cache"""
        self._filter_cache.clear()