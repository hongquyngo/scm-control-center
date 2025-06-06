# pages/3_📊_GAP_Analysis.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import text
import logging

# Import refactored modules
from utils.data_manager import DataManager
from utils.filters import FilterManager
from utils.display_components import DisplayComponents
from utils.formatters import (
    format_number, format_currency, format_percentage,
    check_missing_dates, check_past_dates
)
from utils.helpers import (
    convert_df_to_excel,
    export_multiple_sheets,
    convert_to_period,
    sort_period_columns,
    save_to_session_state,
    get_from_session_state,
    is_past_period,
    parse_week_period,
    parse_month_period,
    create_period_pivot,
    apply_period_indicators
)
from utils.session_state import initialize_session_state
from utils.db import get_db_engine
from utils.smart_filter_manager import SmartFilterManager
from utils.date_mode_component import DateModeComponent

from typing import Tuple, Dict, Any, Optional, List

# Configure logging
logger = logging.getLogger(__name__)

# === Page Config ===
st.set_page_config(
    page_title="GAP Analysis - SCM",
    page_icon="📊",
    layout="wide"
)

# === Initialize Session State ===
initialize_session_state()

# === Constants ===
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]
STYLE_MODES = ["None", "🔴 Highlight Shortage", "🌈 Heatmap"]

# === Initialize Components ===
@st.cache_resource
def get_data_manager():
    return DataManager()

data_manager = get_data_manager()

# === Debug Mode Toggle ===
col_debug1, col_debug2 = st.columns([6, 1])
with col_debug2:
    debug_mode = st.checkbox("🐛 Debug Mode", value=False, key="gap_debug_mode")

if debug_mode:
    st.info("🐛 Debug Mode is ON - Additional information will be displayed")

# === Header with Navigation ===
DisplayComponents.show_page_header(
    title="Supply-Demand GAP Analysis",
    icon="📊",
    prev_page="pages/2_📥_Supply_Analysis.py",
    next_page="pages/4_🧩_Allocation_Plan.py"
)


# Dual date mode selection for demand and supply
st.markdown("### 📅 Date Analysis Mode")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📤 Demand Dates")
    use_adjusted_demand = DateModeComponent.render_date_mode_selector("gap_demand_")

with col2:
    st.markdown("#### 📥 Supply Dates")
    use_adjusted_supply = DateModeComponent.render_date_mode_selector("gap_supply_")

st.markdown("---")

# === Initialize session state for GAP ===
if 'gap_analysis_ran' not in st.session_state:
    st.session_state['gap_analysis_ran'] = False
if 'gap_analysis_data' not in st.session_state:
    st.session_state['gap_analysis_data'] = None

# === Helper Functions ===
def get_demand_date_column(df, use_adjusted):
    """Get appropriate demand date column"""
    return DateModeComponent.get_date_column_for_display(df, 'etd', use_adjusted)

def get_supply_date_column(df, source_type, use_adjusted):
    """Get appropriate supply date column based on source type"""
    date_mapping = {
        'Inventory': 'date_ref',
        'Pending CAN': 'arrival_date',
        'Pending PO': 'eta',
        'Pending WH Transfer': 'transfer_date'
    }
    base_column = date_mapping.get(source_type, 'date_ref')
    return DateModeComponent.get_date_column_for_display(df, base_column, use_adjusted)


# ADD this function sau function get_supply_date_column

def get_supply_date_column_for_gap(df, source_type, use_adjusted):
    """Get supply date column - handle unified dates for All view in GAP"""
    # Check if this is combined data with unified dates
    if 'unified_date' in df.columns and 'source_type' in df.columns and len(df['source_type'].unique()) > 1:
        if use_adjusted and 'unified_date_adjusted' in df.columns:
            return 'unified_date_adjusted'
        else:
            return 'unified_date'
    
    # Otherwise use source-specific dates
    date_mapping = {
        'Inventory': 'date_ref',
        'Pending CAN': 'arrival_date', 
        'Pending PO': 'eta',
        'Pending WH Transfer': 'transfer_date'
    }
    base_column = date_mapping.get(source_type, 'date_ref')
    return DateModeComponent.get_date_column_for_display(df, base_column, use_adjusted)




# === Data Loading Functions ===
def load_and_prepare_demand_data(selected_demand_sources, include_converted):
    """Load and standardize demand data based on source selection"""
    try:
        # Use data_manager to get demand data
        df = data_manager.get_demand_data(sources=selected_demand_sources, include_converted=include_converted)
        
        if debug_mode and not df.empty:
            st.write(f"🐛 Demand data shape: {df.shape}")
            st.write(f"🐛 Demand columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
            st.write(f"🐛 Unique products: {df['pt_code'].nunique()}")
            
            # Check for adjustment columns
            adj_cols = [col for col in df.columns if '_adjusted' in col or '_original' in col]
            if adj_cols:
                st.write(f"🐛 Adjustment columns found: {adj_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading demand data: {str(e)}")
        st.error(f"Failed to load demand data: {str(e)}")
        return pd.DataFrame()

def load_and_prepare_supply_data(selected_supply_sources, exclude_expired=True):
    """Load and standardized supply data based on source selection"""
    try:
        # Use data_manager to get supply data
        df = data_manager.get_supply_data(sources=selected_supply_sources, exclude_expired=exclude_expired)
        
        if debug_mode and not df.empty:
            st.write(f"🐛 Supply data shape: {df.shape}")
            st.write(f"🐛 Supply sources: {df['source_type'].value_counts().to_dict()}")
            
            # Check for adjustment columns
            adj_cols = [col for col in df.columns if '_adjusted' in col or '_original' in col]
            if adj_cols:
                st.write(f"🐛 Adjustment columns found: {adj_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading supply data: {str(e)}")
        st.error(f"Failed to load supply data: {str(e)}")
        return pd.DataFrame()

def enhance_demand_with_allocation_info(df_demand):
    """Add allocation information to demand dataframe - IMPROVED VERSION"""
    engine = get_db_engine()
    
    # Initialize allocation columns first
    df_demand['total_allocated'] = 0
    df_demand['total_delivered'] = 0
    df_demand['undelivered_allocated'] = 0
    
    try:
        # Query all allocations using the view
        allocations_query = """
        SELECT 
            pt_code,
            demand_type,
            demand_reference_id,
            SUM(total_allocated_qty) as total_allocated,
            SUM(total_delivered_qty) as total_delivered,
            SUM(undelivered_qty) as undelivered_allocated
        FROM active_allocations_view
        GROUP BY pt_code, demand_type, demand_reference_id
        """
        
        allocations_df = pd.read_sql(text(allocations_query), engine)
        
        if not allocations_df.empty and 'demand_line_id' in df_demand.columns:
            # Create mapping for demand_reference_id
            df_demand['demand_ref_id'] = df_demand['demand_line_id'].str.extract(r'(\d+)_')[0]
            df_demand['demand_ref_id'] = pd.to_numeric(df_demand['demand_ref_id'], errors='coerce')
            
            # Determine demand type from source_type
            df_demand['alloc_demand_type'] = df_demand['source_type'].map({
                'OC': 'OC',
                'Forecast': 'FORECAST'
            })
            
            # Merge allocations
            df_demand = df_demand.merge(
                allocations_df,
                left_on=['pt_code', 'alloc_demand_type', 'demand_ref_id'],
                right_on=['pt_code', 'demand_type', 'demand_reference_id'],
                how='left',
                suffixes=('', '_alloc')
            )
            
            # Update allocation columns
            for col in ['total_allocated', 'total_delivered', 'undelivered_allocated']:
                col_alloc = f'{col}_alloc'
                if col_alloc in df_demand.columns:
                    df_demand[col] = df_demand[col_alloc].fillna(0)
                    df_demand.drop(columns=[col_alloc], inplace=True)
            
            # Clean up temp columns
            df_demand.drop(columns=['demand_ref_id', 'alloc_demand_type', 'demand_type', 
                                   'demand_reference_id'], errors='ignore', inplace=True)
                
    except Exception as e:
        logger.error(f"Error loading allocation data: {str(e)}")
        # Continue with zero allocations
    
    # Ensure columns are numeric
    for col in ['total_allocated', 'total_delivered', 'undelivered_allocated']:
        df_demand[col] = pd.to_numeric(df_demand[col], errors='coerce').fillna(0)
    
    # Calculate unallocated demand
    df_demand['unallocated_demand'] = (
        df_demand['demand_quantity'] - df_demand['total_allocated']
    ).clip(lower=0)
    
    # Add allocation status with percentage
    df_demand['allocation_status'] = df_demand.apply(
        lambda x: 'Fully Allocated' if x['unallocated_demand'] <= 0 
        else f'Partial ({x["total_allocated"]/x["demand_quantity"]*100:.0f}%)' if x['total_allocated'] > 0 and x['demand_quantity'] > 0
        else 'Not Allocated', 
        axis=1
    )
    
    return df_demand


def adjust_supply_for_allocations(df_supply):
    """Adjust supply quantities based on active allocations"""
    # Get allocations from session or load
    df_allocations = get_from_session_state('active_allocations', pd.DataFrame())
    
    if df_allocations.empty:
        # Load if not in session
        df_allocations = data_manager.load_active_allocations()
    
    if not df_allocations.empty and not df_supply.empty:
        # Adjust available supply by subtracting undelivered allocations
        for _, alloc in df_allocations.iterrows():
            # Match by product and entity
            mask = (
                (df_supply['pt_code'] == alloc.get('pt_code', '')) & 
                (df_supply['legal_entity'] == alloc.get('legal_entity_name', ''))
            )
            
            if mask.any() and 'undelivered_qty' in alloc:
                df_supply.loc[mask, 'quantity'] = df_supply.loc[mask, 'quantity'].apply(
                    lambda x: max(0, x - alloc['undelivered_qty'])
                )
    
    # Remove rows with zero quantity
    df_supply = df_supply[df_supply['quantity'] > 0].copy()
    
    return df_supply

# === Source Selection ===
def select_gap_sources():
    """Select demand and supply sources for GAP analysis"""
    st.markdown("### 📊 Data Source Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Demand Sources")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            demand_oc = st.checkbox("OC", value=True, key="demand_oc")
        with col1_2:
            demand_forecast = st.checkbox("Forecast", value=True, key="demand_forecast")
        
        selected_demand_sources = []
        if demand_oc:
            selected_demand_sources.append("OC")
        if demand_forecast:
            selected_demand_sources.append("Forecast")
        
        if demand_forecast:
            include_converted = st.checkbox(
                "Include Converted Forecasts", 
                value=False,
                help="⚠️ May cause double counting if OC is also selected",
                key="include_converted_forecasts"
            )
        else:
            include_converted = False
        
        # Customer filter - only for demand
        st.markdown("##### Customer Filter")
        
        # Get customers from session state or empty list
        all_customers = st.session_state.get('filter_customers', [])
        
        selected_customers = st.multiselect(
            "Select Customers", 
            options=all_customers,
            key="gap_customer",
            help="Filter demand by specific customers",
            placeholder="Choose customers..." if all_customers else "No customers available"
        )
    
    with col2:
        st.markdown("#### 📥 Supply Sources")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            supply_inv = st.checkbox("Inventory", value=True, key="supply_inv")
            supply_can = st.checkbox("Pending CAN", value=True, key="supply_can")
        with col2_2:
            supply_po = st.checkbox("Pending PO", value=True, key="supply_po")
            supply_wht = st.checkbox("Pending WH Transfer", value=True, key="supply_wht")
        
        exclude_expired = st.checkbox(
            "Exclude Expired", 
            value=True,
            help="Exclude expired inventory items",
            key="exclude_expired_gap"
        )
        
        selected_supply_sources = []
        if supply_inv:
            selected_supply_sources.append("Inventory")
        if supply_can:
            selected_supply_sources.append("Pending CAN")
        if supply_po:
            selected_supply_sources.append("Pending PO")
        if supply_wht:
            selected_supply_sources.append("Pending WH Transfer")
    
    return {
        "demand": selected_demand_sources,
        "supply": selected_supply_sources,
        "include_converted": include_converted,
        "exclude_expired": exclude_expired,
        "selected_customers": selected_customers
    }

# === Filtering Functions ===
# pages/3_📊_GAP_Analysis.py - Replace entire apply_gap_filters function

def apply_gap_filters(df_demand=None, df_supply=None, use_adjusted_demand=True, use_adjusted_supply=True):
    """Apply filters with mode toggle for GAP analysis"""
    # Initialize filter manager
    filter_manager = SmartFilterManager(key_prefix="gap_")
    
    # Render toggle
    use_smart_filters = filter_manager.render_filter_toggle()
    
    
    if use_smart_filters:
        return apply_smart_gap_filters(
            df_demand, df_supply, 
            use_adjusted_demand, use_adjusted_supply, 
            filter_manager
        )
    else:
        return apply_standard_gap_filters(
            df_demand, df_supply,
            use_adjusted_demand, use_adjusted_supply
        )

def apply_standard_gap_filters(df_demand, df_supply, use_adjusted_demand, use_adjusted_supply):
    """Standard filters for GAP analysis"""
    with st.expander("📎 Filters for GAP Analysis", expanded=True):
        # Combine unique values from both demand and supply
        filters = {}
        
        # Get combined unique values
        all_entities = set()
        all_products = []
        all_brands = set()
        all_customers = set()
        
        if df_demand is not None and not df_demand.empty:
            all_entities.update(df_demand['legal_entity'].dropna().unique())
            all_brands.update(df_demand['brand'].dropna().unique())
            if 'customer' in df_demand.columns:
                all_customers.update(df_demand['customer'].dropna().unique())
            # Get products
            if 'pt_code' in df_demand.columns and 'product_name' in df_demand.columns:
                demand_products = df_demand[['pt_code', 'product_name']].drop_duplicates()
                demand_products = demand_products[demand_products['pt_code'].notna() & (demand_products['pt_code'] != 'nan')]
                all_products.extend(demand_products.values.tolist())
        
        if df_supply is not None and not df_supply.empty:
            all_entities.update(df_supply['legal_entity'].dropna().unique())
            all_brands.update(df_supply['brand'].dropna().unique())
            # Get products
            if 'pt_code' in df_supply.columns and 'product_name' in df_supply.columns:
                supply_products = df_supply[['pt_code', 'product_name']].drop_duplicates()
                supply_products = supply_products[supply_products['pt_code'].notna() & (supply_products['pt_code'] != 'nan')]
                all_products.extend(supply_products.values.tolist())
        
        # Remove duplicates and sort
        all_entities = sorted(list(all_entities))
        all_brands = sorted(list(all_brands))
        all_customers = sorted(list(all_customers))
        
        # Create unique product options
        unique_products = {}
        for pt_code, product_name in all_products:
            if pd.notna(pt_code) and str(pt_code) != 'nan':
                unique_products[str(pt_code)] = f"{pt_code} - {str(product_name)[:50] if pd.notna(product_name) else ''}"
        
        product_options = sorted(list(unique_products.values()))
        
        # Row 1: Entity, Product, Brand
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filters['entity'] = st.multiselect(
                "Legal Entity",
                all_entities,
                key="gap_entity_filter_std",
                placeholder="All entities"
            )
        
        with col2:
            selected_products = st.multiselect(
                "Product (PT Code - Name)",
                product_options,
                key="gap_product_filter_std",
                placeholder="All products"
            )
            filters['product'] = [p.split(' - ')[0] for p in selected_products]
        
        with col3:
            filters['brand'] = st.multiselect(
                "Brand",
                list(all_brands),
                key="gap_brand_filter_std",
                placeholder="All brands"
            )
        
        # Row 2: Customer, Data Presence
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if all_customers:
                filters['customer'] = st.multiselect(
                    "Customer (from Demand)",
                    list(all_customers),
                    key="gap_customer_filter_std",
                    placeholder="All customers"
                )
        
        with col5:
            data_presence = st.multiselect(
                "Data Presence",
                ["Demand & Supply", "Demand Only", "Supply Only"],
                key="gap_presence_filter_std",
                placeholder="All types",
                help="Filter products by their data availability"
            )
            filters['data_presence'] = data_presence
        
        # Date range
        st.markdown("#### 📅 Date Range")
        col_date1, col_date2 = st.columns(2)
        
        # Determine date range from both sources
        min_date = datetime.today().date()
        max_date = datetime.today().date()
        
        # Get demand date range
        if df_demand is not None and not df_demand.empty:
            demand_date_col = get_demand_date_column(df_demand, use_adjusted_demand)
            if demand_date_col in df_demand.columns:
                demand_dates = pd.to_datetime(df_demand[demand_date_col], errors='coerce').dropna()
                if len(demand_dates) > 0:
                    min_date = min(min_date, demand_dates.min().date())
                    max_date = max(max_date, demand_dates.max().date())
        
        # Get supply date range
        if df_supply is not None and not df_supply.empty:
            # Check each source type's date column
            for source_type in df_supply['source_type'].unique():
                source_df = df_supply[df_supply['source_type'] == source_type]
                supply_date_col = get_supply_date_column(source_df, source_type, use_adjusted_supply)
                
                if supply_date_col in source_df.columns:
                    supply_dates = pd.to_datetime(source_df[supply_date_col], errors='coerce').dropna()
                    if len(supply_dates) > 0:
                        min_date = min(min_date, supply_dates.min().date())
                        max_date = max(max_date, supply_dates.max().date())
        
        with col_date1:
            filters['start_date'] = st.date_input(
                "From Date",
                value=min_date,
                key="gap_start_date_std"
            )
        
        with col_date2:
            filters['end_date'] = st.date_input(
                "To Date",
                value=max_date,
                key="gap_end_date_std"
            )
        
        # Show active filters
        active_filters = sum(1 for k, v in filters.items() 
                           if k not in ['start_date', 'end_date'] and v and v != [])
        if active_filters > 0:
            st.success(f"🔍 {active_filters} filters active")
    
    return filters

def apply_smart_gap_filters(df_demand, df_supply, use_adjusted_demand, use_adjusted_supply, filter_manager):
    """Apply smart filters for GAP analysis"""
    try:
        # Combine data for unified filter options
        combined_df = pd.DataFrame()
        
        if df_demand is not None and not df_demand.empty:
            demand_subset = df_demand[['legal_entity', 'pt_code', 'product_name', 'brand']].copy()
            demand_subset['data_source'] = 'Demand'
            combined_df = pd.concat([combined_df, demand_subset])
        
        if df_supply is not None and not df_supply.empty:
            supply_subset = df_supply[['legal_entity', 'pt_code', 'product_name', 'brand']].copy()
            supply_subset['data_source'] = 'Supply'
            combined_df = pd.concat([combined_df, supply_subset])
        
        if combined_df.empty:
            return {}
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates()
        
        # Configure filters
        filter_config = {}
        
        # Entity filter
        if 'legal_entity' in combined_df.columns:
            filter_config['entity_selection'] = {
                'column': 'legal_entity',
                'label': 'Legal Entity',
                'help': 'Filter by legal entities',
                'placeholder': 'Choose legal entities...'
            }
        
        # Product filter
        if 'pt_code' in combined_df.columns:
            filter_config['product_selection'] = {
                'column': 'pt_code',
                'label': 'Product (PT Code - Name)',
                'help': 'Search by PT Code or Product Name',
                'placeholder': 'Type to search products...'
            }
        
        # Brand filter
        if 'brand' in combined_df.columns:
            filter_config['brand_selection'] = {
                'column': 'brand',
                'label': 'Brand',
                'help': 'Filter by brands',
                'placeholder': 'Choose brands...'
            }
        
        # Data presence filter
        filter_config['data_presence_selection'] = {
            'column': 'data_source',
            'label': 'Data Presence',
            'help': 'Filter by where products appear',
            'placeholder': 'Choose data presence...'
        }
        
        # Customer filter (from demand only)
        if df_demand is not None and 'customer' in df_demand.columns:
            customers_df = df_demand[['customer']].drop_duplicates()
            customers_df = customers_df[customers_df['customer'].notna()]
            
            if not customers_df.empty:
                # Add customer data to combined
                temp_combined = combined_df.copy()
                temp_combined['customer'] = None
                
                for customer in customers_df['customer'].unique():
                    temp_combined = pd.concat([temp_combined, pd.DataFrame({
                        'customer': [customer],
                        'legal_entity': [None],
                        'pt_code': [None],
                        'product_name': [None],
                        'brand': [None],
                        'data_source': ['Demand']
                    })], ignore_index=True)
                
                combined_df = temp_combined
                
                filter_config['customer_selection'] = {
                    'column': 'customer',
                    'label': 'Customer',
                    'help': 'Filter by customers (from demand)',
                    'placeholder': 'Choose customers...'
                }
        
        # Render smart filters
        with st.container():
            st.markdown("### 📎 Smart Filters for GAP Analysis")
            
            # Add GAP-specific info
            col1, col2, col3 = st.columns(3)
            with col1:
                if df_demand is not None:
                    st.caption(f"📤 Demand: {len(df_demand):,} records")
            with col2:
                if df_supply is not None:
                    st.caption(f"📥 Supply: {len(df_supply):,} records")
            with col3:
                unique_products = combined_df['pt_code'].nunique()
                st.caption(f"🔗 Products: {unique_products:,}")
            
            # Render filters
            filters_result = filter_manager.render_smart_filters(
                df=combined_df,
                filter_config=filter_config,
                show_date_filters=False  # We'll handle dates separately
            )
            
            # Date range filters
            st.markdown("#### 📅 Date Range")
            col1, col2 = st.columns(2)
            
            # Determine date range
            min_date = datetime.today().date()
            max_date = datetime.today().date()
            
            # Get demand date range
            if df_demand is not None:
                demand_date_col = get_demand_date_column(df_demand, use_adjusted_demand)
                if demand_date_col in df_demand.columns:
                    demand_dates = pd.to_datetime(df_demand[demand_date_col], errors='coerce').dropna()
                    if len(demand_dates) > 0:
                        min_date = min(min_date, demand_dates.min().date())
                        max_date = max(max_date, demand_dates.max().date())
            
            # Get supply date range
            if df_supply is not None:
                for source_type in df_supply['source_type'].unique():
                    source_df = df_supply[df_supply['source_type'] == source_type]
                    supply_date_col = get_supply_date_column(source_df, source_type, use_adjusted_supply)
                    
                    if supply_date_col in source_df.columns:
                        supply_dates = pd.to_datetime(source_df[supply_date_col], errors='coerce').dropna()
                        if len(supply_dates) > 0:
                            min_date = min(min_date, supply_dates.min().date())
                            max_date = max(max_date, supply_dates.max().date())
            
            with col1:
                start_date = st.date_input(
                    "From Date",
                    value=min_date,
                    key="gap_start_date_smart"
                )
            
            with col2:
                end_date = st.date_input(
                    "To Date",
                    value=max_date,
                    key="gap_end_date_smart"
                )
            
            # Add date filters to result
            filters_result['date_filters'] = {
                'start_date': start_date,
                'end_date': end_date
            }
        
        # Extract selections
        selections = filters_result.get('selections', {})
        
        # Show active filters summary
        active_filters = {k: len(v) for k, v in selections.items() if v}
        if active_filters:
            st.success(f"🔍 Active filters: {', '.join([f'{k.replace('_selection', '')}: {v}' for k, v in active_filters.items()])}")
        
        # Convert to standard filter format
        filters = {
            'entity': selections.get('entity_selection', []),
            'product': [p.split(' - ')[0] for p in selections.get('product_selection', []) if ' - ' in p],
            'brand': selections.get('brand_selection', []),
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Add customer filter if present
        if 'customer_selection' in selections:
            filters['customer'] = selections['customer_selection']
        
        # Add data presence filter
        if 'data_presence_selection' in selections:
            filters['data_presence'] = selections['data_presence_selection']
        
        return filters
        
    except Exception as e:
        logger.error(f"Smart filter error in GAP: {str(e)}", exc_info=True)
        st.error(f"⚠️ Smart filters error: {str(e)}")
        st.info("💡 Please switch to Standard Filters mode")
        
        return {}


def update_filter_options(df_demand, df_supply):
    """Update filter options based on loaded data"""
    # Get unique values
    all_entities = sorted(set(
        df_demand["legal_entity"].dropna().astype(str).unique().tolist() + 
        df_supply["legal_entity"].dropna().astype(str).unique().tolist()
    ))
    all_entities = [e for e in all_entities if e and e != 'nan']
    
    # Enhanced product options with names
    demand_products = df_demand[['pt_code', 'product_name']].drop_duplicates()
    supply_products = df_supply[['pt_code', 'product_name']].drop_duplicates()
    all_products_df = pd.concat([demand_products, supply_products]).drop_duplicates()
    
    # Filter out invalid PT codes
    all_products_df = all_products_df[
        (all_products_df['pt_code'].notna()) & 
        (all_products_df['pt_code'] != '') &
        (all_products_df['pt_code'] != 'nan')
    ]
    
    product_options = []
    all_pt_codes = []
    for _, row in all_products_df.iterrows():
        pt_code = str(row['pt_code'])
        product_name = str(row['product_name'])[:50] if pd.notna(row['product_name']) else ""
        option = f"{pt_code} - {product_name}"
        product_options.append(option)
        all_pt_codes.append(pt_code)
    
    product_options = sorted(list(set(product_options)))
    all_pt_codes = sorted(list(set(all_pt_codes)))
    
    all_brands = sorted(set(
        df_demand["brand"].dropna().astype(str).unique().tolist() + 
        df_supply["brand"].dropna().astype(str).unique().tolist()
    ))
    all_brands = [b for b in all_brands if b and b != 'nan']
    
    # Get customers from demand only
    all_customers = []
    if 'customer' in df_demand.columns:
        all_customers = sorted(df_demand["customer"].dropna().astype(str).unique().tolist())
        all_customers = [c for c in all_customers if c and c != 'nan']
    
    # Store in session state
    st.session_state['filter_entities'] = all_entities
    st.session_state['filter_products'] = all_pt_codes
    st.session_state['filter_product_options'] = product_options
    st.session_state['filter_brands'] = all_brands
    st.session_state['filter_customers'] = all_customers
    
    if debug_mode:
        st.write(f"🐛 Filter options updated:")
        st.write(f"- Entities: {len(all_entities)}")
        st.write(f"- Products: {len(all_pt_codes)}")
        st.write(f"- Brands: {len(all_brands)}")
        st.write(f"- Customers: {len(all_customers)}")


def apply_filters_to_data(df_demand, df_supply, filters, selected_customers, 
                         use_adjusted_demand, use_adjusted_supply):
    """Apply filters to demand and supply dataframes with date mode awareness"""
    
    # Create a copy to avoid modifying original
    filtered_demand = df_demand.copy()
    filtered_supply = df_supply.copy()
    
    # Clean product codes for consistent matching
    if 'pt_code' in filtered_demand.columns:
        filtered_demand['pt_code'] = filtered_demand['pt_code'].astype(str).str.strip()
    
    if 'pt_code' in filtered_supply.columns:
        filtered_supply['pt_code'] = filtered_supply['pt_code'].astype(str).str.strip()
    
    # === Apply filters to DEMAND ===
    
    # Apply entity filter
    if filters.get('entity'):
        filtered_demand = filtered_demand[filtered_demand['legal_entity'].isin(filters['entity'])]
    
    # Apply product filter with cleaned codes
    if filters.get('product'):
        # Clean filter values too
        clean_products = [str(p).strip() for p in filters['product']]
        filtered_demand = filtered_demand[filtered_demand['pt_code'].isin(clean_products)]
    
    # Apply brand filter
    if filters.get('brand'):
        filtered_demand = filtered_demand[filtered_demand['brand'].isin(filters['brand'])]
    
    # Apply customer filter (from source selection, not from filters)
    if selected_customers and 'customer' in filtered_demand.columns:
        filtered_demand = filtered_demand[filtered_demand['customer'].isin(selected_customers)]
    
    # Apply customer filter from filters (if exists)
    if filters.get('customer') and 'customer' in filtered_demand.columns:
        filtered_demand = filtered_demand[filtered_demand['customer'].isin(filters['customer'])]
    
    # Apply date filter for demand
    demand_date_col = get_demand_date_column(filtered_demand, use_adjusted_demand)
    if demand_date_col in filtered_demand.columns and filters.get('start_date') and filters.get('end_date'):
        # Convert dates to datetime for comparison
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])
        
        # Convert column to datetime
        filtered_demand[demand_date_col] = pd.to_datetime(filtered_demand[demand_date_col], errors='coerce')
        
        # Apply filter (include NaT values)
        date_mask = (
            filtered_demand[demand_date_col].isna() |
            ((filtered_demand[demand_date_col] >= start_date) & 
             (filtered_demand[demand_date_col] <= end_date))
        )
        filtered_demand = filtered_demand[date_mask]
    
    # === Apply filters to SUPPLY ===
    
    # Apply entity filter
    if filters.get('entity'):
        filtered_supply = filtered_supply[filtered_supply['legal_entity'].isin(filters['entity'])]
    
    # Apply product filter with cleaned codes
    if filters.get('product'):
        # Clean filter values
        clean_products = [str(p).strip() for p in filters['product']]
        filtered_supply = filtered_supply[filtered_supply['pt_code'].isin(clean_products)]
        
        # Debug: Check if product exists before other filters
        if debug_mode and filtered_supply.empty:
            st.write("🔍 Product filter eliminated all supply records")
            # Check original data
            original_match = df_supply[df_supply['pt_code'].astype(str).str.strip().isin(clean_products)]
            if not original_match.empty:
                st.write(f"  Product exists in original supply: {len(original_match)} records")
                st.write(f"  Lost due to entity filter")
    
    # Apply brand filter
    if filters.get('brand'):
        filtered_supply = filtered_supply[filtered_supply['brand'].isin(filters['brand'])]
    
    # Apply date filters per source type
    if 'source_type' in filtered_supply.columns and filters.get('start_date') and filters.get('end_date'):
        date_filtered_dfs = []
        
        # Convert filter dates to datetime
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])
        
        # Process each source type separately
        for source_type in filtered_supply['source_type'].unique():
            source_df = filtered_supply[filtered_supply['source_type'] == source_type].copy()
            
            # Get appropriate date column for this source
            supply_date_col = get_supply_date_column(source_df, source_type, use_adjusted_supply)
            
            if supply_date_col in source_df.columns:
                # Convert to datetime
                source_df[supply_date_col] = pd.to_datetime(source_df[supply_date_col], errors='coerce')
                
                # Apply date filter (include NaT values)
                date_mask = (
                    source_df[supply_date_col].isna() |
                    ((source_df[supply_date_col] >= start_date) & 
                     (source_df[supply_date_col] <= end_date))
                )
                source_df = source_df[date_mask]
            
            date_filtered_dfs.append(source_df)
        
        # Combine filtered data
        if date_filtered_dfs:
            filtered_supply = pd.concat(date_filtered_dfs, ignore_index=True)
        else:
            filtered_supply = pd.DataFrame()
    
    # Apply data presence filter if specified
    if filters.get('data_presence'):
        # This is applied after getting initial filtered data
        # Will be handled by the calling function
        pass
    
    return filtered_demand, filtered_supply



# === GAP Calculation ===
def calculate_gap_with_carry_forward(df_demand, df_supply, period_type="Weekly", 
                                    use_adjusted_demand=True, use_adjusted_supply=True):
    """Calculate supply-demand gap with allocation awareness and date mode support"""
    
    # Early return if both empty
    if df_demand.empty and df_supply.empty:
        st.warning("No data available for GAP calculation")
        return pd.DataFrame()
    
    # Special handling for demand-only scenario (no supply)
    if not df_demand.empty and df_supply.empty:
        st.info("📤 Demand-Only Scenario: No supply available for these products")
        
        # Enhance demand with allocation info
        df_demand_enhanced = enhance_demand_with_allocation_info(df_demand)
        
        df_d = df_demand_enhanced.copy()
        
        # Convert demand dates to periods
        demand_date_col = get_demand_date_column(df_d, use_adjusted_demand)
        df_d["period"] = convert_to_period(df_d[demand_date_col], period_type)
        
        # Remove invalid periods
        df_d = df_d[df_d["period"].notna() & (df_d["period"] != "")]
        
        if df_d.empty:
            st.warning("No valid period data for demand")
            return pd.DataFrame()
        
        # Group demand
        demand_grouped = df_d.groupby(
            ["pt_code", "product_name", "package_size", "standard_uom", "period"],
            as_index=False,
            dropna=False
        ).agg({
            "demand_quantity": "sum",
            "unallocated_demand": "sum"
        })
        
        # Use unallocated demand for GAP calculation
        demand_grouped["total_demand_qty"] = demand_grouped["unallocated_demand"]
        
        # Create GAP results with zero supply
        results = []
        for _, row in demand_grouped.iterrows():
            results.append({
                "pt_code": row["pt_code"],
                "product_name": row["product_name"],
                "package_size": row["package_size"],
                "standard_uom": row["standard_uom"],
                "period": row["period"],
                "begin_inventory": 0,
                "supply_in_period": 0,
                "total_available": 0,
                "total_demand_qty": row["total_demand_qty"],
                "gap_quantity": -row["total_demand_qty"],  # Negative because no supply
                "fulfillment_rate_percent": 0,
                "fulfillment_status": "❌ No Supply"
            })
        
        gap_df = pd.DataFrame(results)
        
        if debug_mode:
            st.write(f"🐛 Demand-only GAP calculation complete: {len(gap_df)} rows")
        
        return gap_df
    
    # Special handling for supply-only scenario (no demand)
    if df_demand.empty and not df_supply.empty:
        st.info("📥 Supply-Only Scenario: No demand for these products")
        
        df_s = df_supply.copy()
        
        # Convert supply dates to periods based on source type
        if 'source_type' in df_s.columns:
            # Create period column for each source type
            df_s["period"] = pd.NaT
            
            for source_type in df_s['source_type'].unique():
                source_mask = df_s['source_type'] == source_type
                source_df = df_s[source_mask]
                
                supply_date_col = get_supply_date_column(source_df, source_type, use_adjusted_supply)
                if supply_date_col in source_df.columns:
                    # Convert to period for this source type
                    periods = convert_to_period(source_df[supply_date_col], period_type)
                    df_s.loc[source_mask, "period"] = periods
        else:
            # Fallback - use date_ref if exists, otherwise skip
            if 'date_ref' in df_s.columns:
                df_s["period"] = convert_to_period(df_s["date_ref"], period_type)
            else:
                st.warning("No date column found in supply data")
                return pd.DataFrame()
        
        # Remove invalid periods
        df_s = df_s[df_s["period"].notna() & (df_s["period"] != "")]
        
        if df_s.empty:
            st.warning("No valid period data for supply")
            return pd.DataFrame()
        
        # Group supply
        supply_grouped = df_s.groupby(
            ["pt_code", "product_name", "package_size", "standard_uom", "period"],
            as_index=False,
            dropna=False
        ).agg({
            "quantity": "sum"
        })
        
        # Create GAP results with zero demand
        results = []
        for _, row in supply_grouped.iterrows():
            results.append({
                "pt_code": row["pt_code"],
                "product_name": row["product_name"],
                "package_size": row["package_size"],
                "standard_uom": row["standard_uom"],
                "period": row["period"],
                "begin_inventory": 0,
                "supply_in_period": row["quantity"],
                "total_available": row["quantity"],
                "total_demand_qty": 0,
                "gap_quantity": row["quantity"],  # Positive because no demand
                "fulfillment_rate_percent": 100,
                "fulfillment_status": "✅ Excess Supply"
            })
        
        gap_df = pd.DataFrame(results)
        
        if debug_mode:
            st.write(f"🐛 Supply-only GAP calculation complete: {len(gap_df)} rows")
        
        return gap_df
    
    # Normal case - both demand and supply exist
    if debug_mode:
        st.write(f"🐛 GAP Calculation starting:")
        st.write(f"- Demand records: {len(df_demand)}")
        st.write(f"- Supply records: {len(df_supply)}")
        st.write(f"- Period type: {period_type}")
        st.write(f"- Use adjusted demand: {use_adjusted_demand}")
        st.write(f"- Use adjusted supply: {use_adjusted_supply}")
    
    # Enhance demand with allocation info
    df_demand_enhanced = enhance_demand_with_allocation_info(df_demand)
    
    df_d = df_demand_enhanced.copy()
    df_s = df_supply.copy()
    
    # Convert to periods using appropriate date columns
    demand_date_col = get_demand_date_column(df_d, use_adjusted_demand)
    df_d["period"] = convert_to_period(df_d[demand_date_col], period_type)
    
    # For supply, handle different date columns per source type
    if 'source_type' in df_s.columns:
        for source_type in df_s['source_type'].unique():
            source_mask = df_s['source_type'] == source_type
            source_df = df_s[source_mask]
            
            supply_date_col = get_supply_date_column(source_df, source_type, use_adjusted_supply)
            if supply_date_col in source_df.columns:
                df_s.loc[source_mask, "period"] = convert_to_period(
                    source_df[supply_date_col], 
                    period_type
                )
    else:
        # Fallback if no source_type
        if 'date_ref' in df_s.columns:
            df_s["period"] = convert_to_period(df_s["date_ref"], period_type)
        else:
            st.warning("No appropriate date column found in supply data")
            return pd.DataFrame()
    
    # Remove rows with invalid periods
    df_d = df_d[df_d["period"].notna() & (df_d["period"] != "")]
    df_s = df_s[df_s["period"].notna() & (df_s["period"] != "")]
    
    if debug_mode:
        st.write("🐛 After period conversion:")
        st.write(f"- Demand records with valid periods: {len(df_d)}")
        st.write(f"- Supply records with valid periods: {len(df_s)}")
        st.write(f"- Unique demand products: {df_d['pt_code'].nunique()}")
        st.write(f"- Unique supply products: {df_s['pt_code'].nunique()}")
    
    # Group demand by product and period
    demand_grouped = df_d.groupby(
        ["pt_code", "product_name", "package_size", "standard_uom", "period"],
        as_index=False,
        dropna=False
    ).agg({
        "demand_quantity": "sum",
        "unallocated_demand": "sum"
    })
    
    # Use unallocated demand for GAP calculation
    demand_grouped["total_demand_qty"] = demand_grouped["unallocated_demand"]
    
    # Group supply
    supply_grouped = df_s.groupby(
        ["pt_code", "product_name", "package_size", "standard_uom", "period"],
        as_index=False,
        dropna=False
    ).agg({
        "quantity": "sum"
    })
    supply_grouped.rename(columns={"quantity": "total_supply_qty"}, inplace=True)
    
    # Get all periods and products
    all_periods = get_all_periods(demand_grouped, supply_grouped, period_type)
    all_products = get_all_products(demand_grouped, supply_grouped)
    
    if debug_mode:
        st.write(f"🐛 Total unique products for GAP: {len(all_products)}")
        st.write(f"🐛 Total periods: {len(all_periods)}")
    
    # Add progress tracking for large datasets
    total_products = len(all_products)
    show_progress = total_products > 100
    
    if show_progress:
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.text(f"Calculating GAP for {total_products} products...")
    
    # Calculate gap for each product
    results = []
    for idx, (_, product) in enumerate(all_products.iterrows()):
        # Update progress
        if show_progress:
            progress = (idx + 1) / total_products
            progress_bar.progress(progress)
            progress_text.text(f"Processing product {idx + 1} of {total_products}: {product['pt_code']}")
        
        try:
            product_gap = calculate_product_gap(
                product, 
                all_periods, 
                demand_grouped, 
                supply_grouped
            )
            results.extend(product_gap)
        except Exception as e:
            logger.error(f"Error calculating GAP for product {product['pt_code']}: {str(e)}")
            if debug_mode:
                st.error(f"Error processing {product['pt_code']}: {str(e)}")
    
    # Clear progress indicators
    if show_progress:
        progress_bar.empty()
        progress_text.empty()
    
    gap_df = pd.DataFrame(results)
    
    if debug_mode and not gap_df.empty:
        st.write(f"🐛 Final GAP analysis:")
        st.write(f"- Total GAP rows: {len(gap_df)}")
        st.write(f"- Unique products: {gap_df['pt_code'].nunique()}")
        st.write(f"- Rows with shortage: {len(gap_df[gap_df['gap_quantity'] < 0])}")
    
    return gap_df

# REPLACE phần xử lý supply period conversion trong calculate_gap_with_carry_forward
# Tìm comment "# For supply, handle different date columns per source type" và replace đoạn code sau nó

    # For supply, handle different date columns per source type
    if 'source_type' in df_s.columns:
        # Check if unified dates are available (multiple sources)
        if 'unified_date' in df_s.columns and len(df_s['source_type'].unique()) > 1:
            date_col = 'unified_date_adjusted' if use_adjusted_supply else 'unified_date'
            df_s["period"] = convert_to_period(df_s[date_col], period_type)
        else:
            # Process each source type separately
            for source_type in df_s['source_type'].unique():
                source_mask = df_s['source_type'] == source_type
                source_df = df_s[source_mask]
                
                supply_date_col = get_supply_date_column_for_gap(source_df, source_type, use_adjusted_supply)
                if supply_date_col in source_df.columns:
                    df_s.loc[source_mask, "period"] = convert_to_period(
                        source_df[supply_date_col], 
                        period_type
                    )
    else:
        # Fallback if no source_type
        if 'date_ref' in df_s.columns:
            df_s["period"] = convert_to_period(df_s["date_ref"], period_type)
        else:
            st.warning("No appropriate date column found in supply data")
            return pd.DataFrame()


def get_all_periods(demand_grouped, supply_grouped, period_type):
    """Get all unique periods sorted chronologically"""
    all_periods_raw = list(
        set(demand_grouped["period"]).union(set(supply_grouped["period"]))
    )
    
    # Filter out invalid periods
    all_periods_raw = [
        p for p in all_periods_raw 
        if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"
    ]
    
    # Sort based on period type
    if period_type == "Weekly":
        return sorted(all_periods_raw, key=parse_week_period)
    elif period_type == "Monthly":
        return sorted(all_periods_raw, key=parse_month_period)
    else:  # Daily
        return sorted(all_periods_raw)

def get_all_products(demand_grouped, supply_grouped):
    """Get all unique products - only by pt_code"""
    # Get unique products by pt_code only
    demand_products = demand_grouped[["pt_code"]].drop_duplicates()
    supply_products = supply_grouped[["pt_code"]].drop_duplicates()
    
    # Combine and get unique pt_codes
    all_pt_codes = pd.concat([demand_products, supply_products]).drop_duplicates()
    
    # Now get full product info for each unique pt_code
    products_list = []
    
    for pt_code in all_pt_codes['pt_code'].unique():
        # Try to get from demand first
        demand_match = demand_grouped[demand_grouped['pt_code'] == pt_code].head(1)
        if not demand_match.empty:
            products_list.append(demand_match[["pt_code", "product_name", "package_size", "standard_uom"]])
        else:
            # Get from supply
            supply_match = supply_grouped[supply_grouped['pt_code'] == pt_code].head(1)
            if not supply_match.empty:
                products_list.append(supply_match[["pt_code", "product_name", "package_size", "standard_uom"]])
    
    if products_list:
        products_df = pd.concat(products_list, ignore_index=True)
        # Clean product data
        for col in ["pt_code", "product_name", "package_size", "standard_uom"]:
            products_df[col] = products_df[col].astype(str).str.strip()
        return products_df
    else:
        return pd.DataFrame(columns=["pt_code", "product_name", "package_size", "standard_uom"])

def calculate_product_gap(product, all_periods, demand_grouped, supply_grouped):
    """Calculate gap for a single product across all periods"""
    pt_code = product["pt_code"]
    product_name = product["product_name"]
    package_size = product["package_size"]
    uom = product["standard_uom"]
    
    carry_forward_qty = 0
    results = []
    has_created_row = False
    
    # Get all demand data for this product
    product_demand_data = demand_grouped[demand_grouped["pt_code"] == pt_code]
    
    # Get all supply data for this product  
    product_supply_data = supply_grouped[supply_grouped["pt_code"] == pt_code]
    
    for period in all_periods:
        # Get demand for this period
        demand_in_period = product_demand_data[product_demand_data["period"] == period]
        demand = demand_in_period["total_demand_qty"].sum() if not demand_in_period.empty else 0
        
        # Get supply for this period
        supply_in_period = product_supply_data[product_supply_data["period"] == period]
        supply = supply_in_period["total_supply_qty"].sum() if not supply_in_period.empty else 0
        
        # Calculate total available
        total_available = carry_forward_qty + supply
        
        # Decision logic: Should we create a row?
        should_create_row = False
        
        if demand > 0:
            # Always create row if there's demand
            should_create_row = True
        elif supply > 0 and has_created_row:
            # Create row if there's supply AND we've already created at least one row
            should_create_row = True
        elif carry_forward_qty > 0 and not product_demand_data.empty:
            # Create row if there's carry forward AND product has demand in some period
            should_create_row = True
        
        if should_create_row:
            # Calculate gap
            gap = total_available - demand
            
            # Calculate fulfillment rate
            if demand > 0:
                fulfill_rate = min(100, (total_available / demand * 100))
            else:
                fulfill_rate = 100
                
            status = "✅ Fulfilled" if gap >= 0 else "❌ Shortage"
            
            results.append({
                "pt_code": pt_code,
                "product_name": product_name,
                "package_size": package_size,
                "standard_uom": uom,
                "period": period,
                "begin_inventory": carry_forward_qty,
                "supply_in_period": supply,
                "total_available": total_available,
                "total_demand_qty": demand,
                "gap_quantity": gap,
                "fulfillment_rate_percent": fulfill_rate,
                "fulfillment_status": status,
            })
            
            has_created_row = True
            
            # Update carry forward
            carry_forward_qty = max(0, gap)
        else:
            # No row created, but still update carry forward if there was supply
            if supply > 0:
                carry_forward_qty += supply
    
    return results

# === Display Options ===
def get_gap_display_options():
    """Get display options for GAP analysis"""
    st.markdown("### ⚙️ Display Options")
    
    # First row - existing options
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize defaults
    if 'gap_period_type' not in st.session_state:
        st.session_state['gap_period_type'] = "Weekly"
    
    period_type = col1.selectbox(
        "Group By Period", 
        PERIOD_TYPES, 
        index=PERIOD_TYPES.index(st.session_state.get('gap_period_type', "Weekly")),
        key="gap_period_select"
    )
    # Update session state
    st.session_state['gap_period_type'] = period_type
    
    show_shortage_only = col2.checkbox(
        "🔴 Show only shortages", 
        value=False,
        key="gap_shortage_checkbox"
    )
    
    exclude_zero_demand = col3.checkbox(
        "🚫 Exclude zero demand", 
        value=True,
        key="gap_exclude_zero_demand",
        help="Hide products that have supply but no demand"
    )
    
    exclude_missing_dates = col4.checkbox(
        "📅 Exclude missing dates",
        value=True,
        key="gap_exclude_missing_dates",
        help="Exclude records with missing ETD or reference dates"
    )
    
    # Second row - view options
    col5, col6, col7, col8 = st.columns(4)
    
    show_demand_only = col5.checkbox(
        "📤 Demand Only Products",
        value=False,
        key="gap_show_demand_only",
        help="Show products that have demand but no supply"
    )
    
    show_supply_only = col6.checkbox(
        "📥 Supply Only Products",
        value=False,
        key="gap_show_supply_only",
        help="Show products that have supply but no demand"
    )
    
    show_matched = col7.checkbox(
        "🔗 Matched Products",
        value=True,
        key="gap_show_matched",
        help="Show products that have both demand and supply"
    )
    
    show_data_quality = col8.checkbox(
        "📊 Show Data Quality",
        value=True,
        key="gap_show_data_quality",
        help="Show data quality warnings"
    )
    
    return {
        "period_type": period_type,
        "show_shortage_only": show_shortage_only,
        "exclude_zero_demand": exclude_zero_demand,
        "exclude_missing_dates": exclude_missing_dates,
        "show_demand_only": show_demand_only,
        "show_supply_only": show_supply_only,
        "show_matched": show_matched,   
        "show_data_quality": show_data_quality
    }

# === Display Functions ===

def show_gap_summary(gap_df, display_options, df_demand_filtered=None, df_supply_filtered=None,
                    use_adjusted_demand=True, use_adjusted_supply=True):
    """Show simplified GAP analysis summary with progressive disclosure"""
    st.markdown("### 📊 GAP Analysis Summary")
    
    # Check if gap_df is empty or has required columns
    if gap_df.empty:
        st.warning("No GAP data available for summary.")
        return
    
    # Verify required columns exist
    required_columns = ['pt_code', 'gap_quantity', 'period', 'total_demand_qty', 
                       'total_available', 'supply_in_period', 'fulfillment_rate_percent']
    missing_columns = [col for col in required_columns if col not in gap_df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in GAP data: {missing_columns}")
        st.info("Please ensure GAP analysis has been run successfully.")
        return
    
    # Data Quality Section (if enabled)
    if display_options.get("show_data_quality", True):
        # Pass empty dataframes if None to avoid errors
        demand_data = df_demand_filtered if df_demand_filtered is not None else pd.DataFrame()
        supply_data = df_supply_filtered if df_supply_filtered is not None else pd.DataFrame()
        
        show_data_quality_warnings(
            demand_data, supply_data, 
            use_adjusted_demand, use_adjusted_supply,
            display_options.get("period_type", "Weekly")
        )
    
    # === LEVEL 1: Key Insights Only (Always Show) ===
    st.markdown("#### 🎯 Key Insights")
    
    # Calculate essential metrics
    total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
    shortage_products = gap_df[gap_df['gap_quantity'] < 0]['pt_code'].nunique()
    total_products = gap_df['pt_code'].nunique()
    total_periods = gap_df['period'].nunique()
    
    # Calculate periods with shortage
    periods_with_shortage = gap_df[gap_df['gap_quantity'] < 0]['period'].nunique()
    
    # Determine overall status
    if total_shortage == 0:
        status_color = "#28a745"  # Green
        status_bg_color = "#d4edda"
        status_icon = "✅"
        status_text = "No Shortage Detected"
        status_detail = "Supply meets demand for all products across all periods"
    elif shortage_products / total_products > 0.5 or periods_with_shortage / total_periods > 0.5:
        status_color = "#dc3545"  # Red
        status_bg_color = "#f8d7da"
        status_icon = "🚨"
        status_text = "Critical Shortage"
        status_detail = f"{shortage_products} of {total_products} products need immediate attention"
    else:
        status_color = "#ffc107"  # Orange
        status_bg_color = "#fff3cd"
        status_icon = "⚠️"
        status_text = "Partial Shortage"
        status_detail = f"{shortage_products} of {total_products} products have shortage in some periods"
    
    # Main status card
    st.markdown(f"""
    <div style="background-color: {status_bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {status_color};">
        <h2 style="margin: 0; color: {status_color};">{status_icon} {status_text}</h2>
        <p style="margin: 10px 0 0 0; font-size: 18px; color: #333;">
            {status_detail}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")  # Add spacing
    
    # 3 Essential metrics only
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Shortage Quantity",
            format_number(total_shortage),
            delta=f"{shortage_products} products" if shortage_products > 0 else "No products",
            delta_color="inverse" if shortage_products > 0 else "off",
            help="Sum of all shortage quantities across all products and periods"
        )
    
    with col2:
        coverage_rate = ((total_products - shortage_products) / total_products * 100) if total_products > 0 else 100
        st.metric(
            "Product Coverage Rate",
            f"{coverage_rate:.0f}%",
            delta=f"{total_products - shortage_products} of {total_products} covered",
            delta_color="normal" if coverage_rate >= 80 else "inverse",
            help="Percentage of products with full supply coverage"
        )
    
    with col3:
        # Financial impact if available
        if df_demand_filtered is not None and 'value_in_usd' in df_demand_filtered.columns and shortage_products > 0:
            shortage_value = df_demand_filtered[
                df_demand_filtered['pt_code'].isin(
                    gap_df[gap_df['gap_quantity'] < 0]['pt_code'].unique()
                )
            ]['value_in_usd'].sum()
            st.metric(
                "Revenue at Risk",
                format_currency(shortage_value, 'USD', 0),
                delta=f"{shortage_value/df_demand_filtered['value_in_usd'].sum()*100:.0f}% of total" if df_demand_filtered['value_in_usd'].sum() > 0 else "N/A",
                delta_color="inverse" if shortage_value > 0 else "off",
                help="Potential revenue impact from products with shortage"
            )
        else:
            # Time criticality as alternative metric
            period_type = display_options.get('period_type', 'Weekly')
            past_periods = gap_df[
                gap_df['period'].apply(lambda x: is_past_period(str(x), period_type))
            ]['period'].nunique()
            
            future_periods = total_periods - past_periods
            st.metric(
                "Planning Horizon",
                f"{future_periods} {period_type.lower()} periods",
                delta=f"{past_periods} periods passed" if past_periods > 0 else "All future",
                delta_color="inverse" if past_periods > 0 else "off",
                help="Number of future periods in the analysis"
            )
    
    # === LEVEL 2: Expandable Details ===
    with st.expander("📈 View Detailed Analysis", expanded=shortage_products > 0):
        
        if shortage_products > 0:
            # Quick action summary
            st.markdown("##### 🎯 Action Required")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                # Top shortage products
                st.markdown("**🔴 Products with Shortage:**")
                
                # Get shortage summary by product
                product_shortages = gap_df[gap_df['gap_quantity'] < 0].groupby('pt_code').agg({
                    'gap_quantity': 'sum',
                    'period': 'count'
                }).rename(columns={'period': 'affected_periods'})
                product_shortages['gap_quantity'] = product_shortages['gap_quantity'].abs()
                product_shortages = product_shortages.sort_values('gap_quantity', ascending=False).head(5)
                
                for pt_code, row in product_shortages.iterrows():
                    st.caption(f"• **{pt_code}**: {format_number(row['gap_quantity'])} units ({row['affected_periods']} periods)")
            
            with action_col2:
                # Critical periods
                st.markdown("**📅 Periods with Shortage:**")
                
                period_shortages = gap_df[gap_df['gap_quantity'] < 0].groupby('period').agg({
                    'gap_quantity': 'sum',
                    'pt_code': 'nunique'
                }).rename(columns={'pt_code': 'products_affected'})
                period_shortages['gap_quantity'] = period_shortages['gap_quantity'].abs()
                period_shortages = period_shortages.sort_values('gap_quantity', ascending=False).head(5)
                
                for period, row in period_shortages.iterrows():
                    is_past = is_past_period(str(period), display_options.get('period_type', 'Weekly'))
                    indicator = "🔴" if is_past else "🟡"
                    st.caption(f"{indicator} **{period}**: {format_number(row['gap_quantity'])} units ({row['products_affected']} products)")
        
        # Supply vs Demand Overview
        st.markdown("##### 📊 Supply vs Demand Balance")
        
        total_demand = gap_df['total_demand_qty'].sum()
        total_supply = gap_df['supply_in_period'].sum()
        net_position = total_supply - total_demand
        
        # Create visual balance
        balance_col1, balance_col2, balance_col3 = st.columns([2, 1, 2])
        
        with balance_col1:
            st.metric("Total Demand", format_number(total_demand))
        
        with balance_col2:
            if net_position >= 0:
                st.markdown("<h2 style='text-align: center; color: green;'>→</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='text-align: center; color: red;'>→</h2>", unsafe_allow_html=True)
        
        with balance_col3:
            st.metric(
                "Total Supply", 
                format_number(total_supply),
                delta=format_number(net_position),
                delta_color="normal" if net_position >= 0 else "inverse"
            )
        
        # Simple visualization
        if total_demand > 0:
            supply_rate = min(total_supply / total_demand * 100, 100)
            st.progress(supply_rate / 100)
            st.caption(f"Supply covers {supply_rate:.1f}% of total demand")
    
    # === LEVEL 3: Advanced Analytics (Optional) ===
    if st.checkbox("🔍 Show Advanced Analytics", key="gap_show_advanced"):
        show_advanced_gap_analytics(gap_df, display_options, df_demand_filtered, df_supply_filtered)
    
    # === Action Buttons (Always visible if shortage exists) ===
    # if shortage_products > 0:
    #     st.markdown("---")
    #     st.markdown("#### 🎯 Recommended Actions")
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         if st.button("🧩 Create Allocation Plan", 
    #                     type="primary", 
    #                     use_container_width=True,
    #                     key="gap_create_allocation_btn"):  # Add unique key
    #             st.switch_page("pages/4_🧩_Allocation_Plan.py")
        
    #     with col2:
    #         if st.button("📌 Generate PO Suggestions", 
    #                     type="secondary", 
    #                     use_container_width=True,
    #                     key="gap_generate_po_btn"):  # Add unique key
    #             st.switch_page("pages/5_📌_PO_Suggestions.py")
        
    #     with col3:
    #         # Get shortage details for export
    #         shortage_df = gap_df[gap_df['gap_quantity'] < 0].copy()
    #         shortage_summary = shortage_df.groupby(['pt_code', 'product_name']).agg({
    #             'gap_quantity': 'sum',
    #             'total_demand_qty': 'sum'
    #         }).reset_index()
    #         shortage_summary['gap_quantity'] = shortage_summary['gap_quantity'].abs()
            
    #         # DisplayComponents.show_export_button already has timestamp which makes it unique
    #         DisplayComponents.show_export_button(
    #             df=shortage_summary,
    #             filename="shortage_summary",
    #             button_label="📤 Export Shortage Report"
    #         )

def show_advanced_gap_analytics(gap_df, display_options, df_demand_filtered=None, df_supply_filtered=None):
    """Show advanced analytics for power users"""
    st.markdown("#### 📊 Advanced Analytics")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🎯 Product Details", "📅 Period Analysis", "💡 Insights"])
    
    with tab1:
        # Shortage trend by period
        st.markdown("##### Shortage Trend by Period")
        
        period_summary = gap_df.groupby('period').agg({
            'gap_quantity': lambda x: x[x < 0].sum(),  # Only negative gaps
            'total_demand_qty': 'sum',
            'supply_in_period': 'sum'
        }).reset_index()
        
        # Sort periods properly
        period_type = display_options.get("period_type", "Weekly")
        if period_type == "Weekly":
            period_summary = period_summary.sort_values('period', key=lambda x: x.apply(parse_week_period))
        elif period_type == "Monthly":
            period_summary = period_summary.sort_values('period', key=lambda x: x.apply(parse_month_period))
        
        # Create line chart
        chart_data = period_summary[['period', 'total_demand_qty', 'supply_in_period']].copy()
        chart_data = chart_data.set_index('period')
        st.line_chart(chart_data)
        
        # Show shortage bars
        if (period_summary['gap_quantity'] < 0).any():
            st.markdown("##### Shortage by Period")
            shortage_data = period_summary[period_summary['gap_quantity'] < 0][['period', 'gap_quantity']].copy()
            shortage_data['gap_quantity'] = shortage_data['gap_quantity'].abs()
            shortage_data = shortage_data.set_index('period')
            st.bar_chart(shortage_data)
    
    with tab2:
        # Product performance metrics
        st.markdown("##### Product Performance Metrics")
        
        product_summary = gap_df.groupby(['pt_code', 'product_name']).agg({
            'total_demand_qty': 'sum',
            'total_available': 'sum',
            'gap_quantity': 'sum',
            'fulfillment_rate_percent': 'mean'
        }).reset_index()
        
        # Add status
        product_summary['Status'] = product_summary['gap_quantity'].apply(
            lambda x: '✅ OK' if x >= 0 else '❌ Shortage'
        )
        
        # Sort by gap (worst first)
        product_summary = product_summary.sort_values('gap_quantity')
        
        # Format columns
        product_summary['Demand'] = product_summary['total_demand_qty'].apply(format_number)
        product_summary['Available'] = product_summary['total_available'].apply(format_number)
        product_summary['GAP'] = product_summary['gap_quantity'].apply(format_number)
        product_summary['Fill Rate'] = product_summary['fulfillment_rate_percent'].apply(lambda x: f"{x:.0f}%")
        
        # Display
        display_cols = ['pt_code', 'product_name', 'Demand', 'Available', 'GAP', 'Fill Rate', 'Status']
        st.dataframe(
            product_summary[display_cols],
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        # Period analysis
        st.markdown("##### Period Analysis")
        
        # Get unique periods
        periods = sorted(gap_df['period'].unique(), key=lambda x: parse_week_period(x) if display_options.get("period_type") == "Weekly" else x)
        
        selected_period = st.selectbox("Select Period", periods, key="gap_period_analysis")
        
        if selected_period:
            period_data = gap_df[gap_df['period'] == selected_period]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                products_in_period = period_data['pt_code'].nunique()
                st.metric("Products", products_in_period)
            
            with col2:
                shortage_in_period = period_data[period_data['gap_quantity'] < 0]['pt_code'].nunique()
                st.metric("With Shortage", shortage_in_period)
            
            with col3:
                total_gap = period_data['gap_quantity'].sum()
                st.metric("Net GAP", format_number(total_gap))
            
            # Show products for this period
            st.markdown(f"**Products in {selected_period}:**")
            period_products = period_data[['pt_code', 'product_name', 'total_demand_qty', 'total_available', 'gap_quantity']].copy()
            
            # Format
            for col in ['total_demand_qty', 'total_available', 'gap_quantity']:
                period_products[col] = period_products[col].apply(format_number)
            
            st.dataframe(period_products, use_container_width=True, hide_index=True)
    
    with tab4:
        # Insights and recommendations
        st.markdown("##### 💡 Key Insights")
        
        # Calculate insights
        insights = []
        
        # Insight 1: Coverage
        total_products = gap_df['pt_code'].nunique()
        shortage_products = gap_df[gap_df['gap_quantity'] < 0]['pt_code'].nunique()
        coverage = (total_products - shortage_products) / total_products * 100 if total_products > 0 else 100
        
        if coverage == 100:
            insights.append("✅ **Perfect Coverage**: All products have sufficient supply")
        elif coverage >= 80:
            insights.append(f"👍 **Good Coverage**: {coverage:.0f}% of products have sufficient supply")
        else:
            insights.append(f"⚠️ **Low Coverage**: Only {coverage:.0f}% of products have sufficient supply")
        
        # Insight 2: Timing
        period_type = display_options.get('period_type', 'Weekly')
        first_shortage_period = gap_df[gap_df['gap_quantity'] < 0]['period'].min()
        if first_shortage_period and not is_past_period(str(first_shortage_period), period_type):
            insights.append(f"📅 **First shortage in**: {first_shortage_period} - You have time to prepare")
        elif first_shortage_period:
            insights.append(f"🚨 **Immediate action needed**: Shortage already occurring in {first_shortage_period}")
        
        # Insight 3: Concentration
        if shortage_products > 0:
            top_shortage = gap_df[gap_df['gap_quantity'] < 0].groupby('pt_code')['gap_quantity'].sum()
            top_shortage = top_shortage.sort_values().head(1)
            if len(top_shortage) > 0:
                worst_product = top_shortage.index[0]
                worst_amount = abs(top_shortage.values[0])
                total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].sum()
                concentration = worst_amount / abs(total_shortage) * 100 if total_shortage != 0 else 0
                
                if concentration > 50:
                    insights.append(f"🎯 **Concentrated shortage**: {worst_product} accounts for {concentration:.0f}% of total shortage")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # Recommendations
        st.markdown("##### 📋 Recommendations")
        
        if shortage_products > 0:
            st.write("Based on the analysis, we recommend:")
            st.write("1. **Immediate**: Create allocation plan for products with shortage")
            st.write("2. **Short-term**: Generate PO suggestions to cover future gaps")
            st.write("3. **Long-term**: Review demand forecasting accuracy")
        else:
            st.success("No immediate actions required. Continue monitoring supply-demand balance.")




def show_data_quality_warnings(df_demand, df_supply, use_adjusted_demand, use_adjusted_supply, period_type):
    """Show data quality warnings with date mode awareness - SAFE VERSION"""
    col_dq1, col_dq2, col_dq3 = st.columns(3)
    
    with col_dq1:
        # Check demand missing dates
        if not df_demand.empty:
            demand_date_col = get_demand_date_column(df_demand, use_adjusted_demand)
            if demand_date_col in df_demand.columns:
                demand_missing = check_missing_dates(df_demand, demand_date_col)
                if demand_missing > 0:
                    st.warning(f"⚠️ Demand: {demand_missing} records with missing {demand_date_col}")
            else:
                st.warning(f"⚠️ Demand: date column not found")
        else:
            st.info("No demand data to check")
    
    with col_dq2:
        # Check supply missing dates (handle empty supply)
        if not df_supply.empty:
            supply_missing = 0
            
            # Check if source_type column exists
            if 'source_type' in df_supply.columns:
                for source_type in df_supply['source_type'].unique():
                    source_df = df_supply[df_supply['source_type'] == source_type]
                    supply_date_col = get_supply_date_column(source_df, source_type, use_adjusted_supply)
                    if supply_date_col in source_df.columns:
                        supply_missing += check_missing_dates(source_df, supply_date_col)
            else:
                # No source_type, try common date columns
                date_cols = ['date_ref', 'eta', 'arrival_date', 'transfer_date']
                for col in date_cols:
                    if col in df_supply.columns:
                        supply_missing = check_missing_dates(df_supply, col)
                        break
            
            if supply_missing > 0:
                st.warning(f"⚠️ Supply: {supply_missing} records with missing dates")
        else:
            st.info("No supply data to check")
    
    with col_dq3:
        # Check for past periods in results (handle empty data)
        past_periods_count = 0
        
        if not df_demand.empty:
            demand_date_col = get_demand_date_column(df_demand, use_adjusted_demand)
            if demand_date_col in df_demand.columns:
                for _, row in df_demand.iterrows():
                    if pd.notna(row[demand_date_col]):
                        period = convert_to_period(row[demand_date_col], period_type)
                        if pd.notna(period) and is_past_period(str(period), period_type):
                            past_periods_count += 1
        
        if not df_supply.empty and 'source_type' in df_supply.columns:
            for _, row in df_supply.iterrows():
                source_type = row.get('source_type', '')
                if source_type:
                    date_col = get_supply_date_column(
                        df_supply[df_supply['source_type'] == source_type], 
                        source_type, 
                        use_adjusted_supply
                    )
                    if date_col in row and pd.notna(row[date_col]):
                        period = convert_to_period(row[date_col], period_type)
                        if pd.notna(period) and is_past_period(str(period), period_type):
                            past_periods_count += 1
        
        if past_periods_count > 0:
            st.error(f"🔴 Found {past_periods_count} records with past dates")


def show_date_status_comparison(df_demand, df_supply, use_adjusted_demand, use_adjusted_supply):
    """Show date adjustment impact in a unified, compact layout"""
    st.markdown("#### 📈 Date Adjustment Impact")
    
    # Prepare data for comparison
    impact_data = []
    
    # Demand analysis
    if df_demand is not None and not df_demand.empty:
        if 'etd_original' in df_demand.columns and 'etd_adjusted' in df_demand.columns:
            today = pd.Timestamp.now().normalize()
            
            demand_impact = {
                'Source': '📤 Demand (ETD)',
                'Missing Original': df_demand['etd_original'].isna().sum(),
                'Missing Adjusted': df_demand['etd_adjusted'].isna().sum(),
                'Missing Change': 0,  # Will calculate
                'Past Original': (pd.to_datetime(df_demand['etd_original'], errors='coerce') < today).sum(),
                'Past Adjusted': (pd.to_datetime(df_demand['etd_adjusted'], errors='coerce') < today).sum(),
                'Past Change': 0  # Will calculate
            }
            demand_impact['Missing Change'] = demand_impact['Missing Adjusted'] - demand_impact['Missing Original']
            demand_impact['Past Change'] = demand_impact['Past Adjusted'] - demand_impact['Past Original']
            
            impact_data.append(demand_impact)
    
    # Supply analysis by source
    if df_supply is not None and not df_supply.empty and 'source_type' in df_supply.columns:
        for source_type in sorted(df_supply['source_type'].unique()):
            source_df = df_supply[df_supply['source_type'] == source_type]
            
            # Map to date columns
            date_map = {
                'Inventory': ('date_ref', '📦'),
                'Pending CAN': ('arrival_date', '📥'),
                'Pending PO': ('eta', '📋'),
                'Pending WH Transfer': ('transfer_date', '🚚')
            }
            
            if source_type in date_map:
                base_col, icon = date_map[source_type]
                original_col = f'{base_col}_original'
                adjusted_col = f'{base_col}_adjusted'
                
                if original_col in source_df.columns and adjusted_col in source_df.columns:
                    today = pd.Timestamp.now().normalize()
                    
                    supply_impact = {
                        'Source': f'{icon} {source_type}',
                        'Missing Original': source_df[original_col].isna().sum(),
                        'Missing Adjusted': source_df[adjusted_col].isna().sum(),
                        'Missing Change': 0,
                        'Past Original': (pd.to_datetime(source_df[original_col], errors='coerce') < today).sum(),
                        'Past Adjusted': (pd.to_datetime(source_df[adjusted_col], errors='coerce') < today).sum(),
                        'Past Change': 0
                    }
                    supply_impact['Missing Change'] = supply_impact['Missing Adjusted'] - supply_impact['Missing Original']
                    supply_impact['Past Change'] = supply_impact['Past Adjusted'] - supply_impact['Past Original']
                    
                    impact_data.append(supply_impact)
    
    # Display as a unified table
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        
        # Rename columns for better display
        impact_df = impact_df.rename(columns={
            'Missing Original': 'Missing (Orig)',
            'Missing Adjusted': 'Missing (Adj)',
            'Missing Change': 'Missing Δ',
            'Past Original': 'Past (Orig)',
            'Past Adjusted': 'Past (Adj)',
            'Past Change': 'Past Δ'
        })
        
        # Style the dataframe
        def style_changes(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #dc3545; font-weight: bold'  # Red for increases
                elif val < 0:
                    return 'color: #28a745; font-weight: bold'  # Green for decreases
            return ''
        
        def style_background(row):
            styles = [''] * len(row)
            # Highlight change columns
            for i, col in enumerate(row.index):
                if 'Δ' in col:
                    styles[i] = 'background-color: #f8f9fa'
            return styles
        
        styled_df = impact_df.style.applymap(
            style_changes, 
            subset=['Missing Δ', 'Past Δ']
        ).apply(style_background, axis=1)
        
        # Add number formatting
        styled_df = styled_df.format({
            'Missing (Orig)': '{:,.0f}',
            'Missing (Adj)': '{:,.0f}',
            'Missing Δ': '{:+,.0f}',
            'Past (Orig)': '{:,.0f}',
            'Past (Adj)': '{:,.0f}',
            'Past Δ': '{:+,.0f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=(len(impact_df) + 1) * 35 + 10)
        
        # Summary section
        total_missing_change = impact_df['Missing Δ'].sum()
        total_past_change = impact_df['Past Δ'].sum()
        
        if total_missing_change != 0 or total_past_change != 0:
            st.markdown("**Impact Summary:**")
            
            summary_cols = st.columns(3)
            
            with summary_cols[0]:
                if total_missing_change < 0:
                    st.success(f"✅ {abs(total_missing_change)} missing dates resolved")
                elif total_missing_change > 0:
                    st.warning(f"⚠️ {total_missing_change} new missing dates")
                else:
                    st.info("No change in missing dates")
            
            with summary_cols[1]:
                if total_past_change < 0:
                    st.success(f"✅ {abs(total_past_change)} dates moved to future")
                elif total_past_change > 0:
                    st.warning(f"⚠️ {total_past_change} dates moved to past")
                else:
                    st.info("No change in past dates")
            
            with summary_cols[2]:
                total_adjustments = sum(1 for _, row in impact_df.iterrows() 
                                      if row['Missing Δ'] != 0 or row['Past Δ'] != 0)
                if total_adjustments > 0:
                    st.info(f"📊 {total_adjustments}/{len(impact_df)} sources affected")
                else:
                    st.info("📊 No sources affected")
        else:
            st.success("✅ Date adjustments configured but no impact on missing/past dates")
        
        # Optional: Show which dates are being adjusted
        if st.checkbox("Show adjustment details", key="gap_show_adjustment_details"):
            st.caption("Note: Adjustments may change future dates without affecting missing/past counts")
            
            # Show sample of adjusted records
            samples = []
            
            # Check demand
            if df_demand is not None and not df_demand.empty:
                if 'etd_original' in df_demand.columns and 'etd_adjusted' in df_demand.columns:
                    mask = df_demand['etd_original'] != df_demand['etd_adjusted']
                    if mask.any():
                        sample = df_demand[mask][['pt_code', 'etd_original', 'etd_adjusted']].head(3)
                        for _, row in sample.iterrows():
                            samples.append({
                                'Source': '📤 Demand',
                                'Product': row['pt_code'],
                                'Original Date': pd.to_datetime(row['etd_original']).strftime('%Y-%m-%d') if pd.notna(row['etd_original']) else 'N/A',
                                'Adjusted Date': pd.to_datetime(row['etd_adjusted']).strftime('%Y-%m-%d') if pd.notna(row['etd_adjusted']) else 'N/A'
                            })
            
            # Check supply (limit to avoid too many rows)
            if df_supply is not None and not df_supply.empty and samples.__len__() < 5:
                for source_type in df_supply['source_type'].unique()[:2]:  # Limit to first 2 sources
                    date_map = {
                        'Inventory': 'date_ref',
                        'Pending CAN': 'arrival_date',
                        'Pending PO': 'eta',
                        'Pending WH Transfer': 'transfer_date'
                    }
                    
                    if source_type in date_map:
                        base_col = date_map[source_type]
                        original_col = f'{base_col}_original'
                        adjusted_col = f'{base_col}_adjusted'
                        
                        source_df = df_supply[df_supply['source_type'] == source_type]
                        if original_col in source_df.columns and adjusted_col in source_df.columns:
                            mask = source_df[original_col] != source_df[adjusted_col]
                            if mask.any():
                                sample = source_df[mask][['pt_code', original_col, adjusted_col]].head(2)
                                for _, row in sample.iterrows():
                                    samples.append({
                                        'Source': f'📥 {source_type}',
                                        'Product': row['pt_code'],
                                        'Original Date': pd.to_datetime(row[original_col]).strftime('%Y-%m-%d') if pd.notna(row[original_col]) else 'N/A',
                                        'Adjusted Date': pd.to_datetime(row[adjusted_col]).strftime('%Y-%m-%d') if pd.notna(row[adjusted_col]) else 'N/A'
                                    })
            
            if samples:
                st.dataframe(pd.DataFrame(samples), use_container_width=True)
            else:
                st.info("No date adjustments found in current filtered data")
    
    else:
        st.info("No date adjustments configured or no data available for comparison")


def show_allocation_impact_summary(df_demand_enhanced):
    """Show allocation impact with enhanced UI and progress indicators"""
    if 'allocation_status' not in df_demand_enhanced.columns:
        return
        
    st.markdown("#### 📦 Allocation Impact on Demand")
    
    # Calculate metrics
    total_demand = df_demand_enhanced['demand_quantity'].sum()
    total_allocated = df_demand_enhanced['total_allocated'].sum()
    total_delivered = df_demand_enhanced['total_delivered'].sum()
    total_unallocated = df_demand_enhanced['unallocated_demand'].sum()
    
    # Calculate rates
    allocation_rate = (total_allocated / total_demand * 100) if total_demand > 0 else 0
    delivery_rate = (total_delivered / total_allocated * 100) if total_allocated > 0 else 0
    fulfillment_rate = (total_delivered / total_demand * 100) if total_demand > 0 else 0
    
    # Display metrics with context
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Original Demand",
            format_number(total_demand),
            help="Total demand quantity before allocation"
        )
    
    with col2:
        st.metric(
            "Already Allocated",
            format_number(total_allocated),
            delta=f"{allocation_rate:.1f}%" if allocation_rate > 0 else None,
            help="Quantity allocated in approved plans"
        )
    
    with col3:
        st.metric(
            "Already Delivered", 
            format_number(total_delivered),
            delta=f"{delivery_rate:.1f}% of allocated" if total_allocated > 0 else None,
            help="Quantity delivered to customers"
        )
    
    with col4:
        st.metric(
            "Net Unallocated",
            format_number(total_unallocated),
            delta=f"{100 - allocation_rate:.1f}% remaining" if total_demand > 0 else None,
            delta_color="inverse" if allocation_rate < 50 else "normal",
            help="Quantity still needs allocation"
        )
    
    # Visual progress indicators
    if total_demand > 0:
        st.markdown("##### Allocation Progress")
        
        # Overall progress bar
        progress_col1, progress_col2 = st.columns([4, 1])
        with progress_col1:
            # Create stacked progress bar effect
            fig_html = f"""
            <div style="width: 100%; background-color: #f0f0f0; border-radius: 5px; overflow: hidden;">
                <div style="width: {fulfillment_rate}%; background-color: #28a745; height: 25px; float: left; text-align: center; color: white; line-height: 25px;">
                    {fulfillment_rate:.1f}% Delivered
                </div>
                <div style="width: {allocation_rate - fulfillment_rate}%; background-color: #ffc107; height: 25px; float: left; text-align: center; color: black; line-height: 25px;">
                    {allocation_rate - fulfillment_rate:.1f}% Allocated
                </div>
                <div style="width: {100 - allocation_rate}%; background-color: #dc3545; height: 25px; float: left; text-align: center; color: white; line-height: 25px;">
                    {100 - allocation_rate:.1f}% Unallocated
                </div>
            </div>
            """
            st.markdown(fig_html, unsafe_allow_html=True)
        
        with progress_col2:
            st.caption(f"Total: {format_number(total_demand)}")
        
        # Status messages
        if allocation_rate == 0:
            st.warning("⚠️ No allocations created yet for this demand")
        elif allocation_rate < 50:
            st.info(f"📊 {allocation_rate:.1f}% allocated - significant gap remains")
        elif allocation_rate < 100:
            st.info(f"📊 {allocation_rate:.1f}% allocated - partial coverage")
        else:
            st.success("✅ Fully allocated")
            if delivery_rate < 100:
                st.info(f"🚚 {delivery_rate:.1f}% of allocated quantity has been delivered")
    
    # Show breakdown by allocation status if allocations exist
    if total_allocated > 0:
        with st.expander("View allocation details by status"):
            status_summary = df_demand_enhanced.groupby('allocation_status').agg({
                'pt_code': 'count',
                'demand_quantity': 'sum',
                'total_allocated': 'sum',
                'total_delivered': 'sum',
                'unallocated_demand': 'sum'
            }).rename(columns={'pt_code': 'Products'})
            
            # Format the summary
            for col in ['demand_quantity', 'total_allocated', 'total_delivered', 'unallocated_demand']:
                if col in status_summary.columns:
                    status_summary[col] = status_summary[col].apply(lambda x: format_number(x))
            
            st.dataframe(status_summary, use_container_width=True)
            
            # Show products by status
            if st.checkbox("Show products by allocation status", key="gap_show_products_by_status"):
                for status in df_demand_enhanced['allocation_status'].unique():
                    products = df_demand_enhanced[df_demand_enhanced['allocation_status'] == status]['pt_code'].unique()
                    if len(products) > 0:
                        st.caption(f"**{status}**: {', '.join(products[:10])}{' ...' if len(products) > 10 else ''}")

# === GAP Detail Display Functions ===

def apply_gap_detail_filter(display_df: pd.DataFrame, filter_option: str, display_options: Dict,
                           df_demand_filtered=None, df_supply_filtered=None) -> pd.DataFrame:
    """Apply filter to GAP detail dataframe"""
    if filter_option == "Show All":
        return prepare_gap_detail_display(display_df, display_options, df_demand_filtered, df_supply_filtered)
    
    # Apply different filters
    if filter_option == "Show Shortage Only":
        display_df = display_df[display_df["gap_quantity"] < 0]
    
    elif filter_option == "Show Past Periods Only":
        period_type = display_options.get("period_type", "Weekly")
        display_df = display_df[
            display_df['period'].apply(lambda x: is_past_period(str(x), period_type))
        ]
    
    elif filter_option == "Show Future Periods Only":
        period_type = display_options.get("period_type", "Weekly")
        display_df = display_df[
            ~display_df['period'].apply(lambda x: is_past_period(str(x), period_type))
        ]
    
    elif filter_option == "Show Zero Demand Only":
        display_df = display_df[display_df["total_demand_qty"] == 0]
    
    elif filter_option == "Show Critical Shortage Only":
        display_df = display_df[
            (display_df["gap_quantity"] < 0) & 
            (display_df["fulfillment_rate_percent"] < 50)
        ]
    
    return prepare_gap_detail_display(display_df, display_options, df_demand_filtered, df_supply_filtered)


def prepare_gap_detail_display(display_df: pd.DataFrame, display_options: Dict,
                              df_demand_filtered=None, df_supply_filtered=None) -> pd.DataFrame:
    """Prepare GAP dataframe for detail display"""
    if display_df.empty:
        return display_df
    
    display_df = display_df.copy()
    
    # Add Period Status column
    period_type = display_options.get("period_type", "Weekly")
    display_df['Period Status'] = display_df['period'].apply(
        lambda x: "🔴 Past" if is_past_period(str(x), period_type) else "✅ Future"
    )
    
    # Add Product Type column if we have the data
    if df_demand_filtered is not None and df_supply_filtered is not None:
        demand_products = set()
        supply_products = set()
        
        if not df_demand_filtered.empty and 'pt_code' in df_demand_filtered.columns:
            demand_products = set(df_demand_filtered['pt_code'].unique())
        
        if not df_supply_filtered.empty and 'pt_code' in df_supply_filtered.columns:
            supply_products = set(df_supply_filtered['pt_code'].unique())
        
        if demand_products or supply_products:
            def get_product_type(pt_code):
                if pt_code in demand_products and pt_code in supply_products:
                    return "🔗 Matched"
                elif pt_code in demand_products:
                    return "📤 Demand Only"
                elif pt_code in supply_products:
                    return "📥 Supply Only"
                return "❓ Unknown"
            
            display_df['Product Type'] = display_df['pt_code'].apply(get_product_type)
    
    # Select columns to display
    display_columns = [
        "pt_code", "product_name", "package_size", "standard_uom", "period",
        "Period Status", "begin_inventory", "supply_in_period", "total_available", 
        "total_demand_qty", "gap_quantity", "fulfillment_rate_percent", 
        "fulfillment_status"
    ]
    
    # Add Product Type if exists
    if 'Product Type' in display_df.columns:
        display_columns.append('Product Type')
    
    # Only include columns that exist
    display_columns = [col for col in display_columns if col in display_df.columns]
    
    # Sort by period and product
    if 'period' in display_df.columns and 'pt_code' in display_df.columns:
        display_df = display_df.sort_values(['period', 'pt_code'])
    
    return display_df[display_columns]


def format_gap_display_df(df: pd.DataFrame, display_options: Dict) -> pd.DataFrame:
    """Format GAP dataframe for display"""
    df = df.copy()
    
    # Format numeric columns
    numeric_cols = [
        "begin_inventory", "supply_in_period", "total_available", 
        "total_demand_qty", "gap_quantity"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: format_number(x))
    
    if "fulfillment_rate_percent" in df.columns:
        df["fulfillment_rate_percent"] = df["fulfillment_rate_percent"].apply(
            lambda x: format_percentage(x)
        )
    
    return df


def highlight_gap_rows_enhanced(row):
    """Enhanced highlighting matching Demand/Supply style with priority"""
    styles = [""] * len(row)
    
    try:
        # Priority: shortage > past period > low fulfillment > zero demand
        
        # Check fulfillment status first (highest priority)
        if 'fulfillment_status' in row.index and "❌" in str(row['fulfillment_status']):
            # Shortage - light red background
            return ["background-color: #f8d7da"] * len(row)
        
        # Check if critical shortage (fulfillment < 50%)
        if 'fulfillment_rate_percent' in row.index:
            rate_str = str(row['fulfillment_rate_percent']).replace('%', '').strip()
            try:
                rate = float(rate_str)
                if rate < 50:
                    # Critical shortage - darker red
                    return ["background-color: #f5c6cb"] * len(row)
            except:
                pass
        
        # Check period status
        if 'Period Status' in row.index and "🔴" in str(row['Period Status']):
            # Past period - light gray background
            return ["background-color: #f0f0f0"] * len(row)
        
        # Check zero demand
        if 'total_demand_qty' in row.index:
            try:
                demand_str = str(row['total_demand_qty']).replace(',', '').strip()
                if float(demand_str) == 0:
                    # Zero demand - light blue
                    return ["background-color: #d1ecf1"] * len(row)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error highlighting rows: {str(e)}")
    
    return styles


def show_gap_detail_table(gap_df, display_options, df_demand_filtered=None, df_supply_filtered=None):
    """Show detailed GAP analysis table with enhanced filtering like Demand/Supply pages"""
    st.markdown("### 📄 GAP Details by Product & Period")
    
    # Check if gap_df is empty or missing required columns
    if gap_df.empty:
        st.info("No GAP data to display.")
        return
    
    # Verify required columns in gap_df
    required_gap_columns = ['pt_code', 'product_name', 'package_size', 'standard_uom', 
                           'period', 'gap_quantity', 'total_demand_qty', 'begin_inventory',
                           'supply_in_period', 'total_available', 'fulfillment_rate_percent',
                           'fulfillment_status']
    
    missing_columns = [col for col in required_gap_columns if col not in gap_df.columns]
    if missing_columns:
        st.error(f"Missing required columns in GAP data: {missing_columns}")
        return
    
    # Define filter options
    filter_options = [
        "Show All",
        "Show Shortage Only",
        "Show Past Periods Only", 
        "Show Future Periods Only",
        "Show Zero Demand Only",
        "Show Critical Shortage Only"
    ]
    
    # Use DisplayComponents for detail table with filters
    DisplayComponents.render_detail_table_with_filter(
        df=gap_df,
        filter_options=filter_options,
        filter_apply_func=lambda df, opt: apply_gap_detail_filter(df, opt, display_options, 
                                                                 df_demand_filtered, df_supply_filtered),
        format_func=lambda df: format_gap_display_df(df, display_options),
        style_func=highlight_gap_rows_enhanced,
        height=600,
        key_prefix="gap_detail"
    )


# REPLACE toàn bộ function show_gap_pivot_view

def show_gap_pivot_view(gap_df, display_options):
    """Show GAP pivot view with past period indicators and styling options"""
    st.markdown("### 📊 Pivot View - GAP by Period")
    
    # Apply filters for display
    display_df = gap_df.copy()
    
    if display_options.get("show_shortage_only", False):
        display_df = display_df[display_df["gap_quantity"] < 0]
    
    if display_options.get("exclude_zero_demand", False):
        display_df = display_df[display_df["total_demand_qty"] > 0]
    
    if display_df.empty:
        st.info("No data to display in pivot view with current filters.")
        return
    
    # Style selection
    style_mode = st.radio(
        "🎨 Styling Mode",
        options=STYLE_MODES,
        horizontal=True,
        key="gap_style_mode"
    )
    
    # Create pivot using helper function
    pivot_df = create_period_pivot(
        df=display_df,
        group_cols=["product_name", "pt_code"],
        period_col="period",
        value_col="gap_quantity",
        agg_func="sum",
        period_type=display_options["period_type"],
        show_only_nonzero=display_options.get("show_shortage_only", False),
        fill_value=0
    )
    
    if pivot_df.empty:
        st.info("No data to display after pivoting.")
        return
    
    # Add past period indicators
    display_pivot = apply_period_indicators(
        df=pivot_df,
        period_type=display_options["period_type"],
        exclude_cols=["product_name", "pt_code"],
        indicator="🔴"
    )
    
    # Show legend
    st.info("🔴 = Past period (already occurred)")
    
    # Apply styling based on mode
    if style_mode == "🔴 Highlight Shortage":
        # Create a styled dataframe with custom formatting
        def style_negative(val):
            try:
                # Remove any formatting first
                val_str = str(val).replace(',', '')
                if float(val_str) < 0:
                    return 'background-color: #ffcccc; color: red; font-weight: bold;'
            except:
                pass
            return ''
        
        # Format values first
        for col in display_pivot.columns[2:]:
            display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
        
        styled_df = display_pivot.style.applymap(
            style_negative,
            subset=display_pivot.columns[2:]
        )
        st.dataframe(styled_df, use_container_width=True)
    
    elif style_mode == "🌈 Heatmap":
        # For heatmap, use numeric values from original pivot
        styled_df = pivot_df.style.background_gradient(
            cmap='RdYlGn', 
            subset=pivot_df.columns[2:], 
            axis=1
        ).format("{:,.0f}", subset=pivot_df.columns[2:])
        st.dataframe(styled_df, use_container_width=True)
    
    else:
        # No styling - just format numbers
        for col in display_pivot.columns[2:]:
            display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
        st.dataframe(display_pivot, use_container_width=True)
    
    # Show totals with indicators
    show_gap_totals_with_indicators(display_df, display_options["period_type"])


# ADD this function sau show_gap_pivot_view

def show_gap_totals_with_indicators(df_summary: pd.DataFrame, period: str):
    """Show period totals for GAP with past period indicators"""
    try:
        # Prepare data
        df_grouped = df_summary.copy()
        df_grouped["gap_quantity"] = pd.to_numeric(df_grouped["gap_quantity"], errors='coerce').fillna(0)
        df_grouped["total_demand_qty"] = pd.to_numeric(df_grouped["total_demand_qty"], errors='coerce').fillna(0)
        df_grouped["supply_in_period"] = pd.to_numeric(df_grouped["supply_in_period"], errors='coerce').fillna(0)
        
        # Calculate aggregates by period
        summary_by_period = df_grouped.groupby("period").agg({
            'total_demand_qty': 'sum',
            'supply_in_period': 'sum',
            'gap_quantity': 'sum'
        })
        
        # Create summary DataFrame
        summary_data = {"Metric": ["📤 TOTAL DEMAND", "📥 TOTAL SUPPLY", "📊 NET GAP"]}
        
        # Add all periods to summary_data with indicators
        for period_val in summary_by_period.index:
            col_name = f"🔴 {period_val}" if is_past_period(str(period_val), period) else str(period_val)
            summary_data[col_name] = [
                format_number(summary_by_period.loc[period_val, 'total_demand_qty']),
                format_number(summary_by_period.loc[period_val, 'supply_in_period']),
                format_number(summary_by_period.loc[period_val, 'gap_quantity'])
            ]
        
        display_final = pd.DataFrame(summary_data)
        
        # Sort columns
        metric_cols = ["Metric"]
        period_cols = [col for col in display_final.columns if col not in metric_cols]
        
        # Sort period columns based on original period value
        def get_sort_key(col_name):
            # Remove indicator if present
            clean_name = col_name.replace("🔴 ", "")
            if period == "Weekly":
                return parse_week_period(clean_name)
            elif period == "Monthly":
                return parse_month_period(clean_name)
            else:
                try:
                    return pd.to_datetime(clean_name)
                except:
                    return pd.Timestamp.max
        
        sorted_period_cols = sorted(period_cols, key=get_sort_key)
        display_final = display_final[metric_cols + sorted_period_cols]
        
        st.markdown("#### 🔢 Column Totals")
        st.dataframe(display_final, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error showing totals: {str(e)}")

# === Export Functions ===
def show_export_section(gap_df):
   """Show export options"""
   st.markdown("### 📤 Export Options")
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       # Export detailed GAP
       DisplayComponents.show_export_button(gap_df, "gap_analysis", "📊 Export GAP Details")
   
   with col2:
       # Export shortage summary
       shortage_df = gap_df[gap_df["gap_quantity"] < 0]
       if not shortage_df.empty:
           shortage_summary = shortage_df.groupby(['pt_code', 'product_name']).agg({
               'gap_quantity': 'sum',
               'total_demand_qty': 'sum',
               'total_available': 'sum'
           }).reset_index()
           
           DisplayComponents.show_export_button(shortage_summary, "shortage_summary", "🚨 Export Shortage Summary")
   
   with col3:
       # Export multiple sheets
       if st.button("📑 Export Complete Report"):
           # Prepare multiple dataframes
           sheets_dict = {
               "GAP Details": gap_df,
               "Shortage Summary": shortage_df.groupby(['pt_code', 'product_name']).agg({
                   'gap_quantity': 'sum',
                   'total_demand_qty': 'sum'
               }).reset_index() if not shortage_df.empty else pd.DataFrame(),
               "Product Summary": gap_df.groupby(['pt_code', 'product_name']).agg({
                   'total_demand_qty': 'sum',
                   'total_available': 'sum',
                   'gap_quantity': 'sum'
               }).reset_index()
           }
           
           excel_data = export_multiple_sheets(sheets_dict)
           st.download_button(
               "Download Complete Report",
               data=excel_data,
               file_name=f"gap_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
           )

# === Action Buttons Section ===
def show_action_buttons(gap_df):
    """Show action buttons based on GAP analysis results"""
    st.markdown("---")
    st.header("🎯 Recommended Actions")
    
    # Calculate conditions
    shortage_exists = not gap_df[gap_df['gap_quantity'] < 0].empty
    supply_exists = not gap_df[gap_df['total_available'] > 0].empty
    surplus_exists = not gap_df[gap_df['gap_quantity'] > 0].empty
    
    # Get metrics
    shortage_count = len(gap_df[gap_df['gap_quantity'] < 0]['pt_code'].unique())
    total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
    surplus_count = len(gap_df[gap_df['gap_quantity'] > 0]['pt_code'].unique())
    total_surplus = gap_df[gap_df['gap_quantity'] > 0]['gap_quantity'].sum()
    
    # Show action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Allocation Plan - available when there's supply to allocate
        if supply_exists:
            st.markdown("### 📋 Allocation Planning")
            products_with_supply = len(gap_df[gap_df['total_available'] > 0]['pt_code'].unique())
            st.info(f"Found {products_with_supply} products with available supply")
            
            if st.button("🧩 Create Allocation Plan", 
                        type="primary", 
                        use_container_width=True,
                        key="gap_create_allocation_btn"):
                st.switch_page("pages/4_🧩_Allocation_Plan.py")
        else:
            st.info("✅ No supply available for allocation")
    
    with col2:
        # PO Suggestions - only for shortage
        if shortage_exists:
            st.markdown("### 📦 Replenishment Needed")
            st.warning(f"Total shortage: {format_number(total_shortage)} units")
            
            if st.button("📌 Generate PO Suggestions", 
                        type="secondary", 
                        use_container_width=True,
                        key="gap_generate_po_btn"):
                st.switch_page("pages/5_📌_PO_Suggestions.py")
        else:
            st.success("✅ No shortage detected")
    
    with col3:
        # Export options
        st.markdown("### 📤 Export Reports")
        export_gap_reports(gap_df, shortage_exists, surplus_exists)


def export_gap_reports(gap_df, shortage_exists, surplus_exists):
    """Export various GAP analysis reports"""
    
    # Export complete GAP details
    if st.button("📊 Export GAP Details", 
                use_container_width=True,
                key="export_gap_details"):
        DisplayComponents.show_export_button(
            df=gap_df,
            filename="gap_analysis_details",
            button_label=None  # Hide label since button already shown
        )
    
    # Export shortage summary if exists
    if shortage_exists:
        shortage_df = gap_df[gap_df['gap_quantity'] < 0].copy()
        shortage_summary = shortage_df.groupby(['pt_code', 'product_name']).agg({
            'gap_quantity': 'sum',
            'total_demand_qty': 'sum',
            'total_available': 'sum'
        }).reset_index()
        shortage_summary['gap_quantity'] = shortage_summary['gap_quantity'].abs()
        shortage_summary.rename(columns={'gap_quantity': 'shortage_quantity'}, inplace=True)
        
        if st.button("🚨 Export Shortage Report",
                    use_container_width=True,
                    key="export_shortage_report"):
            DisplayComponents.show_export_button(
                df=shortage_summary,
                filename="shortage_report",
                button_label=None
            )
    
    # Export surplus summary if exists
    if surplus_exists:
        surplus_df = gap_df[gap_df['gap_quantity'] > 0].copy()
        surplus_summary = surplus_df.groupby(['pt_code', 'product_name']).agg({
            'gap_quantity': 'sum',
            'total_available': 'sum',
            'total_demand_qty': 'sum'
        }).reset_index()
        surplus_summary.rename(columns={'gap_quantity': 'surplus_quantity'}, inplace=True)
        
        if st.button("📈 Export Surplus Report",
                    use_container_width=True,
                    key="export_surplus_report"):
            DisplayComponents.show_export_button(
                df=surplus_summary,
                filename="surplus_report",
                button_label=None
            )
    
    # Export complete multi-sheet report
    if st.button("📑 Export Complete Report",
                use_container_width=True,
                key="export_complete_report"):
        export_complete_gap_report(gap_df, shortage_exists, surplus_exists)


def export_complete_gap_report(gap_df, shortage_exists, surplus_exists):
    """Export comprehensive GAP analysis report with multiple sheets"""
    
    sheets_dict = {
        "GAP Overview": create_gap_overview_sheet(gap_df),
        "GAP Details": gap_df
    }
    
    # Add shortage sheet if exists
    if shortage_exists:
        shortage_df = gap_df[gap_df['gap_quantity'] < 0].copy()
        shortage_summary = shortage_df.groupby(['pt_code', 'product_name']).agg({
            'gap_quantity': lambda x: x.abs().sum(),
            'total_demand_qty': 'sum',
            'total_available': 'sum',
            'period': 'count'
        }).reset_index()
        shortage_summary.rename(columns={
            'gap_quantity': 'total_shortage',
            'period': 'affected_periods'
        }, inplace=True)
        sheets_dict["Shortage Summary"] = shortage_summary
    
    # Add surplus sheet if exists
    if surplus_exists:
        surplus_df = gap_df[gap_df['gap_quantity'] > 0].copy()
        surplus_summary = surplus_df.groupby(['pt_code', 'product_name']).agg({
            'gap_quantity': 'sum',
            'total_available': 'sum',
            'total_demand_qty': 'sum',
            'period': 'count'
        }).reset_index()
        surplus_summary.rename(columns={
            'gap_quantity': 'total_surplus',
            'period': 'surplus_periods'
        }, inplace=True)
        sheets_dict["Surplus Summary"] = surplus_summary
    
    # Add product summary
    product_summary = gap_df.groupby(['pt_code', 'product_name']).agg({
        'total_demand_qty': 'sum',
        'total_available': 'sum',
        'gap_quantity': 'sum',
        'fulfillment_rate_percent': 'mean'
    }).reset_index()
    product_summary['status'] = product_summary['gap_quantity'].apply(
        lambda x: 'Shortage' if x < 0 else 'Surplus' if x > 0 else 'Balanced'
    )
    sheets_dict["Product Summary"] = product_summary
    
    # Export to Excel
    excel_data = export_multiple_sheets(sheets_dict)
    st.download_button(
        "📥 Download Complete Report",
        data=excel_data,
        file_name=f"gap_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def create_gap_overview_sheet(gap_df):
    """Create overview summary for GAP analysis"""
    
    # Calculate key metrics
    total_products = gap_df['pt_code'].nunique()
    total_periods = gap_df['period'].nunique()
    
    shortage_products = gap_df[gap_df['gap_quantity'] < 0]['pt_code'].nunique()
    surplus_products = gap_df[gap_df['gap_quantity'] > 0]['pt_code'].nunique()
    balanced_products = total_products - shortage_products - surplus_products
    
    total_demand = gap_df['total_demand_qty'].sum()
    total_supply = gap_df['total_available'].sum()
    total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
    total_surplus = gap_df[gap_df['gap_quantity'] > 0]['gap_quantity'].sum()
    
    avg_fulfillment = gap_df['fulfillment_rate_percent'].mean()
    
    # Create overview dataframe
    overview_data = {
        'Metric': [
            'Analysis Period',
            'Total Products',
            'Total Periods',
            'Products with Shortage',
            'Products with Surplus', 
            'Balanced Products',
            'Total Demand Quantity',
            'Total Supply Available',
            'Total Shortage Quantity',
            'Total Surplus Quantity',
            'Average Fulfillment Rate',
            'Analysis Date'
        ],
        'Value': [
            f"{gap_df['period'].min()} to {gap_df['period'].max()}",
            total_products,
            total_periods,
            shortage_products,
            surplus_products,
            balanced_products,
            format_number(total_demand),
            format_number(total_supply),
            format_number(total_shortage),
            format_number(total_surplus),
            f"{avg_fulfillment:.1f}%",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    
    return pd.DataFrame(overview_data)


# === Main Page Logic ===

# Check if we have saved analysis from session
if st.session_state.get('gap_analysis_result') is not None and not st.session_state.get('gap_analysis_ran', False):
   st.info(f"📅 Using previous analysis from: {st.session_state.get('last_analysis_time', 'Unknown')}")
   if st.button("🔄 Run New Analysis"):
       # Clear previous results
       for key in ['gap_analysis_result', 'demand_filtered', 'supply_filtered', 'gap_df_cached']:
           if key in st.session_state:
               del st.session_state[key]
       st.session_state['gap_analysis_ran'] = False
       st.rerun()

# Pre-load data on first run to populate filters
if 'initial_data_loaded' not in st.session_state:
   with st.spinner("Initializing data..."):
       # Load with default sources
       default_demand_sources = ["OC", "Forecast"]
       default_supply_sources = ["Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"]
       
       # Load demand data
       df_demand_temp = load_and_prepare_demand_data(
           default_demand_sources,
           False  # include_converted = False
       )
       
       # Load supply data
       df_supply_temp = load_and_prepare_supply_data(
           default_supply_sources, 
           True  # exclude_expired = True
       )
       
       if not df_demand_temp.empty and not df_supply_temp.empty:
           # Store in session state
           st.session_state['temp_demand_data'] = df_demand_temp
           st.session_state['temp_supply_data'] = df_supply_temp
           # Update filter options
           update_filter_options(df_demand_temp, df_supply_temp)
       
       st.session_state['initial_data_loaded'] = True
       st.session_state['last_source_key'] = "OC-Forecast_Inventory-Pending CAN-Pending PO-Pending WH Transfer"

# Now show source selection with populated filters
selected_sources = select_gap_sources()

# Display options - ALWAYS SHOW
display_options = get_gap_display_options()

# Check if sources changed and reload if needed
source_key = f"{'-'.join(selected_sources['demand'])}_{'-'.join(selected_sources['supply'])}"

if st.session_state.get('last_source_key') != source_key:
   st.session_state['last_source_key'] = source_key
   # Clear cached data when sources change
   if 'temp_demand_data' in st.session_state:
       del st.session_state['temp_demand_data']
   if 'temp_supply_data' in st.session_state:
       del st.session_state['temp_supply_data']
   
   # Reload data with new sources
   if selected_sources["demand"] and selected_sources["supply"]:
       with st.spinner("Reloading data..."):
           # Load demand data
           df_demand_temp = load_and_prepare_demand_data(
               selected_sources["demand"],
               selected_sources["include_converted"]
           )
           
           # Load supply data
           df_supply_temp = load_and_prepare_supply_data(
               selected_sources["supply"], 
               selected_sources["exclude_expired"]
           )
           
           if not df_demand_temp.empty and not df_supply_temp.empty:
               # Store in session state
               st.session_state['temp_demand_data'] = df_demand_temp
               st.session_state['temp_supply_data'] = df_supply_temp
               # Update filter options
               update_filter_options(df_demand_temp, df_supply_temp)

# Get data from session state for filters
df_for_filters_demand = st.session_state.get('temp_demand_data', pd.DataFrame())
df_for_filters_supply = st.session_state.get('temp_supply_data', pd.DataFrame())

# Filters setup with loaded data
filters = apply_gap_filters(
   df_for_filters_demand, 
   df_for_filters_supply,
   use_adjusted_demand,
   use_adjusted_supply
)

# Load data button
# Thêm debug code vào sau khi Apply filters và trước khi Calculate GAP

if st.button("🚀 Run GAP Analysis", type="primary", use_container_width=True):
    
    if not selected_sources["demand"] or not selected_sources["supply"]:
        st.error("Please select at least one demand source and one supply source.")
    else:
        # Use cached data if available
        if 'temp_demand_data' in st.session_state and 'temp_supply_data' in st.session_state:
            df_demand_all = st.session_state['temp_demand_data']
            df_supply_all = st.session_state['temp_supply_data']
        else:
            # Load demand data
            with st.spinner("Loading demand data..."):
                df_demand_all = load_and_prepare_demand_data(
                    selected_sources["demand"],
                    selected_sources["include_converted"]
                )
            
            # Load supply data
            with st.spinner("Loading supply data..."):
                df_supply_all = load_and_prepare_supply_data(
                    selected_sources["supply"], 
                    selected_sources["exclude_expired"]
                )
        
        # DEBUG: Check data before filtering
        if debug_mode:
            st.write("🐛 DEBUG - Data BEFORE filtering:")
            st.write(f"- Demand ALL: {len(df_demand_all)} records")
            if not df_demand_all.empty:
                st.write(f"  - Unique products: {df_demand_all['pt_code'].nunique()}")
                st.write(f"  - Products: {sorted(df_demand_all['pt_code'].unique())[:10]}...")
            
            st.write(f"- Supply ALL: {len(df_supply_all)} records")
            if not df_supply_all.empty:
                st.write(f"  - Unique products: {df_supply_all['pt_code'].nunique()}")
                st.write(f"  - Products: {sorted(df_supply_all['pt_code'].unique())[:10]}...")
                st.write(f"  - Source types: {df_supply_all['source_type'].value_counts().to_dict() if 'source_type' in df_supply_all.columns else 'No source_type'}")
        
        # Show adjustment summaries
        if use_adjusted_demand and not df_demand_all.empty:
            DateModeComponent.show_adjustment_summary(
                df_demand_all, ['etd'], 'GAP Demand'
            )
        
        if use_adjusted_supply and not df_supply_all.empty:
            date_cols = []
            for source_type in df_supply_all['source_type'].unique() if 'source_type' in df_supply_all.columns else []:
                date_map = {
                    'Inventory': 'date_ref',
                    'Pending CAN': 'arrival_date',
                    'Pending PO': 'eta',
                    'Pending WH Transfer': 'transfer_date'
                }
                if source_type in date_map:
                    date_col = date_map[source_type]
                    if date_col in df_supply_all.columns and date_col not in date_cols:
                        date_cols.append(date_col)
            
            if date_cols:
                DateModeComponent.show_adjustment_summary(
                    df_supply_all, date_cols, 'GAP Supply'
                )
        
        # Adjust supply for allocations
        with st.spinner("Adjusting supply for allocations..."):
            df_supply_adjusted = adjust_supply_for_allocations(df_supply_all.copy())
        
        # DEBUG: Check filters being applied
        if debug_mode:
            st.write("🐛 DEBUG - Filters being applied:")
            st.write(f"- Entity filter: {filters.get('entity', [])}")
            st.write(f"- Product filter: {filters.get('product', [])}")
            st.write(f"- Brand filter: {filters.get('brand', [])}")
            st.write(f"- Customer filter (source): {selected_sources.get('selected_customers', [])}")
            st.write(f"- Date range: {filters.get('start_date')} to {filters.get('end_date')}")
        
        # Apply filters
        df_demand_filtered, df_supply_filtered = apply_filters_to_data(
            df_demand_all, 
            df_supply_adjusted, 
            filters, 
            selected_sources.get("selected_customers", []),
            use_adjusted_demand,
            use_adjusted_supply
        )
        
        # DEBUG: Check data after filtering
        if debug_mode:
            st.write("🐛 DEBUG - Data AFTER filtering:")
            st.write(f"- Demand filtered: {len(df_demand_filtered)} records")
            if not df_demand_filtered.empty:
                st.write(f"  - Products: {df_demand_filtered['pt_code'].unique()}")
            
            st.write(f"- Supply filtered: {len(df_supply_filtered)} records")
            if not df_supply_filtered.empty:
                st.write(f"  - Products: {df_supply_filtered['pt_code'].unique()}")
                if 'source_type' in df_supply_filtered.columns:
                    st.write(f"  - Source types: {df_supply_filtered['source_type'].value_counts().to_dict()}")
            else:
                st.error("⚠️ Supply data is EMPTY after filtering!")
                
                # ENHANCED DEBUG - Check product code issues
                if not df_supply_all.empty and filters.get('product'):
                    st.write("🔍 DEBUG - Checking product code matching:")
                    
                    # Get the filtered product
                    target_product = filters['product'][0] if filters['product'] else None
                    
                    if target_product:
                        st.write(f"  - Looking for product: '{target_product}' (length: {len(target_product)})")
                        
                        # Check exact matches
                        exact_matches = df_supply_all[df_supply_all['pt_code'] == target_product]
                        st.write(f"  - Exact matches in supply: {len(exact_matches)}")
                        
                        # Check with string conversion and strip
                        supply_pt_codes = df_supply_all['pt_code'].astype(str).str.strip()
                        stripped_matches = df_supply_all[supply_pt_codes == target_product.strip()]
                        st.write(f"  - Matches after strip: {len(stripped_matches)}")
                        
                        # Check similar products (contains)
                        similar = df_supply_all[df_supply_all['pt_code'].astype(str).str.contains(target_product[:6], na=False)]
                        st.write(f"  - Similar products (first 6 chars): {len(similar)}")
                        if len(similar) > 0:
                            st.write(f"    Found similar: {similar['pt_code'].unique()[:5].tolist()}")
                        
                        # Show actual pt_code values in supply
                        all_supply_products = df_supply_all['pt_code'].dropna().unique()
                        matching_prefix = [p for p in all_supply_products if str(p).startswith(target_product[:6])]
                        if matching_prefix:
                            st.write(f"  - Products with same prefix: {matching_prefix[:5]}")
                            
                            # Show exact representation
                            for p in matching_prefix[:3]:
                                st.write(f"    '{p}' (type: {type(p)}, len: {len(str(p))}, repr: {repr(p)})")
                        
                        # Check other filters that might exclude the product
                        if len(exact_matches) > 0 or len(stripped_matches) > 0:
                            st.write("  - Product found but filtered out by other criteria:")
                            
                            # Check entity filter
                            if filters.get('entity'):
                                entity_match = df_supply_all[
                                    (df_supply_all['pt_code'] == target_product) & 
                                    (df_supply_all['legal_entity'].isin(filters['entity']))
                                ]
                                st.write(f"    After entity filter: {len(entity_match)} records")
                            
                            # Check date filter
                            product_records = df_supply_all[df_supply_all['pt_code'] == target_product]
                            if not product_records.empty:
                                st.write("    Date info for this product:")
                                for source_type in product_records['source_type'].unique():
                                    source_records = product_records[product_records['source_type'] == source_type]
                                    date_col = get_supply_date_column(source_records, source_type, use_adjusted_supply)
                                    if date_col in source_records.columns:
                                        dates = pd.to_datetime(source_records[date_col], errors='coerce')
                                        st.write(f"      {source_type} - {date_col}: min={dates.min()}, max={dates.max()}")
        
        # Store filtered data
        st.session_state['gap_analysis_data'] = {
            'demand': df_demand_filtered,
            'supply': df_supply_filtered
        }
        st.session_state['gap_analysis_ran'] = True
        st.session_state['gap_display_options'] = display_options
        st.session_state['gap_date_modes'] = {
            'demand': use_adjusted_demand,
            'supply': use_adjusted_supply
        }
        
        # Clear cached GAP results to force recalculation
        if 'gap_df_cached' in st.session_state:
            del st.session_state['gap_df_cached']
        if 'gap_period_type_cache' in st.session_state:
            del st.session_state['gap_period_type_cache']


# Display results if analysis has been run
if st.session_state.get('gap_analysis_ran', False) and st.session_state.get('gap_analysis_data') is not None:
    
    # Get data from session state
    df_demand_filtered = st.session_state['gap_analysis_data']['demand']
    df_supply_filtered = st.session_state['gap_analysis_data']['supply']
    
    # Debug: Check data after filter
    if debug_mode:
        st.write("🐛 After filtering:")
        st.write(f"- Demand records: {len(df_demand_filtered)}")
        st.write(f"- Supply records: {len(df_supply_filtered)}")
        if not df_demand_filtered.empty:
            st.write(f"- Demand products: {df_demand_filtered['pt_code'].unique()[:5]}...")
        if not df_supply_filtered.empty:
            st.write(f"- Supply products: {df_supply_filtered['pt_code'].unique()[:5]}...")
            st.write(f"- Supply columns: {list(df_supply_filtered.columns)}")
    
    # Get date modes
    date_modes = st.session_state.get('gap_date_modes', {
        'demand': use_adjusted_demand,
        'supply': use_adjusted_supply
    })
    
    # Get display options from session state or use current
    stored_display_options = st.session_state.get('gap_display_options', display_options)
    
    # Initialize display dataframes
    df_demand_filtered_display = df_demand_filtered.copy()
    df_supply_filtered_display = df_supply_filtered.copy()
    
    # Apply date exclusion if requested
    if stored_display_options.get("exclude_missing_dates", True):
        # Filter demand - safe check
        if not df_demand_filtered_display.empty:
            demand_date_col = get_demand_date_column(df_demand_filtered_display, date_modes['demand'])
            if demand_date_col and demand_date_col in df_demand_filtered_display.columns:
                mask = df_demand_filtered_display[demand_date_col].notna()
                df_demand_filtered_display = df_demand_filtered_display[mask]
                if debug_mode:
                    st.write(f"🐛 Demand after date filter: {len(df_demand_filtered_display)} records")
        
        # Filter supply - safe check
        if not df_supply_filtered_display.empty:
            # Check if source_type column exists
            if 'source_type' in df_supply_filtered_display.columns:
                filtered_supply_dfs = []
                source_types = df_supply_filtered_display['source_type'].unique()
                
                for source_type in source_types:
                    source_df = df_supply_filtered_display[df_supply_filtered_display['source_type'] == source_type].copy()
                    
                    # Get date column for this source type
                    supply_date_col = get_supply_date_column(source_df, source_type, date_modes['supply'])
                    
                    if supply_date_col and supply_date_col in source_df.columns:
                        # Filter out missing dates
                        source_df = source_df[source_df[supply_date_col].notna()]
                    
                    if not source_df.empty:
                        filtered_supply_dfs.append(source_df)
                
                # Combine results
                if filtered_supply_dfs:
                    df_supply_filtered_display = pd.concat(filtered_supply_dfs, ignore_index=True)
                else:
                    df_supply_filtered_display = pd.DataFrame()
            else:
                # No source_type column - try common date columns
                date_cols = ['date_ref', 'date_ref_adjusted', 'eta', 'arrival_date', 'transfer_date']
                date_col_found = None
                
                for col in date_cols:
                    if col in df_supply_filtered_display.columns:
                        date_col_found = col
                        break
                
                if date_col_found:
                    df_supply_filtered_display = df_supply_filtered_display[
                        df_supply_filtered_display[date_col_found].notna()
                    ]
            
            if debug_mode:
                st.write(f"🐛 Supply after date filter: {len(df_supply_filtered_display)} records")
    
    # Final validation - check if we have data
    if df_demand_filtered_display.empty and df_supply_filtered_display.empty:
        st.error("❌ No data available after applying filters.")
        st.info("💡 This could happen because:")
        st.write("- The filtered product has no matching records")
        st.write("- All dates are missing for the selected criteria")
        st.write("- The date range excludes all records")
        
        # Show what data we started with
        if debug_mode:
            st.write("🐛 Original data before date filtering:")
            st.write(f"- Demand: {len(df_demand_filtered)} records")
            st.write(f"- Supply: {len(df_supply_filtered)} records")
        
        st.stop()
    
    # Show date status comparison
    show_date_status_comparison(
        df_demand_filtered, 
        df_supply_filtered,
        date_modes['demand'],
        date_modes['supply']
    )
    
    # Show allocation impact
    df_demand_enhanced = enhance_demand_with_allocation_info(df_demand_filtered_display)
    show_allocation_impact_summary(df_demand_enhanced)
    
    # Check if we need to recalculate GAP based on period change
    current_period = display_options["period_type"]
    cached_period = st.session_state.get('gap_period_type_cache', None)
    
    if 'gap_df_cached' not in st.session_state or cached_period != current_period:
        # Calculate GAP
        with st.spinner("Calculating supply-demand gaps..."):
            gap_df = calculate_gap_with_carry_forward(
                df_demand_filtered_display, 
                df_supply_filtered_display, 
                current_period,
                date_modes['demand'],
                date_modes['supply']
            )
            # Cache the results
            st.session_state['gap_df_cached'] = gap_df
            st.session_state['gap_period_type_cache'] = current_period
    else:
        gap_df = st.session_state['gap_df_cached']
    
    # Save to session state for other pages
    save_to_session_state('gap_analysis_result', gap_df)
    save_to_session_state('demand_filtered', df_demand_filtered_display)
    save_to_session_state('supply_filtered', df_supply_filtered_display)
    save_to_session_state('last_gap_analysis', gap_df)
    save_to_session_state('last_analysis_time', datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    if not gap_df.empty:
        # Display results with current display options
        show_gap_summary(
            gap_df, 
            display_options, 
            df_demand_filtered_display, 
            df_supply_filtered_display,
            date_modes['demand'],
            date_modes['supply']
        )
        show_gap_detail_table(gap_df, display_options, df_demand_filtered_display, df_supply_filtered_display)
        show_gap_pivot_view(gap_df, display_options)
        
        # Single unified action section (as you have it)
        show_action_buttons(gap_df)
    else:
        st.warning("No data available for the selected filters and sources.")

# Debug info panel
if debug_mode:
    with st.expander("🐛 Debug Information", expanded=True):
        st.markdown("### Debug Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Date Modes:**")
            st.write(f"- Demand: {'Adjusted' if use_adjusted_demand else 'Original'}")
            st.write(f"- Supply: {'Adjusted' if use_adjusted_supply else 'Original'}")
            
            if 'gap_analysis_data' in st.session_state:
                demand_data = st.session_state['gap_analysis_data'].get('demand', pd.DataFrame())
                supply_data = st.session_state['gap_analysis_data'].get('supply', pd.DataFrame())
                
                st.write("\n**Data Shapes:**")
                st.write(f"- Demand: {len(demand_data)} rows")
                st.write(f"- Supply: {len(supply_data)} rows")
                
                if 'gap_df_cached' in st.session_state:
                    gap_data = st.session_state['gap_df_cached']
                    st.write(f"- GAP: {len(gap_data)} rows")
        
        with col2:
            st.write("**Session State Keys:**")
            gap_keys = [k for k in st.session_state.keys() if 'gap' in k.lower()]
            for key in gap_keys[:10]:  # Show first 10
                st.write(f"- {key}")
            
            if len(gap_keys) > 10:
                st.write(f"... and {len(gap_keys) - 10} more")



# Help section
DisplayComponents.show_help_section(
   "Understanding GAP Analysis",
   """
   ### How GAP Analysis Works
   
   **Date Mode Selection:**
   - **Dual Control**: Separate date modes for demand and supply
   - **Demand Dates**: Controls ETD adjustments
   - **Supply Dates**: Controls arrival/ready date adjustments per source
   - Mix and match based on your analysis needs
   
   **Allocation Awareness:**
   - Supply is adjusted for allocated but undelivered quantities
   - Demand shows both original and unallocated amounts
   - GAP calculation uses net unallocated demand
   
   **Carry-Forward Logic:**
   - Excess inventory from one period carries to the next
   - Only positive inventory is carried forward
   - Reflects real warehouse operations
   
   **Date Columns by Source:**
   - **OC/Forecast**: ETD (Expected Time of Delivery)
   - **Inventory**: date_ref (current date)
   - **Pending CAN**: arrival_date
   - **Pending PO**: eta
   - **Pending WH Transfer**: transfer_date
   
   **Key Metrics:**
   - **Begin Inventory**: Carried from previous period
   - **Supply in Period**: New arrivals in this period
   - **Total Available**: Begin + Supply
   - **GAP**: Available - Demand (negative = shortage)
   - **Fulfillment Rate**: % of demand that can be met
   
   **Product Analysis:**
   - **Matched**: Products with both demand and supply
   - **Demand Only**: Need to source (no supply)
   - **Supply Only**: Potential dead stock (no demand)
   
   **Visual Indicators:**
   - 🔴 Past Period: Already occurred
   - ❌ Shortage: Insufficient supply
   - ✅ Fulfilled: Adequate supply
   
   **Common Issues:**
   - **Missing Dates**: Records excluded or shown as warnings
   - **Converted Forecasts**: May double-count with OC
   - **Past Periods**: Historical data included in analysis
   
   **Next Steps:**
   1. **Shortage → Allocation Plan**: Distribute limited supply
   2. **Shortage → PO Suggestions**: Order more inventory
   3. **Surplus → Reallocation**: Move excess to other locations
   
   **Tips:**
   - Configure time adjustments for accurate planning
   - Use date modes to compare scenarios
   - Export pivot view for management reporting
   - Monitor product matching rates
   """
)

# Footer
st.markdown("---")
st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")