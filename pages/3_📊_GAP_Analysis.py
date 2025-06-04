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

# === Date Mode Selection ===
st.markdown("---")

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
    """Add allocation information to demand dataframe"""
    engine = get_db_engine()
    
    try:
        oc_allocations = pd.read_sql(text("""
            SELECT 
                demand_reference_id,
                SUM(allocated_qty) as total_allocated,
                SUM(delivered_qty) as total_delivered,
                SUM(allocated_qty - delivered_qty) as undelivered_allocated
            FROM allocation_details ad
            JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
            WHERE ap.status IN ('APPROVED', 'EXECUTED')
              AND ad.demand_type = 'OC'
              AND ad.status NOT IN ('CANCELLED')
            GROUP BY demand_reference_id
        """), engine)
        
        if not oc_allocations.empty and 'demand_line_id' in df_demand.columns:
            # Extract OC IDs for merging
            df_demand['merge_id'] = df_demand['demand_line_id'].str.extract(r'(\d+)_OC')
            df_demand['merge_id'] = pd.to_numeric(df_demand['merge_id'], errors='coerce')
            
            # Merge with allocation data
            df_demand = df_demand.merge(
                oc_allocations,
                left_on='merge_id',
                right_on='demand_reference_id',
                how='left',
                suffixes=('', '_alloc')
            )
            
            # Clean up
            df_demand.drop(columns=['merge_id', 'demand_reference_id'], inplace=True, errors='ignore')
    except:
        # If allocation query fails, continue without allocation data
        pass
    
    # Fill allocation columns
    allocation_cols = ['total_allocated', 'total_delivered', 'undelivered_allocated']
    for col in allocation_cols:
        if col not in df_demand.columns:
            df_demand[col] = 0
        else:
            df_demand[col] = df_demand[col].fillna(0)
    
    # Calculate unallocated demand
    df_demand['unallocated_demand'] = df_demand['demand_quantity'] - df_demand['total_allocated']
    df_demand['unallocated_demand'] = df_demand['unallocated_demand'].clip(lower=0)
    
    # Add allocation status
    df_demand['allocation_status'] = df_demand.apply(
        lambda x: 'Fully Allocated' if x['unallocated_demand'] <= 0 
        else 'Partial' if x['total_allocated'] > 0 
        else 'Not Allocated', axis=1
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
    
    st.markdown("---")
    
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
    """Show GAP analysis summary with enhanced metrics, data quality, and visualizations"""
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
    
    # Quick Summary Cards
    show_gap_quick_summary(gap_df)
    
    # Product Analysis Metrics
    # Always try to show product analysis, function will handle None/empty data
    show_product_analysis_metrics(df_demand_filtered, df_supply_filtered)
    
    # Show filtering info
    show_filtering_info(gap_df, display_options)
    
    # GAP Analysis Visualizations
    if not gap_df.empty:
        st.markdown("#### 📈 GAP Analysis Visualizations")
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Shortage by Period Chart
            shortage_by_period = gap_df[gap_df['gap_quantity'] < 0].groupby('period')['gap_quantity'].sum().abs()
            
            if not shortage_by_period.empty:
                st.markdown("**📉 Shortage by Period**")
                
                # Create DataFrame for chart
                chart_df = pd.DataFrame({
                    'Period': shortage_by_period.index,
                    'Shortage': shortage_by_period.values
                })
                
                # Sort periods properly based on period type
                period_type = display_options.get("period_type", "Weekly")
                if period_type == "Weekly":
                    chart_df = chart_df.sort_values('Period', key=lambda x: x.apply(parse_week_period))
                elif period_type == "Monthly":
                    chart_df = chart_df.sort_values('Period', key=lambda x: x.apply(parse_month_period))
                else:
                    chart_df = chart_df.sort_values('Period')
                
                # Add color coding for past periods
                chart_df['Color'] = chart_df['Period'].apply(
                    lambda x: '#FF6B6B' if is_past_period(str(x), period_type) else '#4ECDC4'
                )
                
                # Display chart
                st.bar_chart(chart_df.set_index('Period')['Shortage'])
                
                # Show legend
                st.caption("🔴 Red = Past periods | 🟢 Green = Future periods")
            else:
                st.info("✅ No shortages detected in any period!")
        
        with col2:
            # Top Shortage Products Chart
            top_shortage = gap_df[gap_df['gap_quantity'] < 0].groupby(['pt_code', 'product_name'])['gap_quantity'].sum()
            top_shortage = top_shortage.sort_values().head(10).abs()
            
            if not top_shortage.empty:
                st.markdown("**🚨 Top 10 Products with Shortage**")
                
                # Create DataFrame for chart
                chart_df = pd.DataFrame({
                    'Product': [f"{idx[0]}" for idx in top_shortage.index],  # Show PT code only for space
                    'Product_Full': [f"{idx[0]} - {idx[1][:30]}..." for idx in top_shortage.index],  # Full name for tooltip
                    'Shortage': top_shortage.values
                })
                
                # Display chart
                st.bar_chart(chart_df.set_index('Product')['Shortage'])
                
                # Show full names below
                with st.expander("View full product names"):
                    for _, row in chart_df.iterrows():
                        st.caption(f"• {row['Product_Full']}: {format_number(row['Shortage'])}")
            else:
                st.info("✅ No product shortages detected!")
        
        # Additional Analysis Section
        st.markdown("#### 📊 Period Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Fulfillment Rate by Period
            period_fulfillment = gap_df[gap_df['total_demand_qty'] > 0].groupby('period').agg({
                'total_available': 'sum',
                'total_demand_qty': 'sum'
            })
            
            if not period_fulfillment.empty:
                period_fulfillment['fulfillment_rate'] = (
                    period_fulfillment['total_available'] / period_fulfillment['total_demand_qty'] * 100
                ).clip(upper=100)
                
                st.markdown("**📈 Fulfillment Rate by Period**")
                
                # Create chart data
                chart_df = pd.DataFrame({
                    'Period': period_fulfillment.index,
                    'Fulfillment %': period_fulfillment['fulfillment_rate'].round(1)
                })
                
                # Sort periods
                period_type = display_options.get("period_type", "Weekly")
                if period_type == "Weekly":
                    chart_df = chart_df.sort_values('Period', key=lambda x: x.apply(parse_week_period))
                elif period_type == "Monthly":
                    chart_df = chart_df.sort_values('Period', key=lambda x: x.apply(parse_month_period))
                
                # Display line chart
                st.line_chart(chart_df.set_index('Period'))
                
                # Show average
                avg_fulfillment = period_fulfillment['fulfillment_rate'].mean()
                st.metric("Average Fulfillment Rate", f"{avg_fulfillment:.1f}%")
        
        with col4:
            # Supply vs Demand Summary
            st.markdown("**📊 Supply vs Demand Summary**")
            
            total_metrics = gap_df.groupby('period').agg({
                'total_demand_qty': 'sum',
                'supply_in_period': 'sum',
                'gap_quantity': 'sum'
            })
            
            if not total_metrics.empty:
                # Create summary DataFrame
                summary_df = pd.DataFrame({
                    'Total Demand': total_metrics['total_demand_qty'].sum(),
                    'Total Supply': total_metrics['supply_in_period'].sum(),
                    'Net GAP': total_metrics['gap_quantity'].sum()
                }, index=['Value'])
                
                # Display as metrics
                col4_1, col4_2, col4_3 = st.columns(3)
                
                with col4_1:
                    st.metric(
                        "Total Demand",
                        format_number(summary_df['Total Demand'].iloc[0])
                    )
                
                with col4_2:
                    st.metric(
                        "Total Supply",
                        format_number(summary_df['Total Supply'].iloc[0])
                    )
                
                with col4_3:
                    net_gap = summary_df['Net GAP'].iloc[0]
                    st.metric(
                        "Net GAP",
                        format_number(abs(net_gap)),
                        delta="Shortage" if net_gap < 0 else "Surplus",
                        delta_color="inverse" if net_gap < 0 else "normal"
                    )
                
                # Show percentage breakdown
                if summary_df['Total Demand'].iloc[0] > 0:
                    supply_rate = (summary_df['Total Supply'].iloc[0] / summary_df['Total Demand'].iloc[0] * 100)
                    st.progress(min(supply_rate / 100, 1.0))
                    st.caption(f"Supply covers {supply_rate:.1f}% of total demand")
    
    # Executive Summary Text
    st.markdown("---")
    st.markdown("#### 💼 Executive Summary")
    
    # Generate summary text
    if not gap_df.empty:
        total_products = gap_df['pt_code'].nunique()
        shortage_products = gap_df[gap_df['gap_quantity'] < 0]['pt_code'].nunique()
        shortage_percentage = (shortage_products / total_products * 100) if total_products > 0 else 0
        
        total_periods = gap_df['period'].nunique()
        periods_with_shortage = gap_df[gap_df['gap_quantity'] < 0]['period'].nunique()
        
        # Calculate financial impact if value data available
        value_text = ""
        if df_demand_filtered is not None and not df_demand_filtered.empty and 'value_in_usd' in df_demand_filtered.columns:
            shortage_value = df_demand_filtered[
                df_demand_filtered['pt_code'].isin(
                    gap_df[gap_df['gap_quantity'] < 0]['pt_code'].unique()
                )
            ]['value_in_usd'].sum()
            value_text = f" with potential revenue impact of **{format_currency(shortage_value, 'USD', 0)}**"
        
        summary_text = f"""
        The GAP analysis covers **{total_products}** products across **{total_periods}** {display_options.get('period_type', 'Weekly').lower()} periods.
        
        **{shortage_products}** products ({shortage_percentage:.1f}%) have shortages in at least one period{value_text}.
        
        Shortages occur in **{periods_with_shortage}** out of **{total_periods}** periods analyzed.
        """
        
        st.info(summary_text)
        
        # Recommendations
        if shortage_products > 0:
            st.markdown("**🎯 Recommended Actions:**")
            
            recommendations = []
            
            if shortage_percentage > 50:
                recommendations.append("🚨 **Critical**: Over 50% of products have shortages. Immediate action required.")
            elif shortage_percentage > 25:
                recommendations.append("⚠️ **High Priority**: Significant shortage detected. Review allocation priorities.")
            
            if periods_with_shortage > total_periods * 0.5:
                recommendations.append("📅 **Widespread Issue**: Shortages span multiple periods. Consider bulk purchasing.")
            
            if df_demand_filtered is not None and df_supply_filtered is not None:
                if 'pt_code' in df_demand_filtered.columns and 'pt_code' in df_supply_filtered.columns:
                    demand_only = set(df_demand_filtered['pt_code'].unique()) - set(df_supply_filtered['pt_code'].unique())
                    if len(demand_only) > 0:
                        recommendations.append(f"📦 **No Supply**: {len(demand_only)} products have demand but zero supply.")
            
            if not recommendations:
                recommendations.append("✅ **Good Coverage**: Minor shortages detected. Use allocation plan to optimize.")
            
            for rec in recommendations:
                st.write(rec)
    else:
        st.warning("No GAP data available to generate summary.")


def show_gap_quick_summary(gap_df):
    """Show quick summary cards for executives"""
    if gap_df.empty:
        return
    
    st.markdown("#### 🎯 Quick Summary")
    
    # Calculate key metrics
    total_demand = gap_df['total_demand_qty'].sum()
    total_supply = gap_df['supply_in_period'].sum()
    total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
    total_surplus = gap_df[gap_df['gap_quantity'] > 0]['gap_quantity'].sum()
    
    # Calculate service level
    fulfilled_demand = gap_df.apply(
        lambda row: min(row['total_demand_qty'], row['total_available']) if row['total_demand_qty'] > 0 else 0, 
        axis=1
    ).sum()
    service_level = (fulfilled_demand / total_demand * 100) if total_demand > 0 else 100
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Demand",
            format_number(total_demand),
            help="Total demand quantity across all products and periods"
        )
    
    with col2:
        st.metric(
            "Total Supply",
            format_number(total_supply),
            delta=format_number(total_supply - total_demand),
            delta_color="normal" if total_supply >= total_demand else "inverse",
            help="Total supply quantity across all products and periods"
        )
    
    with col3:
        st.metric(
            "Service Level",
            f"{service_level:.1f}%",
            delta=f"{service_level - 100:.1f}%" if service_level < 100 else "Full coverage",
            delta_color="normal" if service_level >= 95 else "inverse",
            help="Percentage of demand that can be fulfilled"
        )
    
    with col4:
        if total_shortage > total_surplus:
            st.metric(
                "Net Position",
                format_number(total_shortage),
                delta="Shortage",
                delta_color="inverse",
                help="Overall shortage after considering all surplus"
            )
        else:
            st.metric(
                "Net Position",
                format_number(total_surplus - total_shortage),
                delta="Surplus",
                delta_color="normal",
                help="Overall surplus after covering all shortages"
            )
    
    # Additional row for critical metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        critical_shortage = gap_df[
            (gap_df['gap_quantity'] < 0) & 
            (gap_df['fulfillment_rate_percent'] < 50)
        ]['pt_code'].nunique()
        
        st.metric(
            "Critical Shortages",
            critical_shortage,
            delta="Products <50% fulfilled",
            delta_color="inverse" if critical_shortage > 0 else "off",
            help="Products with less than 50% fulfillment rate"
        )
    
    with col6:
        perfect_fulfillment = gap_df[
            gap_df['fulfillment_rate_percent'] >= 100
        ]['pt_code'].nunique()
        total_products = gap_df['pt_code'].nunique()
        
        st.metric(
            "Perfect Fulfillment",
            perfect_fulfillment,
            delta=f"{perfect_fulfillment/total_products*100:.1f}%" if total_products > 0 else "0%",
            help="Products with 100% fulfillment"
        )
    
    with col7:
        avg_gap_percentage = (
            gap_df[gap_df['total_demand_qty'] > 0]['gap_quantity'] / 
            gap_df[gap_df['total_demand_qty'] > 0]['total_demand_qty'] * 100
        ).mean()
        
        st.metric(
            "Avg GAP %",
            f"{avg_gap_percentage:.1f}%",
            delta="Surplus" if avg_gap_percentage > 0 else "Shortage",
            delta_color="normal" if avg_gap_percentage >= 0 else "inverse",
            help="Average GAP as percentage of demand"
        )
    
    with col8:
        past_periods = gap_df[
            gap_df['period'].apply(lambda x: is_past_period(str(x), gap_df['period'].iloc[0]))
        ]['period'].nunique()
        
        if past_periods > 0:
            st.metric(
                "Past Periods",
                past_periods,
                delta="Already occurred",
                delta_color="off",
                help="Number of periods that have already passed"
            )
        else:
            st.metric(
                "Planning Horizon",
                gap_df['period'].nunique(),
                delta="Future periods",
                help="Number of future periods in analysis"
            )


def show_gap_quick_summary(gap_df):
    """Show quick summary cards for executives"""
    if gap_df.empty:
        return
    
    st.markdown("#### 🎯 Quick Summary")
    
    # Calculate key metrics
    total_demand = gap_df['total_demand_qty'].sum()
    total_supply = gap_df['supply_in_period'].sum()
    total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
    total_surplus = gap_df[gap_df['gap_quantity'] > 0]['gap_quantity'].sum()
    
    # Calculate service level
    fulfilled_demand = gap_df.apply(
        lambda row: min(row['total_demand_qty'], row['total_available']) if row['total_demand_qty'] > 0 else 0, 
        axis=1
    ).sum()
    service_level = (fulfilled_demand / total_demand * 100) if total_demand > 0 else 100
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Demand",
            format_number(total_demand),
            help="Total demand quantity across all products and periods"
        )
    
    with col2:
        st.metric(
            "Total Supply",
            format_number(total_supply),
            delta=format_number(total_supply - total_demand),
            delta_color="normal" if total_supply >= total_demand else "inverse",
            help="Total supply quantity across all products and periods"
        )
    
    with col3:
        st.metric(
            "Service Level",
            f"{service_level:.1f}%",
            delta=f"{service_level - 100:.1f}%" if service_level < 100 else "Full coverage",
            delta_color="normal" if service_level >= 95 else "inverse",
            help="Percentage of demand that can be fulfilled"
        )
    
    with col4:
        if total_shortage > total_surplus:
            st.metric(
                "Net Position",
                format_number(total_shortage),
                delta="Shortage",
                delta_color="inverse",
                help="Overall shortage after considering all surplus"
            )
        else:
            st.metric(
                "Net Position",
                format_number(total_surplus - total_shortage),
                delta="Surplus",
                delta_color="normal",
                help="Overall surplus after covering all shortages"
            )
    
    # Additional row for critical metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        critical_shortage = gap_df[
            (gap_df['gap_quantity'] < 0) & 
            (gap_df['fulfillment_rate_percent'] < 50)
        ]['pt_code'].nunique()
        
        st.metric(
            "Critical Shortages",
            critical_shortage,
            delta="Products <50% fulfilled",
            delta_color="inverse" if critical_shortage > 0 else "off",
            help="Products with less than 50% fulfillment rate"
        )
    
    with col6:
        perfect_fulfillment = gap_df[
            gap_df['fulfillment_rate_percent'] >= 100
        ]['pt_code'].nunique()
        total_products = gap_df['pt_code'].nunique()
        
        st.metric(
            "Perfect Fulfillment",
            perfect_fulfillment,
            delta=f"{perfect_fulfillment/total_products*100:.1f}%" if total_products > 0 else "0%",
            help="Products with 100% fulfillment"
        )
    
    with col7:
        avg_gap_percentage = (
            gap_df[gap_df['total_demand_qty'] > 0]['gap_quantity'] / 
            gap_df[gap_df['total_demand_qty'] > 0]['total_demand_qty'] * 100
        ).mean()
        
        st.metric(
            "Avg GAP %",
            f"{avg_gap_percentage:.1f}%",
            delta="Surplus" if avg_gap_percentage > 0 else "Shortage",
            delta_color="normal" if avg_gap_percentage >= 0 else "inverse",
            help="Average GAP as percentage of demand"
        )
    
    with col8:
        past_periods = gap_df[
            gap_df['period'].apply(lambda x: is_past_period(str(x), gap_df['period'].iloc[0]))
        ]['period'].nunique()
        
        if past_periods > 0:
            st.metric(
                "Past Periods",
                past_periods,
                delta="Already occurred",
                delta_color="off",
                help="Number of periods that have already passed"
            )
        else:
            st.metric(
                "Planning Horizon",
                gap_df['period'].nunique(),
                delta="Future periods",
                help="Number of future periods in analysis"
            )

def show_gap_quick_summary(gap_df):
    """Show quick summary cards for executives"""
    if gap_df.empty:
        return
    
    st.markdown("#### 🎯 Quick Summary")
    
    # Calculate key metrics
    total_demand = gap_df['total_demand_qty'].sum()
    total_supply = gap_df['supply_in_period'].sum()
    total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
    total_surplus = gap_df[gap_df['gap_quantity'] > 0]['gap_quantity'].sum()
    
    # Calculate service level
    fulfilled_demand = gap_df.apply(
        lambda row: min(row['total_demand_qty'], row['total_available']) if row['total_demand_qty'] > 0 else 0, 
        axis=1
    ).sum()
    service_level = (fulfilled_demand / total_demand * 100) if total_demand > 0 else 100
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Demand",
            format_number(total_demand),
            help="Total demand quantity across all products and periods"
        )
    
    with col2:
        st.metric(
            "Total Supply",
            format_number(total_supply),
            delta=format_number(total_supply - total_demand),
            delta_color="normal" if total_supply >= total_demand else "inverse",
            help="Total supply quantity across all products and periods"
        )
    
    with col3:
        st.metric(
            "Service Level",
            f"{service_level:.1f}%",
            delta=f"{service_level - 100:.1f}%" if service_level < 100 else "Full coverage",
            delta_color="normal" if service_level >= 95 else "inverse",
            help="Percentage of demand that can be fulfilled"
        )
    
    with col4:
        if total_shortage > total_surplus:
            st.metric(
                "Net Position",
                format_number(total_shortage),
                delta="Shortage",
                delta_color="inverse",
                help="Overall shortage after considering all surplus"
            )
        else:
            st.metric(
                "Net Position",
                format_number(total_surplus - total_shortage),
                delta="Surplus",
                delta_color="normal",
                help="Overall surplus after covering all shortages"
            )
    
    # Additional row for critical metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        critical_shortage = gap_df[
            (gap_df['gap_quantity'] < 0) & 
            (gap_df['fulfillment_rate_percent'] < 50)
        ]['pt_code'].nunique()
        
        st.metric(
            "Critical Shortages",
            critical_shortage,
            delta="Products <50% fulfilled",
            delta_color="inverse" if critical_shortage > 0 else "off",
            help="Products with less than 50% fulfillment rate"
        )
    
    with col6:
        perfect_fulfillment = gap_df[
            gap_df['fulfillment_rate_percent'] >= 100
        ]['pt_code'].nunique()
        total_products = gap_df['pt_code'].nunique()
        
        st.metric(
            "Perfect Fulfillment",
            perfect_fulfillment,
            delta=f"{perfect_fulfillment/total_products*100:.1f}%" if total_products > 0 else "0%",
            help="Products with 100% fulfillment"
        )
    
    with col7:
        avg_gap_percentage = (
            gap_df[gap_df['total_demand_qty'] > 0]['gap_quantity'] / 
            gap_df[gap_df['total_demand_qty'] > 0]['total_demand_qty'] * 100
        ).mean()
        
        st.metric(
            "Avg GAP %",
            f"{avg_gap_percentage:.1f}%",
            delta="Surplus" if avg_gap_percentage > 0 else "Shortage",
            delta_color="normal" if avg_gap_percentage >= 0 else "inverse",
            help="Average GAP as percentage of demand"
        )
    
    with col8:
        past_periods = gap_df[
            gap_df['period'].apply(lambda x: is_past_period(str(x), gap_df['period'].iloc[0]))
        ]['period'].nunique()
        
        if past_periods > 0:
            st.metric(
                "Past Periods",
                past_periods,
                delta="Already occurred",
                delta_color="off",
                help="Number of periods that have already passed"
            )
        else:
            st.metric(
                "Planning Horizon",
                gap_df['period'].nunique(),
                delta="Future periods",
                help="Number of future periods in analysis"
            )


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


def show_product_analysis_metrics(df_demand, df_supply):
    """Show detailed product analysis metrics - SAFE VERSION"""
    # Initialize empty sets for safety
    demand_products = set()
    supply_products = set()
    
    # Get unique products from demand
    if df_demand is not None and not df_demand.empty and 'pt_code' in df_demand.columns:
        demand_products = set(df_demand['pt_code'].dropna().unique())
    
    # Get unique products from supply
    if df_supply is not None and not df_supply.empty and 'pt_code' in df_supply.columns:
        supply_products = set(df_supply['pt_code'].dropna().unique())
    
    # Calculate intersections
    matched_products = demand_products.intersection(supply_products)
    demand_only_products = demand_products - supply_products
    supply_only_products = supply_products - demand_products
    
    # Calculate values
    demand_only_value = 0
    supply_only_value = 0
    
    if df_demand is not None and not df_demand.empty and 'pt_code' in df_demand.columns and 'value_in_usd' in df_demand.columns:
        if demand_only_products:
            demand_only_value = df_demand[
                df_demand['pt_code'].isin(demand_only_products)
            ]['value_in_usd'].sum()
    
    if df_supply is not None and not df_supply.empty and 'pt_code' in df_supply.columns and 'value_in_usd' in df_supply.columns:
        if supply_only_products:
            supply_only_value = df_supply[
                df_supply['pt_code'].isin(supply_only_products)
            ]['value_in_usd'].sum()
    
    # Display metrics
    st.markdown("#### 📈 Product Analysis")
    
    # First row - Product counts
    metrics1 = [
        {
            "title": "📤 Demand Products",
            "value": len(demand_products),
            "format_type": "number",
            "help_text": "Total unique products in demand"
        },
        {
            "title": "📥 Supply Products",
            "value": len(supply_products),
            "format_type": "number",
            "help_text": "Total unique products in supply"
        },
        {
            "title": "🔗 Matched Products",
            "value": len(matched_products),
            "format_type": "number",
            "delta": f"{len(matched_products)/len(demand_products)*100:.1f}%" if len(demand_products) > 0 else "0%",
            "help_text": "Products with both demand and supply"
        },
        {
            "title": "📊 Total Unique Products",
            "value": len(demand_products.union(supply_products)),
            "format_type": "number",
            "help_text": "All unique products across demand and supply"
        }
    ]
    
    DisplayComponents.show_summary_metrics(metrics1)
    
    # Second row - Unmatched products
    metrics2 = [
        {
            "title": "🚨 Demand Only",
            "value": len(demand_only_products),
            "format_type": "number",
            "delta": f"-{format_currency(demand_only_value, 'USD', 0)}",
            "delta_color": "inverse",
            "help_text": "Products with demand but no supply"
        },
        {
            "title": "📦 Supply Only",
            "value": len(supply_only_products),
            "format_type": "number",
            "delta": format_currency(supply_only_value, 'USD', 0),
            "help_text": "Products with supply but no demand"
        }
    ]
    
    # Add shortage metrics if gap_df is available
    gap_df = st.session_state.get('gap_df_cached', pd.DataFrame())
    if not gap_df.empty:
        shortage_products = 0
        avg_fulfillment = 100
        
        if 'pt_code' in gap_df.columns and 'gap_quantity' in gap_df.columns:
            shortage_products = len(gap_df[gap_df["gap_quantity"] < 0]["pt_code"].unique())
        
        if 'total_demand_qty' in gap_df.columns and 'fulfillment_rate_percent' in gap_df.columns:
            demand_rows = gap_df[gap_df["total_demand_qty"] > 0]
            if not demand_rows.empty:
                avg_fulfillment = demand_rows["fulfillment_rate_percent"].clip(upper=100).mean()
        
        metrics2.extend([
            {
                "title": "❌ Products with Shortage",
                "value": shortage_products,
                "format_type": "number",
                "delta": f"{shortage_products/len(matched_products)*100:.1f}%" if len(matched_products) > 0 else "0%"
            },
            {
                "title": "📊 Avg Fulfillment Rate",
                "value": avg_fulfillment if pd.notna(avg_fulfillment) else 0,
                "format_type": "percentage"
            }
        ])
    
    DisplayComponents.show_summary_metrics(metrics2)


def show_basic_gap_metrics(gap_df):
    """Show basic GAP metrics when detailed data not available"""
    total_products = gap_df["pt_code"].nunique()
    shortage_products = len(gap_df[gap_df["gap_quantity"] < 0]["pt_code"].unique())
    total_shortage = gap_df["gap_quantity"].where(gap_df["gap_quantity"] < 0, 0).abs().sum()
    
    fulfillment_rates = gap_df[gap_df["total_demand_qty"] > 0]["fulfillment_rate_percent"].copy()
    fulfillment_rates = fulfillment_rates.clip(upper=100)
    avg_fulfillment = fulfillment_rates.mean() if len(fulfillment_rates) > 0 else 0
    
    metrics = [
        {"title": "Total Products", "value": total_products, "format_type": "number"},
        {
            "title": "Products with Shortage",
            "value": shortage_products,
            "format_type": "number",
            "delta": f"{shortage_products/total_products*100:.1f}%" if total_products > 0 else "0%"
        },
        {"title": "Total Shortage Qty", "value": total_shortage, "format_type": "number"},
        {"title": "Avg Fulfillment Rate", "value": avg_fulfillment, "format_type": "percentage"}
    ]
    
    DisplayComponents.show_summary_metrics(metrics)

def show_filtering_info(gap_df, display_options):
    """Show information about filtered view"""
    if display_options["show_shortage_only"] or display_options["exclude_zero_demand"]:
        filtered_df = gap_df.copy()
        
        if display_options["show_shortage_only"]:
            filtered_df = filtered_df[filtered_df["gap_quantity"] < 0]
        if display_options["exclude_zero_demand"]:
            filtered_df = filtered_df[filtered_df["total_demand_qty"] > 0]
        
        st.info(f"🔍 Showing {len(filtered_df)} items out of {len(gap_df)} total items (filters applied)")


def show_date_status_comparison(df_demand, df_supply, use_adjusted_demand, use_adjusted_supply):
    """Show comparison of date adjustments impact - SAFE VERSION"""
    st.markdown("#### 📈 Date Adjustment Impact")
    
    col1, col2 = st.columns(2)
    
    # Demand date comparison
    with col1:
        st.markdown("**📤 Demand Date Analysis**")
        if not df_demand.empty and 'etd_original' in df_demand.columns and 'etd_adjusted' in df_demand.columns:
            today = pd.Timestamp.now().normalize()
            
            original_missing = df_demand['etd_original'].isna().sum()
            adjusted_missing = df_demand['etd_adjusted'].isna().sum()
            
            original_past = (pd.to_datetime(df_demand['etd_original'], errors='coerce') < today).sum()
            adjusted_past = (pd.to_datetime(df_demand['etd_adjusted'], errors='coerce') < today).sum()
            
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.caption("Original ETD")
                st.metric("Missing", original_missing, label_visibility="collapsed")
                st.metric("Past", original_past, label_visibility="collapsed")
            
            with col1_2:
                st.caption("Adjusted ETD")
                st.metric(
                    "Missing", 
                    adjusted_missing,
                    delta=adjusted_missing - original_missing if adjusted_missing != original_missing else None,
                    label_visibility="collapsed"
                )
                st.metric(
                    "Past", 
                    adjusted_past,
                    delta=adjusted_past - original_past if adjusted_past != original_past else None,
                    label_visibility="collapsed"
                )
        else:
            st.info("No date adjustments configured for demand")
    
    # Supply date comparison - FIXED TO HANDLE EMPTY SUPPLY
    with col2:
        st.markdown("**📥 Supply Date Analysis**")
        
        # Check if supply data exists and not empty
        if df_supply.empty:
            st.warning("No supply data available for comparison")
        elif 'source_type' not in df_supply.columns:
            st.warning("Supply data missing source_type information")
        else:
            # Aggregate across source types
            supply_stats = []
            
            for source_type in df_supply['source_type'].unique():
                source_df = df_supply[df_supply['source_type'] == source_type]
                
                # Map to date columns
                date_map = {
                    'Inventory': 'date_ref',
                    'Pending CAN': 'arrival_date',
                    'Pending PO': 'eta',
                    'Pending WH Transfer': 'transfer_date'
                }
                
                base_col = date_map.get(source_type, 'date_ref')
                original_col = f'{base_col}_original'
                adjusted_col = f'{base_col}_adjusted'
                
                if original_col in source_df.columns and adjusted_col in source_df.columns:
                    today = pd.Timestamp.now().normalize()
                    
                    original_missing = source_df[original_col].isna().sum()
                    adjusted_missing = source_df[adjusted_col].isna().sum()
                    
                    original_past = (pd.to_datetime(source_df[original_col], errors='coerce') < today).sum()
                    adjusted_past = (pd.to_datetime(source_df[adjusted_col], errors='coerce') < today).sum()
                    
                    supply_stats.append({
                        'source': source_type,
                        'missing_delta': adjusted_missing - original_missing,
                        'past_delta': adjusted_past - original_past
                    })
            
            if supply_stats:
                total_missing_delta = sum(s['missing_delta'] for s in supply_stats)
                total_past_delta = sum(s['past_delta'] for s in supply_stats)
                
                st.metric("Missing Date Changes", total_missing_delta)
                st.metric("Past Date Changes", total_past_delta)
                
                with st.expander("Details by Source"):
                    for stat in supply_stats:
                        st.caption(f"**{stat['source']}**: Missing {stat['missing_delta']:+d}, Past {stat['past_delta']:+d}")
            else:
                st.info("No date adjustments configured for supply")


def show_allocation_impact_summary(df_demand_enhanced):
    """Show how allocation affects the GAP analysis"""
    if 'allocation_status' in df_demand_enhanced.columns:
        st.markdown("#### 📦 Allocation Impact on Demand")
        
        total_demand_original = df_demand_enhanced['demand_quantity'].sum()
        total_allocated = df_demand_enhanced['total_allocated'].sum()
        total_delivered = df_demand_enhanced['total_delivered'].sum()
        total_unallocated = df_demand_enhanced['unallocated_demand'].sum()
        
        allocation_metrics = [
            {"title": "Original Demand", "value": total_demand_original, "format_type": "number"},
            {"title": "Already Allocated", "value": total_allocated, "format_type": "number"},
            {"title": "Already Delivered", "value": total_delivered, "format_type": "number"},
            {"title": "Net Unallocated", "value": total_unallocated, "format_type": "number"}
        ]
        
        DisplayComponents.show_summary_metrics(allocation_metrics)



def show_gap_detail_table(gap_df, display_options, df_demand_filtered=None, df_supply_filtered=None):
    """Show detailed GAP analysis table with enhanced filtering"""
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
    
    # Apply display filters
    display_df = gap_df.copy()
    
    # Apply existing filters
    if display_options["show_shortage_only"]:
        display_df = display_df[display_df["gap_quantity"] < 0]
    
    if display_options["exclude_zero_demand"]:
        display_df = display_df[display_df["total_demand_qty"] > 0]
    
    # Apply product type filters (with safe checks)
    demand_products = set()
    supply_products = set()
    
    # Safely extract product sets
    if df_demand_filtered is not None and not df_demand_filtered.empty and 'pt_code' in df_demand_filtered.columns:
        demand_products = set(df_demand_filtered['pt_code'].unique())
    
    if df_supply_filtered is not None and not df_supply_filtered.empty and 'pt_code' in df_supply_filtered.columns:
        supply_products = set(df_supply_filtered['pt_code'].unique())
    
    # Apply product type filters only if we have data
    if demand_products or supply_products:
        matched_products = demand_products.intersection(supply_products)
        demand_only_products = demand_products - supply_products
        supply_only_products = supply_products - demand_products
        
        products_to_show = set()
        
        if display_options.get("show_matched", True):
            products_to_show.update(matched_products)
        
        if display_options.get("show_demand_only", False):
            products_to_show.update(demand_only_products)
            
        if display_options.get("show_supply_only", False):
            products_to_show.update(supply_only_products)
        
        if products_to_show:
            display_df = display_df[display_df["pt_code"].isin(products_to_show)]
    
    if display_df.empty:
        st.info("No data to display with current filter settings.")
        return
    
    # Format display
    display_df_formatted = display_df.copy()
    
    # Format numeric columns
    numeric_cols = [
        "begin_inventory", "supply_in_period", "total_available", 
        "total_demand_qty", "gap_quantity"
    ]
    
    for col in numeric_cols:
        if col in display_df_formatted.columns:
            display_df_formatted[col] = display_df_formatted[col].apply(lambda x: format_number(x))
    
    if "fulfillment_rate_percent" in display_df_formatted.columns:
        display_df_formatted["fulfillment_rate_percent"] = display_df_formatted["fulfillment_rate_percent"].apply(
            lambda x: format_percentage(x)
        )
    
    # Add past period indicator
    display_df_formatted['Period Status'] = display_df_formatted['period'].apply(
        lambda x: "🔴 Past" if is_past_period(str(x), display_options.get("period_type", "Weekly")) else "✅ Future"
    )
    
    # Add product type indicator (only if we have the data)
    if demand_products or supply_products:
        def get_product_type(pt_code):
            if pt_code in demand_products and pt_code in supply_products:
                return "🔗 Matched"
            elif pt_code in demand_products:
                return "📤 Demand Only"
            elif pt_code in supply_products:
                return "📥 Supply Only"
            return "❓ Unknown"
        
        display_df_formatted['Product Type'] = display_df_formatted['pt_code'].apply(get_product_type)
    
    # Select columns to display
    display_columns = [
        "pt_code", "product_name", "package_size", "standard_uom", "period",
        "Period Status", "begin_inventory", "supply_in_period", "total_available", 
        "total_demand_qty", "gap_quantity", "fulfillment_rate_percent", 
        "fulfillment_status"
    ]
    
    # Add Product Type column if it exists
    if 'Product Type' in display_df_formatted.columns:
        display_columns.append('Product Type')
    
    # Only include columns that exist
    display_columns = [col for col in display_columns if col in display_df_formatted.columns]
    
    # Apply row highlighting
    styled_df = display_df_formatted[display_columns].style.apply(
        highlight_gap_rows, axis=1
    )
    
    # Display
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        height=400
    )



def highlight_gap_rows(row):
    """Highlight rows based on gap status and period"""
    styles = [""] * len(row)
    
    try:
        # Check period status first
        if 'Period Status' in row.index and "🔴" in str(row['Period Status']):
            # Past period - light gray background
            base_style = "background-color: #f0f0f0"
        else:
            base_style = ""
        
        # Check fulfillment status
        if 'fulfillment_status' in row.index:
            if "❌" in str(row['fulfillment_status']):
                # Shortage - red background
                styles = ["background-color: #ffcccc"] * len(row)
            elif base_style:
                styles = [base_style] * len(row)
        elif base_style:
            styles = [base_style] * len(row)
        
    except Exception as e:
        logger.error(f"Error highlighting rows: {str(e)}")
    
    return styles

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
    
    # Create pivot
    pivot_gap = display_df.pivot_table(
        index=["product_name", "pt_code"], 
        columns="period",
        values="gap_quantity",
        aggfunc="sum",
        fill_value=0
    ).reset_index()
    
    # Sort columns
    pivot_gap = sort_period_columns(pivot_gap, display_options["period_type"], ["product_name", "pt_code"])
    
    # Create display version with past period indicators
    display_pivot = pivot_gap.copy()
    
    # Rename columns with indicators for past periods
    renamed_columns = {}
    for col in pivot_gap.columns:
        if col not in ["product_name", "pt_code"]:
            if is_past_period(str(col), display_options["period_type"]):
                renamed_columns[col] = f"🔴 {col}"
    
    if renamed_columns:
        display_pivot = display_pivot.rename(columns=renamed_columns)
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
        # For heatmap, use numeric values
        heatmap_pivot = pivot_gap.copy()
        if renamed_columns:
            heatmap_pivot = heatmap_pivot.rename(columns=renamed_columns)
        
        styled_df = heatmap_pivot.style.background_gradient(
            cmap='RdYlGn', 
            subset=heatmap_pivot.columns[2:], 
            axis=1
        ).format("{:,.0f}", subset=heatmap_pivot.columns[2:])
        st.dataframe(styled_df, use_container_width=True)
    
    else:
        # No styling - just format numbers
        for col in display_pivot.columns[2:]:
            display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
        st.dataframe(display_pivot, use_container_width=True)
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
   st.header("🎯 Next Actions")
   
   shortage_exists = not gap_df[gap_df['gap_quantity'] < 0].empty
   surplus_exists = not gap_df[gap_df['gap_quantity'] > 0].empty
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       if shortage_exists:
           st.markdown("### 🚨 Shortage Detected")
           shortage_count = len(gap_df[gap_df['gap_quantity'] < 0]['pt_code'].unique())
           st.info(f"Found {shortage_count} products with shortage")
           
           if st.button("🧩 Create Allocation Plan", type="primary", use_container_width=True):
               st.switch_page("pages/4_🧩_Allocation_Plan.py")
       else:
           st.success("✅ No shortage detected!")
   
   with col2:
       if shortage_exists:
           st.markdown("### 📦 Replenishment Needed")
           total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
           st.info(f"Total shortage: {format_number(total_shortage)} units")
           
           if st.button("📌 Generate PO Suggestions", type="secondary", use_container_width=True):
               st.switch_page("pages/5_📌_PO_Suggestions.py")
   
   with col3:
       if surplus_exists:
           st.markdown("### 📈 Surplus Available")
           surplus_count = len(gap_df[gap_df['gap_quantity'] > 0]['pt_code'].unique())
           st.info(f"Found {surplus_count} products with surplus")
           
           if st.button("🔄 Reallocation Options", use_container_width=True):
               save_to_session_state('show_reallocation', True)
               st.switch_page("pages/5_📌_PO_Suggestions.py")

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
        
        # Export section
        st.markdown("---")
        show_export_section(gap_df)
        
        # Action buttons
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