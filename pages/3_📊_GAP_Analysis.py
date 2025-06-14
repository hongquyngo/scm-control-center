# pages/3_📊_GAP_Analysis.py

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import AuthManager

# Authentication check
auth_manager = AuthManager()
if not auth_manager.check_session():
    st.switch_page("pages/0_🔐_Login.py")
    st.stop()

import pandas as pd
from datetime import datetime
import logging

# Import refactored modules
from utils.data_manager import DataManager
from utils.display_components import DisplayComponents
from utils.formatters import (
    format_number, format_currency, format_percentage,
    check_missing_dates
)
from utils.helpers import (
    export_multiple_sheets,
    convert_to_period,
    save_to_session_state,
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
from utils.period_processor import PeriodBasedGAPProcessor

from typing import Tuple, Dict, Any, Optional, List

def ensure_numeric_columns(df, columns):
    """Ensure columns are numeric type to prevent Arrow errors"""
    for col in columns:
        if col in df.columns:
            # Convert to numeric, replacing errors with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df



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



# === Data Loading Functions ===



def load_and_prepare_demand_data(selected_demand_sources, include_converted):
    """Load and standardize demand data based on source selection"""
    try:
        # Use data_manager to get demand data
        df = data_manager.get_demand_data(sources=selected_demand_sources, include_converted=include_converted)

        # ADD THIS - Fix numeric columns
        numeric_cols = ['demand_quantity', 'buying_quantity', 'value_in_usd', 
                       'total_allocated', 'total_delivered', 'undelivered_allocated']
        df = ensure_numeric_columns(df, numeric_cols)
        
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

        # ADD THIS - Fix numeric columns
        numeric_cols = ['quantity', 'buying_quantity', 'value_in_usd']
        df = ensure_numeric_columns(df, numeric_cols)
        
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
# Trong 3_📊_GAP_Analysis.py
def calculate_gap_with_carry_forward(df_demand, df_supply, period_type="Weekly", 
                                    use_adjusted_demand=True, use_adjusted_supply=True,
                                    track_backlog=True):
    """
    Enhanced version with proper backlog tracking
    
    Args:
        track_backlog: If True, tracks negative carry forward (unfulfilled demand)
                      If False, uses original logic (only positive carry forward)
    """
    
    # Early return if both empty
    if df_demand.empty and df_supply.empty:
        st.warning("No data available for GAP calculation")
        return pd.DataFrame()
    
    # Get allocations data
    allocations_df = data_manager.load_active_allocations()
    
    # Initialize processor
    processor = PeriodBasedGAPProcessor(period_type)
    
    # Process all data by period
    with st.spinner("Processing data by period..."):
        period_data = processor.process_for_gap(
            df_demand, 
            df_supply,
            allocations_df,
            use_adjusted_demand,
            use_adjusted_supply
        )
    
    if period_data.empty:
        st.warning("No valid period data for GAP calculation")
        return pd.DataFrame()
    
    # Apply carry forward logic with proper backlog tracking
    results = []
    products = period_data['pt_code'].unique()
    
    for product in products:
        product_data = period_data[period_data['pt_code'] == product].copy()
        
        # Sort by period
        if period_type == "Weekly":
            product_data['sort_key'] = product_data['period'].apply(parse_week_period)
        elif period_type == "Monthly":
            product_data['sort_key'] = product_data['period'].apply(parse_month_period)
        else:
            product_data['sort_key'] = pd.to_datetime(product_data['period'])
        
        product_data = product_data.sort_values('sort_key')
        
        # Initialize tracking variables
        carry_forward = 0  # Positive inventory carried forward
        backlog = 0        # Negative balance (unfulfilled demand)
        previous_gap = 0   # Track previous period's gap
        
        for idx, (_, row) in enumerate(product_data.iterrows()):
            # Store the begin inventory (carry forward from previous period)
            begin_inventory = carry_forward
            
            if track_backlog:
                # ENHANCED LOGIC: Track both positive and negative balances
                
                # Calculate effective demand (current demand + backlog)
                effective_demand = row['unallocated_demand'] + backlog
                
                # Total available = current supply + carried forward inventory
                total_available = row['available_supply'] + carry_forward
                
                # Calculate gap
                gap = total_available - effective_demand
                
                # Update tracking variables for NEXT period
                if gap >= 0:
                    # Surplus: clear backlog, set carry forward
                    carry_forward = gap
                    backlog = 0
                else:
                    # Shortage: clear carry forward, accumulate backlog
                    carry_forward = 0
                    backlog = abs(gap)
                
                # Calculate fulfillment rate based on effective demand
                if effective_demand > 0:
                    fulfillment_rate = min(100, (total_available / effective_demand * 100))
                else:
                    fulfillment_rate = 100 if total_available > 0 else 0
                
                # For display purposes
                display_demand = effective_demand
                current_period_backlog = backlog  # Backlog that will carry to NEXT period
                
            else:
                # ORIGINAL LOGIC: Only positive carry forward
                total_available = row['available_supply'] + carry_forward
                gap = total_available - row['unallocated_demand']
                
                if row['unallocated_demand'] > 0:
                    fulfillment_rate = min(100, (total_available / row['unallocated_demand'] * 100))
                else:
                    fulfillment_rate = 100 if total_available > 0 else 0
                
                # Update carry forward (original logic)
                carry_forward = max(0, gap)
                
                # For display purposes
                display_demand = row['unallocated_demand']
                current_period_backlog = 0
            
            # Determine status
            if gap >= 0:
                status = "✅ Fulfilled"
            else:
                status = "❌ Shortage"
            
            # Build result row
            result_row = {
                'pt_code': row['pt_code'],
                'product_name': row.get('product_name', ''),
                'package_size': row.get('package_size', ''),
                'standard_uom': row.get('standard_uom', ''),
                'period': row['period'],
                'begin_inventory': begin_inventory,  # Use the stored value
                'supply_in_period': row['supply_quantity'],
                'total_available': total_available,
                'original_demand_qty': row['demand_quantity'],
                'total_demand_qty': display_demand,
                'gap_quantity': gap,
                'fulfillment_rate_percent': fulfillment_rate,
                'fulfillment_status': status
            }
            
            # Add backlog info if tracking
            if track_backlog:
                # Show backlog FROM PREVIOUS period that was added to current demand
                result_row['backlog_qty'] = backlog if idx > 0 else 0
                result_row['effective_demand'] = effective_demand
                # Track backlog that will go to NEXT period
                result_row['backlog_to_next'] = current_period_backlog
            
            results.append(result_row)
    
    gap_df = pd.DataFrame(results)
    
    if debug_mode and not gap_df.empty:
        st.write(f"🐛 GAP calculation complete:")
        st.write(f"- Total rows: {len(gap_df)}")
        st.write(f"- Unique products: {gap_df['pt_code'].nunique()}")
        st.write(f"- Shortage rows: {len(gap_df[gap_df['gap_quantity'] < 0])}")
        if track_backlog:
            st.write(f"- Rows with backlog: {len(gap_df[gap_df.get('backlog_qty', 0) > 0])}")
    
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
# Replace the entire get_gap_display_options function with these two functions:

def get_gap_calculation_options():
    """Get options that affect GAP calculation logic - must be set before running"""
    st.markdown("### ⚙️ Calculation Options")
    st.caption("These options affect how GAP is calculated. Changes require re-running the analysis.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize defaults
    if 'gap_period_type' not in st.session_state:
        st.session_state['gap_period_type'] = "Weekly"
    
    with col1:
        period_type = st.selectbox(
            "Group By Period", 
            ["Daily", "Weekly", "Monthly"], 
            index=["Daily", "Weekly", "Monthly"].index(st.session_state.get('gap_period_type', "Weekly")),
            key="gap_period_select"
        )
        st.session_state['gap_period_type'] = period_type
    
    with col2:
        exclude_missing_dates = st.checkbox(
            "📅 Exclude missing dates",
            value=True,
            key="gap_exclude_missing_dates",
            help="Exclude records with missing ETD or reference dates from calculation"
        )
    
    with col3:
        include_draft_allocations = st.checkbox(
            "📝 Include DRAFT allocations",
            value=False,
            key="gap_include_draft",
            help="Include uncommitted allocations in calculation (may show conservative view)"
        )
    
    with col4:
        track_backlog = st.checkbox(
            "📊 Track Backlog",
            value=True,
            key="gap_track_backlog",
            help="Track negative carry forward (backlog) from shortage periods"
        )
    
    return {
        "period_type": period_type,
        "exclude_missing_dates": exclude_missing_dates,
        "include_draft_allocations": include_draft_allocations,
        "track_backlog": track_backlog
    }

def get_gap_display_filters():
    """Get unified display filters that can be changed dynamically after GAP calculation"""
    if not st.session_state.get('gap_analysis_ran', False):
        return None
        
    st.markdown("### 🔍 Display Filters")
    st.caption("Filter the calculated results. Changes apply immediately.")
    
    # Product Type Filters
    st.markdown("#### Product Types")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_matched = st.checkbox(
            "🔗 Matched Products",
            value=False,
            key="gap_show_matched",
            help="Products with both demand and supply"
        )
    
    with col2:
        show_demand_only = st.checkbox(
            "📤 Demand Only",
            value=False,
            key="gap_show_demand_only",
            help="Products with demand but no supply"
        )
    
    with col3:
        show_supply_only = st.checkbox(
            "📥 Supply Only",
            value=False,
            key="gap_show_supply_only",
            help="Products with supply but no demand"
        )
    
    # Period/Status Filters
    st.markdown("#### Period & Status Filters")
    period_filter = st.radio(
        "Show:",
        options=[
            "All",
            "Shortage Only",
            "Past Periods Only",
            "Future Periods Only",
            "Critical Shortage Only"
        ],
        horizontal=True,
        key="gap_period_filter",
        help="Filter by period status or shortage condition"
    )
    
    # View Options
    st.markdown("#### View Options")
    col4, col5 = st.columns(2)
    
    with col4:
        show_data_quality = st.checkbox(
            "📊 Show Data Quality",
            value=True,
            key="gap_show_data_quality",
            help="Show data quality warnings"
        )
    
    with col5:
        enable_row_highlighting = st.checkbox(
            "🎨 Enable Row Highlighting",
            value=False,
            key="gap_row_highlighting",
            help="Highlight rows based on status (may affect performance for large datasets)"
        )
    
    return {
        # Product type filters
        "show_matched": show_matched,
        "show_demand_only": show_demand_only,
        "show_supply_only": show_supply_only,
        # Period filter
        "period_filter": period_filter,
        # View options
        "show_data_quality": show_data_quality,
        "enable_row_highlighting": enable_row_highlighting
    }

# Add these helper functions after the display options functions:

def filter_gap_by_product_type(gap_df: pd.DataFrame, display_filters: Dict, 
                               df_demand=None, df_supply=None) -> pd.DataFrame:
    """Filter GAP dataframe by product type options"""
    if gap_df.empty:
        return gap_df
    
    # Get product sets
    demand_products = set()
    supply_products = set()
    
    if df_demand is not None and not df_demand.empty and 'pt_code' in df_demand.columns:
        demand_products = set(df_demand['pt_code'].unique())
    
    if df_supply is not None and not df_supply.empty and 'pt_code' in df_supply.columns:
        supply_products = set(df_supply['pt_code'].unique())
    
    # Determine which products to show
    products_to_show = set()
    
    if display_filters.get('show_matched', True):
        matched = demand_products & supply_products
        products_to_show.update(matched)
    
    if display_filters.get('show_demand_only', False):
        demand_only = demand_products - supply_products
        products_to_show.update(demand_only)
    
    if display_filters.get('show_supply_only', False):
        supply_only = supply_products - demand_products
        products_to_show.update(supply_only)
    
    # Filter
    if products_to_show:
        return gap_df[gap_df['pt_code'].isin(products_to_show)]
    else:
        # No product types selected
        return pd.DataFrame()

def apply_display_filters(gap_df: pd.DataFrame, display_filters: Dict, 
                         df_demand=None, df_supply=None,
                         period_type: str = "Weekly") -> pd.DataFrame:
    """Apply all post-calculation display filters"""
    if gap_df.empty or not display_filters:
        return gap_df
    
    filtered_df = gap_df.copy()
    
    # 1. Apply product type filters
    filtered_df = filter_gap_by_product_type(
        filtered_df, 
        display_filters,
        df_demand,
        df_supply
    )
    
    # 2. Apply period/status filter
    period_filter = display_filters.get("period_filter", "All")
    
    if period_filter == "Shortage Only":
        filtered_df = filtered_df[filtered_df["gap_quantity"] < 0]
    
    elif period_filter == "Past Periods Only":
        filtered_df = filtered_df[
            filtered_df['period'].apply(lambda x: is_past_period(str(x), period_type))
        ]
    
    elif period_filter == "Future Periods Only":
        filtered_df = filtered_df[
            ~filtered_df['period'].apply(lambda x: is_past_period(str(x), period_type))
        ]
    
    elif period_filter == "Critical Shortage Only":
        filtered_df = filtered_df[
            (filtered_df["gap_quantity"] < 0) & 
            (filtered_df["fulfillment_rate_percent"] < 50)
        ]
    
    return filtered_df

# === Display Functions ===

def show_gap_summary(gap_df, display_options, df_demand_filtered=None, df_supply_filtered=None,
                    use_adjusted_demand=True, use_adjusted_supply=True):
    """Show GAP analysis summary with backlog tracking support"""
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
    
    # Calculate backlog metrics if tracking
    track_backlog = display_options.get('track_backlog', True)
    if track_backlog and 'backlog_qty' in gap_df.columns:
        # Get final backlog for each product (last period's backlog)
        final_backlog_by_product = gap_df.groupby('pt_code')['backlog_qty'].last()
        total_backlog = final_backlog_by_product.sum()
        products_with_backlog = (final_backlog_by_product > 0).sum()
        
        # Get max backlog ever for each product
        max_backlog_by_product = gap_df.groupby('pt_code')['backlog_qty'].max()
        peak_total_backlog = max_backlog_by_product.sum()
        products_ever_had_backlog = (max_backlog_by_product > 0).sum()
    else:
        total_backlog = 0
        products_with_backlog = 0
        peak_total_backlog = 0
        products_ever_had_backlog = 0
    
    # Determine overall status
    if total_shortage == 0 and total_backlog == 0:
        status_color = "#28a745"  # Green
        status_bg_color = "#d4edda"
        status_icon = "✅"
        status_text = "No Shortage Detected"
        status_detail = "Supply meets demand for all products across all periods"
    elif products_with_backlog > 0:
        status_color = "#fd7e14"  # Orange
        status_bg_color = "#fff3cd"
        status_icon = "⚠️"
        status_text = "Backlog Detected"
        status_detail = f"{products_with_backlog} of {total_products} products have unfulfilled demand carried forward"
    elif shortage_products / total_products > 0.5 or periods_with_shortage / total_periods > 0.5:
        status_color = "#dc3545"  # Red
        status_bg_color = "#f8d7da"
        status_icon = "🚨"
        status_text = "Critical Shortage"
        status_detail = f"{shortage_products} of {total_products} products need immediate attention"
    else:
        status_color = "#ffc107"  # Yellow
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
    
    # Show tracking mode info
    if track_backlog:
        st.info("📊 **Backlog Tracking: ON** - Unfulfilled demand accumulates to next periods")
    else:
        st.info("📊 **Backlog Tracking: OFF** - Each period calculated independently")
    
    # Essential metrics - 4 columns if tracking backlog, 3 otherwise
    if track_backlog and 'backlog_qty' in gap_df.columns:
        col1, col2, col3, col4 = st.columns(4)
    else:
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
        if track_backlog and 'backlog_qty' in gap_df.columns:
            st.metric(
                "Current Backlog",
                format_number(total_backlog),
                delta=f"{products_with_backlog} products" if products_with_backlog > 0 else "All clear",
                delta_color="inverse" if total_backlog > 0 else "off",
                help="Total unfulfilled demand at end of analysis period"
            )
        else:
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
    
    # 4th column for peak backlog if tracking
    if track_backlog and 'backlog_qty' in gap_df.columns:
        with col4:
            st.metric(
                "Peak Backlog",
                format_number(peak_total_backlog),
                delta=f"{products_ever_had_backlog} products affected",
                delta_color="inverse" if peak_total_backlog > total_backlog else "normal",
                help="Maximum backlog reached during analysis period"
            )
    
    # === LEVEL 2: Expandable Details ===
    with st.expander("📈 View Detailed Analysis", expanded=shortage_products > 0 or products_with_backlog > 0):
        
        if shortage_products > 0 or products_with_backlog > 0:
            # Quick action summary
            st.markdown("##### 🎯 Action Required")
            
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                # Top shortage products
                st.markdown("**🔴 Products with Issues:**")
                
                # Combine shortage and backlog info
                if track_backlog and 'backlog_qty' in gap_df.columns:
                    # Get both shortage and backlog info
                    product_issues = gap_df.groupby('pt_code').agg({
                        'gap_quantity': lambda x: x[x < 0].sum() if any(x < 0) else 0,
                        'backlog_qty': 'last',
                        'period': 'count'
                    }).rename(columns={'period': 'affected_periods'})
                    
                    product_issues['total_issue'] = product_issues['gap_quantity'].abs() + product_issues['backlog_qty']
                    product_issues = product_issues[product_issues['total_issue'] > 0]
                    product_issues = product_issues.sort_values('total_issue', ascending=False).head(5)
                    
                    for pt_code, row in product_issues.iterrows():
                        issue_text = []
                        if row['gap_quantity'] < 0:
                            issue_text.append(f"Shortage: {format_number(abs(row['gap_quantity']))}")
                        if row['backlog_qty'] > 0:
                            issue_text.append(f"Backlog: {format_number(row['backlog_qty'])}")
                        st.caption(f"• **{pt_code}**: {' | '.join(issue_text)} ({row['affected_periods']} periods)")
                else:
                    # Original logic for shortage only
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
                st.markdown("**📅 Periods with Issues:**")
                
                if track_backlog and 'backlog_qty' in gap_df.columns:
                    # Show periods with either shortage or high backlog
                    period_issues = gap_df.groupby('period').agg({
                        'gap_quantity': lambda x: x[x < 0].sum() if any(x < 0) else 0,
                        'backlog_qty': 'sum',
                        'pt_code': 'nunique'
                    }).rename(columns={'pt_code': 'products_affected'})
                    
                    period_issues['has_issue'] = (period_issues['gap_quantity'] < 0) | (period_issues['backlog_qty'] > 0)
                    period_issues = period_issues[period_issues['has_issue']]
                    period_issues['total_issue'] = period_issues['gap_quantity'].abs() + period_issues['backlog_qty']
                    period_issues = period_issues.sort_values('total_issue', ascending=False).head(5)
                    
                    for period, row in period_issues.iterrows():
                        is_past = is_past_period(str(period), display_options.get('period_type', 'Weekly'))
                        indicator = "🔴" if is_past else "🟡"
                        issue_parts = []
                        if row['gap_quantity'] < 0:
                            issue_parts.append(f"Gap: {format_number(abs(row['gap_quantity']))}")
                        if row['backlog_qty'] > 0:
                            issue_parts.append(f"Backlog: {format_number(row['backlog_qty'])}")
                        st.caption(f"{indicator} **{period}**: {' | '.join(issue_parts)} ({row['products_affected']} products)")
                else:
                    # Original shortage-only logic
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
        
        # Calculate totals based on tracking mode
        if track_backlog and 'effective_demand' in gap_df.columns:
            # Use effective demand when tracking backlog
            total_demand = gap_df.groupby(['pt_code', 'period'])['effective_demand'].first().sum()
            display_demand_label = "Total Effective Demand"
        else:
            total_demand = gap_df['total_demand_qty'].sum()
            display_demand_label = "Total Demand"
        
        total_supply = gap_df['supply_in_period'].sum()
        net_position = total_supply - total_demand
        
        # Create visual balance
        balance_col1, balance_col2, balance_col3 = st.columns([2, 1, 2])
        
        with balance_col1:
            st.metric(display_demand_label, format_number(total_demand))
        
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
            st.caption(f"Supply covers {supply_rate:.1f}% of total {display_demand_label.lower()}")
        
        # Backlog progression if tracking
        if track_backlog and 'backlog_qty' in gap_df.columns and products_ever_had_backlog > 0:
            st.markdown("##### 📈 Backlog Progression")
            
            # Show backlog trend
            backlog_by_period = gap_df.groupby('period')['backlog_qty'].sum()
            
            # Create simple text chart
            st.caption("Backlog trend by period:")
            for period, backlog in backlog_by_period.items():
                if backlog > 0:
                    bar_length = int(backlog / backlog_by_period.max() * 20)
                    st.text(f"{period}: {'█' * bar_length} {format_number(backlog)}")
    
    # === LEVEL 3: Advanced Analytics (Optional) ===
    if st.checkbox("🔍 Show Advanced Analytics", key="gap_show_advanced"):
        show_advanced_gap_analytics(gap_df, display_options, df_demand_filtered, df_supply_filtered)

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
        
        styled_df = impact_df.style.map(
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
    if filter_option == "All":
        return prepare_gap_detail_display(display_df, display_options, df_demand_filtered, df_supply_filtered)
    
    # Apply different filters
    if filter_option == "Shortage Only":
        display_df = display_df[display_df["gap_quantity"] < 0]
    
    elif filter_option == "Past Periods Only":
        period_type = display_options.get("period_type", "Weekly")
        display_df = display_df[
            display_df['period'].apply(lambda x: is_past_period(str(x), period_type))
        ]
    
    elif filter_option == "Future Periods Only":
        period_type = display_options.get("period_type", "Weekly")
        display_df = display_df[
            ~display_df['period'].apply(lambda x: is_past_period(str(x), period_type))
        ]
    
    elif filter_option == "Zero Demand Only":
        display_df = display_df[display_df["total_demand_qty"] == 0]
    
    elif filter_option == "Critical Shortage Only":
        display_df = display_df[
            (display_df["gap_quantity"] < 0) & 
            (display_df["fulfillment_rate_percent"] < 50)
        ]
    
    return prepare_gap_detail_display(display_df, display_options, df_demand_filtered, df_supply_filtered)

def prepare_gap_detail_display(display_df: pd.DataFrame, display_options: Dict,
                              df_demand_filtered=None, df_supply_filtered=None) -> pd.DataFrame:
    """Prepare GAP dataframe for detail display with backlog support"""
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
    
    # Add Backlog Status column if tracking backlog
    if 'backlog_qty' in display_df.columns and display_options.get('track_backlog', True):
        def get_backlog_status(row):
            try:
                backlog = float(str(row.get('backlog_qty', 0)).replace(',', ''))
                if backlog > 0:
                    return "⚠️ Has Backlog"
                else:
                    return "✅ No Backlog"
            except:
                return "✅ No Backlog"
        
        display_df['Backlog Status'] = display_df.apply(get_backlog_status, axis=1)
    
    # Build display columns based on what's available and options
    display_columns = [
        "pt_code", 
        "product_name", 
        "package_size", 
        "standard_uom", 
        "period",
        "Period Status"
    ]
    
    # Add inventory/supply columns
    display_columns.extend([
        "begin_inventory", 
        "supply_in_period", 
        "total_available"
    ])
    
    # Add demand columns based on backlog tracking
    if 'backlog_qty' in display_df.columns and display_options.get('track_backlog', True):
        # When tracking backlog, show original demand, backlog, and effective demand
        display_columns.extend([
            "original_demand_qty",
            "backlog_qty", 
            "effective_demand"
        ])
        # Rename for clarity in display
        if 'original_demand_qty' in display_df.columns:
            display_df = display_df.rename(columns={
                'original_demand_qty': 'Period Demand',
                'backlog_qty': 'Backlog from Previous',
                'effective_demand': 'Total Demand (Period + Backlog)'
            })
            # Update column names in list
            idx = display_columns.index("original_demand_qty")
            display_columns[idx] = "Period Demand"
            idx = display_columns.index("backlog_qty")
            display_columns[idx] = "Backlog from Previous"
            idx = display_columns.index("effective_demand")
            display_columns[idx] = "Total Demand (Period + Backlog)"
    else:
        # Original logic - just show total demand
        display_columns.append("total_demand_qty")
    
    # Add result columns
    display_columns.extend([
        "gap_quantity", 
        "fulfillment_rate_percent", 
        "fulfillment_status"
    ])
    
    # Add optional columns if they exist
    if 'Product Type' in display_df.columns:
        display_columns.append('Product Type')
    
    if 'Backlog Status' in display_df.columns:
        display_columns.append('Backlog Status')
    
    # Only include columns that exist in the dataframe
    display_columns = [col for col in display_columns if col in display_df.columns]
    
    # Sort by period and product
    if 'period' in display_df.columns and 'pt_code' in display_df.columns:
        # Sort based on period type
        if period_type == "Weekly":
            display_df['sort_key'] = display_df['period'].apply(parse_week_period)
        elif period_type == "Monthly":
            display_df['sort_key'] = display_df['period'].apply(parse_month_period)
        else:
            display_df['sort_key'] = pd.to_datetime(display_df['period'])
        
        display_df = display_df.sort_values(['pt_code', 'sort_key'])
        
        # Remove sort key from final display
        if 'sort_key' in display_df.columns:
            display_df = display_df.drop(columns=['sort_key'])
    
    # Reorder columns for better readability
    # Ensure key columns are at the beginning
    key_columns = ['pt_code', 'product_name', 'period', 'Period Status']
    other_columns = [col for col in display_columns if col not in key_columns]
    
    final_column_order = []
    for col in key_columns:
        if col in display_columns:
            final_column_order.append(col)
    final_column_order.extend(other_columns)
    
    return display_df[final_column_order]

def format_gap_display_df(df: pd.DataFrame, display_options: Dict) -> pd.DataFrame:
    """Format GAP dataframe for display with backlog support"""
    df = df.copy()
    
    # Define numeric columns to format
    numeric_cols = [
        "begin_inventory", 
        "supply_in_period", 
        "total_available", 
        "total_demand_qty", 
        "gap_quantity", 
        "original_demand_qty",
        "Period Demand",  # Renamed column
        "Backlog from Previous",  # Renamed column
        "Total Demand (Period + Backlog)"  # Renamed column
    ]
    
    # Add backlog columns if present (in case they weren't renamed)
    if 'backlog_qty' in df.columns:
        numeric_cols.append("backlog_qty")
    if 'effective_demand' in df.columns:
        numeric_cols.append("effective_demand")
    
    # Format numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: format_number(x))
    
    # Format percentage column
    if "fulfillment_rate_percent" in df.columns:
        df["fulfillment_rate_percent"] = df["fulfillment_rate_percent"].apply(
            lambda x: format_percentage(x)
        )
        # Rename for better display
        df = df.rename(columns={"fulfillment_rate_percent": "Fulfillment %"})
    
    # Format other columns for better display
    rename_map = {
        "pt_code": "PT Code",
        "product_name": "Product Name",
        "package_size": "Pack Size",
        "standard_uom": "UOM",
        "period": "Period",
        "begin_inventory": "Begin Inventory",
        "supply_in_period": "Supply In",
        "total_available": "Total Available",
        "total_demand_qty": "Demand Qty",
        "gap_quantity": "GAP",
        "fulfillment_status": "Status"
    }
    
    # Apply rename only for columns that exist
    rename_map_filtered = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map_filtered)
    
    return df

def highlight_gap_rows_enhanced(row):
    """Enhanced highlighting matching Demand/Supply style with priority including backlog"""
    styles = [""] * len(row)
    
    try:
        # Priority: shortage > has backlog > past period > low fulfillment > zero demand
        
        # Check fulfillment status first (highest priority)
        if 'Status' in row.index and "❌" in str(row['Status']):
            # Shortage - light red background
            return ["background-color: #f8d7da"] * len(row)
        
        # Check for backlog (second priority)
        backlog_cols = ['Backlog from Previous', 'backlog_qty', 'Backlog Status']
        for col in backlog_cols:
            if col in row.index:
                if col == 'Backlog Status' and "⚠️" in str(row[col]):
                    # Has backlog - orange background
                    return ["background-color: #fff3cd"] * len(row)
                elif col in ['Backlog from Previous', 'backlog_qty']:
                    try:
                        backlog_val = float(str(row[col]).replace(',', '').strip())
                        if backlog_val > 0:
                            # Has backlog - orange background
                            return ["background-color: #fff3cd"] * len(row)
                    except:
                        pass
        
        # Check if critical shortage (fulfillment < 50%)
        fulfillment_cols = ['Fulfillment %', 'fulfillment_rate_percent']
        for col in fulfillment_cols:
            if col in row.index:
                rate_str = str(row[col]).replace('%', '').strip()
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
        demand_cols = ['Total Demand (Period + Backlog)', 'Demand Qty', 'total_demand_qty']
        for col in demand_cols:
            if col in row.index:
                try:
                    demand_str = str(row[col]).replace(',', '').strip()
                    if float(demand_str) == 0:
                        # Zero demand - light blue
                        return ["background-color: #d1ecf1"] * len(row)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error highlighting rows: {str(e)}")
    
    return styles

# Replace the entire show_gap_detail_table function with this simplified version:

def show_gap_detail_table(gap_df, display_filters, df_demand_filtered=None, df_supply_filtered=None):
    """Show detailed GAP analysis table - simplified version without duplicate filters"""
    st.markdown("### 📄 GAP Details by Product & Period")
    
    # Check if gap_df is empty
    if gap_df.empty:
        st.info("No data matches the selected filters.")
        return
    
    # Show record count
    st.caption(f"Showing {len(gap_df):,} records")
    
    # Prepare display dataframe
    display_df = prepare_gap_detail_display(
        gap_df, 
        display_filters, 
        df_demand_filtered, 
        df_supply_filtered
    )
    
    # Format the dataframe
    formatted_df = format_gap_display_df(display_df, display_filters)
    
    # Apply row highlighting if enabled
    if display_filters.get("enable_row_highlighting", False):
        styled_df = formatted_df.style.apply(highlight_gap_rows_enhanced, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)
    else:
        st.dataframe(formatted_df, use_container_width=True, height=600)

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
    
    # UPDATED LOGIC: Count products with BOTH demand AND supply
    # Same logic as prepare_products_data()
    actionable_products = gap_df[
        (gap_df['total_demand_qty'] > 0) &   # Has unallocated demand
        (gap_df['total_available'] > 0)       # Has available supply
    ]['pt_code'].unique()
    
    products_for_allocation = len(actionable_products)
    
    # Show action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Allocation Plan - available when there are actionable products
        if products_for_allocation > 0:
            st.markdown("### 📋 Allocation Planning")
            
            # Updated message to be clearer
            st.info(f"Found {products_for_allocation} products with demand and available supply")
            
            # Additional metrics for clarity
            actionable_df = gap_df[gap_df['pt_code'].isin(actionable_products)]
            total_actionable_demand = actionable_df['total_demand_qty'].sum()
            total_actionable_supply = actionable_df['total_available'].sum()
            
            # Show breakdown
            st.caption(f"📤 Total demand: {format_number(total_actionable_demand)} units")
            st.caption(f"📥 Total supply: {format_number(total_actionable_supply)} units")
            
            # Calculate average fulfillment capability
            avg_fulfillment = (total_actionable_supply / total_actionable_demand * 100) if total_actionable_demand > 0 else 0
            st.caption(f"📊 Average fulfillment capability: {avg_fulfillment:.1f}%")
            
            if st.button("🧩 Create Allocation Plan", 
                        type="primary", 
                        use_container_width=True,
                        key="gap_create_allocation_btn"):
                st.switch_page("pages/4_🧩_Allocation_Plan.py")
        else:
            st.info("✅ No products require allocation (no items with both demand and supply)")
    
    with col2:
        # PO Suggestions - only for shortage
        if shortage_exists:
            st.markdown("### 📦 Replenishment Needed")
            st.warning(f"Total shortage: {format_number(total_shortage)} units")
            
            # Show products that need PO (demand but no supply)
            products_need_po = gap_df[
                (gap_df['total_demand_qty'] > 0) &   # Has demand
                (gap_df['total_available'] <= 0)      # No supply
            ]['pt_code'].nunique()
            
            if products_need_po > 0:
                st.caption(f"🚨 {products_need_po} products have demand but no supply")
            
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

# === NEW: Calculation Options (Pre-run) ===
calculation_options = get_gap_calculation_options()

# Load data button
if st.button("🚀 Run GAP Analysis", type="primary", use_container_width=True):
    
    if not selected_sources["demand"] or not selected_sources["supply"]:
        st.error("Please select at least one demand source and one supply source.")
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
        
        # IMPORTANT: Enhance BEFORE filtering
        with st.spinner("Enhancing data with allocation info..."):
            # Enhance demand with allocations
            df_demand_enhanced = data_manager.enhance_demand_with_allocations(df_demand_all)
            
            # Enhance supply with allocations (with DRAFT option from calculation_options)
            include_drafts = calculation_options.get("include_draft_allocations", False)
            df_supply_enhanced = data_manager.enhance_supply_with_allocations(
                df_supply_all,
                include_drafts=include_drafts
            )
        
        # Apply filters on ENHANCED data
        df_demand_filtered, df_supply_filtered = apply_filters_to_data(
            df_demand_enhanced,  # Use enhanced
            df_supply_enhanced,  # Use enhanced
            filters, 
            selected_sources.get("selected_customers", []),
            use_adjusted_demand,
            use_adjusted_supply
        )
        
        # Store filtered data
        st.session_state['gap_analysis_data'] = {
            'demand': df_demand_filtered,
            'supply': df_supply_filtered
        }
        st.session_state['gap_analysis_ran'] = True
        st.session_state['gap_calculation_options'] = calculation_options  # Store calculation options
        st.session_state['gap_date_modes'] = {
            'demand': use_adjusted_demand,
            'supply': use_adjusted_supply
        }
        
        # Clear cached GAP results to force recalculation
        if 'gap_df_cached' in st.session_state:
            del st.session_state['gap_df_cached']
        if 'gap_period_type_cache' in st.session_state:
            del st.session_state['gap_period_type_cache']

# === NEW: Display Filters (Post-run) ===
if st.session_state.get('gap_analysis_ran', False):
    display_filters = get_gap_display_filters()
    
    # Display results if analysis has been run and filters are available
    if st.session_state.get('gap_analysis_data') is not None and display_filters:
        
        # Get data from session state
        df_demand_filtered = st.session_state['gap_analysis_data']['demand']
        df_supply_filtered = st.session_state['gap_analysis_data']['supply']
        
        # Get date modes
        date_modes = st.session_state.get('gap_date_modes', {
            'demand': use_adjusted_demand,
            'supply': use_adjusted_supply
        })
        
        # Get stored calculation options
        calc_options = st.session_state.get('gap_calculation_options', calculation_options)
        
        # Initialize display dataframes
        df_demand_filtered_display = df_demand_filtered.copy()
        df_supply_filtered_display = df_supply_filtered.copy()
        
        # Apply date exclusion if requested (from calculation options)
        if calc_options.get("exclude_missing_dates", True):
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
        df_demand_enhanced = data_manager.enhance_demand_with_allocations(df_demand_filtered_display)
        show_allocation_impact_summary(df_demand_enhanced)
        
        # Check if we need to recalculate GAP based on period change
        current_period = calc_options["period_type"]
        cached_period = st.session_state.get('gap_period_type_cache', None)
        
        if 'gap_df_cached' not in st.session_state or cached_period != current_period:
            # Calculate GAP
            with st.spinner("Calculating supply-demand gaps..."):
                gap_df = calculate_gap_with_carry_forward(
                    df_demand_filtered_display, 
                    df_supply_filtered_display, 
                    current_period,
                    date_modes['demand'],
                    date_modes['supply'],
                    track_backlog=calc_options.get('track_backlog', True)
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
            # === NEW: Apply display filters to GAP data ===
            gap_df_filtered = apply_display_filters(
                gap_df,
                display_filters,
                df_demand_filtered_display,
                df_supply_filtered_display,
                calc_options.get('period_type', 'Weekly')
            )
            
            if gap_df_filtered.empty:
                st.warning("No products match the selected display filters.")
                st.info("💡 Try adjusting the Product Types or Period filters above.")
            else:
                # Merge calculation options and display filters for views
                all_options = {**calc_options, **display_filters}
                
                # Display results with filtered data
                show_gap_summary(
                    gap_df_filtered,  # Use filtered data
                    all_options, 
                    df_demand_filtered_display, 
                    df_supply_filtered_display,
                    date_modes['demand'],
                    date_modes['supply']
                )
                
                show_gap_detail_table(
                    gap_df_filtered,  # Use filtered data
                    display_filters,  # Only display filters needed here
                    df_demand_filtered_display, 
                    df_supply_filtered_display
                )
                
                show_gap_pivot_view(
                    gap_df_filtered,  # Use filtered data
                    all_options
                )
                
                # Action buttons with filtered data
                show_action_buttons(gap_df_filtered)
        else:
            st.warning("No data available for the selected filters and sources.")


# Debug info panel - COMPLETE REPLACEMENT
if debug_mode:
    with st.expander("🐛 Debug Information", expanded=True):
        st.markdown("### Debug Information")
        
        # Create debug info safely
        debug_info = {
            "Date Modes": {
                "Demand": 'Adjusted' if use_adjusted_demand else 'Original',
                "Supply": 'Adjusted' if use_adjusted_supply else 'Original'
            },
            "Data Status": {},
            "Session Keys": {}
        }
        
        # Check gap_analysis_data safely
        try:
            gap_data = st.session_state.get('gap_analysis_data')
            if gap_data is not None:
                if isinstance(gap_data, dict):
                    demand_data = gap_data.get('demand', pd.DataFrame())
                    supply_data = gap_data.get('supply', pd.DataFrame())
                    debug_info["Data Status"]["Demand"] = f"{len(demand_data)} rows" if isinstance(demand_data, pd.DataFrame) else "Invalid type"
                    debug_info["Data Status"]["Supply"] = f"{len(supply_data)} rows" if isinstance(supply_data, pd.DataFrame) else "Invalid type"
                else:
                    debug_info["Data Status"]["gap_analysis_data"] = f"Type: {type(gap_data).__name__}"
            else:
                debug_info["Data Status"]["gap_analysis_data"] = "None"
        except Exception as e:
            debug_info["Data Status"]["Error"] = str(e)
        
        # Check gap_df_cached safely
        try:
            gap_df = st.session_state.get('gap_df_cached')
            if gap_df is not None and hasattr(gap_df, 'shape'):
                debug_info["Data Status"]["GAP Results"] = f"{len(gap_df)} rows"
            else:
                debug_info["Data Status"]["GAP Results"] = "Not available"
        except Exception as e:
            debug_info["Data Status"]["GAP Error"] = str(e)
        
        # Get session state keys
        gap_keys = sorted([k for k in st.session_state.keys() if 'gap' in k.lower()])
        for key in gap_keys[:10]:
            try:
                value = st.session_state.get(key)
                if value is None:
                    debug_info["Session Keys"][key] = "None"
                elif isinstance(value, pd.DataFrame):
                    debug_info["Session Keys"][key] = f"DataFrame({len(value)} rows)"
                elif isinstance(value, dict):
                    debug_info["Session Keys"][key] = f"Dict({len(value)} keys)"
                elif isinstance(value, (str, int, float, bool)):
                    debug_info["Session Keys"][key] = str(value)
                else:
                    debug_info["Session Keys"][key] = type(value).__name__
            except:
                debug_info["Session Keys"][key] = "Error reading"
        
        if len(gap_keys) > 10:
            debug_info["Session Keys"]["..."] = f"{len(gap_keys) - 10} more keys"
        
        # Display debug info in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Date Modes & Data Status:**")
            for category in ["Date Modes", "Data Status"]:
                if category in debug_info:
                    for key, value in debug_info[category].items():
                        st.write(f"- {key}: {value}")
        
        with col2:
            st.write("**Session State Keys:**")
            if "Session Keys" in debug_info:
                for key, value in debug_info["Session Keys"].items():
                    st.write(f"- {key}: {value}")
        
        # Show timestamp
        st.caption(f"🕐 Debug generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

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