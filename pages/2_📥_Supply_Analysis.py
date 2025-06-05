# pages/2_📥_Supply_Analysis.py

import streamlit as st
import pandas as pd
from datetime import datetime, date
import logging
from typing import Tuple, Dict, Any, Optional, List

# Import refactored modules
from utils.data_manager import DataManager
from utils.filters import FilterManager
from utils.display_components import DisplayComponents
from utils.formatters import format_number, format_currency, format_percentage, check_missing_dates
from utils.helpers import (
    convert_df_to_excel,
    convert_to_period,
    sort_period_columns,
    save_to_session_state,
    is_past_period,
    parse_week_period,
    parse_month_period,
    create_period_pivot,
    apply_period_indicators
)
from utils.session_state import initialize_session_state
from utils.smart_filter_manager import SmartFilterManager
from utils.date_mode_component import DateModeComponent

# Configure logging
logger = logging.getLogger(__name__)

# === Constants ===
SUPPLY_SOURCES = ["Inventory Only", "Pending CAN Only", "Pending PO Only", "Pending WH Transfer Only", "All"]
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]

# === Page Config ===
st.set_page_config(
    page_title="Supply Analysis - SCM",
    page_icon="📥",
    layout="wide"
)

# === Initialize Session State ===
initialize_session_state()

# === Initialize Components ===
@st.cache_resource
def get_data_manager():
    return DataManager()

data_manager = get_data_manager()

# === Debug Mode Toggle ===
col_debug1, col_debug2 = st.columns([6, 1])
with col_debug2:
    debug_mode = st.checkbox("🐛 Debug Mode", value=False, key="supply_debug_mode")

if debug_mode:
    st.info("🐛 Debug Mode is ON - Additional information will be displayed")

# === Header with Navigation ===
DisplayComponents.show_page_header(
    title="Inbound Supply Analysis",
    icon="📥",
    prev_page="pages/1_📤_Demand_Analysis.py",
    next_page="pages/3_📊_GAP_Analysis.py"
)

# === Date Mode Selection ===
use_adjusted_dates = DateModeComponent.render_date_mode_selector("supply_")

# === Data Source Selection ===
def select_supply_source() -> Tuple[str, bool]:
    """Allow user to choose supply data source"""
    # Use DisplayComponents for source selection
    source_config = DisplayComponents.render_source_selector(
        options=SUPPLY_SOURCES,
        default_index=4,  # Default to "All"
        radio_label="Select Supply Source:",
        key="supply_source_selector",
        additional_options={
            'exclude_expired': {
                'type': 'checkbox',
                'label': '📅 Exclude expired inventory',
                'default': True,
                'help': 'Hide inventory items that have passed their expiry date',
                'key': 'supply_expired_checkbox'
            }
        }
    )
    
    source = source_config['source']
    exclude_expired = source_config.get('exclude_expired', True)
    
    if debug_mode:
        st.write(f"🐛 Selected source: {source}, Exclude expired: {exclude_expired}")
    
    return source, exclude_expired

# === Helper Functions ===
def get_supply_date_column(df: pd.DataFrame, source_type: str, use_adjusted_dates: bool) -> str:
    """Get appropriate date column for supply based on source type"""
    # Map source types to their primary date columns
    date_mapping = {
        'Inventory': 'date_ref',  # Always TODAY for inventory
        'Pending CAN': 'arrival_date',
        'Pending PO': 'eta',
        'Pending WH Transfer': 'transfer_date'
    }
    
    base_column = date_mapping.get(source_type, 'date_ref')
    return DateModeComponent.get_date_column_for_display(df, base_column, use_adjusted_dates)


# === Data Loading Functions ===
def load_and_prepare_supply_data(supply_source: str, exclude_expired: bool = True, 
                                use_adjusted_dates: bool = True) -> pd.DataFrame:
    """Load and standardize supply data based on source selection"""
    try:
        # Convert source selection to list format for data_manager
        sources = []
        if supply_source == "All":
            sources = ["Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"]
        elif supply_source == "Inventory Only":
            sources = ["Inventory"]
        elif supply_source == "Pending CAN Only":
            sources = ["Pending CAN"]
        elif supply_source == "Pending PO Only":
            sources = ["Pending PO"]
        elif supply_source == "Pending WH Transfer Only":
            sources = ["Pending WH Transfer"]
        
        # Use data_manager to get supply data
        df = data_manager.get_supply_data(sources=sources, exclude_expired=exclude_expired)
        
        # FIX: Reset index to ensure no duplicates
        if not df.empty:
            df = df.reset_index(drop=True)
        
        if debug_mode and not df.empty:
            debug_info = {
                "Total rows": len(df),
                "Unique products": df['pt_code'].nunique() if 'pt_code' in df.columns else 0,
                "Supply sources": df['source_type'].value_counts().to_dict() if 'source_type' in df.columns else {},
                "Date columns": [col for col in df.columns if any(x in col.lower() for x in 
                               ['date', 'arrival', 'eta', 'transfer', 'created', 'receipt', 'stockin'])],
                "Index is unique": df.index.is_unique,  # Add this check
                "Duplicate rows": df.duplicated().sum()  # Add this check
            }
            DisplayComponents.show_debug_info(debug_info)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading supply data: {str(e)}")
        st.error(f"Failed to load supply data: {str(e)}")
        return pd.DataFrame()


# === Filtering Functions ===
def apply_supply_filters(df: pd.DataFrame, use_adjusted_dates: bool = True) -> Tuple[pd.DataFrame, date, date, Dict]:
    """Apply filters with mode toggle"""
    if df.empty:
        return df, date.today(), date.today(), {}
    
    # Initialize filter manager
    filter_manager = SmartFilterManager(key_prefix="supply_")
    
    # Render toggle and get mode
    use_smart_filters = filter_manager.render_filter_toggle()
    
    if use_smart_filters:
        return apply_smart_supply_filters(df, use_adjusted_dates, filter_manager)
    else:
        return apply_standard_supply_filters(df, use_adjusted_dates)


def apply_standard_supply_filters(df: pd.DataFrame, use_adjusted_dates: bool = True) -> Tuple[pd.DataFrame, date, date, Dict]:
    """Standard independent filters for supply"""
    with st.expander("📎 Filters", expanded=True):

        # Always use date_ref for filtering (it's unified for All tab)
        date_column = DateModeComponent.get_date_column_for_display(df, 'date_ref', use_adjusted_dates)

        # For filtering, use appropriate date column based on what's available
        if "unified_date" in df.columns:
            # For "All" view, use unified_date
            date_column = DateModeComponent.get_date_column_for_display(df, 'unified_date', use_adjusted_dates)
        else:
            # For single source, use date_ref
            date_column = DateModeComponent.get_date_column_for_display(df, 'date_ref', use_adjusted_dates)
        
        # Row 1: Entity, Product, Brand
        col1, col2, col3 = st.columns(3)
        
        filters = {}
        filter_params = {}
        
        with col1:
            if 'legal_entity' in df.columns:
                entities = sorted(df["legal_entity"].dropna().unique())
                filters['entity'] = st.multiselect(
                    "Legal Entity",
                    entities,
                    key="supply_entity_filter_std",
                    placeholder="All entities"
                )
        
        with col2:
            if 'pt_code' in df.columns and 'product_name' in df.columns:
                # Create product options
                products_df = df[['pt_code', 'product_name']].drop_duplicates()
                products_df = products_df[products_df['pt_code'].notna() & (products_df['pt_code'] != 'nan')]
                
                product_options = []
                for _, row in products_df.iterrows():
                    pt_code = str(row['pt_code'])
                    product_name = str(row['product_name'])[:50] if pd.notna(row['product_name']) else ""
                    product_options.append(f"{pt_code} - {product_name}")
                
                selected_products = st.multiselect(
                    "Product (PT Code - Name)",
                    sorted(product_options),
                    key="supply_product_filter_std",
                    placeholder="All products"
                )
                
                filters['product'] = [p.split(' - ')[0] for p in selected_products]
        
        with col3:
            if 'brand' in df.columns:
                brands = sorted(df["brand"].dropna().unique())
                filters['brand'] = st.multiselect(
                    "Brand",
                    brands,
                    key="supply_brand_filter_std",
                    placeholder="All brands"
                )
        
        # Row 2: Source Type, Vendor, Warehouse
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if 'source_type' in df.columns:
                sources = sorted(df['source_type'].unique())
                filters['source_type'] = st.multiselect(
                    "Supply Source",
                    sources,
                    key="supply_source_filter_std",
                    placeholder="All sources"
                )
        
        with col5:
            if 'vendor' in df.columns and df['vendor'].notna().any():
                vendors = sorted(df['vendor'].dropna().unique())
                filters['vendor'] = st.multiselect(
                    "Vendor",
                    vendors,
                    key="supply_vendor_filter_std",
                    placeholder="All vendors"
                )
        
        with col6:
            if 'from_warehouse' in df.columns and df['from_warehouse'].notna().any():
                warehouses = sorted(df['from_warehouse'].dropna().unique())
                filters['from_warehouse'] = st.multiselect(
                    "From Warehouse",
                    warehouses,
                    key="supply_from_wh_filter_std",
                    placeholder="All warehouses"
                )
        
        # Supply-specific filters
        st.markdown("#### 🏭 Supply-Specific Filters")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            if "days_until_expiry" in df.columns:
                filter_params['expiry_warning_days'] = st.number_input(
                    "Show items expiring within (days)",
                    min_value=0,
                    max_value=365,
                    value=30,
                    key="supply_expiry_days_std"
                )
        
        with col_s2:
            if "days_until_expiry" in df.columns:
                filter_params['only_expiring'] = st.checkbox(
                    "Only show expiring items",
                    key="supply_only_expiring_std"
                )
        
        with col_s3:
            if "quantity" in df.columns:
                filter_params['min_stock'] = st.number_input(
                    "Minimum stock quantity",
                    min_value=0,
                    value=0,
                    key="supply_min_stock_std"
                )
        
        # Date range
        st.markdown("#### 📅 Date Range")
        col_date1, col_date2 = st.columns(2)
        
        # Get date range from appropriate column
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
            min_date = dates.min().date() if len(dates) > 0 else date.today()
            max_date = dates.max().date() if len(dates) > 0 else date.today()
        else:
            min_date = max_date = date.today()
        
        with col_date1:
            start_date = st.date_input(
                f"From Date",
                value=min_date,
                key="supply_start_date_std"
            )
        
        with col_date2:
            end_date = st.date_input(
                f"To Date",
                value=max_date,
                key="supply_end_date_std"
            )
        
        # Show active filters
        active_filters = sum(1 for v in filters.values() if v and v != [])
        if active_filters > 0:
            st.success(f"🔍 {active_filters} filters active")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply selection filters
    if filters.get('entity'):
        filtered_df = filtered_df[filtered_df['legal_entity'].isin(filters['entity'])]
    
    if filters.get('product'):
        filtered_df = filtered_df[filtered_df['pt_code'].isin(filters['product'])]
    
    if filters.get('brand'):
        filtered_df = filtered_df[filtered_df['brand'].isin(filters['brand'])]
    
    if filters.get('source_type'):
        filtered_df = filtered_df[filtered_df['source_type'].isin(filters['source_type'])]
    
    if filters.get('vendor'):
        filtered_df = filtered_df[filtered_df['vendor'].isin(filters['vendor'])]
    
    if filters.get('from_warehouse'):
        filtered_df = filtered_df[filtered_df['from_warehouse'].isin(filters['from_warehouse'])]
    
    # Apply date filter on appropriate column
    if date_column in filtered_df.columns:
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors='coerce')
        filtered_df = filtered_df[
            (filtered_df[date_column].isna()) |
            ((filtered_df[date_column] >= pd.to_datetime(start_date)) & 
             (filtered_df[date_column] <= pd.to_datetime(end_date)))
        ]
    
    # Apply supply-specific filters
    if filter_params.get('only_expiring') and "days_until_expiry" in filtered_df.columns:
        expiry_days = filter_params.get('expiry_warning_days', 30)
        filtered_df = filtered_df[
            (filtered_df["days_until_expiry"] >= 0) &
            (filtered_df["days_until_expiry"] <= expiry_days)
        ]
    
    if filter_params.get('min_stock', 0) > 0 and "quantity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["quantity"] >= filter_params['min_stock']]
    
    # Show filtering result
    if len(filtered_df) < len(df):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"🔍 Filtered: {len(df):,} → {len(filtered_df):,} records")
        with col2:
            retention = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
            st.metric("Retention", f"{retention:.1f}%")
    
    return filtered_df, start_date, end_date, filter_params


def apply_smart_supply_filters(df: pd.DataFrame, use_adjusted_dates: bool, 
                             filter_manager: SmartFilterManager) -> Tuple[pd.DataFrame, date, date, Dict]:
    """Apply smart interactive filters to supply dataframe"""
    try:
        # Use appropriate date column
        if "unified_date" in df.columns:
            # For "All" view, use unified_date
            date_column = DateModeComponent.get_date_column_for_display(df, 'unified_date', use_adjusted_dates)
        else:
            # For single source, use date_ref
            date_column = DateModeComponent.get_date_column_for_display(df, 'date_ref', use_adjusted_dates)
        
        # Validate date column exists
        if date_column not in df.columns:
            st.warning(f"Date column '{date_column}' not found.")
            # Fallback logic
            if "unified_date" in df.columns:
                date_column = "unified_date"
            elif "date_ref" in df.columns:
                date_column = "date_ref"
            else:
                date_column = df.columns[0]
        
        # Configure filters based on available columns
        filter_config = {}
        
        # Entity filter
        if 'legal_entity' in df.columns:
            filter_config['entity_selection'] = {
                'column': 'legal_entity',
                'label': 'Legal Entity',
                'help': 'Filter by legal entities',
                'placeholder': 'Choose legal entities...'
            }
        
        # Product filter with PT Code + Name
        if 'pt_code' in df.columns:
            filter_config['product_selection'] = {
                'column': 'pt_code',
                'label': 'Product (PT Code - Name)',
                'help': 'Search by PT Code or Product Name',
                'placeholder': 'Type to search products...'
            }
        
        # Brand filter
        if 'brand' in df.columns:
            filter_config['brand_selection'] = {
                'column': 'brand',
                'label': 'Brand',
                'help': 'Filter by brands',
                'placeholder': 'Choose brands...'
            }
        
        # Source type filter
        if 'source_type' in df.columns:
            unique_sources = df['source_type'].unique()
            if len(unique_sources) > 1:
                filter_config['source_selection'] = {
                    'column': 'source_type',
                    'label': 'Source Type',
                    'help': 'Filter by supply source type',
                    'placeholder': 'Choose source types...'
                }
        
        # Vendor filter (for PO)
        if 'vendor' in df.columns and df['vendor'].notna().any():
            filter_config['vendor_selection'] = {
                'column': 'vendor',
                'label': 'Vendor',
                'help': 'Filter by vendor (for PO)',
                'placeholder': 'Choose vendors...'
            }
        
        # Warehouse filters
        if 'from_warehouse' in df.columns and df['from_warehouse'].notna().any():
            filter_config['from_warehouse_selection'] = {
                'column': 'from_warehouse',
                'label': 'From Warehouse',
                'help': 'Filter by source warehouse',
                'placeholder': 'Choose warehouse...'
            }
        
        # Render smart filters
        with st.container():
            # Show filter header
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("### 📎 Smart Filters")
            with col2:
                st.caption(f"📊 {len(df):,} total records")
            
            # Render filters
            filters_result = filter_manager.render_smart_filters(
                df=df,
                filter_config=filter_config,
                show_date_filters=True,
                date_column=date_column
            )
        
        # Validate filter result
        if not isinstance(filters_result, dict):
            raise ValueError(f"Invalid filter result type: {type(filters_result)}")
        
        # Extract components
        selections = filters_result.get('selections', {})
        date_filters = filters_result.get('date_filters', {})
        
        # Apply filters
        filtered_df = filter_manager.apply_filters_to_dataframe(df, filters_result)
        
        # Get dates
        start_date = date_filters.get('start_date', datetime.today().date())
        end_date = date_filters.get('end_date', datetime.today().date())
        
        # Validate dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Additional supply-specific filters
        filter_params = {}
        
        # Expiry warning filter
        st.markdown("#### 🏭 Supply-Specific Filters")
        
        if "days_until_expiry" in df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_params['expiry_warning_days'] = st.number_input(
                    "Show items expiring within (days)",
                    min_value=0,
                    max_value=365,
                    value=30,
                    key="supply_expiry_warning_days_smart",
                    help="Highlight items nearing expiry"
                )
            
            with col2:
                # Apply expiry filter
                if st.checkbox("Only show expiring items", key="supply_only_expiring_smart"):
                    filtered_df = filtered_df[
                        (filtered_df["days_until_expiry"] <= filter_params['expiry_warning_days']) &
                        (filtered_df["days_until_expiry"] >= 0)
                    ]
                    filter_params['only_expiring'] = True
                else:
                    filter_params['only_expiring'] = False
        
            # Stock status filter
            with col3:
                if "quantity" in filtered_df.columns:
                    min_stock = st.number_input(
                        "Minimum stock quantity",
                        min_value=0,
                        value=0,
                        key="supply_min_stock_smart",
                        help="Filter by minimum stock level"
                    )
                    if min_stock > 0:
                        filtered_df = filtered_df[filtered_df["quantity"] >= min_stock]
                    filter_params['min_stock'] = min_stock
        else:
            # If no expiry column, still check for quantity filter
            if "quantity" in filtered_df.columns:
                min_stock = st.number_input(
                    "Minimum stock quantity",
                    min_value=0,
                    value=0,
                    key="supply_min_stock_smart_alt",
                    help="Filter by minimum stock level"
                )
                if min_stock > 0:
                    filtered_df = filtered_df[filtered_df["quantity"] >= min_stock]
                filter_params['min_stock'] = min_stock
        
        # Show filter summary
        if filtered_df.shape[0] < df.shape[0]:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.info(f"🔍 Filtered: {len(df):,} → {len(filtered_df):,} records")
            with col2:
                active_count = sum(1 for v in selections.values() if v)
                # Add supply-specific filters to count
                if filter_params.get('only_expiring'):
                    active_count += 1
                if filter_params.get('min_stock', 0) > 0:
                    active_count += 1
                if active_count > 0:
                    st.success(f"✅ {active_count} filters active")
            with col3:
                retention = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
                st.metric("Retained", f"{retention:.1f}%")
        
        return filtered_df, start_date, end_date, filter_params
        
    except Exception as e:
        logger.error(f"Smart filter error in supply: {str(e)}", exc_info=True)
        st.error(f"⚠️ Smart filters error: {str(e)}")
        st.info("💡 Please switch to Standard Filters mode")
        
        # Return original data with default values
        return df, date.today(), date.today(), {}


# === Display Functions ===
def show_supply_summary(filtered_df: pd.DataFrame, filter_params: Dict, use_adjusted_dates: bool = True):
    """Show supply summary metrics with status comparison"""
    if filtered_df.empty:
        DisplayComponents.show_no_data_message("No supply data to display")
        return
    
    try:
        # Calculate metrics by source type
        source_summary = filtered_df.groupby('source_type').agg({
            'quantity': 'sum',
            'value_in_usd': 'sum',
            'pt_code': 'nunique'
        }).reset_index()
        
        
        # Display overall metrics
        metrics = [
            {
                "title": "Total Unique Products",
                "value": filtered_df["pt_code"].nunique(),
                "format_type": "number"
            },
            {
                "title": "Total Quantity",
                "value": filtered_df["quantity"].sum(),
                "format_type": "number"
            },
            {
                "title": "Total Value",
                "value": filtered_df["value_in_usd"].sum(),
                "format_type": "currency"
            }
        ]
        
        # Render summary section with additional content
        DisplayComponents.render_summary_section(
            metrics=metrics,
            title="### 📊 Supply Summary",
            additional_content=lambda: show_additional_supply_info(filtered_df, source_summary, filter_params, use_adjusted_dates)
        )
        
    except Exception as e:
        logger.error(f"Error showing supply summary: {str(e)}")
        st.error(f"Error displaying summary: {str(e)}")


def show_additional_supply_info(filtered_df: pd.DataFrame, source_summary: pd.DataFrame, 
                               filter_params: Dict, use_adjusted_dates: bool):
    """Show additional supply information - WITH DATE COMPARISON"""
    # Source breakdown
    st.markdown("#### 📦 Supply by Source")
    
    # Create columns for each source type
    if not source_summary.empty:
        source_cols = st.columns(len(source_summary))
        for idx, (col, row) in enumerate(zip(source_cols, source_summary.itertuples())):
            with col:
                st.markdown(f"**{row.source_type}**")
                st.metric("Products", f"{row.pt_code:,}")
                st.metric("Quantity", format_number(row.quantity))
                st.metric("Value", format_currency(row.value_in_usd, "$", 0))
    
    # Show Date Status Comparison (if applicable)
    show_date_status_comparison(filtered_df, use_adjusted_dates)

    # Special warnings
    show_supply_warnings(filtered_df, filter_params)


def show_date_status_comparison(filtered_df: pd.DataFrame, use_adjusted_dates: bool = True):
    """Show date status comparison between original and adjusted - SIMILAR TO DEMAND PAGE"""
    # Get unique source types
    source_types = filtered_df['source_type'].unique() if 'source_type' in filtered_df.columns else []
    
    if len(source_types) == 0:
        return
    
    st.markdown("#### 📈 Date Status Comparison")
    
    today = pd.Timestamp.now().normalize()
    
    # Collect comparison data for each source type
    comparison_data = []
    
    # For "All" tab - aggregate all sources
    total_missing_original = 0
    total_past_original = 0
    total_missing_adjusted = 0
    total_past_adjusted = 0
    
    for source_type in source_types:
        source_df = filtered_df[filtered_df['source_type'] == source_type]
        
        # Get date columns for this source
        date_mapping = {
            'Inventory': 'date_ref',
            'Pending CAN': 'arrival_date',
            'Pending PO': 'eta',
            'Pending WH Transfer': 'transfer_date'
        }
        
        base_col = date_mapping.get(source_type, 'date_ref')
        adjusted_col = f"{base_col}_adjusted"
        
        # Check if both columns exist
        if base_col in source_df.columns and adjusted_col in source_df.columns:
            original_series = pd.to_datetime(source_df[base_col], errors='coerce')
            adjusted_series = pd.to_datetime(source_df[adjusted_col], errors='coerce')
            
            missing_original = original_series.isna().sum()
            past_original = (original_series < today).sum()
            
            missing_adjusted = adjusted_series.isna().sum()
            past_adjusted = (adjusted_series < today).sum()
            
            # Add to totals for "All" tab
            total_missing_original += missing_original
            total_past_original += past_original
            total_missing_adjusted += missing_adjusted
            total_past_adjusted += past_adjusted
            
            comparison_data.append({
                'source': source_type,
                'base_col': base_col,
                'adjusted_col': adjusted_col,
                'missing_original': missing_original,
                'past_original': past_original,
                'missing_adjusted': missing_adjusted,
                'past_adjusted': past_adjusted
            })
    
    if not comparison_data:
        return
    
    # Create tabs: All (default) + individual sources
    tab_labels = ["All"] + [data['source'] for data in comparison_data]
    tabs = st.tabs(tab_labels)
    
    # All tab (first tab - default)
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Original Dates (All Sources)**")
            st.metric("Missing", total_missing_original)
            st.metric("Past", total_past_original)
        
        with col2:
            st.success("**Adjusted Dates (All Sources)**")
            st.metric("Missing", total_missing_adjusted)
            st.metric("Past", total_past_adjusted)
        
        # Show overall improvement/degradation
        if total_missing_original != total_missing_adjusted or total_past_original != total_past_adjusted:
            st.caption("📊 Overall impact of adjustments:")
            if total_missing_adjusted < total_missing_original:
                st.success(f"✅ {total_missing_original - total_missing_adjusted} missing dates resolved across all sources")
            if total_past_adjusted != total_past_original:
                diff = total_past_adjusted - total_past_original
                if diff > 0:
                    st.warning(f"⚠️ {diff} more records moved to past due to adjustments")
                else:
                    st.info(f"ℹ️ {abs(diff)} records moved from past to future due to adjustments")
    
    # Individual source tabs
    for idx, data in enumerate(comparison_data):
        with tabs[idx + 1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Original {data['base_col'].replace('_', ' ').title()}**")
                st.metric("Missing", data['missing_original'])
                st.metric("Past", data['past_original'])
            
            with col2:
                st.success(f"**Adjusted {data['base_col'].replace('_', ' ').title()}**")
                st.metric("Missing", data['missing_adjusted'])
                st.metric("Past", data['past_adjusted'])
            
            # Show improvement/degradation for each source
            if data['missing_original'] != data['missing_adjusted'] or data['past_original'] != data['past_adjusted']:
                st.caption("📊 Impact of adjustments:")
                if data['missing_adjusted'] < data['missing_original']:
                    st.success(f"✅ {data['missing_original'] - data['missing_adjusted']} missing dates resolved")
                if data['past_adjusted'] != data['past_original']:
                    diff = data['past_adjusted'] - data['past_original']
                    if diff > 0:
                        st.warning(f"⚠️ {diff} more records moved to past")
                    else:
                        st.info(f"ℹ️ {abs(diff)} records moved from past to future")


def show_supply_warnings(filtered_df: pd.DataFrame, filter_params: Dict):
    """Show supply-specific warnings"""
    warnings = []
    
    # Expiry warning for inventory and WH transfer
    if "days_until_expiry" in filtered_df.columns and filter_params.get('expiry_warning_days'):
        expiry_warning_days = filter_params['expiry_warning_days']
        
        # Already expired
        expired = filtered_df[filtered_df["days_until_expiry"] < 0]
        if not expired.empty:
            warnings.append(f"💀 {len(expired)} items already expired!")
        
        # Expiring soon
        expiring_soon = filtered_df[
            (filtered_df["days_until_expiry"] >= 0) &
            (filtered_df["days_until_expiry"] <= expiry_warning_days)
        ]
        if not expiring_soon.empty:
            warnings.append(f"⚠️ {len(expiring_soon)} items expiring within {expiry_warning_days} days!")
    
    # Delayed CAN warning
    if "days_since_arrival" in filtered_df.columns:
        delayed_cans = filtered_df[
            (filtered_df["source_type"] == "Pending CAN") & 
            (filtered_df["days_since_arrival"] > 7)
        ]
        if not delayed_cans.empty:
            warnings.append(f"📦 {len(delayed_cans)} CAN items pending stock-in for >7 days!")
    
    # Long transfer warning
    if "days_in_transfer" in filtered_df.columns:
        long_transfers = filtered_df[
            (filtered_df["source_type"] == "Pending WH Transfer") & 
            (filtered_df["days_in_transfer"] > 3)
        ]
        if not long_transfers.empty:
            warnings.append(f"🚚 {len(long_transfers)} warehouse transfers taking >3 days!")
    
    # Display warnings
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("✅ No critical supply issues detected")

def show_supply_detail_table(filtered_df: pd.DataFrame, use_adjusted_dates: bool = True):
    """Show detailed supply table with enhanced filtering and date highlighting"""
    st.markdown("### 🔍 Supply Details")
    
    if filtered_df.empty:
        DisplayComponents.show_no_data_message("No data to display")
        return
    
    try:
        # Define filter options FIRST
        filter_options = [
            "Show All",
            "Show Missing Original Date Only", 
            "Show Past Original Date Only",
            "Show Missing Adjusted Date Only",
            "Show Past Adjusted Date Only"
        ]
        
        # Create tabs for different source types
        source_types = filtered_df["source_type"].unique()
        
        if len(source_types) > 1:
            # Multiple source types - create tabs
            tab_labels = ["All"] + list(source_types)
            tabs = st.tabs(tab_labels)
            
            # All tab
            with tabs[0]:
                display_supply_detail_for_source(
                    filtered_df, 
                    "All", 
                    filter_options, 
                    use_adjusted_dates
                )
            
            # Individual source tabs
            for idx, source in enumerate(source_types):
                with tabs[idx + 1]:
                    source_df = filtered_df[filtered_df["source_type"] == source]
                    display_supply_detail_for_source(
                        source_df, 
                        source, 
                        filter_options, 
                        use_adjusted_dates
                    )
        else:
            # Single source type - no tabs needed
            display_supply_detail_for_source(
                filtered_df, 
                source_types[0] if len(source_types) > 0 else "All", 
                filter_options, 
                use_adjusted_dates
            )
        
    except Exception as e:
        logger.error(f"Error showing supply detail table: {str(e)}")
        st.error(f"Error displaying table: {str(e)}")

def display_supply_detail_for_source(df: pd.DataFrame, source_label: str, 
                                   filter_options: List[str], use_adjusted_dates: bool):
    """Display detail table for a specific source"""
    if df.empty:
        DisplayComponents.show_no_data_message(f"No {source_label} data available")
        return
    
    # Get date columns for this source - IMPORTANT: This will include unified dates for All tab
    date_columns_added = get_date_columns_for_source(df, source_label)
    
    # Debug info
    if st.session_state.get('supply_debug_mode', False):
        st.write(f"🐛 Date columns for {source_label}: {date_columns_added}")
        st.write(f"🐛 Available columns: {[col for col in date_columns_added if col in df.columns]}")
    
    # Use DisplayComponents for detail table
    DisplayComponents.render_detail_table_with_filter(
        df=df,
        filter_options=filter_options,
        filter_apply_func=lambda df, opt: apply_supply_detail_filter(df, opt, date_columns_added),
        format_func=lambda df: format_supply_display_df_enhanced(df, date_columns_added),
        style_func=lambda row: highlight_supply_issues_enhanced(row, date_columns_added),
        height=600,
        key_prefix=f"supply_{source_label.lower().replace(' ', '_')}"
    )

def get_date_columns_for_source(df: pd.DataFrame, source_type: str) -> List[str]:
    """Get appropriate date columns for a source type - ALWAYS SHOW BOTH ORIGINAL AND ADJUSTED"""
    date_columns = []
    
    if source_type == "All":
        # For All tab, always show unified_date and unified_date_adjusted
        if "unified_date" in df.columns:
            date_columns.append("unified_date")
        if "unified_date_adjusted" in df.columns:
            date_columns.append("unified_date_adjusted")
            
        # If unified columns don't exist, fall back to showing all available date columns
        if not date_columns:
            # Check for any date columns
            possible_dates = ["date_ref", "arrival_date", "eta", "transfer_date"]
            for date_col in possible_dates:
                if date_col in df.columns:
                    date_columns.append(date_col)
                if f"{date_col}_adjusted" in df.columns:
                    date_columns.append(f"{date_col}_adjusted")
    else:
        # Map source types to their primary date column
        date_mapping = {
            "Inventory": "date_ref", 
            "Pending CAN": "arrival_date",
            "Pending PO": "eta",
            "Pending WH Transfer": "transfer_date"
        }
        
        base_col = date_mapping.get(source_type, "date_ref")
        
        # Always add both original and adjusted columns if they exist
        if base_col in df.columns:
            date_columns.append(base_col)
        
        # Add adjusted column
        adjusted_col = f"{base_col}_adjusted"
        if adjusted_col in df.columns:
            date_columns.append(adjusted_col)
    
    return date_columns


def apply_supply_detail_filter(display_df: pd.DataFrame, filter_option: str, 
                             date_columns: List[str]) -> pd.DataFrame:
    """Apply filter to supply detail dataframe - UPDATED FOR UNIFIED DATES"""
    if filter_option == "Show All":
        return prepare_supply_detail_display(display_df)
    
    today = pd.Timestamp.now().normalize()
    
    # Initialize filter mask
    filter_mask = pd.Series([False] * len(display_df), index=display_df.index)
    
    # Check if we're in "All" view by looking for unified_date columns
    is_all_view = "unified_date" in display_df.columns or "unified_date_adjusted" in display_df.columns
    
    if is_all_view:
        # For All tab - use unified date columns
        if filter_option == "Show Missing Original Date Only":
            if "unified_date" in display_df.columns:
                date_series = pd.to_datetime(display_df["unified_date"], errors='coerce')
                filter_mask = date_series.isna()
        
        elif filter_option == "Show Past Original Date Only":
            if "unified_date" in display_df.columns:
                date_series = pd.to_datetime(display_df["unified_date"], errors='coerce')
                filter_mask = date_series.notna() & (date_series < today)
        
        elif filter_option == "Show Missing Adjusted Date Only":
            if "unified_date_adjusted" in display_df.columns:
                date_series = pd.to_datetime(display_df["unified_date_adjusted"], errors='coerce')
                filter_mask = date_series.isna()
            else:
                st.warning("Adjusted date column not available")
                return pd.DataFrame()
                
        elif filter_option == "Show Past Adjusted Date Only":
            if "unified_date_adjusted" in display_df.columns:
                date_series = pd.to_datetime(display_df["unified_date_adjusted"], errors='coerce')
                filter_mask = date_series.notna() & (date_series < today)
            else:
                st.warning("Adjusted date column not available")
                return pd.DataFrame()
    else:
        # For source-specific tabs - use source-specific date columns
        # Get base date column (first one without adjusted suffix)
        base_date_col = None
        adjusted_date_col = None
        
        for col in date_columns:
            if '_adjusted' in col:
                adjusted_date_col = col
            elif '_original' not in col:  # Skip _original columns
                base_date_col = col
        
        if filter_option == "Show Missing Original Date Only":
            if base_date_col and base_date_col in display_df.columns:
                date_series = pd.to_datetime(display_df[base_date_col], errors='coerce')
                filter_mask = date_series.isna()
        
        elif filter_option == "Show Past Original Date Only":
            if base_date_col and base_date_col in display_df.columns:
                date_series = pd.to_datetime(display_df[base_date_col], errors='coerce')
                filter_mask = date_series.notna() & (date_series < today)
        
        elif filter_option == "Show Missing Adjusted Date Only":
            if adjusted_date_col and adjusted_date_col in display_df.columns:
                date_series = pd.to_datetime(display_df[adjusted_date_col], errors='coerce')
                filter_mask = date_series.isna()
            else:
                st.warning("Adjusted date column not available")
                return pd.DataFrame()
                
        elif filter_option == "Show Past Adjusted Date Only":
            if adjusted_date_col and adjusted_date_col in display_df.columns:
                date_series = pd.to_datetime(display_df[adjusted_date_col], errors='coerce')
                filter_mask = date_series.notna() & (date_series < today)
            else:
                st.warning("Adjusted date column not available")
                return pd.DataFrame()
    
    # Apply the filter
    if filter_mask.any():
        display_df = display_df[filter_mask]
    elif filter_option != "Show All":
        display_df = pd.DataFrame()
    
    return prepare_supply_detail_display(display_df)


# === Additional utility function to ensure data integrity ===
def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has unique index"""
    if not df.index.is_unique:
        logger.warning("DataFrame has duplicate index, resetting...")
        return df.reset_index(drop=True)
    return df


# Add this to prepare_supply_detail_display in Supply Analysis page
# This will show PO line details better

def prepare_supply_detail_display(display_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare supply dataframe for detail display - WITH PROPER DATE COLUMNS"""
    if display_df.empty:
        return display_df
    
    # Get source type
    source_type = display_df["source_type"].iloc[0] if len(display_df["source_type"].unique()) == 1 else "All"
    
    # Prepare display columns based on source type
    base_columns = [
        "pt_code", "product_name", "brand", "package_size", "standard_uom"
    ]
    
    # Add source_type for "All" view
    if source_type == "All":
        base_columns.insert(0, "source_type")
    
    # Add date columns based on source type
    date_columns = get_date_columns_for_source(display_df, source_type)
    
    # Insert date columns after standard_uom but before quantity
    insert_index = base_columns.index("standard_uom") + 1
    for date_col in date_columns:
        if date_col in display_df.columns:
            base_columns.insert(insert_index, date_col)
            insert_index += 1
    
    # Add remaining standard columns
    base_columns.extend(["quantity", "value_in_usd", "legal_entity"])
    
    # Add source-specific additional columns
    additional_columns = []
    
    if "expiry_date" in display_df.columns:
        additional_columns.extend(["expiry_date", "days_until_expiry"])
    
    if "days_since_arrival" in display_df.columns and source_type in ["Pending CAN", "All"]:
        additional_columns.append("days_since_arrival")
    
    if "vendor" in display_df.columns and source_type in ["Pending PO", "All"]:
        additional_columns.append("vendor")
    
    # For Pending PO, show more details
    if source_type in ["Pending PO", "All"]:
        po_specific_cols = ["po_number", "buying_quantity", "buying_uom", "purchase_unit_cost"]
        for col in po_specific_cols:
            if col in display_df.columns and col not in base_columns and col not in additional_columns:
                additional_columns.append(col)
    
    if source_type in ["Pending WH Transfer", "All"]:
        if "transfer_route" in display_df.columns:
            additional_columns.append("transfer_route")
        if "days_in_transfer" in display_df.columns:
            additional_columns.append("days_in_transfer")
    
    # Add supply_number to identify unique lines
    if "supply_number" in display_df.columns:
        idx = base_columns.index("legal_entity") if "legal_entity" in base_columns else len(base_columns)
        base_columns.insert(idx, "supply_number")
    
    # Combine columns
    display_columns = base_columns + additional_columns
    
    # Filter columns that actually exist
    display_columns = [col for col in display_columns if col in display_df.columns]
    
    display_df = display_df[display_columns].copy()
    
    # Sort by appropriate columns
    sort_columns = []
    
    # For All view, sort by source_type first
    if source_type == "All" and "source_type" in display_df.columns:
        sort_columns.append("source_type")
    
    # Sort by PO number if available
    if "po_number" in display_df.columns:
        sort_columns.append("po_number")
    
    # Sort by appropriate date column
    for col in date_columns:
        if col in display_df.columns and 'adjusted' in col:
            sort_columns.append(col)
            break
    else:
        # Fallback to original date if no adjusted found
        for col in date_columns:
            if col in display_df.columns:
                sort_columns.append(col)
                break
    
    # Add product code as final sort
    if "pt_code" in display_df.columns:
        sort_columns.append("pt_code")
    
    if sort_columns:
        display_df = display_df.sort_values(sort_columns)
    
    return display_df


def format_supply_display_df_enhanced(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Enhanced formatting for supply dataframe with multiple date columns"""
    df = df.copy()
    today = pd.Timestamp.now().normalize()
    
    try:
        # Format quantity and value
        if "quantity" in df.columns:
            df["quantity"] = df["quantity"].apply(lambda x: format_number(x))
        if "value_in_usd" in df.columns:
            df["value_in_usd"] = df["value_in_usd"].apply(lambda x: format_currency(x))
        
        # Enhanced date formatting with indicators for all date columns
        def format_date_with_status(x):
            if pd.isna(x):
                return "❌ Missing"
            elif isinstance(x, pd.Timestamp) and x < today:
                days_past = (today - x).days
                return f"🔴 {x.strftime('%Y-%m-%d')} ({days_past}d ago)"
            elif isinstance(x, pd.Timestamp):
                return x.strftime("%Y-%m-%d")
            else:
                return str(x)
        
        # Apply to all date columns including unified dates
        for date_col in date_columns:
            if date_col in df.columns:
                df[date_col] = df[date_col].apply(format_date_with_status)
        
        # Also check for unified_date columns if not in date_columns
        if "unified_date" in df.columns and "unified_date" not in date_columns:
            df["unified_date"] = df["unified_date"].apply(format_date_with_status)
        if "unified_date_adjusted" in df.columns and "unified_date_adjusted" not in date_columns:
            df["unified_date_adjusted"] = df["unified_date_adjusted"].apply(format_date_with_status)
        
        # Format expiry date
        if "expiry_date" in df.columns:
            df["expiry_date"] = df["expiry_date"].apply(
                lambda x: "" if pd.isna(x) else x.strftime("%Y-%m-%d")
            )
        
        # Format days columns with color coding
        if "days_until_expiry" in df.columns:
            def format_days_until_expiry(x):
                if pd.isna(x):
                    return ""
                days = int(x)
                if days < 0:
                    return f"💀 Expired {abs(days)}d ago"
                elif days <= 7:
                    return f"🔴 {days} days"
                elif days <= 30:
                    return f"🟡 {days} days"
                else:
                    return f"✅ {days} days"
            
            df["days_until_expiry"] = df["days_until_expiry"].apply(format_days_until_expiry)
        
        if "days_since_arrival" in df.columns:
            def format_days_since_arrival(x):
                if pd.isna(x):
                    return ""
                days = int(x)
                if days > 7:
                    return f"🔴 {days} days"
                elif days > 3:
                    return f"🟡 {days} days"
                else:
                    return f"✅ {days} days"
            
            df["days_since_arrival"] = df["days_since_arrival"].apply(format_days_since_arrival)
        
        if "days_in_transfer" in df.columns:
            def format_days_in_transfer(x):
                if pd.isna(x):
                    return ""
                days = int(x)
                if days > 3:
                    return f"🔴 {days} days"
                elif days > 2:
                    return f"🟡 {days} days"
                else:
                    return f"✅ {days} days"
            
            df["days_in_transfer"] = df["days_in_transfer"].apply(format_days_in_transfer)
        
    except Exception as e:
        logger.error(f"Error formatting display dataframe: {str(e)}")
    
    return df


def highlight_supply_issues_enhanced(row: pd.Series, date_columns: List[str]) -> List[str]:
    """Enhanced highlighting for supply issues with proper priority"""
    styles = [""] * len(row)
    
    try:
        # Priority order: expired > very urgent expiry > urgent expiry > past date > missing date
        
        # Check expiry status first (highest priority)
        if "days_until_expiry" in row.index:
            cell_value = str(row["days_until_expiry"])
            if "💀" in cell_value:  # Expired
                return ["background-color: #f8d7da"] * len(row)  # Light red
            elif "🔴" in cell_value:  # Very urgent (≤7 days)
                return ["background-color: #f8d7da"] * len(row)  # Light red
            elif "🟡" in cell_value:  # Urgent (≤30 days)
                return ["background-color: #fff3cd"] * len(row)  # Light yellow
        
        # Check transfer/arrival delays (medium priority)
        if "days_since_arrival" in row.index:
            cell_value = str(row["days_since_arrival"])
            if "🔴" in cell_value:  # Long delay
                return ["background-color: #ffe4e1"] * len(row)  # Light pink
        
        if "days_in_transfer" in row.index:
            cell_value = str(row["days_in_transfer"])
            if "🔴" in cell_value:  # Long transfer
                return ["background-color: #ffe4e1"] * len(row)  # Light pink
        
        # Check date issues in any date column (lower priority)
        has_past = False
        has_missing = False
        
        for date_col in date_columns:
            if date_col in row.index:
                cell_value = str(row[date_col])
                if "🔴" in cell_value and "(" in cell_value:  # Past date with days ago
                    has_past = True
                elif "❌" in cell_value:  # Missing date
                    has_missing = True
        
        # Apply highlighting based on what was found
        if has_past:
            return ["background-color: #ffe4e1"] * len(row)  # Light pink for past dates
        elif has_missing:
            return ["background-color: #fff3cd"] * len(row)  # Light yellow for missing dates
        
    except Exception as e:
        logger.error(f"Error highlighting rows: {str(e)}")
    
    return styles


# Replace show_supply_grouped_view in Supply Analysis with this version
# This uses the same helper functions as Demand Analysis for consistency

def show_supply_grouped_view(filtered_df: pd.DataFrame, start_date: date, 
                           end_date: date, use_adjusted_dates: bool = True):
    """Show grouped supply by period with proper date handling for each source"""
    st.markdown("### 📊 Grouped Supply by Product")
    st.markdown(f"📅 Period: **{start_date}** to **{end_date}**")
    
    if filtered_df.empty:
        DisplayComponents.show_no_data_message("No data to display")
        return
    
    try:
        # Display options
        display_options = DisplayComponents.render_display_options(
            page_name="supply_grouped",
            show_period=True,
            show_filters=["nonzero"]
        )
        
        period = display_options.get('period_type', 'Weekly')
        show_only_nonzero = display_options.get('show_only_nonzero', True)
        
        # Reset index to avoid duplicate issues
        df_summary = filtered_df.reset_index(drop=True).copy()
        
        # Create grouping date column
        df_summary["grouping_date"] = pd.NaT
        
        # Check if this is "All" view with unified dates
        if "unified_date" in df_summary.columns and df_summary["unified_date"].notna().any():
            # Use unified date for grouping in "All" view
            if use_adjusted_dates and "unified_date_adjusted" in df_summary.columns:
                df_summary["grouping_date"] = pd.to_datetime(df_summary["unified_date_adjusted"], errors='coerce')
            else:
                df_summary["grouping_date"] = pd.to_datetime(df_summary["unified_date"], errors='coerce')
            
            if debug_mode:
                non_null_count = df_summary["grouping_date"].notna().sum()
                st.write(f"🐛 Using unified dates for grouping: {non_null_count} non-null dates")
        else:
            # Process each source type using specific dates
            for source_type in df_summary['source_type'].unique():
                source_indices = df_summary.index[df_summary['source_type'] == source_type]
                
                if source_type == 'Inventory':
                    # Inventory is always available NOW
                    df_summary.loc[source_indices, "grouping_date"] = pd.Timestamp.now().normalize()
                                
                elif source_type == 'Pending CAN':
                    if use_adjusted_dates and 'arrival_date_adjusted' in df_summary.columns:
                        valid_indices = df_summary.index[
                            (df_summary['source_type'] == source_type) & 
                            df_summary['arrival_date_adjusted'].notna()
                        ]
                        if len(valid_indices) > 0:
                            df_summary.loc[valid_indices, "grouping_date"] = pd.to_datetime(
                                df_summary.loc[valid_indices, 'arrival_date_adjusted'], errors='coerce'
                            )
                    elif 'arrival_date' in df_summary.columns:
                        valid_indices = df_summary.index[
                            (df_summary['source_type'] == source_type) & 
                            df_summary['arrival_date'].notna()
                        ]
                        if len(valid_indices) > 0:
                            df_summary.loc[valid_indices, "grouping_date"] = pd.to_datetime(
                                df_summary.loc[valid_indices, 'arrival_date'], errors='coerce'
                            )
                        
                elif source_type == 'Pending PO':
                    if use_adjusted_dates and 'eta_adjusted' in df_summary.columns:
                        valid_indices = df_summary.index[
                            (df_summary['source_type'] == source_type) & 
                            df_summary['eta_adjusted'].notna()
                        ]
                        if len(valid_indices) > 0:
                            df_summary.loc[valid_indices, "grouping_date"] = pd.to_datetime(
                                df_summary.loc[valid_indices, 'eta_adjusted'], errors='coerce'
                            )
                    elif 'eta' in df_summary.columns:
                        valid_indices = df_summary.index[
                            (df_summary['source_type'] == source_type) & 
                            df_summary['eta'].notna()
                        ]
                        if len(valid_indices) > 0:
                            df_summary.loc[valid_indices, "grouping_date"] = pd.to_datetime(
                                df_summary.loc[valid_indices, 'eta'], errors='coerce'
                            )
                        
                elif source_type == 'Pending WH Transfer':
                    if use_adjusted_dates and 'transfer_date_adjusted' in df_summary.columns:
                        valid_indices = df_summary.index[
                            (df_summary['source_type'] == source_type) & 
                            df_summary['transfer_date_adjusted'].notna()
                        ]
                        if len(valid_indices) > 0:
                            df_summary.loc[valid_indices, "grouping_date"] = pd.to_datetime(
                                df_summary.loc[valid_indices, 'transfer_date_adjusted'], errors='coerce'
                            )
                    elif 'transfer_date' in df_summary.columns:
                        valid_indices = df_summary.index[
                            (df_summary['source_type'] == source_type) & 
                            df_summary['transfer_date'].notna()
                        ]
                        if len(valid_indices) > 0:
                            df_summary.loc[valid_indices, "grouping_date"] = pd.to_datetime(
                                df_summary.loc[valid_indices, 'transfer_date'], errors='coerce'
                            )
        
        # Filter out rows with missing grouping dates
        df_summary = df_summary[df_summary["grouping_date"].notna()].copy()
        
        if df_summary.empty:
            DisplayComponents.show_no_data_message("No data with valid dates for grouping")
            return
        
        # Create period column using helper function
        df_summary["period"] = convert_to_period(df_summary["grouping_date"], period)
        
        # Remove any rows where period conversion failed
        df_summary = df_summary[df_summary["period"].notna()]
        
        if debug_mode:
            debug_info = {
                "Records with periods": len(df_summary),
                "Unique periods": df_summary['period'].nunique(),
                "Period distribution by source": {
                    source: df_summary[df_summary['source_type'] == source]['period'].nunique()
                    for source in df_summary['source_type'].unique()
                },
                "Sample periods": df_summary['period'].value_counts().head(5).to_dict()
            }
            DisplayComponents.show_debug_info(debug_info)
        
        # Get unique source types
        source_types = df_summary["source_type"].unique() if "source_type" in df_summary.columns else ["All"]
        
        # Create tabs
        if len(source_types) > 1:
            tabs = st.tabs(["All"] + list(source_types))
            
            # All tab
            with tabs[0]:
                display_supply_pivot(df_summary, period, show_only_nonzero, "All Sources")
            
            # Individual source tabs
            for idx, source in enumerate(source_types):
                with tabs[idx + 1]:
                    source_df = df_summary[df_summary["source_type"] == source]
                    if not source_df.empty:
                        display_supply_pivot(source_df, period, show_only_nonzero, source)
                    else:
                        DisplayComponents.show_no_data_message(f"No data for {source}")
        else:
            # Single source
            display_supply_pivot(df_summary, period, show_only_nonzero, source_types[0] if source_types else "All Sources")
            
    except Exception as e:
        logger.error(f"Error showing grouped view: {str(e)}", exc_info=True)
        st.error(f"Error displaying grouped view: {str(e)}")
        
        # Show debug information on error
        if debug_mode:
            st.write("Debug - Error Details:")
            st.write(f"- DataFrame shape: {filtered_df.shape}")
            st.write(f"- DataFrame columns: {list(filtered_df.columns)}")
            st.write(f"- Source types: {filtered_df['source_type'].unique() if 'source_type' in filtered_df.columns else 'No source_type column'}")
            
            # Check date columns
            date_cols = [col for col in filtered_df.columns if any(x in col.lower() for x in ['date', 'eta', 'arrival', 'transfer'])]
            st.write(f"- Date columns found: {date_cols}")
            
            # Sample data
            if not filtered_df.empty:
                st.write("- Sample data (first 5 rows):")
                display_cols = ['source_type', 'pt_code'] + date_cols[:3]
                display_cols = [col for col in display_cols if col in filtered_df.columns]
                if display_cols:
                    st.dataframe(filtered_df[display_cols].head())

def display_supply_pivot(df_summary: pd.DataFrame, period: str, show_only_nonzero: bool, title: str):
    """Display supply pivot table with past period indicators - using helper functions"""
    if df_summary.empty:
        DisplayComponents.show_no_data_message(f"No data to display for {title}")
        return
    
    # Create pivot table using the SAME helper function as Demand Analysis
    pivot_df = create_period_pivot(
        df=df_summary,
        group_cols=["product_name", "pt_code"],
        period_col="period",
        value_col="quantity",
        agg_func="sum",
        period_type=period,
        show_only_nonzero=show_only_nonzero,
        fill_value=0
    )
    
    if pivot_df.empty:
        DisplayComponents.show_no_data_message(f"No data to display for {title}")
        return
    
    # Add past period indicators
    display_pivot = apply_period_indicators(
        df=pivot_df,
        period_type=period,
        exclude_cols=["product_name", "pt_code"],
        indicator="🔴"
    )
    
    # Format for display
    for col in display_pivot.columns:
        if col not in ["product_name", "pt_code"]:
            display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    
    # Show legend
    st.info("🔴 = Past period (already occurred)")
    
    # Display pivot
    st.dataframe(display_pivot, use_container_width=True, height=400)
    
    # Show totals with indicators
    show_supply_totals_with_indicators(df_summary, period)
    
    # Export button (without indicators)
    # Use the same format_pivot_for_display helper as Demand Analysis
    excel_pivot = format_pivot_for_display(pivot_df)
    DisplayComponents.show_export_button(
        excel_pivot, 
        f"supply_{title.lower().replace(' ', '_')}", 
        f"📤 Export {title} to Excel"
    )

def format_pivot_for_display(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Format pivot table for display - PRESERVE COLUMN ORDER"""
    display_pivot = pivot_df.copy()
    
    # Remember original column order
    original_columns = list(display_pivot.columns)
    
    # Format only numeric columns (skip first 2 columns: product_name and pt_code)
    for col in original_columns[2:]:  
        display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    
    # IMPORTANT: Return with original column order preserved
    return display_pivot[original_columns]

def show_supply_totals_with_indicators(df_summary: pd.DataFrame, period: str):
    """Show period totals for supply with past period indicators"""
    try:
        # Prepare data
        df_grouped = df_summary.copy()
        df_grouped["quantity"] = pd.to_numeric(df_grouped["quantity"], errors='coerce').fillna(0)
        df_grouped["value_in_usd"] = pd.to_numeric(df_grouped["value_in_usd"], errors='coerce').fillna(0)
        
        # Calculate aggregates
        qty_by_period = df_grouped.groupby("period")["quantity"].sum()
        val_by_period = df_grouped.groupby("period")["value_in_usd"].sum()
        
        # Create summary DataFrame
        summary_data = {"Metric": ["🔢 TOTAL QUANTITY", "💵 TOTAL VALUE (USD)"]}
        
        # Add all periods to summary_data with indicators
        for period_val in qty_by_period.index:
            col_name = f"🔴 {period_val}" if is_past_period(str(period_val), period) else str(period_val)
            summary_data[col_name] = [
                format_number(qty_by_period[period_val]),
                format_currency(val_by_period[period_val])
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

# === Main Page Logic ===
def main():
    """Main function for supply analysis page"""
   # DEBUG: Check if time adjustments are working
    if debug_mode:
        st.write("🐛 DEBUG - TimeAdjustmentIntegration Check:")
        from utils.adjustments.time_adjustment_integration import TimeAdjustmentIntegration
        tai = TimeAdjustmentIntegration()
        
        # Check mapping
        for source in ["Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"]:
            date_col = tai._get_date_column(source)
            st.write(f"- {source} → {date_col}")
    
    # Source selection and options
    supply_source, exclude_expired = select_supply_source()
    
    # Load and prepare data
    df_all = DisplayComponents.show_data_loading_spinner(
        load_and_prepare_supply_data,
        "Loading supply data...",
        supply_source, 
        exclude_expired, 
        use_adjusted_dates
    )
    
    if df_all.empty:
        DisplayComponents.show_no_data_message(
            "No supply data available for the selected source.",
            "Please check your data source and try again."
        )
        st.stop()
    
    # Show adjustment summary if in adjusted mode
    if use_adjusted_dates:
        # Identify date columns for supply
        date_columns = []
        
        # Check by source type
        if 'source_type' in df_all.columns:
            for source_type in df_all['source_type'].unique():
                date_map = {
                    'Inventory': 'date_ref',
                    'Pending CAN': 'arrival_date',
                    'Pending PO': 'eta',
                    'Pending WH Transfer': 'transfer_date'
                }
                
                if source_type in date_map:
                    date_col = date_map[source_type]
                    if date_col in df_all.columns and date_col not in date_columns:
                        date_columns.append(date_col)
        
        # Show summary if we have date columns
        if date_columns:
            DateModeComponent.show_adjustment_summary(df_all, date_columns, 'Supply')
    
    # Apply filters
    filtered_df, start_date, end_date, filter_params = apply_supply_filters(df_all, use_adjusted_dates)
    
    # Save to session state for other pages
    save_to_session_state('supply_analysis_data', filtered_df)
    save_to_session_state('supply_analysis_filters', {
        'source': supply_source,
        'exclude_expired': exclude_expired,
        'start_date': start_date,
        'end_date': end_date,
        'use_adjusted_dates': use_adjusted_dates
    })
    
    # Display sections
    show_supply_summary(filtered_df, filter_params, use_adjusted_dates)
    show_supply_detail_table(filtered_df, use_adjusted_dates)
    show_supply_grouped_view(filtered_df, start_date, end_date, use_adjusted_dates)
    
    # Additional Page Features
    st.markdown("---")
    
    # Quick Actions
    actions = [
        {
            "label": "📊 Go to GAP Analysis",
            "type": "primary",
            "page": "pages/3_📊_GAP_Analysis.py"
        },
        {
            "label": "📤 View Demand",
            "page": "pages/1_📤_Demand_Analysis.py"
        },
        {
            "label": "📈 View Dashboard",
            "page": "main.py"
        },
        {
            "label": "🔄 Refresh Data",
            "callback": lambda: (data_manager.clear_cache(), st.rerun())
        }
    ]
    
    DisplayComponents.show_action_buttons(actions)
    
    # Debug info panel
    if debug_mode:
        with st.expander("🐛 Debug Information", expanded=True):
            st.markdown("### Debug Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Shape:**")
                st.write(f"- Total rows: {len(df_all)}")
                st.write(f"- Filtered rows: {len(filtered_df)}")
                st.write(f"- Unique products: {filtered_df['pt_code'].nunique() if 'pt_code' in filtered_df.columns else 0}")
                
                if 'source_type' in filtered_df.columns:
                    st.write("\n**Source Distribution:**")
                    for source, count in filtered_df['source_type'].value_counts().items():
                        st.write(f"- {source}: {count}")
            
            with col2:
                st.write("**Date Columns Available:**")
                date_cols = [col for col in filtered_df.columns if any(x in col.lower() for x in 
                           ['date', 'arrival', 'eta', 'transfer'])]
                for col in date_cols:
                    st.write(f"- {col}")
                
                st.write("\n**Sample Data:**")
                if not filtered_df.empty:
                    display_cols = ['pt_code', 'source_type', 'quantity', 'date_ref']
                    display_cols = [col for col in display_cols if col in filtered_df.columns]
                    if display_cols:
                        st.write(filtered_df[display_cols].head())
    
    # Help section
    DisplayComponents.show_help_section(
        "How to use Supply Analysis",
        """
        ### Understanding Supply Data
        
        **Date Mode Selection:**
        - **Adjusted Dates**: Uses dates modified by time adjustment rules
        - **Original Dates**: Uses raw dates from source data
        - Each supply source may have different date adjustments
        
        **Date References by Source:**
        - **Inventory**: Default TODAY (current availability)
        - Can be adjusted for planning scenarios
        - Useful for simulating delayed availability
        - **Pending CAN**: arrival_date (when goods arrived)
        - **Pending PO**: eta (estimated time of arrival)
        - **Pending WH Transfer**: transfer_date (when transfer initiated)

        **Inventory Date Adjustments:**
        While inventory typically represents immediate availability (TODAY),
        adjustments can be useful for:
        - Scenario planning and what-if analysis
        - Modeling inventory constraints or holds
        - Planning for ownership transfers
        - Testing different GAP analysis scenarios
        
        **Date References by Source:**
        - Each source type uses its specific date column
        - Adjustments can be configured per source type
        - Date status comparison shows impact of adjustments
        
        **Visual Indicators:**
        - ❌ Missing Date: No date provided
        - 🔴 Past Date: Should have arrived/completed
        - 💀 Expired: Past expiry date
        - 🔴 Expiring ≤7 days: Critical expiry warning
        - 🟡 Expiring ≤30 days: Expiry warning
        
        **Filter Options:**
        - **Show All**: Complete view
        - **Show Missing Dates Only**: Data quality check
        - **Show Past Dates Only**: Overdue items
        - **Show Expiring Soon**: Items needing attention
        - **Show Expired Only**: Items to remove
        
        **Row Highlighting Priority:**
        1. Expired items (dark red)
        2. Critical expiry ≤7 days (red)
        3. Expiry warning ≤30 days (yellow)
        4. Past dates (light pink)
        5. Missing dates (light orange)
        
        **Tips:**
        - Configure date adjustments in Settings
        - Use tabs to analyze each source separately
        - Export grouped data for planning
        - Monitor expiry dates closely
        - Check delayed CAN and transfers
        """
    )
    
    # Footer
    st.markdown("---")
    st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run main function
if __name__ == "__main__":
    main()