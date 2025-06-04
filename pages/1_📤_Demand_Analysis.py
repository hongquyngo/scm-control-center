import streamlit as st
import pandas as pd
from datetime import datetime, date
import logging
from typing import Tuple, Dict, Any, Optional

# Import refactored modules
from utils.data_manager import DataManager
from utils.filters import FilterManager
from utils.display_components import DisplayComponents
from utils.formatters import format_number, format_currency, format_percentage
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
from utils.date_mode_component import DateModeComponent
from utils.smart_filter_manager import SmartFilterManager

# Configure logging
logger = logging.getLogger(__name__)

# === Constants ===
DEMAND_SOURCES = ["OC Only", "Forecast Only", "Both"]
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]

# === Page Config ===
st.set_page_config(
    page_title="Demand Analysis - SCM",
    page_icon="📤",
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
    debug_mode = st.checkbox("🐛 Debug Mode", value=False, key="demand_debug_mode")

if debug_mode:
    st.info("🐛 Debug Mode is ON - Additional information will be displayed")

# === Header with Navigation ===
DisplayComponents.show_page_header(
    title="Outbound Demand Analysis",
    icon="📤",
    prev_page=None,
    next_page="pages/2_📥_Supply_Analysis.py"
)

use_adjusted_dates = DateModeComponent.render_date_mode_selector("demand_")

# === Data Source Selection ===
def select_demand_source() -> str:
    """Allow user to choose demand data source"""
    source_config = DisplayComponents.render_source_selector(
        options=DEMAND_SOURCES,
        default_index=2,  # Default to "Both"
        radio_label="Select Outbound Demand Source:",
        key="demand_source_selector"
    )
    
    source = source_config['source']
    
    if debug_mode:
        st.write(f"🐛 Selected source: {source}")
    
    return source

# === Data Loading Functions ===
def load_and_prepare_demand_data(source: str, use_adjusted_dates: bool = True) -> pd.DataFrame:
    """Load and standardize demand data based on source selection"""
    try:
        # Convert source selection to list format for data_manager
        sources = []
        if source in ["OC Only", "Both"]:
            sources.append("OC")
        if source in ["Forecast Only", "Both"]:
            sources.append("Forecast")
        
        # Use data_manager to get demand data
        df = data_manager.get_demand_data(sources=sources, include_converted=True)
        
        if debug_mode and not df.empty:
            debug_info = {
                "Total rows": len(df),
                "Unique products": df['pt_code'].nunique() if 'pt_code' in df.columns else 0,
                "Data sources": df['source_type'].value_counts().to_dict() if 'source_type' in df.columns else {},
                "Date columns": [col for col in df.columns if 'etd' in col.lower()]
            }
            DisplayComponents.show_debug_info(debug_info)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading demand data: {str(e)}")
        st.error(f"Failed to load demand data: {str(e)}")
        return pd.DataFrame()

# === Filtering Functions ===
def apply_demand_filters(df: pd.DataFrame, use_adjusted_dates: bool = True) -> Tuple[pd.DataFrame, date, date]:
    """Apply filters to demand dataframe with mode toggle"""
    if df.empty:
        return df, date.today(), date.today()
    
    # Initialize filter manager
    filter_manager = SmartFilterManager(key_prefix="demand_")
    
    # Render toggle and get mode
    use_smart_filters = filter_manager.render_filter_toggle()
    
    if use_smart_filters:
        # Use existing smart filter logic
        return apply_smart_demand_filters(df, use_adjusted_dates, filter_manager)
    else:
        # Use standard filters
        return apply_standard_demand_filters(df, use_adjusted_dates)

def apply_standard_demand_filters(df: pd.DataFrame, use_adjusted_dates: bool = True) -> Tuple[pd.DataFrame, date, date]:
    """Apply standard independent filters"""
    with st.expander("📎 Filters", expanded=True):
        # Get appropriate date column
        date_column = DateModeComponent.get_date_column_for_display(
            df, 'etd', use_adjusted_dates
        )
        
        # Row 1: Entity, Customer, Product
        col1, col2, col3 = st.columns(3)
        
        filters = {}
        
        with col1:
            if 'legal_entity' in df.columns:
                entities = sorted(df["legal_entity"].dropna().unique())
                filters['entity'] = st.multiselect(
                    "Legal Entity", 
                    entities,
                    key="demand_entity_filter_std",
                    placeholder="All entities"
                )
        
        with col2:
            if 'customer' in df.columns:
                customers = sorted(df["customer"].dropna().unique())
                filters['customer'] = st.multiselect(
                    "Customer", 
                    customers,
                    key="demand_customer_filter_std",
                    placeholder="All customers"
                )
        
        with col3:
            if 'pt_code' in df.columns and 'product_name' in df.columns:
                # Create product options with PT Code + Name
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
                    key="demand_product_filter_std",
                    placeholder="All products"
                )
                
                # Extract PT codes from selection
                filters['product'] = [p.split(' - ')[0] for p in selected_products]
        
        # Row 2: Brand, Status, Source
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if 'brand' in df.columns:
                brands = sorted(df["brand"].dropna().unique())
                filters['brand'] = st.multiselect(
                    "Brand", 
                    brands,
                    key="demand_brand_filter_std",
                    placeholder="All brands"
                )
        
        with col5:
            if 'is_converted_to_oc' in df.columns:
                statuses = sorted(df["is_converted_to_oc"].dropna().astype(str).unique())
                filters['conversion_status'] = st.multiselect(
                    "Conversion Status",
                    statuses,
                    key="demand_conversion_filter_std",
                    placeholder="All statuses"
                )
        
        with col6:
            if 'source_type' in df.columns:
                sources = sorted(df["source_type"].unique())
                if len(sources) > 1:
                    filters['source_type'] = st.multiselect(
                        "Data Source",
                        sources,
                        key="demand_source_filter_std",
                        placeholder="All sources"
                    )
        
        # Date range filters
        st.markdown("#### 📅 Date Range")
        col_date1, col_date2 = st.columns(2)
        
        # Get date range
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
            min_date = dates.min().date() if len(dates) > 0 else date.today()
            max_date = dates.max().date() if len(dates) > 0 else date.today()
        else:
            min_date = max_date = date.today()
        
        with col_date1:
            start_date = st.date_input(
                f"From Date ({date_column.replace('_', ' ').title()})",
                value=min_date,
                key="demand_start_date_std"
            )
        
        with col_date2:
            end_date = st.date_input(
                f"To Date ({date_column.replace('_', ' ').title()})",
                value=max_date,
                key="demand_end_date_std"
            )
        
        # Show active filters summary
        active_filters = sum(1 for v in filters.values() if v and v != [])
        if active_filters > 0:
            st.success(f"🔍 {active_filters} filters active")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply selection filters
    if filters.get('entity'):
        filtered_df = filtered_df[filtered_df['legal_entity'].isin(filters['entity'])]
    
    if filters.get('customer'):
        filtered_df = filtered_df[filtered_df['customer'].isin(filters['customer'])]
    
    if filters.get('product'):
        filtered_df = filtered_df[filtered_df['pt_code'].isin(filters['product'])]
    
    if filters.get('brand'):
        filtered_df = filtered_df[filtered_df['brand'].isin(filters['brand'])]
    
    if filters.get('conversion_status'):
        filtered_df = filtered_df[filtered_df['is_converted_to_oc'].astype(str).isin(filters['conversion_status'])]
    
    if filters.get('source_type'):
        filtered_df = filtered_df[filtered_df['source_type'].isin(filters['source_type'])]
    
    # Apply date filter
    if date_column in filtered_df.columns:
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors='coerce')
        filtered_df = filtered_df[
            (filtered_df[date_column].isna()) |
            ((filtered_df[date_column] >= pd.to_datetime(start_date)) & 
             (filtered_df[date_column] <= pd.to_datetime(end_date)))
        ]
    
    # Show filtering result
    if len(filtered_df) < len(df):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"🔍 Filtered: {len(df):,} → {len(filtered_df):,} records")
        with col2:
            retention = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
            st.metric("Retention", f"{retention:.1f}%")
    
    return filtered_df, start_date, end_date

def apply_smart_demand_filters(df: pd.DataFrame, use_adjusted_dates: bool, 
                              filter_manager: SmartFilterManager) -> Tuple[pd.DataFrame, date, date]:
    """Apply smart interactive filters to demand dataframe"""
    try:
        # Get appropriate date column based on mode
        date_column = DateModeComponent.get_date_column_for_display(
            df, 'etd', use_adjusted_dates
        )
        
        # Validate date column exists
        if date_column not in df.columns:
            st.warning(f"Date column '{date_column}' not found. Using 'etd' as fallback.")
            date_column = 'etd' if 'etd' in df.columns else df.columns[0]
        
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
        
        # Customer filter (only for demand data)
        if 'customer' in df.columns:
            filter_config['customer_selection'] = {
                'column': 'customer',
                'label': 'Customer',
                'help': 'Filter by customers',
                'placeholder': 'Choose customers...'
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
        
        # Conversion status filter (for forecasts)
        if 'is_converted_to_oc' in df.columns:
            filter_config['status_selection'] = {
                'column': 'is_converted_to_oc',
                'label': 'Conversion Status',
                'help': 'Filter by forecast conversion status',
                'placeholder': 'Choose conversion status...'
            }
        
        # Source type filter (if mixed OC and Forecast)
        if 'source_type' in df.columns:
            unique_sources = df['source_type'].unique()
            if len(unique_sources) > 1:
                filter_config['source_selection'] = {
                    'column': 'source_type',
                    'label': 'Data Source',
                    'help': 'Filter by OC or Forecast',
                    'placeholder': 'Choose data source...'
                }
        
        # Render smart filters
        with st.container():
            # Show filter header with summary
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("### 📎 Smart Filters")
            with col2:
                st.caption(f"📊 {len(df):,} total records")
            
            # Render the smart filter interface
            filters_result = filter_manager.render_smart_filters(
                df=df,
                filter_config=filter_config,
                show_date_filters=True,
                date_column=date_column
            )
        
        # Validate filter result
        if not isinstance(filters_result, dict):
            raise ValueError(f"Invalid filter result type: {type(filters_result)}")
        
        # Extract components from filter result
        selections = filters_result.get('selections', {})
        date_filters = filters_result.get('date_filters', {})
        
        # Apply filters to dataframe
        filtered_df = filter_manager.apply_filters_to_dataframe(df, filters_result)
        
        # Get date range
        start_date = date_filters.get('start_date', date.today())
        end_date = date_filters.get('end_date', date.today())
        
        # Validate dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Show filter summary
        if filtered_df.shape[0] < df.shape[0]:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.info(f"🔍 Filtered: {len(df):,} → {len(filtered_df):,} records")
            with col2:
                # Count active filters
                active_count = sum(1 for v in selections.values() if v)
                if active_count > 0:
                    st.success(f"✅ {active_count} filters active")
            with col3:
                # Percentage retained
                retention = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
                st.metric("Retained", f"{retention:.1f}%")
        
        return filtered_df, start_date, end_date
        
    except Exception as e:
        # Log the full error
        logger.error(f"Smart filter error: {str(e)}", exc_info=True)
        
        # Show user-friendly error message
        st.error(f"⚠️ Smart filters encountered an error: {str(e)}")
        st.info("💡 Please switch to Standard Filters mode")
        
        # Return original data
        return df, date.today(), date.today()

# === Display Functions ===
def show_demand_summary(filtered_df: pd.DataFrame, use_adjusted_dates: bool = True):
    """Show demand summary metrics - backlog information only"""
    if filtered_df.empty:
        DisplayComponents.show_no_data_message("No demand data to display")
        return
    
    try:
        # Calculate main metrics
        metrics = [
            {
                "title": "Total Backlog Customers",
                "value": filtered_df["customer"].nunique() if "customer" in filtered_df.columns else 0,
                "format_type": "number"
            },
            {
                "title": "Total Backlog OC",
                "value": filtered_df[filtered_df["source_type"] == "OC"]["demand_number"].nunique() 
                        if "source_type" in filtered_df.columns and "demand_number" in filtered_df.columns else 0,
                "format_type": "number"
            },
            {
                "title": "Total Backlog Products",
                "value": filtered_df["pt_code"].nunique() if "pt_code" in filtered_df.columns else 0,
                "format_type": "number"
            },
            {
                "title": "Total Pipeline Value (USD)",
                "value": filtered_df["value_in_usd"].sum() if "value_in_usd" in filtered_df.columns else 0,
                "format_type": "currency"
            }
        ]
        
        # Render summary section with additional content
        DisplayComponents.render_summary_section(
            metrics=metrics,
            title="### 📊 Demand Summary",
            additional_content=lambda: show_additional_summaries(filtered_df, use_adjusted_dates)
        )
            
    except Exception as e:
        logger.error(f"Error showing demand summary: {str(e)}")
        st.error(f"Error displaying summary: {str(e)}")

def show_additional_summaries(filtered_df: pd.DataFrame, use_adjusted_dates: bool):
    """Show ETD comparison and forecast conversion summaries"""
    # Show ETD Status Comparison
    show_etd_status_comparison(filtered_df, use_adjusted_dates)
    
    # Conversion status for forecasts
    if "source_type" in filtered_df.columns and filtered_df["source_type"].str.contains("Forecast").any():
        show_forecast_conversion_summary(filtered_df)

def show_etd_status_comparison(filtered_df: pd.DataFrame, use_adjusted_dates: bool = True):
    """Show ETD status comparison between original and adjusted"""
    # Only show if we have both original and adjusted columns
    if 'etd_original' not in filtered_df.columns or 'etd_adjusted' not in filtered_df.columns:
        return
    
    st.markdown("#### 📈 ETD Status Comparison")
    
    today = pd.Timestamp.now().normalize()
    
    # Calculate missing and past for both
    original_series = pd.to_datetime(filtered_df['etd_original'], errors='coerce')
    adjusted_series = pd.to_datetime(filtered_df['etd_adjusted'], errors='coerce')
    
    missing_original = original_series.isna().sum()
    past_original = (original_series < today).sum()
    
    missing_adjusted = adjusted_series.isna().sum()
    past_adjusted = (adjusted_series < today).sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Original ETD**")
        st.metric("Missing", missing_original)
        st.metric("Past", past_original)
    
    with col2:
        st.success("**Adjusted ETD**")
        st.metric("Missing", missing_adjusted)
        st.metric("Past", past_adjusted)
    
    # Show improvement/degradation
    if missing_original != missing_adjusted or past_original != past_adjusted:
        st.caption("📊 Impact of adjustments:")
        if missing_adjusted < missing_original:
            st.success(f"✅ {missing_original - missing_adjusted} missing ETDs resolved by adjustments")
        if past_adjusted != past_original:
            diff = past_adjusted - past_original
            if diff > 0:
                st.warning(f"⚠️ {diff} more records moved to past due to adjustments")
            else:
                st.info(f"ℹ️ {abs(diff)} records moved from past to future due to adjustments")

def show_forecast_conversion_summary(filtered_df: pd.DataFrame):
    """Show forecast conversion statistics"""
    try:
        forecast_df = filtered_df[filtered_df["source_type"] == "Forecast"]
        
        if not forecast_df.empty and 'is_converted_to_oc' in forecast_df.columns:
            st.markdown("#### 📈 Forecast Conversion Status")
            
            total_forecast = len(forecast_df)
            converted = len(forecast_df[forecast_df["is_converted_to_oc"] == "Yes"])
            not_converted = len(forecast_df[forecast_df["is_converted_to_oc"] == "No"])
            conversion_rate = (converted / total_forecast * 100) if total_forecast > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Forecast Lines", f"{total_forecast:,}")
            with col2:
                st.metric("Converted to OC", f"{converted:,} ({conversion_rate:.1f}%)")
            with col3:
                st.metric("Not Converted", f"{not_converted:,}")
                
    except Exception as e:
        logger.error(f"Error showing forecast conversion summary: {str(e)}")

def show_demand_detail_table(filtered_df: pd.DataFrame, use_adjusted_dates: bool = True):
    """Show detailed demand table with enhanced filtering and highlighting"""
    st.markdown("### 🔍 Demand Details")
    
    if filtered_df.empty:
        DisplayComponents.show_no_data_message("No data to display")
        return
    
    try:
        # Define filter options
        filter_options = [
            "Show All",
            "Show Missing Original ETD Only", 
            "Show Past Original ETD Only",
            "Show Missing Adjusted ETD Only",
            "Show Past Adjusted ETD Only"
        ]
        
        # Get appropriate date columns
        date_columns_added = []
        if 'etd_original' in filtered_df.columns:
            date_columns_added.append('etd_original')
        elif 'etd' in filtered_df.columns and 'etd_adjusted' not in filtered_df.columns:
            date_columns_added.append('etd')
        
        if 'etd_adjusted' in filtered_df.columns:
            date_columns_added.append('etd_adjusted')
        
        # Use DisplayComponents for detail table
        DisplayComponents.render_detail_table_with_filter(
            df=filtered_df,
            filter_options=filter_options,
            filter_apply_func=lambda df, opt: apply_detail_filter(df, opt, use_adjusted_dates),
            format_func=lambda df: format_demand_display_df_enhanced(df, date_columns_added),
            style_func=lambda row: highlight_etd_issues_enhanced(row, date_columns_added),
            height=600,
            key_prefix="demand_detail"
        )
        
    except Exception as e:
        logger.error(f"Error showing demand detail table: {str(e)}")
        st.error(f"Error displaying table: {str(e)}")

def apply_detail_filter(display_df: pd.DataFrame, filter_option: str, 
                       use_adjusted_dates: bool) -> pd.DataFrame:
    """Apply filter to detail dataframe"""
    if filter_option == "Show All":
        return prepare_detail_display(display_df, use_adjusted_dates)
    
    today = pd.Timestamp.now().normalize()
    
    # Apply selected filter
    if filter_option == "Show Missing Original ETD Only":
        filter_column = 'etd_original' if 'etd_original' in display_df.columns else 'etd'
        if filter_column in display_df.columns:
            display_df[filter_column] = pd.to_datetime(display_df[filter_column], errors='coerce')
            display_df = display_df[display_df[filter_column].isna()]
    
    elif filter_option == "Show Past Original ETD Only":
        filter_column = 'etd_original' if 'etd_original' in display_df.columns else 'etd'
        if filter_column in display_df.columns:
            display_df[filter_column] = pd.to_datetime(display_df[filter_column], errors='coerce')
            display_df = display_df[
                display_df[filter_column].notna() & 
                (display_df[filter_column] < today)
            ]
    
    elif filter_option == "Show Missing Adjusted ETD Only":
        if 'etd_adjusted' in display_df.columns:
            display_df['etd_adjusted'] = pd.to_datetime(display_df['etd_adjusted'], errors='coerce')
            display_df = display_df[display_df['etd_adjusted'].isna()]
        else:
            st.warning("Adjusted ETD column not available")
            return pd.DataFrame()
            
    elif filter_option == "Show Past Adjusted ETD Only":
        if 'etd_adjusted' in display_df.columns:
            display_df['etd_adjusted'] = pd.to_datetime(display_df['etd_adjusted'], errors='coerce')
            display_df = display_df[
                display_df['etd_adjusted'].notna() & 
                (display_df['etd_adjusted'] < today)
            ]
        else:
            st.warning("Adjusted ETD column not available")
            return pd.DataFrame()
    
    return prepare_detail_display(display_df, use_adjusted_dates)

def prepare_detail_display(display_df: pd.DataFrame, use_adjusted_dates: bool) -> pd.DataFrame:
    """Prepare dataframe for detail display"""
    if display_df.empty:
        return display_df
    
    # Prepare display columns
    base_columns = [
        "pt_code", "product_name", "brand", "package_size", "standard_uom"
    ]
    
    # Add date columns based on what's available
    date_columns = []
    if 'etd_original' in display_df.columns:
        base_columns.append('etd_original')
        date_columns.append('etd_original')
    elif 'etd' in display_df.columns and 'etd_adjusted' not in display_df.columns:
        base_columns.append('etd')
        date_columns.append('etd')
    
    if 'etd_adjusted' in display_df.columns:
        base_columns.append('etd_adjusted')
        date_columns.append('etd_adjusted')
    
    # Add remaining columns
    remaining_columns = [
        "demand_quantity", "value_in_usd", "source_type", "demand_number", 
        "customer", "legal_entity"
    ]
    base_columns.extend(remaining_columns)
    
    if 'is_converted_to_oc' in display_df.columns:
        base_columns.append("is_converted_to_oc")
    
    # Filter to only existing columns
    display_columns = [col for col in base_columns if col in display_df.columns]
    display_df = display_df[display_columns].copy()
    
    # Sort by appropriate date column
    sort_column = None
    if use_adjusted_dates and 'etd_adjusted' in display_df.columns:
        sort_column = 'etd_adjusted'
    elif not use_adjusted_dates and 'etd_original' in display_df.columns:
        sort_column = 'etd_original'
    elif 'etd' in display_df.columns:
        sort_column = 'etd'
    
    # Create sort key for proper ordering
    if sort_column and sort_column in display_df.columns:
        today = pd.Timestamp.now().normalize()
        display_df["etd_sort_key"] = display_df[sort_column].apply(
            lambda x: (0, pd.NaT) if pd.isna(x) else 
                      (1, x) if x < today else 
                      (2, x)
        )
        
        # Sort: Missing ETD first, then Past ETD, then Future ETD
        display_df = display_df.sort_values(["etd_sort_key"], key=lambda x: [(v[0], v[1]) for v in x])
        display_df = display_df.drop("etd_sort_key", axis=1)
    
    return display_df

def format_demand_display_df_enhanced(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """Format dataframe columns for display with multiple date columns"""
    df = df.copy()
    today = pd.Timestamp.now().normalize()
    
    try:
        # Format numeric columns
        if "demand_quantity" in df.columns:
            df["demand_quantity"] = df["demand_quantity"].apply(lambda x: format_number(x))
        if "value_in_usd" in df.columns:
            df["value_in_usd"] = df["value_in_usd"].apply(lambda x: format_currency(x))
        
        # Enhanced ETD formatting for all date columns
        def format_etd(x):
            if pd.isna(x):
                return "❌ Missing"
            elif isinstance(x, pd.Timestamp) and x < today:
                return f"🔴 {x.strftime('%Y-%m-%d')}"
            elif isinstance(x, pd.Timestamp):
                return x.strftime("%Y-%m-%d")
            else:
                return str(x)
        
        # Apply formatting to all date columns
        for date_col in date_columns:
            if date_col in df.columns:
                df[date_col] = df[date_col].apply(format_etd)
                
    except Exception as e:
        logger.error(f"Error formatting display dataframe: {str(e)}")
    
    return df

def highlight_etd_issues_enhanced(row: pd.Series, date_columns: list) -> list:
    """Highlight rows based on ETD issues across multiple date columns"""
    styles = [""] * len(row)
    
    try:
        # Check if any date column has issues
        has_missing = False
        has_past = False
        
        for date_col in date_columns:
            if date_col in row.index:
                cell_value = str(row[date_col])
                if "❌ Missing" in cell_value:
                    has_missing = True
                elif "🔴" in cell_value:
                    has_past = True
        
        # Apply highlighting based on priority: missing > past > normal
        if has_missing:
            # Light orange/yellow background for missing ETD
            styles = ["background-color: #fff3cd"] * len(row)
        elif has_past:
            # Light red background for past ETD
            styles = ["background-color: #f8d7da"] * len(row)
            
    except Exception as e:
        logger.error(f"Error highlighting rows: {str(e)}")
    
    return styles

def show_demand_grouped_view(filtered_df: pd.DataFrame, start_date: date, 
                           end_date: date, use_adjusted_dates: bool = True):
    """Show grouped demand by period with past period indicators"""
    st.markdown("### 📦 Grouped Demand by Product")
    st.markdown(f"📅 Period: **{start_date}** to **{end_date}**")
    
    if filtered_df.empty:
        DisplayComponents.show_no_data_message("No data to display")
        return
    
    try:
        # Get appropriate ETD column
        etd_column = DateModeComponent.get_date_column_for_display(
            filtered_df, 'etd', use_adjusted_dates
        )
        
        # Validate column exists
        if etd_column not in filtered_df.columns:
            st.error(f"Date column '{etd_column}' not found")
            return
        
        # Display options
        display_options = DisplayComponents.render_display_options(
            page_name="demand_grouped",
            show_period=True,
            show_filters=["nonzero"]
        )
        
        period = display_options.get('period_type', 'Weekly')
        show_only_nonzero = display_options.get('show_only_nonzero', True)
        
        # Filter out missing dates for grouping
        df_summary = filtered_df[filtered_df[etd_column].notna()].copy()
        
        if df_summary.empty:
            DisplayComponents.show_no_data_message("No data with valid ETD dates for grouping")
            return
        
        # Create period column using appropriate date
        df_summary["period"] = convert_to_period(df_summary[etd_column], period)
        
        if debug_mode:
            DisplayComponents.show_debug_info({
                "Sample periods created": df_summary[[etd_column, "period"]].head(10).to_dict()
            })
        
        # Create pivot table
        pivot_df = create_period_pivot(
            df=df_summary,
            group_cols=["product_name", "pt_code"],
            period_col="period",
            value_col="demand_quantity",
            agg_func="sum",
            period_type=period,
            show_only_nonzero=show_only_nonzero,
            fill_value=0
        )
        
        if pivot_df.empty:
            DisplayComponents.show_no_data_message("No data to display after grouping")
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
        
        # Show totals with past period indicators
        show_demand_totals_with_indicators(df_summary, period)
        
        # Export button (without indicators)
        excel_pivot = format_pivot_for_display(pivot_df)
        DisplayComponents.show_export_button(excel_pivot, "grouped_demand", "📤 Export to Excel")
        
    except Exception as e:
        logger.error(f"Error showing grouped view: {str(e)}")
        st.error(f"Error displaying grouped view: {str(e)}")

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

def show_demand_totals_with_indicators(df_summary: pd.DataFrame, period: str):
    """Show period totals for demand with past period indicators"""
    try:
        # Prepare data
        df_grouped = df_summary.copy()
        df_grouped["demand_quantity"] = pd.to_numeric(df_grouped["demand_quantity"], errors='coerce').fillna(0)
        df_grouped["value_in_usd"] = pd.to_numeric(df_grouped["value_in_usd"], errors='coerce').fillna(0)
        
        # Calculate aggregates
        qty_by_period = df_grouped.groupby("period")["demand_quantity"].sum()
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
        
        # Sort columns (need to handle renamed columns)
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
                return clean_name
        
        sorted_period_cols = sorted(period_cols, key=get_sort_key)
        display_final = display_final[metric_cols + sorted_period_cols]
        
        st.markdown("#### 🔢 Column Totals")
        st.dataframe(display_final, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error showing totals: {str(e)}")

# === Main Page Logic ===
def main():
    
    # Source selection
    source = select_demand_source()
    
    # Load and prepare data
    df_all = DisplayComponents.show_data_loading_spinner(
        load_and_prepare_demand_data,
        "Loading demand data...",
        source, 
        use_adjusted_dates
    )
    
    if df_all.empty:
        DisplayComponents.show_no_data_message(
            "No outbound demand data available.",
            "Please check your data source and try again."
        )
        st.stop()
    
    # Show adjustment summary if in adjusted mode
    if use_adjusted_dates:
        DateModeComponent.show_adjustment_summary(df_all, ['etd'], 'Demand')
    
    # Apply filters with use_adjusted_dates parameter
    filtered_df, start_date, end_date = apply_demand_filters(df_all, use_adjusted_dates)
    
    # Save to session state for other pages
    save_to_session_state('demand_analysis_data', filtered_df)
    save_to_session_state('demand_analysis_filters', {
        'source': source,
        'start_date': start_date,
        'end_date': end_date,
        'use_adjusted_dates': use_adjusted_dates
    })
    
    # Display sections with use_adjusted_dates parameter
    show_demand_summary(filtered_df, use_adjusted_dates)
    show_demand_detail_table(filtered_df, use_adjusted_dates)
    show_demand_grouped_view(filtered_df, start_date, end_date, use_adjusted_dates)
    
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
            "label": "📥 View Supply",
            "page": "pages/2_📥_Supply_Analysis.py"
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
                
                # Show date column being used
                etd_col = DateModeComponent.get_date_column_for_display(
                    filtered_df, 'etd', use_adjusted_dates
                )
                if etd_col in filtered_df.columns:
                    st.write(f"- Using date column: {etd_col}")
                    valid_dates = pd.to_datetime(filtered_df[etd_col], errors='coerce').dropna()
                    if len(valid_dates) > 0:
                        st.write(f"- Date range: {valid_dates.min()} to {valid_dates.max()}")
            
            with col2:
                st.write("**Sample Data:**")
                etd_col = DateModeComponent.get_date_column_for_display(
                    filtered_df, 'etd', use_adjusted_dates
                )
                display_cols = ['pt_code', etd_col, 'demand_quantity']
                display_cols = [col for col in display_cols if col in filtered_df.columns]
                if display_cols and not filtered_df.empty:
                    st.write(filtered_df[display_cols].head())
    
    # Help section
    DisplayComponents.show_help_section(
        "How to use Demand Analysis",
        """
        ### Understanding Demand Data
        
        **Date Mode Selection:**
        - **Adjusted Dates**: Uses dates that have been modified by time adjustment rules
        - **Original Dates**: Uses the raw dates from the source data
        - Toggle between modes using the selector at the top of the page
        
        **Data Sources:**
        - **OC (Order Confirmation)**: Confirmed customer orders
        - **Forecast**: Customer demand predictions
        - **Both**: Combined view (watch for duplicates if forecast is converted to OC!)
        
        **Key Metrics:**
        - **Total Backlog Customers**: Number of unique customers with pending orders
        - **Total Backlog OC**: Number of unique order confirmations
        - **Total Backlog Products**: Number of unique products in demand
        - **Total Pipeline Value**: Total value of all pending orders
        
        **ETD Status Comparison:**
        - Shows missing and past ETD counts for both original and adjusted dates
        - Helps understand the impact of time adjustments
        - Only appears when time adjustments are configured
        
        **Filter Options in Details:**
        - **Show All**: Display all records
        - **Show Missing Original/Adjusted ETD Only**: Focus on data quality issues
        - **Show Past Original/Adjusted ETD Only**: Identify overdue deliveries
        
        **Visual Indicators:**
        - ❌ Missing ETD: No date provided (needs attention)
        - 🔴 Past ETD: Overdue for delivery
        - 🔴 Past Period: In grouped view, indicates periods that have passed
        
        **Tips:**
        - Configure time adjustments to see their impact on ETD dates
        - Use radio filters to focus on specific data issues
        - Export grouped data for planning meetings
        - Monitor forecast conversion rates
        
        **Common Actions:**
        1. Review missing ETD dates
        2. Follow up on past ETD orders
        3. Analyze demand patterns by period
        4. Check forecast accuracy
        5. Export data for further analysis
        """
    )
    
    # Footer
    st.markdown("---")
    st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run main function
if __name__ == "__main__":
    main()