import streamlit as st
import pandas as pd
from datetime import datetime

# Import refactored modules
from utils.data_manager import DataManager
from utils.filters import FilterManager
from utils.display_components import DisplayComponents
from utils.formatters import format_number, format_currency, check_missing_dates
from utils.helpers import (
    convert_df_to_excel,
    convert_to_period,
    sort_period_columns,
    save_to_session_state,
    is_past_period,
    parse_week_period,
    parse_month_period
)
from utils.session_state import initialize_session_state

# === Page Config ===
st.set_page_config(
    page_title="Supply Analysis - SCM",
    page_icon="📥",
    layout="wide"
)

# === Initialize Session State ===
initialize_session_state()

# === Constants ===
SUPPLY_SOURCES = ["Inventory Only", "Pending CAN Only", "Pending PO Only", "Pending WH Transfer Only", "All"]
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]

# === Initialize Components ===
@st.cache_resource
def get_data_manager():
    return DataManager()

data_manager = get_data_manager()

# === Header with Navigation ===
DisplayComponents.show_page_header(
    title="Inbound Supply Analysis",
    icon="📥",
    prev_page="pages/1_📤_Demand_Analysis.py",
    next_page="pages/3_📊_GAP_Analysis.py"
)

# === Data Source Selection ===
def select_supply_source():
    """Allow user to choose supply data source"""
    col1, col2 = st.columns([3, 1])
    with col1:
        source = st.radio(
            "Select Supply Source:",
            SUPPLY_SOURCES,
            index=4,  # Default to "All" (now index 4)
            horizontal=True,
            key="supply_source_radio"
        )
    with col2:
        exclude_expired = st.checkbox(
            "📅 Exclude expired inventory", 
            value=True, 
            key="supply_expired_checkbox",
            help="Hide inventory items that have passed their expiry date"
        )
    return source, exclude_expired

# === Data Loading Functions ===
def load_and_prepare_supply_data(supply_source, exclude_expired=True):
    """Load and standardize supply data based on source selection"""
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
    
    return df

# === Filtering Functions ===
def apply_supply_filters(df):
    """Apply filters to supply dataframe with enhanced product search"""
    with st.expander("📎 Filters", expanded=True):
        # Row 1: Entity, Brand, Product
        col1, col2, col3 = st.columns(3)
        
        filters = {}
        
        with col1:
            if 'legal_entity' in df.columns:
                entities = df["legal_entity"].dropna().unique().tolist()
                filters['entity'] = st.multiselect(
                    "Legal Entity", 
                    sorted(entities), 
                    key="supply_legal_entity"
                )
        
        with col2:
            if 'brand' in df.columns:
                brands = df["brand"].dropna().unique().tolist()
                filters['brand'] = st.multiselect(
                    "Brand", 
                    sorted(brands), 
                    key="supply_brand"
                )
        
        with col3:
            # Use FilterManager for product filter
            filters['product'] = FilterManager.create_product_filter(df, "supply_")
        
        # Row 2: Source type, Date range
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if 'source_type' in df.columns:
                source_types = df["source_type"].unique().tolist()
                filters['source_type'] = st.multiselect(
                    "Source Type",
                    source_types,
                    default=source_types,
                    key="supply_source_type_filter"
                )
        
        # Date range
        with col5:
            if 'date_ref' in df.columns and df['date_ref'].notna().any():
                default_start = df["date_ref"].min().date()
            else:
                default_start = datetime.today().date()
            start_date = st.date_input("From Date (Reference)", default_start, key="supply_start_date")
        
        with col6:
            if 'date_ref' in df.columns and df['date_ref'].notna().any():
                default_end = df["date_ref"].max().date()
            else:
                default_end = datetime.today().date()
            end_date = st.date_input("To Date (Reference)", default_end, key="supply_end_date")
        
        filters['start_date'] = start_date
        filters['end_date'] = end_date
        
        # Row 3: Additional filters based on source
        filter_params = {}
        if "days_until_expiry" in df.columns:
            col7, col8, col9 = st.columns(3)
            with col7:
                expiry_warning_days = st.number_input(
                    "Show items expiring within (days)",
                    min_value=0,
                    max_value=365,
                    value=30,
                    key="expiry_warning_days"
                )
                filter_params['expiry_warning_days'] = expiry_warning_days
    
    # Apply filters using FilterManager
    filtered_df = FilterManager.apply_filters(df, filters, date_column="date_ref")
    
    # Apply source type filter if different from default
    if 'source_type' in filters and filters['source_type'] and len(filters['source_type']) < len(df["source_type"].unique()):
        filtered_df = filtered_df[filtered_df["source_type"].isin(filters['source_type'])]
    
    return filtered_df, start_date, end_date, filter_params

# === Display Functions ===
def show_supply_summary(filtered_df, filter_params):
    """Show supply summary metrics"""
    st.markdown("### 📊 Supply Summary")
    
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
        },
        {
            "title": "⚠️ Missing Dates",
            "value": filtered_df["date_ref"].isna().sum(),
            "format_type": "number",
            "delta": "records" if filtered_df["date_ref"].isna().sum() > 0 else None,
            "delta_color": "inverse"
        }
    ]
    
    DisplayComponents.show_summary_metrics(metrics)
    
    # Source breakdown
    st.markdown("#### 📦 Supply by Source")
    
    # Create columns for each source type
    if not source_summary.empty:
        source_cols = st.columns(len(source_summary))
        for idx, (col, row) in enumerate(zip(source_cols, source_summary.itertuples())):
            with col:
                st.markdown(f"**{row.source_type}**")
                st.metric("Products", f"{row.pt_code:,}", label_visibility="collapsed")
                st.metric("Quantity", format_number(row.quantity), label_visibility="collapsed")
                st.metric("Value", format_currency(row.value_in_usd, "USD", 0), label_visibility="collapsed")
    
    # Special warnings
    show_supply_warnings(filtered_df, filter_params)

def show_supply_warnings(filtered_df, filter_params):
    """Show supply-specific warnings"""
    warning_shown = False
    
    # Expiry warning for inventory and WH transfer
    if "days_until_expiry" in filtered_df.columns and filter_params.get('expiry_warning_days'):
        expiry_warning_days = filter_params['expiry_warning_days']
        expiring_soon = filtered_df[
            (filtered_df["source_type"].isin(["Inventory", "Pending WH Transfer"])) & 
            (filtered_df["days_until_expiry"] <= expiry_warning_days) &
            (filtered_df["days_until_expiry"] >= 0)
        ]
        
        if not expiring_soon.empty:
            st.warning(f"⚠️ {len(expiring_soon)} items expiring within {expiry_warning_days} days!")
            warning_shown = True
    
    # Delayed CAN warning
    if "days_since_arrival" in filtered_df.columns:
        delayed_cans = filtered_df[
            (filtered_df["source_type"] == "Pending CAN") & 
            (filtered_df["days_since_arrival"] > 7)
        ]
        
        if not delayed_cans.empty:
            st.warning(f"⚠️ {len(delayed_cans)} CAN items pending stock-in for more than 7 days!")
            warning_shown = True
    
    # Long transfer warning
    if "days_in_transfer" in filtered_df.columns:
        long_transfers = filtered_df[
            (filtered_df["source_type"] == "Pending WH Transfer") & 
            (filtered_df["days_in_transfer"] > 3)
        ]
        
        if not long_transfers.empty:
            st.warning(f"⚠️ {len(long_transfers)} warehouse transfers taking more than 3 days!")
            warning_shown = True
    
    if not warning_shown:
        st.success("✅ No critical supply issues detected")

def show_supply_detail_table(filtered_df):
    """Show detailed supply table with All tab as default"""
    st.markdown("### 🔍 Supply Details")
    
    # Tab view for different source types
    source_types = filtered_df["source_type"].unique()
    if len(source_types) > 1:
        # Create tabs with "All" first
        tabs = st.tabs(["All"] + list(source_types))
        
        # All tab (now first)
        with tabs[0]:
            display_source_table(filtered_df, "All")
        
        # Individual source tabs
        for idx, source in enumerate(source_types):
            with tabs[idx + 1]:
                source_df = filtered_df[filtered_df["source_type"] == source]
                display_source_table(source_df, source)
    else:
        display_source_table(filtered_df, source_types[0] if len(source_types) > 0 else "All")

def display_source_table(df, source_type):
    """Display table for specific source type"""
    if df.empty:
        st.info(f"No {source_type} data available")
        return
    
    # Prepare display columns based on source type
    base_columns = [
        "pt_code", "product_name", "brand", "package_size", "standard_uom",
        "date_ref", "quantity", "value_in_usd", "legal_entity"
    ]
    
    # Add source-specific columns
    additional_columns = []
    
    if source_type == "Inventory" and "days_until_expiry" in df.columns:
        additional_columns = ["expiry_date", "days_until_expiry"]
    elif source_type == "Pending CAN" and "days_since_arrival" in df.columns:
        additional_columns = ["days_since_arrival"]
    elif source_type == "Pending PO" and "vendor" in df.columns:
        additional_columns = ["vendor"]
    elif source_type == "Pending WH Transfer":
        if "transfer_route" in df.columns:
            additional_columns.append("transfer_route")
        if "days_in_transfer" in df.columns:
            additional_columns.append("days_in_transfer")
        if "expiry_date" in df.columns:
            additional_columns.extend(["expiry_date", "days_until_expiry"])
    elif source_type == "All":
        additional_columns = ["source_type"]
        # Add all optional columns if they exist
        for col in ["expiry_date", "days_until_expiry", "days_since_arrival", 
                   "vendor", "transfer_route", "days_in_transfer"]:
            if col in df.columns and col not in additional_columns:
                additional_columns.append(col)
    
    # Add supply_number if exists
    if "supply_number" in df.columns:
        base_columns.insert(base_columns.index("legal_entity"), "supply_number")
    
    # Combine columns
    display_columns = base_columns + additional_columns
    
    # Filter columns that actually exist
    display_columns = [col for col in display_columns if col in df.columns]
    
    display_df = df[display_columns].copy()
    
    # Sort by date ref
    if "date_ref" in display_df.columns:
        display_df = display_df.sort_values("date_ref", ascending=True)
    
    # Format columns
    display_df = format_supply_display_df(display_df)
    
    # Apply conditional formatting
    if "days_until_expiry" in display_df.columns:
        styled_df = display_df.style.apply(highlight_expiry_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)

def format_supply_display_df(df):
    """Format dataframe columns for display"""
    df = df.copy()
    
    # Format quantity and value
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].apply(lambda x: format_number(x))
    if "value_in_usd" in df.columns:
        df["value_in_usd"] = df["value_in_usd"].apply(lambda x: format_currency(x, "USD"))
    
    # Format dates
    if "date_ref" in df.columns:
        df["date_ref"] = df["date_ref"].apply(
            lambda x: "❌ Missing" if pd.isna(x) else x.strftime("%Y-%m-%d")
        )
    
    if "expiry_date" in df.columns:
        df["expiry_date"] = df["expiry_date"].apply(
            lambda x: "" if pd.isna(x) else x.strftime("%Y-%m-%d")
        )
    
    # Format days columns
    if "days_until_expiry" in df.columns:
        df["days_until_expiry"] = df["days_until_expiry"].apply(
            lambda x: f"{int(x)} days" if pd.notna(x) else ""
        )
    
    if "days_since_arrival" in df.columns:
        df["days_since_arrival"] = df["days_since_arrival"].apply(
            lambda x: f"{int(x)} days" if pd.notna(x) else ""
        )
        
    if "days_in_transfer" in df.columns:
        df["days_in_transfer"] = df["days_in_transfer"].apply(
            lambda x: f"{int(x)} days" if pd.notna(x) else ""
        )
    
    return df

def highlight_expiry_rows(row):
    """Highlight rows based on expiry status"""
    if "days_until_expiry" in row:
        days_str = row["days_until_expiry"]
        if days_str and "days" in days_str:
            try:
                days = int(days_str.split()[0])
                if days <= 7:
                    return ["background-color: #ffcccc"] * len(row)  # Red for very urgent
                elif days <= 30:
                    return ["background-color: #ffe6cc"] * len(row)  # Orange for urgent
            except:
                pass
    return [""] * len(row)

def show_supply_grouped_view(filtered_df, start_date, end_date):
    """Show grouped supply by period with tabs for each source"""
    st.markdown("### 📊 Grouped Supply by Product")
    st.markdown(f"📅 Period: **{start_date}** to **{end_date}**")
    
    # Controls (remove the checkbox for group_by_source)
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Group by Period", PERIOD_TYPES, index=1)
    with col2:
        show_only_nonzero = st.checkbox("Show only products with quantity > 0", value=True, key="supply_show_nonzero")
    
    # Filter out missing dates for grouping
    df_summary = filtered_df[filtered_df["date_ref"].notna()].copy()
    
    if df_summary.empty:
        st.info("No data with valid dates for grouping")
        return
    
    # Create period column
    df_summary["period"] = convert_to_period(df_summary["date_ref"], period)
    
    # Get unique source types
    source_types = df_summary["source_type"].unique()
    
    # Always show tabs (remove the checkbox condition)
    if len(source_types) > 1:
        # Create tabs with "All" first
        tabs = st.tabs(["All"] + list(source_types))
        
        # All tab (first)
        with tabs[0]:
            display_supply_pivot(df_summary, period, show_only_nonzero, "All Sources")
        
        # Individual source tabs
        for idx, source in enumerate(source_types):
            with tabs[idx + 1]:
                source_df = df_summary[df_summary["source_type"] == source]
                display_supply_pivot(source_df, period, show_only_nonzero, source)
    else:
        # Only one source type, just display it
        display_supply_pivot(df_summary, period, show_only_nonzero, "All Sources")

def display_supply_pivot(df_summary, period, show_only_nonzero, title):
    """Display supply pivot table with past period indicators"""
    # Create pivot table
    pivot_df = create_supply_pivot(df_summary, show_only_nonzero)
    pivot_df = sort_period_columns(pivot_df, period, ["product_name", "pt_code"])
    
    # Create display version with past period indicators
    display_pivot = pivot_df.copy()
    
    # Format numeric columns first
    for col in pivot_df.columns[2:]:  # Skip product_name and pt_code
        display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    
    # Then rename columns with indicators for past periods
    renamed_columns = {}
    for col in pivot_df.columns:
        if col not in ["product_name", "pt_code"]:
            if is_past_period(str(col), period):
                renamed_columns[col] = f"🔴 {col}"
    
    if renamed_columns:
        display_pivot = display_pivot.rename(columns=renamed_columns)
    
    # Show legend if there are past periods
    if renamed_columns:
        st.info("🔴 = Past period (already occurred)")
    
    # Display pivot
    st.dataframe(display_pivot, use_container_width=True)
    
    # Show totals with indicators
    show_supply_totals_with_indicators(df_summary, period)
    
    # Export button (use original pivot without indicators)
    export_pivot = pivot_df.copy()
    for col in pivot_df.columns[2:]:
        export_pivot[col] = export_pivot[col].apply(lambda x: format_number(x))
    
    DisplayComponents.show_export_button(
        export_pivot, 
        f"supply_{title.lower().replace(' ', '_')}", 
        f"📤 Export {title} to Excel"
    )

def create_supply_pivot(df_summary, show_only_nonzero):
    """Create pivot table for supply"""
    pivot_df = (
        df_summary
        .groupby(["product_name", "pt_code", "period"])
        .agg(total_quantity=("quantity", "sum"))
        .reset_index()
        .pivot(index=["product_name", "pt_code"], columns="period", values="total_quantity")
        .fillna(0)
        .reset_index()
    )
    
    if show_only_nonzero:
        pivot_df = pivot_df[pivot_df.iloc[:, 2:].sum(axis=1) > 0]
    
    return pivot_df

def show_supply_totals_with_indicators(df_summary, period):
    """Show period totals for supply with past period indicators"""
    # Calculate aggregates
    qty_by_period = df_summary.groupby("period")["quantity"].sum()
    val_by_period = df_summary.groupby("period")["value_in_usd"].sum()
    
    # Create summary DataFrame
    summary_data = {"Metric": ["🔢 TOTAL QUANTITY", "💰 TOTAL VALUE (USD)"]}
    
    # Keep track of original period names for sorting
    period_mapping = {}  # renamed -> original
    
    # Add all periods to summary_data with indicators
    for period_val in qty_by_period.index:
        if is_past_period(str(period_val), period):
            col_name = f"🔴 {period_val}"
            period_mapping[col_name] = period_val
        else:
            col_name = str(period_val)
            period_mapping[col_name] = period_val
            
        summary_data[col_name] = [
            format_number(qty_by_period[period_val]),
            format_currency(val_by_period[period_val], "USD")
        ]
    
    display_final = pd.DataFrame(summary_data)
    
    # Sort columns properly
    metric_cols = ["Metric"]
    period_cols = [col for col in display_final.columns if col not in metric_cols]
    
    # Sort using original period names
    def get_sort_key(col_name):
        original_period = period_mapping.get(col_name, col_name)
        if period == "Weekly":
            return parse_week_period(original_period)
        elif period == "Monthly":
            return parse_month_period(original_period)
        else:
            try:
                return pd.to_datetime(original_period)
            except:
                return pd.Timestamp.max
    
    sorted_period_cols = sorted(period_cols, key=get_sort_key)
    display_final = display_final[metric_cols + sorted_period_cols]
    
    st.markdown("#### 🔢 Column Totals")
    st.dataframe(display_final, use_container_width=True)

# === Main Page Logic ===
st.subheader("📥 Inbound Supply Capability")

# Source selection and options
supply_source, exclude_expired = select_supply_source()

# Load and prepare data
with st.spinner("Loading supply data..."):
    df_all = load_and_prepare_supply_data(supply_source, exclude_expired)

if df_all.empty:
    st.info("No supply data available for the selected source.")
    st.stop()

# Apply filters
filtered_df, start_date, end_date, filter_params = apply_supply_filters(df_all)

# Save to session state for other pages
save_to_session_state('supply_analysis_data', filtered_df)
save_to_session_state('supply_analysis_filters', {
    'source': supply_source,
    'exclude_expired': exclude_expired,
    'start_date': start_date,
    'end_date': end_date
})

# Display sections
show_supply_summary(filtered_df, filter_params)
show_supply_detail_table(filtered_df)
show_supply_grouped_view(filtered_df, start_date, end_date)

# === Additional Page Features ===
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

# Help section
DisplayComponents.show_help_section(
    "How to use Supply Analysis",
    """
    ### Understanding Supply Data
    
    **Data Sources:**
    - **Inventory**: Current stock on hand with expiry tracking
    - **Pending CAN**: Goods arrived but not yet stocked in
    - **Pending PO**: Purchase orders in transit
    - **Pending WH Transfer**: Goods being transferred between warehouses
    - **All**: Combined view of total supply pipeline
    
    **Key Metrics:**
    - **Date Reference**: 
      - Inventory: Today's date
      - CAN: Arrival date
      - PO: Cargo ready date
      - WH Transfer: Transfer start date + lead time
    - **Days Until Expiry**: For inventory and transfer items (red < 7 days, orange < 30 days)
    - **Days Since Arrival**: For CAN items pending stock-in
    - **Days in Transfer**: For warehouse transfers in progress
    
    **Important Features:**
    - **Exclude Expired**: Toggle to hide/show expired inventory
    - **Source Breakdown**: See quantity and value by each supply type
    - **Period Grouping**: Analyze supply availability over time
    
    **Common Actions:**
    1. Check inventory items nearing expiry
    2. Follow up on delayed CAN stock-ins
    3. Monitor incoming PO pipeline
    4. Track warehouse transfers in progress
    5. Compare supply timing with demand (go to GAP Analysis)
    
    **Tips:**
    - Use tabs in grouped view for detailed analysis by source
    - Export period data for supply planning meetings
    - Set expiry warning days based on your product shelf life
    - Monitor long warehouse transfers (>3 days) for potential delays
    """
)

# Footer
st.markdown("---")
st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")