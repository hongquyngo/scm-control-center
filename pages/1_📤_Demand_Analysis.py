import streamlit as st
import pandas as pd
from datetime import datetime, date
from utils.data_loader import load_outbound_demand_data, load_customer_forecast_data
from utils.helpers import (
    convert_df_to_excel, 
    convert_to_period, 
    sort_period_columns,
    format_number,
    format_currency,
    check_missing_dates,
    check_past_dates,
    is_past_period,
    save_to_session_state,
    parse_week_period,
    parse_month_period
)

# === Page Config ===
st.set_page_config(
    page_title="Demand Analysis - SCM",
    page_icon="📤",
    layout="wide"
)

# === Constants ===
DEMAND_SOURCES = ["OC Only", "Forecast Only", "Both"]
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]

# === Debug Mode Toggle ===
col_debug1, col_debug2 = st.columns([6, 1])
with col_debug2:
    debug_mode = st.checkbox("🐛 Debug Mode", value=False, key="demand_debug_mode")

if debug_mode:
    st.info("🐛 Debug Mode is ON - Additional information will be displayed")

# === Header with Navigation ===
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("🏠 Dashboard"):
        st.switch_page("main.py")
with col2:
    st.title("📤 Outbound Demand Analysis")
with col3:
    if st.button("Next: Supply →"):
        st.switch_page("pages/2_📥_Supply_Analysis.py")

st.markdown("---")

# === Data Source Selection ===
def select_demand_source():
    """Allow user to choose demand data source"""
    source = st.radio(
        "Select Outbound Demand Source:",
        DEMAND_SOURCES,
        index=2,  # Default to "Both"
        horizontal=True
    )
    
    if debug_mode:
        st.write(f"🐛 Selected source: {source}")
    
    return source

# === Data Loading Functions ===
def load_and_prepare_demand_data(source):
    """Load and standardize demand data based on source selection"""
    df_parts = []
    
    if source in ["OC Only", "Both"]:
        df_oc = load_outbound_demand_data()
        if not df_oc.empty:
            df_oc["source_type"] = "OC"
            df_parts.append(standardize_demand_df(df_oc, is_forecast=False))
            
            if debug_mode:
                st.write(f"🐛 OC data loaded: {len(df_oc)} rows")
    
    if source in ["Forecast Only", "Both"]:
        df_fc = load_customer_forecast_data()
        if not df_fc.empty:
            df_fc["source_type"] = "Forecast"
            df_parts.append(standardize_demand_df(df_fc, is_forecast=True))
            
            if debug_mode:
                st.write(f"🐛 Forecast data loaded: {len(df_fc)} rows")
    
    result_df = pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()
    
    if debug_mode and not result_df.empty:
        st.write(f"🐛 Combined data: {len(result_df)} rows, {result_df['pt_code'].nunique()} unique products")
    
    return result_df

def standardize_demand_df(df, is_forecast):
    """Standardize column names and data types for demand dataframes"""
    df = df.copy()
    
    # Date columns
    df["etd"] = pd.to_datetime(df["etd"], errors="coerce")
    df["oc_date"] = pd.to_datetime(df.get("oc_date", pd.NaT), errors="coerce")
    
    # Quantity and value columns
    if is_forecast:
        df['demand_quantity'] = pd.to_numeric(df['standard_quantity'], errors='coerce').fillna(0)
        df['value_in_usd'] = pd.to_numeric(df.get('total_amount_usd', 0), errors='coerce').fillna(0)
        df['demand_number'] = df.get('forecast_number', '')
        df['is_converted_to_oc'] = df.get('is_converted_to_oc', 'No')
    else:
        df['demand_quantity'] = pd.to_numeric(df['pending_standard_delivery_quantity'], errors='coerce').fillna(0)
        df['value_in_usd'] = pd.to_numeric(df.get('outstanding_amount_usd', 0), errors='coerce').fillna(0)
        df['demand_number'] = df.get('oc_number', '')
        df['is_converted_to_oc'] = 'N/A'
    
    # String columns
    string_cols = ['product_name', 'pt_code', 'brand', 'legal_entity', 'customer']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = ""
    
    df['standard_uom'] = df.get('standard_uom', '')
    df['package_size'] = df.get('package_size', '')
    
    return df

# === Filtering Functions ===
def apply_demand_filters(df):
    """Apply filters to demand dataframe with enhanced product search"""
    with st.expander("📎 Filters", expanded=True):
        # Row 1: Entity, Customer, Product
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_entity = st.multiselect(
                "Legal Entity", 
                sorted(df["legal_entity"].dropna().unique()),
                key="demand_entity_filter"
            )
        with col2:
            selected_customer = st.multiselect(
                "Customer", 
                sorted(df["customer"].dropna().unique()),
                key="demand_customer_filter"
            )
        with col3:
            # Enhanced product filter with both PT Code and Product Name
            unique_products = df[['pt_code', 'product_name']].drop_duplicates()
            product_options = [f"{row['pt_code']} - {row['product_name'][:50]}" 
                             for _, row in unique_products.iterrows()]
            
            selected_products = st.multiselect(
                "Product (PT Code - Name)", 
                sorted(product_options),
                key="demand_product_filter",
                help="Search by PT Code or Product Name"
            )
            
            # Extract selected PT codes
            selected_pt_codes = [p.split(' - ')[0] for p in selected_products]
        
        # Row 2: Brand, Date range
        col4, col5, col6 = st.columns(3)
        with col4:
            selected_brand = st.multiselect(
                "Brand", 
                sorted(df["brand"].dropna().unique()),
                key="demand_brand_filter"
            )
        with col5:
            default_start = df["etd"].min().date() if pd.notnull(df["etd"].min()) else datetime.today().date()
            start_date = st.date_input("From Date (ETD)", default_start, key="demand_start_date")
        with col6:
            default_end = df["etd"].max().date() if pd.notnull(df["etd"].max()) else datetime.today().date()
            end_date = st.date_input("To Date (ETD)", default_end, key="demand_end_date")
        
        # Row 3: Conversion status (if applicable)
        if 'is_converted_to_oc' in df.columns:
            col7, col8, col9 = st.columns(3)
            with col7:
                conversion_options = df["is_converted_to_oc"].dropna().unique().tolist()
                selected_conversion = st.multiselect(
                    "Conversion Status", 
                    sorted(conversion_options),
                    key="demand_conversion_filter"
                )
    
    # Apply filters
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    if selected_entity:
        filtered_df = filtered_df[filtered_df["legal_entity"].isin(selected_entity)]
    if selected_customer:
        filtered_df = filtered_df[filtered_df["customer"].isin(selected_customer)]
    if selected_products:
        filtered_df = filtered_df[filtered_df["pt_code"].isin(selected_pt_codes)]
    if selected_brand:
        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brand)]
    
    if 'is_converted_to_oc' in df.columns and 'selected_conversion' in locals() and selected_conversion:
        filtered_df = filtered_df[filtered_df["is_converted_to_oc"].isin(selected_conversion)]
    
    # Date filter (include null ETD)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    filtered_df = filtered_df[
        filtered_df["etd"].isna() |
        ((filtered_df["etd"] >= start_ts) & (filtered_df["etd"] <= end_ts))
    ]
    
    if debug_mode:
        st.write(f"🐛 Filtered: {original_count} → {len(filtered_df)} rows")
    
    return filtered_df, start_date, end_date

# === Display Functions ===
def show_demand_summary(filtered_df):
    """Show demand summary metrics with past ETD tracking"""
    st.markdown("### 📊 Demand Summary")
    
    today = pd.Timestamp.now().normalize()
    
    # Calculate metrics
    total_products = filtered_df["pt_code"].nunique()
    total_value = filtered_df["value_in_usd"].sum()
    missing_etd = filtered_df["etd"].isna().sum()
    
    # NEW: Calculate past ETD
    past_etd = filtered_df[filtered_df["etd"] < today].shape[0]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Unique Products", f"{total_products:,}")
    with col2:
        st.metric("Total Value (USD)", format_currency(total_value, "USD"))
    with col3:
        if missing_etd > 0:
            st.metric("⚠️ Missing ETD", f"{missing_etd} records", delta_color="inverse")
    with col4:
        if past_etd > 0:
            st.metric("🔴 Past ETD", f"{past_etd} records", delta_color="inverse", 
                     help="Orders with ETD before today")
    
    # Conversion status for forecasts
    if filtered_df["source_type"].str.contains("Forecast").any():
        show_forecast_conversion_summary(filtered_df)

def show_forecast_conversion_summary(filtered_df):
    """Show forecast conversion statistics"""
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



def show_demand_detail_table(filtered_df):
    """Show detailed demand table with enhanced filtering and highlighting"""
    st.markdown("### 🔍 Demand Details")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_missing_only = st.checkbox("Show Missing ETD Only", key="show_missing_etd")
    with col2:
        show_past_only = st.checkbox("Show Past ETD Only", key="show_past_etd")
    with col3:
        # Display warnings for both missing and past ETD
        missing_etd_count = check_missing_dates(filtered_df, "etd", show_warning=False)
        past_etd_count = check_past_dates(filtered_df, "etd", show_warning=False)
        
        if missing_etd_count > 0:
            st.warning(f"⚠️ Found {missing_etd_count} records with missing ETD")
        if past_etd_count > 0:
            st.error(f"🔴 Found {past_etd_count} records with past ETD")
    
    # Apply display filters
    display_df = filtered_df.copy()
    today = pd.Timestamp.now().normalize()
    
    # FIXED LOGIC: Handle both checkboxes properly
    if show_missing_only and show_past_only:
        # Show both missing AND past ETD
        display_df = display_df[
            display_df["etd"].isna() | 
            (display_df["etd"] < today)
        ]
    elif show_missing_only:
        # Show only missing ETD
        display_df = display_df[display_df["etd"].isna()]
    elif show_past_only:
        # Show only past ETD (not missing)
        display_df = display_df[display_df["etd"] < today]
    
    # Prepare display columns
    base_columns = [
        "pt_code", "product_name", "brand", "package_size", "standard_uom",
        "etd", "demand_quantity", "value_in_usd", "source_type", "demand_number", 
        "customer", "legal_entity"
    ]
    
    if 'is_converted_to_oc' in display_df.columns:
        display_columns = base_columns + ["is_converted_to_oc"]
    else:
        display_columns = base_columns
    
    display_df = display_df[display_columns].copy()
    
    # Create sort key for proper ordering
    display_df["etd_sort_key"] = display_df["etd"].apply(
        lambda x: (0, pd.NaT) if pd.isna(x) else 
                  (1, x) if x < today else 
                  (2, x)
    )
    
    # Sort: Missing ETD first, then Past ETD, then Future ETD
    display_df = display_df.sort_values(["etd_sort_key"], key=lambda x: [(v[0], v[1]) for v in x])
    display_df = display_df.drop("etd_sort_key", axis=1)
    
    # Format columns
    display_df = format_demand_display_df(display_df)
    
    # Apply styling with enhanced highlighting
    styled_df = display_df.style.apply(highlight_etd_issues, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=600)



def format_demand_display_df(df):
    """Format dataframe columns for display"""
    df = df.copy()
    today = pd.Timestamp.now().normalize()
    
    # Format numeric columns
    df["demand_quantity"] = df["demand_quantity"].apply(lambda x: format_number(x))
    df["value_in_usd"] = df["value_in_usd"].apply(lambda x: format_currency(x, "USD"))
    
    # Enhanced ETD formatting
    def format_etd(x):
        if pd.isna(x):
            return "❌ Missing"
        elif x < today:
            return f"🔴 {x.strftime('%Y-%m-%d')}"
        else:
            return x.strftime("%Y-%m-%d")
    
    df["etd"] = df["etd"].apply(format_etd)
    
    return df

def highlight_etd_issues(row):
    """Highlight rows based on ETD issues"""
    styles = [""] * len(row)
    
    if "❌ Missing" in str(row.get("etd", "")):
        # Light orange/yellow background for missing ETD (matches warning color)
        styles = ["background-color: #fff3cd"] * len(row)
    elif "🔴" in str(row.get("etd", "")):
        # Light red background for past ETD (matches error color)
        styles = ["background-color: #f8d7da"] * len(row)
    
    return styles


# Replace the show_demand_grouped_view function with this updated version
def show_demand_grouped_view(filtered_df, start_date, end_date):
    """Show grouped demand by period with past period indicators"""
    st.markdown("### 📦 Grouped Demand by Product")
    st.markdown(f"📅 Period: **{start_date}** to **{end_date}**")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Group By Period", PERIOD_TYPES, index=1)
    with col2:
        show_only_nonzero = st.checkbox("Show only products with quantity > 0", value=True)
    
    # Filter out missing ETD for grouping
    df_summary = filtered_df[filtered_df["etd"].notna()].copy()
    
    if df_summary.empty:
        st.info("No data with valid ETD dates for grouping")
        return
    
    # Create period column
    df_summary["period"] = convert_to_period(df_summary["etd"], period)
    
    if debug_mode:
        st.write("🐛 Sample periods created:")
        st.write(df_summary[["etd", "period"]].head(10))
    
    # Create pivot table
    pivot_df = create_demand_pivot(df_summary, show_only_nonzero)
    
    if debug_mode:
        st.write("🐛 Pivot columns BEFORE sorting:")
        st.write(list(pivot_df.columns))
    
  # Sort columns
    pivot_df = sort_period_columns(pivot_df, period, ["product_name", "pt_code"])
    
    if debug_mode:
        st.write("🐛 Pivot columns AFTER sorting:")
        st.write(list(pivot_df.columns))
        
        # ADD THIS DEBUG SECTION
        st.write("🐛 Checking past periods:")
        for col in pivot_df.columns:
            if col not in ["product_name", "pt_code"]:
                is_past = is_past_period(str(col), period)
                st.write(f"  - '{col}' → is_past = {is_past}")
        
        # Check current date
        st.write(f"🐛 Current date: {datetime.now().date()}")
    
    # Add past period indicators to column names
    display_pivot = pivot_df.copy()
    renamed_columns = {}
    
    for col in display_pivot.columns:
        if col not in ["product_name", "pt_code"]:
            if is_past_period(str(col), period):
                # Add red circle emoji to indicate past period
                renamed_columns[col] = f"🔴 {col}"
            else:
                renamed_columns[col] = col
    
    # Rename columns with indicators
    if renamed_columns:
        display_pivot = display_pivot.rename(columns=renamed_columns)
    
    # Format for display
    for col in display_pivot.columns:
        if col not in ["product_name", "pt_code"]:
            display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    
    if debug_mode:
        st.write("🐛 Display columns AFTER formatting:")
        st.write(list(display_pivot.columns))
    
    # Show legend
    st.info("🔴 = Past period (already occurred)")
    
    # Display pivot
    st.dataframe(display_pivot, use_container_width=True, height=400)
    
    # Show totals with past period indicators
    show_demand_totals_with_indicators(df_summary, period)
    
    # Export button (without indicators)
    excel_pivot = format_pivot_for_display(pivot_df)
    excel_data = convert_df_to_excel(excel_pivot, "Grouped Demand")
    st.download_button(
        label="📤 Export to Excel",
        data=excel_data,
        file_name=f"grouped_demand_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def create_demand_pivot(df_summary, show_only_nonzero):
    """Create pivot table for demand"""
    pivot_df = (
        df_summary
        .groupby(["product_name", "pt_code", "period"])
        .agg(total_quantity=("demand_quantity", "sum"))
        .reset_index()
        .pivot(index=["product_name", "pt_code"], columns="period", values="total_quantity")
        .fillna(0)
        .reset_index()
    )
    
    if show_only_nonzero:
        pivot_df = pivot_df[pivot_df.iloc[:, 2:].sum(axis=1) > 0]
    
    return pivot_df

def format_pivot_for_display(pivot_df):
    """Format pivot table for display - PRESERVE COLUMN ORDER"""
    display_pivot = pivot_df.copy()
    
    # Remember original column order
    original_columns = list(display_pivot.columns)
    
    if debug_mode:
        st.write("🐛 Formatting columns, original order:")
        st.write(original_columns)
    
    # Format only numeric columns (skip first 2 columns: product_name and pt_code)
    for col in original_columns[2:]:  
        display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    
    # IMPORTANT: Return with original column order preserved
    return display_pivot[original_columns]



# Add this new function for totals with indicators
def show_demand_totals_with_indicators(df_summary, period):
    """Show period totals for demand with past period indicators"""
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
            format_currency(val_by_period[period_val], "USD")
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
    
    if debug_mode:
        st.write("🐛 Summary columns AFTER sorting:")
        st.write(list(display_final.columns))
    
    st.markdown("#### 🔢 Column Totals")
    st.dataframe(display_final, use_container_width=True)




# === Main Page Logic ===
st.subheader("📤 Outbound Demand by Period")

# Source selection
source = select_demand_source()

# Load and prepare data
with st.spinner("Loading demand data..."):
    df_all = load_and_prepare_demand_data(source)

if df_all.empty:
    st.info("No outbound demand data available.")
    st.stop()

# Apply filters
filtered_df, start_date, end_date = apply_demand_filters(df_all)

# Save to session state for other pages
save_to_session_state('demand_analysis_data', filtered_df)
save_to_session_state('demand_analysis_filters', {
    'source': source,
    'start_date': start_date,
    'end_date': end_date
})

# Display sections
show_demand_summary(filtered_df)
show_demand_detail_table(filtered_df)
show_demand_grouped_view(filtered_df, start_date, end_date)

# === Additional Page Features ===
st.markdown("---")

# Quick Actions
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Go to GAP Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/3_📊_GAP_Analysis.py")

with col2:
    if st.button("📥 View Supply", use_container_width=True):
        st.switch_page("pages/2_📥_Supply_Analysis.py")

with col3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Debug info panel
if debug_mode:
    with st.expander("🐛 Debug Information", expanded=True):
        st.markdown("### Debug Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Shape:**")
            st.write(f"- Total rows: {len(df_all)}")
            st.write(f"- Filtered rows: {len(filtered_df)}")
            st.write(f"- Unique products: {filtered_df['pt_code'].nunique()}")
            st.write(f"- Date range: {filtered_df['etd'].min()} to {filtered_df['etd'].max()}")
        
        with col2:
            st.write("**Sample Data:**")
            st.write(filtered_df[['pt_code', 'etd', 'demand_quantity']].head())

# Help section
with st.expander("ℹ️ How to use Demand Analysis", expanded=False):
    st.markdown("""
    ### Understanding Demand Data
    
    **Data Sources:**
    - **OC (Order Confirmation)**: Confirmed customer orders
    - **Forecast**: Customer demand predictions
    - **Both**: Combined view (watch for duplicates if forecast is converted to OC!)
    
    **Key Metrics:**
    - **Pending Delivery**: Orders not yet shipped
    - **ETD**: Estimated Time of Departure (when goods should leave warehouse)
    - **Missing ETD**: Orders without ETD dates (❌ Red background)
    - **Past ETD**: Orders with ETD before today (🔴 Orange background)
    - **Conversion Status**: Whether forecast has been converted to actual order
    
    **Enhanced Features:**
    - **Product Search**: Filter by PT Code or Product Name
    - **ETD Status Tracking**: Automatic detection of missing and past ETD
    - **Smart Sorting**: Missing ETD → Past ETD → Future ETD
    - **Quick Filters**: Show only missing or past ETD records
    
    **Visual Indicators:**
    - ❌ Red background: Missing ETD - needs immediate attention
    - 🔴 Orange background: Past ETD - overdue for delivery
    - Normal: Future ETD - on schedule
    
    **Tips:**
    - Use "Show Missing ETD Only" to focus on data quality issues
    - Use "Show Past ETD Only" to identify overdue deliveries
    - Monitor forecast conversion rates to improve planning accuracy
    - Export grouped data for demand planning meetings
    
    **Common Actions:**
    1. Review and fix missing ETD dates
    2. Follow up on past ETD orders
    3. Check forecast accuracy by comparing converted vs non-converted
    4. Identify products with consistent demand patterns
    """)

# Footer
st.markdown("---")
st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")