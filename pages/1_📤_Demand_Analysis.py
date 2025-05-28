import streamlit as st
import pandas as pd
from datetime import datetime
from utils.data_loader import load_outbound_demand_data, load_customer_forecast_data
from utils.helpers import (
    convert_df_to_excel, 
    convert_to_period, 
    sort_period_columns,
    format_number,
    format_currency,
    check_missing_dates,
    save_to_session_state
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
    return st.radio(
        "Select Outbound Demand Source:",
        DEMAND_SOURCES,
        index=2,  # Default to "Both"
        horizontal=True
    )

# === Data Loading Functions ===
def load_and_prepare_demand_data(source):
    """Load and standardize demand data based on source selection"""
    df_parts = []
    
    if source in ["OC Only", "Both"]:
        df_oc = load_outbound_demand_data()
        if not df_oc.empty:
            df_oc["source_type"] = "OC"
            df_parts.append(standardize_demand_df(df_oc, is_forecast=False))
    
    if source in ["Forecast Only", "Both"]:
        df_fc = load_customer_forecast_data()
        if not df_fc.empty:
            df_fc["source_type"] = "Forecast"
            df_parts.append(standardize_demand_df(df_fc, is_forecast=True))
    
    return pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()

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
        df[col] = df[col].astype(str)
    
    df['standard_uom'] = df.get('standard_uom', '')
    df['package_size'] = df.get('package_size', '')
    
    return df

# === Filtering Functions ===
def apply_demand_filters(df):
    """Apply filters to demand dataframe"""
    with st.expander("📎 Filters", expanded=True):
        # Row 1: Entity, Customer, Product
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_entity = st.multiselect(
                "Legal Entity", 
                sorted(df["legal_entity"].dropna().unique())
            )
        with col2:
            selected_customer = st.multiselect(
                "Customer", 
                sorted(df["customer"].dropna().unique())
            )
        with col3:
            selected_product = st.multiselect(
                "PT Code", 
                sorted(df["pt_code"].dropna().unique())
            )
        
        # Row 2: Brand, Date range
        col4, col5, col6 = st.columns(3)
        with col4:
            selected_brand = st.multiselect(
                "Brand", 
                sorted(df["brand"].dropna().unique())
            )
        with col5:
            default_start = df["etd"].min().date() if pd.notnull(df["etd"].min()) else datetime.today().date()
            start_date = st.date_input("From Date (ETD)", default_start)
        with col6:
            default_end = df["etd"].max().date() if pd.notnull(df["etd"].max()) else datetime.today().date()
            end_date = st.date_input("To Date (ETD)", default_end)
        
        # Row 3: Conversion status (if applicable)
        if 'is_converted_to_oc' in df.columns:
            col7, col8, col9 = st.columns(3)
            with col7:
                conversion_options = df["is_converted_to_oc"].dropna().unique().tolist()
                selected_conversion = st.multiselect(
                    "Conversion Status", 
                    sorted(conversion_options)
                )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_entity:
        filtered_df = filtered_df[filtered_df["legal_entity"].isin(selected_entity)]
    if selected_customer:
        filtered_df = filtered_df[filtered_df["customer"].isin(selected_customer)]
    if selected_product:
        filtered_df = filtered_df[filtered_df["pt_code"].isin(selected_product)]
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
    
    return filtered_df, start_date, end_date

# === Display Functions ===
def show_demand_summary(filtered_df):
    """Show demand summary metrics"""
    st.markdown("### 📊 Demand Summary")
    
    # Calculate metrics
    total_products = filtered_df["pt_code"].nunique()
    total_value = filtered_df["value_in_usd"].sum()
    missing_etd = filtered_df["etd"].isna().sum()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Products", f"{total_products:,}")
    with col2:
        st.metric("Total Value (USD)", format_currency(total_value, "USD"))
    with col3:
        if missing_etd > 0:
            st.metric("⚠️ Missing ETD", f"{missing_etd} records", delta_color="inverse")
    
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
    """Show detailed demand table with highlighting"""
    st.markdown("### 🔍 Demand Details")
    
    # Warning for missing ETD
    missing_etd_count = check_missing_dates(filtered_df, "etd")
    
    # Prepare display columns
    base_columns = [
        "pt_code", "product_name", "brand", "package_size", "standard_uom",
        "etd", "demand_quantity", "value_in_usd", "source_type", "demand_number", 
        "customer", "legal_entity"
    ]
    
    if 'is_converted_to_oc' in filtered_df.columns:
        display_columns = base_columns + ["is_converted_to_oc"]
    else:
        display_columns = base_columns
    
    display_df = filtered_df[display_columns].copy()
    
    # Sort with missing ETD first
    display_df["etd_is_null"] = display_df["etd"].isna()
    display_df = display_df.sort_values(["etd_is_null", "etd"], ascending=[False, True])
    display_df = display_df.drop("etd_is_null", axis=1)
    
    # Format columns
    display_df = format_demand_display_df(display_df)
    
    # Apply styling
    styled_df = display_df.style.apply(highlight_missing_dates, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def format_demand_display_df(df):
    """Format dataframe columns for display"""
    df = df.copy()
    df["demand_quantity"] = df["demand_quantity"].apply(lambda x: format_number(x))
    df["value_in_usd"] = df["value_in_usd"].apply(lambda x: format_currency(x, "USD"))
    df["etd"] = df["etd"].apply(
        lambda x: "❌ Missing" if pd.isna(x) else x.strftime("%Y-%m-%d")
    )
    return df

def highlight_missing_dates(row):
    """Highlight rows with missing dates"""
    if row["etd"] == "❌ Missing":
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)

def show_demand_grouped_view(filtered_df, start_date, end_date):
    """Show grouped demand by period"""
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
    
    # Create pivot table
    pivot_df = create_demand_pivot(df_summary, show_only_nonzero)
    pivot_df = sort_period_columns(pivot_df, period, ["product_name", "pt_code"])
    
    # Display pivot
    display_pivot = format_pivot_for_display(pivot_df)
    st.dataframe(display_pivot, use_container_width=True)
    
    # Show totals
    show_demand_totals(df_summary, period)
    
    # Export button
    excel_data = convert_df_to_excel(display_pivot, "Grouped Demand")
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
    """Format pivot table for display"""
    display_pivot = pivot_df.copy()
    for col in display_pivot.columns[2:]:  # Skip product_name and pt_code
        display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    return display_pivot

def show_demand_totals(df_summary, period):
    """Show period totals for demand"""
    # Prepare data
    df_grouped = df_summary.copy()
    df_grouped["demand_quantity"] = pd.to_numeric(df_grouped["demand_quantity"], errors='coerce').fillna(0)
    df_grouped["value_in_usd"] = pd.to_numeric(df_grouped["value_in_usd"], errors='coerce').fillna(0)
    
    # Calculate aggregates
    qty_by_period = df_grouped.groupby("period")["demand_quantity"].sum()
    val_by_period = df_grouped.groupby("period")["value_in_usd"].sum()
    
    # Create summary DataFrame
    summary_data = {"Metric": ["🔢 TOTAL QUANTITY", "💵 TOTAL VALUE (USD)"]}
    
    for period in qty_by_period.index:
        summary_data[period] = [
            format_number(qty_by_period[period]),
            format_currency(val_by_period[period], "USD")
        ]
    
    display_final = pd.DataFrame(summary_data)
    display_final = sort_period_columns(display_final, period, ["Metric"])
    
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
    - **Conversion Status**: Whether forecast has been converted to actual order
    
    **Tips:**
    - Check for missing ETD dates - these need immediate attention
    - Monitor forecast conversion rates to improve planning accuracy
    - Group by period (Weekly/Monthly) for better demand planning
    - Use filters to focus on specific customers or products
    
    **Common Actions:**
    1. Review high-value pending orders
    2. Identify products with consistent demand patterns
    3. Check forecast accuracy by comparing converted vs non-converted
    4. Export grouped data for demand planning meetings
    """)

# Footer
st.markdown("---")
st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")