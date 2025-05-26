import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from data_loader import load_outbound_demand_data, load_customer_forecast_data

# === Constants ===
DEMAND_SOURCES = ["OC Only", "Forecast Only", "Both"]
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]

# === Main Entry Point ===
def show_outbound_demand_tab():
    """Main entry point for Outbound Demand tab"""
    st.subheader("📤 Outbound Demand by Period")
    
    # Source selection
    source = select_demand_source()
    
    # Load and prepare data
    df_all = load_and_prepare_demand_data(source)
    
    if df_all.empty:
        st.info("No outbound demand data available.")
        return
    
    # Apply filters
    filtered_df, start_date, end_date = apply_demand_filters(df_all)
    
    # Display sections
    show_demand_summary(filtered_df)
    show_demand_detail_table(filtered_df)
    show_demand_grouped_view(filtered_df, start_date, end_date)

# === Data Source Selection ===
def select_demand_source():
    """Allow user to choose demand data source"""
    return st.radio(
        "Select Outbound Demand Source:",
        DEMAND_SOURCES,
        index=2,  # Default to "Both"
        horizontal=True
    )

# === Data Loading ===
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

# === Filtering ===
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
        st.metric("Total Value (USD)", f"${total_value:,.2f}")
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
    missing_etd_count = filtered_df["etd"].isna().sum()
    if missing_etd_count > 0:
        st.warning(f"⚠️ Found {missing_etd_count} records with missing ETD dates")
    
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
    df_summary["period"] = create_period_column(df_summary["etd"], period)
    
    # Create pivot table
    pivot_df = create_demand_pivot(df_summary, show_only_nonzero)
    pivot_df = sort_period_columns(pivot_df, period)
    
    # Display pivot
    display_pivot = format_pivot_for_display(pivot_df)
    st.dataframe(display_pivot, use_container_width=True)
    
    # Show totals
    show_demand_totals(df_summary, period)
    
    # Export button
    excel_data = convert_df_to_excel(display_pivot)
    st.download_button(
        label="📤 Export to Excel",
        data=excel_data,
        file_name="grouped_outbound_demand.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# === Helper Functions ===
def format_demand_display_df(df):
    """Format dataframe columns for display"""
    df = df.copy()
    df["demand_quantity"] = df["demand_quantity"].apply(lambda x: f"{x:,.0f}")
    df["value_in_usd"] = df["value_in_usd"].apply(lambda x: f"${x:,.2f}")
    df["etd"] = df["etd"].apply(
        lambda x: "❌ Missing" if pd.isna(x) else x.strftime("%Y-%m-%d")
    )
    return df

def highlight_missing_dates(row):
    """Highlight rows with missing dates"""
    if row["etd"] == "❌ Missing":
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)

def create_period_column(date_series, period_type):
    """Create period column based on period type"""
    if period_type == "Daily":
        return date_series.dt.strftime("%Y-%m-%d")
    elif period_type == "Weekly":
        year = date_series.dt.isocalendar().year
        week = date_series.dt.isocalendar().week
        return "Week " + week.astype(str).str.zfill(2) + " - " + year.astype(str)
    else:  # Monthly
        return date_series.dt.to_period("M").dt.strftime("%b %Y")

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
        display_pivot[col] = display_pivot[col].apply(lambda x: f"{x:,.0f}")
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
            f"{qty_by_period[period]:,.0f}",
            f"${val_by_period[period]:,.2f}"
        ]
    
    display_final = pd.DataFrame(summary_data)
    display_final = sort_period_columns(display_final, period)
    
    st.markdown("#### 🔢 Column Totals")
    st.dataframe(display_final, use_container_width=True)

def sort_period_columns(df, period_type):
    """Sort dataframe columns by period"""
    # Identify info columns
    info_cols = ["Metric"] if "Metric" in df.columns else ["product_name", "pt_code"]
    
    # Get period columns
    period_cols = [col for col in df.columns if col not in info_cols]
    period_cols = [p for p in period_cols if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"]
    
    # Sort based on period type
    if period_type == "Weekly":
        def parse_week_key(x):
            try:
                parts = str(x).split(" - ")
                if len(parts) == 2:
                    week = int(parts[0].replace("Week", "").strip())
                    year = int(parts[1].strip())
                    return (year, week)
            except:
                pass
            return (9999, 99)
        sorted_periods = sorted(period_cols, key=parse_week_key)
    
    elif period_type == "Monthly":
        def parse_month_key(x):
            try:
                return pd.to_datetime("01 " + str(x), format="%d %b %Y")
            except:
                return pd.Timestamp.max
        sorted_periods = sorted(period_cols, key=parse_month_key)
    
    else:  # Daily
        sorted_periods = sorted(period_cols)
    
    return df[info_cols + sorted_periods]

def convert_df_to_excel(df):
    """Convert dataframe to Excel bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
        
        # Auto-adjust column widths
        worksheet = writer.sheets["Data"]
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, max_len)
    
    return output.getvalue()