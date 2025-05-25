import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO

from data_loader import load_outbound_demand_data, load_customer_forecast_data

# === Main Entry Point for Outbound Tab ===
def show_outbound_demand_tab():
    st.subheader("📤 Outbound Demand by Period")

    output_source = select_data_source()
    df_all = load_and_prepare_data(output_source)

    if df_all.empty:
        st.info("No outbound demand data available.")
        return

    filtered_df, start_date, end_date = apply_outbound_filters(df_all)
    show_outbound_summary(filtered_df)
    show_grouped_demand_summary(filtered_df, start_date, end_date)


# === Allow user to choose data source: OC / Forecast / Both ===
def select_data_source():
    return st.radio(
        "Select Outbound Demand Source:",
        ["OC Only", "Forecast Only", "Both"],
        horizontal=True
    )


def load_and_prepare_data(source):
    df_oc, df_fc = pd.DataFrame(), pd.DataFrame()

    if source in ["OC Only", "Both"]:
        df_oc = load_outbound_demand_data()
        df_oc["source_type"] = "OC"

    if source in ["Forecast Only", "Both"]:
        df_fc = load_customer_forecast_data()
        df_fc["source_type"] = "Forecast"

    df_parts = []
    if not df_oc.empty:
        df_parts.append(standardize_df(df_oc, is_forecast=False))
    if not df_fc.empty:
        df_parts.append(standardize_df(df_fc, is_forecast=True))

    if not df_parts:
        return pd.DataFrame()

    return pd.concat(df_parts, ignore_index=True)


# === Unify column format between OC and Forecast sources ===
def standardize_df(df, is_forecast):
    df = df.copy()
    df["etd"] = pd.to_datetime(df["etd"], errors="coerce")
    df["oc_date"] = pd.to_datetime(df.get("oc_date", pd.NaT), errors="coerce")

    if is_forecast:
        df['demand_quantity'] = pd.to_numeric(df['standard_quantity'], errors='coerce').fillna(0)
        df['value_in_usd'] = pd.to_numeric(df.get('total_amount_usd', 0), errors='coerce').fillna(0)
        df['demand_number'] = df.get('forecast_number', '')
    else:
        df['demand_quantity'] = pd.to_numeric(df['pending_standard_delivery_quantity'], errors='coerce').fillna(0)
        df['value_in_usd'] = pd.to_numeric(df.get('outstanding_amount_usd', 0), errors='coerce').fillna(0)
        df['demand_number'] = df.get('oc_number', '')

    df['product_name'] = df['product_name'].astype(str)
    df['pt_code'] = df['pt_code'].astype(str)
    df['brand'] = df['brand'].astype(str)
    df['legal_entity'] = df['legal_entity'].astype(str)
    df['customer'] = df['customer'].astype(str)
    df['standard_uom'] = df.get('standard_uom', '')
    df['package_size'] = df.get('package_size', '')

    return df


# === UI filters for legal entity, customer, product, brand, ETD ===
def apply_outbound_filters(df):
    with st.expander("📎 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_entity = st.multiselect("Legal Entity", sorted(df["legal_entity"].dropna().unique()))
        with col2:
            selected_customer = st.multiselect("Customer", sorted(df["customer"].dropna().unique()))
        with col3:
            selected_product = st.multiselect("PT Code", sorted(df["pt_code"].dropna().unique()))

        col4, col5, col6 = st.columns(3)
        with col4:
            selected_brand = st.multiselect("Brand", sorted(df["brand"].dropna().unique()))
        with col5:
            default_start = df["etd"].min().date() if pd.notnull(df["etd"].min()) else datetime.today().date()
            start_date = st.date_input("From Date (ETD)", default_start)
        with col6:
            default_end = df["etd"].max().date() if pd.notnull(df["etd"].max()) else datetime.today().date()
            end_date = st.date_input("To Date (ETD)", default_end)

    filtered_df = df.copy()
    if selected_entity:
        filtered_df = filtered_df[filtered_df["legal_entity"].isin(selected_entity)]
    if selected_customer:
        filtered_df = filtered_df[filtered_df["customer"].isin(selected_customer)]
    if selected_product:
        filtered_df = filtered_df[filtered_df["pt_code"].isin(selected_product)]
    if selected_brand:
        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brand)]

    # Convert to Timestamp for compatibility
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # Include rows with null ETD to avoid filtering them out
    filtered_df = filtered_df[
        filtered_df["etd"].isna() |
        ((filtered_df["etd"] >= start_ts) & (filtered_df["etd"] <= end_ts))
    ]

    return filtered_df, start_date, end_date


# === Render outbound demand table + summary statistics ===
def show_outbound_summary(filtered_df):
    st.markdown("### 🔍 Outbound Demand Details")

    total_unique_products = filtered_df["pt_code"].nunique()
    total_value_usd = filtered_df["value_in_usd"].sum()

    st.markdown(f"🔢 Total Unique Products: **{int(total_unique_products):,}**  💵 Total Value (USD): **${total_value_usd:,.2f}**")

    display_df = filtered_df[[ 
        "pt_code", "product_name", "brand", "package_size", "standard_uom",
        "etd", "demand_quantity", "value_in_usd", "source_type", "demand_number", 
        "customer", "legal_entity"
    ]].copy()

    display_df["demand_quantity"] = display_df["demand_quantity"].apply(lambda x: f"{x:,.0f}")
    display_df["value_in_usd"] = display_df["value_in_usd"].apply(lambda x: f"${x:,.2f}")
    display_df["etd"] = pd.to_datetime(display_df["etd"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.dataframe(display_df, use_container_width=True)


# === Group and pivot demand data by day/week/month ===
def show_grouped_demand_summary(filtered_df, start_date, end_date):
    st.markdown("### 📦 Grouped Demand by Product (Pivot View)")
    st.markdown(f"📅 Showing demand from **{start_date}** to **{end_date}**")

    col_period, col_filter = st.columns(2)
    with col_period:
        period = st.selectbox("Group By Period", ["Daily", "Weekly", "Monthly"], index=1)
    with col_filter:
        show_only_nonzero = st.checkbox("Show only products with quantity > 0", value=True)

    df_summary = filtered_df.copy()

    # Create "period" field depending on time aggregation level
    if period == "Daily":
        df_summary["period"] = df_summary["etd"].dt.strftime("%Y-%m-%d")
    elif period == "Weekly":
        # Use ISO week for proper week numbering
        df_summary["year"] = df_summary["etd"].dt.isocalendar().year
        df_summary["week"] = df_summary["etd"].dt.isocalendar().week
        df_summary["period"] = "Week " + df_summary["week"].astype(str).str.zfill(2) + " - " + df_summary["year"].astype(str)
    else:  # Monthly
        df_summary["period"] = df_summary["etd"].dt.to_period("M").dt.strftime("%b %Y")

    # Pivot table: product x period
    pivot_df = (
        df_summary
        .groupby(["product_name", "pt_code", "period"])
        .agg(total_quantity=("demand_quantity", "sum"))
        .reset_index()
        .pivot(index=["product_name", "pt_code"], columns="period", values="total_quantity")
        .fillna(0)
        .reset_index()
    )

    pivot_df = sort_period_columns(pivot_df, period)

    if show_only_nonzero:
        pivot_df = pivot_df[pivot_df.iloc[:, 2:].sum(axis=1) > 0]

    # Format quantities
    pivot_df.iloc[:, 2:] = pivot_df.iloc[:, 2:].applymap(lambda x: f"{x:,.0f}")
    st.dataframe(pivot_df, use_container_width=True)

    # Summary row: total quantity + value across all products per period
    df_grouped = df_summary.copy()
    df_grouped["demand_quantity"] = pd.to_numeric(df_grouped["demand_quantity"], errors='coerce').fillna(0)
    df_grouped["value_in_usd"] = pd.to_numeric(df_grouped["value_in_usd"], errors='coerce').fillna(0)

    pivot_qty = df_grouped.groupby("period").agg(total_quantity=("demand_quantity", "sum")).T
    pivot_val = df_grouped.groupby("period").agg(total_value_usd=("value_in_usd", "sum")).T

    pivot_qty.index = ["🔢 TOTAL QUANTITY"]
    pivot_val.index = ["💵 TOTAL VALUE (USD)"]

    pivot_final = pd.concat([pivot_qty, pivot_val])
    pivot_final = pivot_final.reset_index().rename(columns={"index": "Metric"})

    # Format the summary values
    for col in pivot_final.columns[1:]:
        if "🔢 TOTAL QUANTITY" in pivot_final["Metric"].values:
            pivot_final.loc[pivot_final["Metric"] == "🔢 TOTAL QUANTITY", col] = (
                pivot_final.loc[pivot_final["Metric"] == "🔢 TOTAL QUANTITY", col]
                .astype(float)
                .map("{:,.0f}".format)
            )
        if "💵 TOTAL VALUE (USD)" in pivot_final["Metric"].values:
            pivot_final.loc[pivot_final["Metric"] == "💵 TOTAL VALUE (USD)", col] = (
                pivot_final.loc[pivot_final["Metric"] == "💵 TOTAL VALUE (USD)", col]
                .astype(float)
                .map("${:,.2f}".format)
            )

    pivot_final = sort_period_columns(pivot_final, period)

    st.markdown("🔢 Column Total (All Products)")
    st.dataframe(pivot_final, use_container_width=True)

    # === Export Excel ===
    excel_data = convert_df_to_excel(pivot_df)
    st.download_button(
        label="📤 Export to Excel",
        data=excel_data,
        file_name="grouped_outbound_demand.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# === Helper: Sort period columns chronologically ===
def sort_period_columns(df, period_type):
    """Sort dataframe columns by period"""
    # Identify non-period columns
    if "Metric" in df.columns:
        info_cols = ["Metric"]
    else:
        info_cols = ["product_name", "pt_code"]
    
    # Get period columns and filter out invalid ones
    period_cols = [col for col in df.columns if col not in info_cols]
    period_cols = [p for p in period_cols if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"]

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
    else:
        # Daily: sort as strings (YYYY-MM-DD format sorts correctly)
        sorted_periods = sorted(period_cols)

    return df[info_cols + sorted_periods]


# === Helper: Convert pivoted DataFrame to downloadable Excel ===
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Grouped Demand")
        
        # Auto-adjust column widths
        worksheet = writer.sheets["Grouped Demand"]
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, max_len)
            
    return output.getvalue()