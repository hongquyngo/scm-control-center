import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO

from data_loader import (
    load_inventory_data,
    load_pending_can_data,
    load_pending_po_data
)

def show_inbound_supply_tab():
    st.subheader("📥 Inbound Supply Capability")

    supply_source = select_supply_source()
    exclude_expired = st.checkbox("📅 Exclude expired inventory", value=True, key="supply_expired_checkbox")
    df_all = load_and_prepare_supply_data(supply_source, exclude_expired)

    if df_all.empty:
        st.info("No supply data available.")
        return

    filtered_df, start_date, end_date = apply_supply_filters(df_all)
    show_supply_detail_table(filtered_df)
    show_grouped_supply_summary(filtered_df, start_date, end_date)


def select_supply_source():
    return st.radio(
        "Select Supply Source:",
        ["Inventory Only", "Pending CAN Only", "Pending PO Only", "All"],
        horizontal=True,
        key="supply_source_radio"
    )


def load_and_prepare_supply_data(supply_source, exclude_expired=True):
    today = pd.to_datetime("today").normalize()
    df_parts = []

    if supply_source in ["Inventory Only", "All"]:
        inv_df = load_inventory_data()
        inv_df["source_type"] = "Inventory"
        inv_df["supply_number"] = inv_df["inventory_history_id"].astype(str)
        inv_df["date_ref"] = today
        inv_df["quantity"] = pd.to_numeric(inv_df["remaining_quantity"], errors="coerce").fillna(0)
        inv_df["value_in_usd"] = pd.to_numeric(inv_df["inventory_value_usd"], errors="coerce").fillna(0)
        inv_df["legal_entity"] = inv_df["owning_company_name"]
        inv_df["expiry_date"] = pd.to_datetime(inv_df["expiry_date"], errors="coerce")
        if exclude_expired:
            inv_df = inv_df[(inv_df["expiry_date"].isna()) | (inv_df["expiry_date"] >= today)]
        df_parts.append(inv_df)

    if supply_source in ["Pending CAN Only", "All"]:
        can_df = load_pending_can_data()
        can_df["source_type"] = "Pending CAN"
        can_df["supply_number"] = can_df["arrival_note_number"].astype(str)
        can_df["date_ref"] = pd.to_datetime(can_df["arrival_date"], errors="coerce")
        can_df["quantity"] = pd.to_numeric(can_df["pending_quantity"], errors="coerce").fillna(0)
        can_df["value_in_usd"] = pd.to_numeric(can_df["pending_value_usd"], errors="coerce").fillna(0)
        can_df["legal_entity"] = can_df["consignee"]
        df_parts.append(can_df)

    if supply_source in ["Pending PO Only", "All"]:
        po_df = load_pending_po_data()
        po_df["source_type"] = "Pending PO"
        po_df["supply_number"] = po_df["po_number"].astype(str)
        po_df["date_ref"] = pd.to_datetime(po_df["cargo_ready_date"], errors="coerce")
        po_df["quantity"] = pd.to_numeric(po_df["pending_standard_arrival_quantity"], errors="coerce").fillna(0)
        po_df["value_in_usd"] = pd.to_numeric(po_df["outstanding_arrival_amount_usd"], errors="coerce").fillna(0)
        po_df["legal_entity"] = po_df["legal_entity"]
        df_parts.append(po_df)

    if not df_parts:
        return pd.DataFrame()

    def standardize(df):
        df["pt_code"] = df["pt_code"].astype(str)
        df["product_name"] = df.get("product_name", "").astype(str)
        df["brand"] = df.get("brand", "").astype(str)
        df["standard_uom"] = df.get("standard_uom", "")
        df["package_size"] = df.get("package_size", "")
        return df[[
            "source_type", "supply_number", "pt_code", "product_name", "brand", 
            "package_size", "standard_uom", "legal_entity", "date_ref", 
            "quantity", "value_in_usd"
        ]]

    return pd.concat([standardize(df) for df in df_parts], ignore_index=True)


def apply_supply_filters(df):
    with st.expander("📎 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_entity = st.multiselect("Legal Entity", sorted(df["legal_entity"].dropna().unique()), key="supply_legal_entity")
        with col2:
            selected_brand = st.multiselect("Brand", sorted(df["brand"].dropna().unique()), key="supply_brand")
        with col3:
            selected_pt = st.multiselect("PT Code", sorted(df["pt_code"].dropna().unique()), key="supply_pt_code")

        col4, col5 = st.columns(2)
        with col4:
            default_start = df["date_ref"].min().date() if pd.notnull(df["date_ref"].min()) else datetime.today().date()
            start_date = st.date_input("From Date (Reference)", default_start, key="supply_start_date")
        with col5:
            default_end = df["date_ref"].max().date() if pd.notnull(df["date_ref"].max()) else datetime.today().date()
            end_date = st.date_input("To Date (Reference)", default_end, key="supply_end_date")

    filtered_df = df.copy()
    if selected_entity:
        filtered_df = filtered_df[filtered_df["legal_entity"].isin(selected_entity)]
    if selected_brand:
        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brand)]
    if selected_pt:
        filtered_df = filtered_df[filtered_df["pt_code"].isin(selected_pt)]

    # Convert dates for comparison
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    filtered_df = filtered_df[
        (filtered_df["date_ref"] >= start_ts) & 
        (filtered_df["date_ref"] <= end_ts)
    ]

    return filtered_df, start_date, end_date


def show_supply_detail_table(df):
    st.markdown("### 📄 Supply Capability Detail")

    total_unique_products = df["pt_code"].nunique()
    total_value_usd = df["value_in_usd"].sum()

    st.markdown(f"🔢 Total Unique Products: **{int(total_unique_products):,}**  💵 Total Value (USD): **${total_value_usd:,.2f}**")

    df_disp = df.copy()
    df_disp["quantity"] = df_disp["quantity"].apply(lambda x: f"{x:,.0f}")
    df_disp["value_in_usd"] = df_disp["value_in_usd"].apply(lambda x: f"${x:,.2f}")
    df_disp["date_ref"] = pd.to_datetime(df_disp["date_ref"], errors="coerce").dt.strftime("%Y-%m-%d")

    df_disp.rename(columns={"quantity": "supply_quantity"}, inplace=True)

    st.dataframe(df_disp[[
        "pt_code", "product_name", "brand", "package_size", "standard_uom",
        "date_ref", "supply_quantity", "value_in_usd", "source_type", 
        "supply_number", "legal_entity"
    ]], use_container_width=True)


def show_grouped_supply_summary(df, start_date, end_date):
    st.markdown("### 📊 Grouped Supply by Period")
    st.markdown(f"📅 From **{start_date}** to **{end_date}**")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Group by Period", ["Daily", "Weekly", "Monthly"], index=1)
    with col2:
        show_only_nonzero = st.checkbox("Show only products with quantity > 0", value=True, key="supply_show_nonzero")

    df_summary = df.copy()

    # Create period column
    if period == "Daily":
        df_summary["period"] = df_summary["date_ref"].dt.strftime("%Y-%m-%d")
    elif period == "Weekly":
        # Use ISO week for proper week numbering
        df_summary["year"] = df_summary["date_ref"].dt.isocalendar().year
        df_summary["week"] = df_summary["date_ref"].dt.isocalendar().week
        df_summary["period"] = "Week " + df_summary["week"].astype(str).str.zfill(2) + " - " + df_summary["year"].astype(str)
    else:  # Monthly
        df_summary["period"] = df_summary["date_ref"].dt.to_period("M").dt.strftime("%b %Y")

    pivot_df = (
        df_summary
        .groupby(["product_name", "pt_code", "period"])
        .agg(total_quantity=("quantity", "sum"))
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

    # === Totals ===
    df_grouped = df_summary.copy()
    df_grouped["quantity"] = pd.to_numeric(df_grouped["quantity"], errors="coerce").fillna(0)
    df_grouped["value_in_usd"] = pd.to_numeric(df_grouped["value_in_usd"], errors="coerce").fillna(0)

    pivot_qty = df_grouped.groupby("period").agg(total_quantity=("quantity", "sum")).T
    pivot_val = df_grouped.groupby("period").agg(total_value_usd=("value_in_usd", "sum")).T

    pivot_qty.index = ["🔢 TOTAL QUANTITY"]
    pivot_val.index = ["💰 TOTAL VALUE (USD)"]

    pivot_final = pd.concat([pivot_qty, pivot_val])
    pivot_final = pivot_final.reset_index().rename(columns={"index": "Metric"})

    # Format totals
    for col in pivot_final.columns[1:]:
        if "🔢 TOTAL QUANTITY" in pivot_final["Metric"].values:
            pivot_final.loc[pivot_final["Metric"] == "🔢 TOTAL QUANTITY", col] = (
                pivot_final.loc[pivot_final["Metric"] == "🔢 TOTAL QUANTITY", col]
                .astype(float).map("{:,.0f}".format)
            )
        if "💰 TOTAL VALUE (USD)" in pivot_final["Metric"].values:
            pivot_final.loc[pivot_final["Metric"] == "💰 TOTAL VALUE (USD)", col] = (
                pivot_final.loc[pivot_final["Metric"] == "💰 TOTAL VALUE (USD)", col]
                .astype(float).map("${:,.2f}".format)
            )

    pivot_final = sort_period_columns(pivot_final, period)

    st.markdown("🔢 Column Total (All Products)")
    st.dataframe(pivot_final, use_container_width=True)

    export_data = convert_df_to_excel(pivot_df)
    st.download_button(
        label="📤 Export Grouped Supply to Excel",
        data=export_data,
        file_name="grouped_supply_by_period.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


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


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Grouped Supply")
        
        # Auto-adjust column widths
        worksheet = writer.sheets["Grouped Supply"]
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, max_len)
            
    return output.getvalue()