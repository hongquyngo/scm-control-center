#data_loader.py

import pandas as pd
from sqlalchemy import text
from db import get_db_engine
import streamlit as st

@st.cache_data(ttl=1800)
def load_data():
    """
    Load all required datasets from the database.

    Returns:
        tuple: (inv_df, inv_by_kpi_center_df, backlog_df, backlog_by_kpi_center_df)
    """
    engine = get_db_engine()

    inv_query = """
        SELECT *
        FROM prostechvn.sales_invoice_full_looker_view
        WHERE DATE(inv_date) >= DATE_FORMAT(CURDATE(), '%Y-01-01')
          AND inv_date < DATE_ADD(DATE_FORMAT(CURDATE(), '%Y-%m-%d'), INTERVAL 1 DAY);
    """

    inv_by_kpi_center_query = """
        SELECT *
        FROM prostechvn.sales_report_by_kpi_center_flat_looker_view
        WHERE DATE(inv_date) >= DATE_FORMAT(CURDATE(), '%Y-01-01')
          AND inv_date < DATE_ADD(DATE_FORMAT(CURDATE(), '%Y-%m-%d'), INTERVAL 1 DAY);
    """

    backlog_query = """
        SELECT *
        FROM prostechvn.order_confirmation_full_looker_view
        WHERE IFNULL(total_invoiced_selling_quantity, 0) < selling_quantity;
    """

    backlog_by_kpi_center_query = """
        SELECT *
        FROM prostechvn.backlog_by_kpi_center_flat_looker_view
        WHERE IFNULL(total_invoiced_selling_quantity, 0) < selling_quantity;
    """

    inv_df = pd.read_sql(text(inv_query), engine)
    inv_by_kpi_center_df = pd.read_sql(text(inv_by_kpi_center_query), engine)
    backlog_df = pd.read_sql(text(backlog_query), engine)
    backlog_by_kpi_center_df = pd.read_sql(text(backlog_by_kpi_center_query), engine)

    return inv_df, inv_by_kpi_center_df, backlog_df, backlog_by_kpi_center_df


@st.cache_data(ttl=1800)
def load_sales_performance_data():

    engine = get_db_engine()
    """
    Load data for sales performance page.
    """
    sales_data_by_salesperson_query = """
        SELECT * FROM prostechvn.sales_gp1_by_split_view
    """
    backlog_data_by_salesperson_query = """
        SELECT * FROM prostechvn.backlog_by_salesperson_looker_view
    """
    kpi_by_salesperson_query = """
        SELECT * FROM prostechvn.sales_employee_kpi_assignments_view
    """

    sales_report_by_salesperson_df = pd.read_sql(text(sales_data_by_salesperson_query), engine)
    backlog_report_by_salesperson_df = pd.read_sql(text(backlog_data_by_salesperson_query), engine)
    kpi_by_salesperson_df = pd.read_sql(text(kpi_by_salesperson_query), engine)


    return sales_report_by_salesperson_df, backlog_report_by_salesperson_df, kpi_by_salesperson_df


@st.cache_data(ttl=1800)
def load_outbound_demand_data():
    engine = get_db_engine()
    query = """
        SELECT * FROM prostechvn.outbound_oc_pending_delivery_view;
    """
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=1800)
def load_customer_forecast_data():
    engine = get_db_engine()
    query = """
        SELECT * FROM prostechvn.customer_demand_forecast_full_view;
    """
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=1800)
def load_inventory_data():
    engine = get_db_engine()
    query = "SELECT * FROM prostechvn.inventory_detailed_view"
    return pd.read_sql(text(query), engine)

@st.cache_data(ttl=1800)
def load_pending_can_data():
    engine = get_db_engine()
    query = "SELECT * FROM prostechvn.can_pending_stockin_view"
    return pd.read_sql(text(query), engine)

@st.cache_data(ttl=1800)
def load_pending_po_data():
    engine = get_db_engine()
    query = """
    SELECT * FROM prostechvn.purchase_order_full_view
    WHERE pending_standard_arrival_quantity > 0
    """
    return pd.read_sql(text(query), engine)
