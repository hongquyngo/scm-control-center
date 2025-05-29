import pandas as pd
from sqlalchemy import text
from utils.db import get_db_engine
import streamlit as st

# === DEMAND DATA LOADERS ===

@st.cache_data(ttl=1800)
def load_outbound_demand_data():
    """Load OC (Order Confirmation) pending delivery data"""
    engine = get_db_engine()
    query = """
        SELECT * FROM prostechvn.outbound_oc_pending_delivery_view;
    """
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=1800)
def load_customer_forecast_data():
    """Load customer demand forecast data"""
    engine = get_db_engine()
    query = """
        SELECT * FROM prostechvn.customer_demand_forecast_full_view;
    """
    return pd.read_sql(text(query), engine)


# === SUPPLY DATA LOADERS ===

@st.cache_data(ttl=1800)
def load_inventory_data():
    """Load current inventory data"""
    engine = get_db_engine()
    query = "SELECT * FROM prostechvn.inventory_detailed_view"
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=1800)
def load_pending_can_data():
    """Load pending CAN (Container Arrival Note) data"""
    engine = get_db_engine()
    query = "SELECT * FROM prostechvn.can_pending_stockin_view"
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=1800)
def load_pending_po_data():
    """Load pending Purchase Order data"""
    engine = get_db_engine()
    query = """
    SELECT * FROM prostechvn.purchase_order_full_view
    WHERE pending_standard_arrival_quantity > 0
    """
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=1800)
def load_pending_wh_transfer_data():
    """Load pending Warehouse Transfer data"""
    engine = get_db_engine()
    query = """
    SELECT 
    *
    FROM prostechvn.warehouse_transfer_details_view wtdv
    WHERE wtdv.is_completed = 0  -- Only pending transfers
    """
    return pd.read_sql(text(query), engine)


# === MASTER DATA LOADERS ===

@st.cache_data(ttl=3600)
def load_product_master():
    """Load product master data"""
    engine = get_db_engine()
    query = """
    SELECT 
        p.id as product_id,
        p.pt_code,
        p.name as product_name,
        p.description,
        b.brand_name as brand,
        p.package_size,
        p.uom as standard_uom,
        p.shelf_life,
        p.shelf_life_uom,
        p.storage_condition,
        p.hs_code,
        p.legacy_pt_code
    FROM products p
    LEFT JOIN brands b ON p.brand_id = b.id
    WHERE p.delete_flag = 0
    """
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=3600)
def load_customer_master():
    """Load customer master data"""
    engine = get_db_engine()
    query = """

    SELECT 
        c.id as customer_id,
        c.english_name as customer_name,
        c.company_code as customer_code,
        c.registration_code,
        c.local_name,
        tnc.limit_credit as creadit_limit,
        cur.code as creadit_limit_currency,
        pt.name as payment_term_days
        
    FROM companies c
    JOIN companies_company_types cct ON c.id = cct.companies_id
    JOIN company_types ct ON cct.company_type_id = ct.id
    JOIN term_and_conditions tnc ON c.customer_term_condition_id = tnc.id
    JOIN payment_terms pt ON tnc.payment_term_id = pt.id
    JOIN currencies cur ON tnc.credit_currency_id = cur.id
    
    WHERE c.delete_flag = 0
    AND ct.name = 'Customer'

    """
    return pd.read_sql(text(query), engine)


@st.cache_data(ttl=3600)
def load_vendor_master():
    """Load vendor/supplier master data"""
    engine = get_db_engine()
    query = """

    SELECT 
        c.id as vendor_id,
        c.english_name as vendor_name,
        c.company_code as vendor_code,
        c.registration_code,
        c.local_name
    FROM companies c
    JOIN companies_company_types cct ON c.id = cct.companies_id
    JOIN company_types ct ON cct.company_type_id = ct.id
    WHERE c.delete_flag = 0
    AND ct.name = 'Vendor'

    """
    return pd.read_sql(text(query), engine)



@st.cache_data(ttl=1800)
def load_active_allocations():
    """Load active allocations affecting supply"""
    engine = get_db_engine()
    query = """
    SELECT * FROM active_allocations_view
    WHERE undelivered_qty > 0
    """
    return pd.read_sql(text(query), engine)

@st.cache_data(ttl=1800)
def load_allocation_history(product_id=None, customer_id=None, days_back=30):
    """Load allocation history with filters"""
    engine = get_db_engine()
    
    conditions = ["1=1"]
    params = {}
    
    if product_id:
        conditions.append("ad.product_id = :product_id")
        params['product_id'] = product_id
        
    if customer_id:
        conditions.append("ad.customer_id = :customer_id")
        params['customer_id'] = customer_id
    
    if days_back:
        conditions.append("ap.allocation_date >= DATE_SUB(CURRENT_DATE(), INTERVAL :days DAY)")
        params['days'] = days_back
    
    query = f"""
    SELECT 
        ap.allocation_number,
        ap.allocation_date,
        ap.allocation_method,
        ap.status as plan_status,
        ad.*,
        (ad.allocated_qty - ad.delivered_qty) as remaining_qty,
        ROUND(ad.delivered_qty / NULLIF(ad.allocated_qty, 0) * 100, 1) as fulfillment_rate
    FROM allocation_details ad
    JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
    WHERE {' AND '.join(conditions)}
    ORDER BY ap.allocation_date DESC, ad.id
    """
    
    return pd.read_sql(text(query), engine, params=params)


# === UTILITY FUNCTIONS ===

def refresh_all_data():
    """Clear cache to refresh all data"""
    st.cache_data.clear()
    st.success("✅ All data caches cleared. Data will be refreshed on next load.")


def get_data_last_updated():
    """Get timestamp of last data update"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")