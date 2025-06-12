import streamlit as st
from utils.auth import AuthManager

# === Authentication Check ===
# Initialize auth manager
auth_manager = AuthManager()

# Check if user is authenticated
if not auth_manager.check_session():
    st.switch_page("pages/0_🔐_Login.py")  # Updated path
    st.stop()

import pandas as pd
from datetime import datetime

# Import refactored modules
from utils.data_manager import DataManager
from utils.settings_manager import SettingsManager
from utils.formatters import format_number, format_currency, format_percentage
from utils.helpers import save_to_session_state
from utils.session_state import initialize_session_state




# === Page Config ===
st.set_page_config(
    page_title="Supply Chain Control Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Initialize Session State ===
initialize_session_state()

# === Initialize Components ===
@st.cache_resource
def get_data_manager():
    """Get singleton DataManager instance"""
    return DataManager()

@st.cache_resource
def get_settings_manager():
    """Get singleton SettingsManager instance"""
    return SettingsManager()

data_manager = get_data_manager()
settings_manager = get_settings_manager()


# === Sidebar Configuration ===
with st.sidebar:
    # User info section
    st.markdown("### 👤 User Information")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{auth_manager.get_user_display_name()}**")
        st.caption(f"Role: {st.session_state.get('user_role', 'user').title()}")
    with col2:
        if st.button("🚪", help="Logout"):
            auth_manager.logout()
            st.switch_page("pages/0_🔐_Login.py")
    
    # Session info
    if 'login_time' in st.session_state:
        elapsed = datetime.now() - st.session_state.login_time
        hours, remainder = divmod(elapsed.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        st.caption(f"Session time: {hours}h {minutes}m")
        
        # Warning if session is about to expire (7+ hours)
        if hours >= 7:
            st.warning("⚠️ Session expires in less than 1 hour")
    
    st.markdown("---")
    
    # Original Configuration section
    st.header("⚙️ Configuration")
    
    # Manual refresh
    if st.button("🔄 Refresh All Data", use_container_width=True, type="primary"):
        data_manager.clear_cache()
        st.rerun()
    
    # Auto refresh option
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Interval (minutes)", 5, 60, 15, 5)
        st.info(f"⏱️ Auto-refresh every {refresh_interval} min")


# === Main Dashboard ===
st.title("🏭 Supply Chain Control Center")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === 1. DATA LOADING SECTION ===
st.header("📊 Data Loading Status")

# Load all data with progress tracking
with st.spinner("Loading all data sources..."):
    all_data = data_manager.preload_all_data()
    
    # Store in session state for other pages
    save_to_session_state('all_data_loaded', True)
    save_to_session_state('data_load_time', datetime.now())

# Show data loading status
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📤 Demand Data")
    demand_oc = all_data.get('demand_oc', pd.DataFrame())
    demand_forecast = all_data.get('demand_forecast', pd.DataFrame())
    
    if not demand_oc.empty or not demand_forecast.empty:
        st.success("✅ Loaded successfully")
        st.caption(f"OC Records: {len(demand_oc):,}")
        st.caption(f"Forecast Records: {len(demand_forecast):,}")
    else:
        st.error("❌ No demand data available")

with col2:
    st.markdown("### 📥 Supply Data")
    inventory = all_data.get('supply_inventory', pd.DataFrame())
    pending_can = all_data.get('supply_can', pd.DataFrame())
    pending_po = all_data.get('supply_po', pd.DataFrame())
    pending_wht = all_data.get('supply_wh_transfer', pd.DataFrame())
    
    if not inventory.empty or not pending_can.empty or not pending_po.empty or not pending_wht.empty:
        st.success("✅ Loaded successfully")
        st.caption(f"Inventory: {len(inventory):,}")
        st.caption(f"Pending CAN: {len(pending_can):,}")
        st.caption(f"Pending PO: {len(pending_po):,}")
        st.caption(f"Pending Transfer: {len(pending_wht):,}")
    else:
        st.error("❌ No supply data available")

with col3:
    st.markdown("### 🗂️ Master Data")
    products = all_data.get('master_products', pd.DataFrame())
    customers = all_data.get('master_customers', pd.DataFrame())
    
    if not products.empty:
        st.success("✅ Loaded successfully")
        st.caption(f"Products: {len(products):,}")
        st.caption(f"Customers: {len(customers):,}")
    else:
        st.warning("⚠️ Master data incomplete")

# === 2. KEY INSIGHTS SECTION ===
st.header("🔍 Supply Chain Insights")

# Get calculated insights
insights = data_manager.get_insights()

# Tab layout for different insight categories
tab1, tab2, tab3 = st.tabs(["📤 Demand Insights", "📥 Supply Insights", "⚠️ Risk Alerts"])

with tab1:

    # Add scope note
    st.caption("📌 Based on Order Confirmations (OC) only")

    # Demand metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Pending Orders",
            format_number(insights.get('demand_oc_pending_count', 0)),
            help="Total OC pending delivery"
        )
        st.caption(f"Value: {format_currency(insights.get('demand_oc_pending_value', 0), 'USD')}")
    
    with col2:
        overdue_count = insights.get('demand_overdue_count', 0)
        overdue_value = insights.get('demand_overdue_value', 0)
        st.metric(
            "⏰ Overdue Orders",
            format_number(overdue_count),
            delta=f"{format_currency(overdue_value, 'USD')}" if overdue_count > 0 else None,
            delta_color="inverse"
        )
    
    with col3:
        critical_count = insights.get('critical_shortage_count', 0)
        critical_value = insights.get('critical_shortage_value', 0)
        st.metric(
            "🚨 Critical (3 days)",
            format_number(critical_count),
            help="Orders due in next 3 days"
        )
        if critical_count > 0:
            st.caption(f"Value: {format_currency(critical_value, 'USD')}")
    
    with col4:
        missing_etd = insights.get('demand_missing_etd', 0)
        if missing_etd > 0:
            st.metric(
                "⚠️ Missing ETD",
                format_number(missing_etd),
                delta="Data quality issue",
                delta_color="inverse"
            )
        else:
            st.metric("✅ Data Quality", "Good", help="All orders have ETD")

with tab2:

    # Add scope note
    st.caption("📌 Based on current inventory stock only")

    # Supply metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Inventory Value",
            format_currency(insights.get('inventory_total_value', 0), 'USD'),
            help="Current stock value"
        )
    
    with col2:
        expired_count = insights.get('expired_items_count', 0)
        expired_value = insights.get('expired_items_value', 0)
        st.metric(
            "💀 Expired Items",
            format_number(expired_count),
            delta=f"-{format_currency(expired_value, 'USD')}" if expired_count > 0 else None,
            delta_color="inverse"
        )
    
    with col3:
        near_expiry_count = insights.get('near_expiry_7d_count', 0)
        near_expiry_value = insights.get('near_expiry_7d_value', 0)
        st.metric(
            "📅 Expiring Soon (7d)",
            format_number(near_expiry_count),
            help="Items expiring in 7 days"
        )
        if near_expiry_count > 0:
            st.caption(f"Value: {format_currency(near_expiry_value, 'USD')}")
    
    with col4:
        excess_count = insights.get('excess_inventory_count', 0)
        excess_value = insights.get('excess_inventory_value', 0)
        if excess_count > 0:
            st.metric(
                "📦 Excess Inventory",
                format_number(excess_count),
                delta=format_currency(excess_value, 'USD'),
                help="Stock > 6 months"
            )
        else:
            st.metric("✅ Inventory Health", "Good")

with tab3:
    # Supply Chain Risk Alerts
    st.markdown("### 🚨 Critical Supply Chain Risks")
    
    # Get alerts
    critical_alerts = data_manager.get_critical_alerts()
    warnings = data_manager.get_warnings()
    
    if critical_alerts:
        for alert in critical_alerts[:5]:  # Top 5 critical alerts
            col1, col2 = st.columns([4, 1])
            with col1:
                st.error(f"{alert['icon']} **{alert['message']}** - {alert.get('action', '')}")
            with col2:
                if alert.get('value'):
                    st.metric("Impact", alert['value'], label_visibility="collapsed")
    else:
        st.success("✅ No critical issues detected")
    
    st.markdown("### ⚠️ Warnings")
    if warnings:
        for warning in warnings[:5]:  # Top 5 warnings
            col1, col2 = st.columns([4, 1])
            with col1:
                st.warning(f"{warning['icon']} {warning['message']}")
            with col2:
                if warning.get('value'):
                    st.metric("Impact", warning['value'], label_visibility="collapsed")
    else:
        st.info("ℹ️ No warnings at this time")

# === 3. PRODUCT MATCHING ANALYSIS ===
st.header("🔗 Product Matching Analysis")

# Add scope indicator
st.caption("📌 **Scope**: OC orders vs Current inventory only • Excludes: Forecasts, Pending supply (PO/CAN/Transfers)")

# Product matching metrics
matched_products = insights.get('matched_products', set())
demand_only = insights.get('demand_only_products', set())
supply_only = insights.get('supply_only_products', set())

total_products = len(matched_products) + len(demand_only) + len(supply_only)

if total_products > 0:
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        match_rate = len(matched_products) / total_products * 100 if total_products > 0 else 0
        st.metric(
            "Match Rate",
            format_percentage(match_rate),
            help="Products with both demand (OC) and supply (inventory)"
        )
    
    with col2:
        if len(demand_only) > 0:
            st.metric(
                "📤 OC Only",  # Changed from "Demand Only"
                format_number(len(demand_only)),
                delta=f"-{format_currency(insights.get('demand_only_value', 0), 'USD')}",
                delta_color="inverse",
                help="Products with OC orders but no current inventory"
            )
        else:
            st.metric("📤 OC Only", "0", help="No unmatched OC orders")
    
    with col3:
        if len(supply_only) > 0:
            st.metric(
                "📥 Inventory Only",  # Changed from "Supply Only"
                format_number(len(supply_only)),
                delta=format_currency(insights.get('supply_only_value', 0), 'USD'),
                help="Products in inventory but no OC orders"
            )
        else:
            st.metric("📥 Inventory Only", "0", help="No unmatched inventory")
    
    with col4:
        st.metric(
            "🔗 Matched",
            format_number(len(matched_products)),
            help="Products with both OC orders and inventory"
        )
    
    # Add info box for full analysis
    st.info("💡 **For comprehensive analysis** including Forecasts and Pending Supply (PO, CAN, Transfers), use [GAP Analysis →](pages/3_📊_GAP_Analysis.py)")
    
    # Visual breakdown
    if st.checkbox("Show detailed breakdown"):
        st.markdown("#### Product Distribution (OC vs Inventory)")
        
        # Create a simple bar chart data
        chart_data = pd.DataFrame({
            'Category': ['Matched', 'OC Only', 'Inventory Only'],
            'Count': [len(matched_products), len(demand_only), len(supply_only)],
            'Value (USD)': [
                0,  # Matched products don't have "risk" value
                insights.get('demand_only_value', 0),
                insights.get('supply_only_value', 0)
            ]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(chart_data.set_index('Category')['Count'])
        with col2:
            st.bar_chart(chart_data[chart_data['Value (USD)'] > 0].set_index('Category')['Value (USD)'])

else:
    st.warning("⚠️ No product data available for matching analysis")

# === 4. QUICK ACTIONS ===
st.header("🎯 Recommended Actions")

# Add context box
st.info("""
**📊 Dashboard shows**: Current snapshot (OC orders vs Stock on hand)  
**🔍 For full analysis**: Include forecasts & pending supply → Use GAP Analysis
""")


col1, col2, col3 = st.columns(3)

with col1:
    # Check if we have demand-supply imbalance
    if len(demand_only) > 0 or insights.get('critical_shortage_count', 0) > 0:
        st.markdown("### 🚨 Address Shortages")
        st.write("Critical items need immediate attention")
        if st.button("→ Go to GAP Analysis", type="primary", use_container_width=True):
            st.switch_page("pages/3_📊_GAP_Analysis.py")
        if st.button("→ Create PO Plan", use_container_width=True):
            st.switch_page("pages/5_📌_PO_Suggestions.py")

with col2:
    # Check for allocation needs
    if insights.get('demand_overdue_count', 0) > 0:
        st.markdown("### 📦 Manage Allocations")
        st.write("Overdue orders need allocation")
        if st.button("→ Allocation Planning", type="primary", use_container_width=True):
            st.switch_page("pages/4_🧩_Allocation_Plan.py")

with col3:
    # Check for inventory issues
    if insights.get('expired_items_count', 0) > 0 or insights.get('near_expiry_7d_count', 0) > 0:
        st.markdown("### 🗑️ Inventory Cleanup")
        st.write("Handle expired/expiring items")
        if st.button("→ Review Inventory", type="primary", use_container_width=True):
            st.switch_page("pages/2_📥_Supply_Analysis.py")

# === Auto-refresh Logic ===
if auto_refresh:
    import time
    st.empty()  # Placeholder for countdown
    time.sleep(refresh_interval * 60)
    data_manager.clear_cache()
    st.rerun()


# === Footer ===
st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.caption(f"Supply Chain Control Center v1.0")

with col2:
    st.caption(f"Logged in as: {auth_manager.get_user_display_name()} ({st.session_state.get('user_role', 'user')})")

with col3:
    st.caption(f"Data freshness: {datetime.now().strftime('%H:%M:%S')}")

# Update the debug section to include user info:
# In the debug mode section at the bottom:

# Debug mode (hidden)
if st.checkbox("🐛", value=False, label_visibility="collapsed"):
    with st.expander("Debug Information"):
        st.write("**User Session:**")
        st.write(f"- User ID: {st.session_state.get('user_id')}")
        st.write(f"- Username: {st.session_state.get('username')}")
        st.write(f"- Role: {st.session_state.get('user_role')}")
        st.write(f"- Login Time: {st.session_state.get('login_time')}")
        
        st.write("\n**Loaded Data:**")
        for key, df in all_data.items():
            st.write(f"- {key}: {len(df)} rows, {df.shape[1] if not df.empty else 0} columns")
        st.write(f"**Cache Status:** Active")
        st.write(f"**Settings Applied:** {len(settings_manager.get_applied_adjustments())}")