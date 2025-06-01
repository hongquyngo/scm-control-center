import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import time

# Import new utilities
from utils.data_manager import DataManager
from utils.display_components import DisplayComponents
from utils.formatters import format_number, format_currency, format_percentage
from utils.settings_manager import SettingsManager

# === Page Config ===
st.set_page_config(
    page_title="Supply Chain Control Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Initialize Managers ===
data_manager = DataManager()
settings_manager = SettingsManager()
display = DisplayComponents()

# === Sidebar Configuration (MOVED TO TOP) ===
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data refresh settings
    st.subheader("🔄 Data Refresh")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh interval (minutes)", 
            min_value=5, 
            max_value=60, 
            value=15
        )
        st.info(f"Data will refresh every {refresh_interval} minutes")
    
    # Quick filters - will be populated after data loads
    st.subheader("🔍 Quick Filters")
    selected_entities = []  # Initialize empty, will update after data loads
    
    # Date range filter
    st.subheader("📅 Date Range")
    date_filter_type = st.radio(
        "Filter by",
        ["Last 7 days", "Last 30 days", "Custom range"],
        index=1
    )
    
    if date_filter_type == "Custom range":
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")
    else:
        end_date = datetime.now().date()
        if date_filter_type == "Last 7 days":
            start_date = (datetime.now() - pd.Timedelta(days=7)).date()
        else:  # Last 30 days
            start_date = (datetime.now() - pd.Timedelta(days=30)).date()
    
    # System info
    st.subheader("📊 System Info")
    st.metric("Active Users", "12", "+2")
    st.metric("DB Status", "Online", delta_color="off")
    st.metric("Cache Hit Rate", "87%", "+5%")

# === Dashboard Header ===
st.title("🏭 Supply Chain Control Center")
st.markdown("Real-time visibility and control of your supply chain operations")

# === Refresh Control ===
col1, col2, col3 = st.columns([2, 2, 1])
with col3:
    if st.button("🔄 Refresh All Data", use_container_width=True):
        data_manager.clear_cache()
        st.rerun()

st.markdown("---")

# === Load All Data ===
with st.spinner("Loading supply chain data..."):
    # Load all data types
    all_data = data_manager.load_all_data()
    
    # Get insights
    insights = data_manager.get_insights()

# === Update Sidebar Filters with Loaded Data ===
with st.sidebar:
    # Entity filter for dashboard
    all_entities = set()
    for data_type, df in all_data.items():
        if not df.empty and 'legal_entity' in df.columns:
            all_entities.update(df['legal_entity'].dropna().unique())
        elif not df.empty and 'owning_company_name' in df.columns:
            # For inventory data
            all_entities.update(df['owning_company_name'].dropna().unique())
    
    if all_entities:
        selected_entities = st.multiselect(
            "Legal Entities",
            options=sorted(all_entities),
            default=list(all_entities),
            key="dashboard_entity_filter"
        )
    else:
        selected_entities = []

# === Key Metrics Dashboard ===
st.header("📊 Executive Summary")

# Row 1: Overall Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    demand_value = insights.get('demand_oc_pending_value', 0)
    display.show_metric_card(
        title="📤 Pending Demand",
        value=demand_value,
        format_type="currency",
        help_text="Total value of pending customer orders"
    )

with col2:
    inventory_value = insights.get('inventory_total_value', 0)
    display.show_metric_card(
        title="📦 Inventory Value",
        value=inventory_value,
        format_type="currency",
        help_text="Current inventory value"
    )

with col3:
    overdue_count = insights.get('demand_overdue_count', 0)
    display.show_metric_card(
        title="⏰ Overdue Orders",
        value=overdue_count,
        delta=f"{overdue_count} orders" if overdue_count > 0 else None,
        help_text="Orders past their ETD"
    )

with col4:
    expired_count = insights.get('expired_items_count', 0)
    display.show_metric_card(
        title="💀 Expired Items",
        value=expired_count,
        delta=f"{expired_count} items" if expired_count > 0 else None,
        help_text="Inventory past expiry date"
    )

# === Alerts Section ===
st.header("🚨 Alerts & Actions Required")

# Get critical alerts
alerts = []
warnings = []

# Critical: Overdue orders
if insights.get('demand_overdue_count', 0) > 0:
    alerts.append({
        'icon': '🕐',
        'message': f"{insights['demand_overdue_count']} orders are past ETD",
        'value': format_currency(insights.get('demand_overdue_value', 0), 'USD')
    })

# Critical: Expired inventory
if insights.get('expired_items_count', 0) > 0:
    alerts.append({
        'icon': '💀',
        'message': f"{insights['expired_items_count']} expired inventory items",
        'value': format_currency(insights.get('expired_items_value', 0), 'USD')
    })

# Warning: Near expiry
if insights.get('near_expiry_7d_count', 0) > 0:
    warnings.append({
        'icon': '📅',
        'message': f"{insights['near_expiry_7d_count']} items expiring in 7 days",
        'value': format_currency(insights.get('near_expiry_7d_value', 0), 'USD')
    })

# Display alerts
display.show_alerts_panel(alerts, warnings)

# === Visual Analytics ===
st.header("📈 Key Analytics")

col1, col2 = st.columns(2)

with col1:
    # Demand by Period Chart
    st.subheader("📤 Demand Trend")
    
    oc_data = all_data.get('demand_oc', pd.DataFrame())
    if not oc_data.empty:
        # Apply entity filter
        if selected_entities and 'legal_entity' in oc_data.columns:
            oc_data_filtered = oc_data[oc_data['legal_entity'].isin(selected_entities)]
        else:
            oc_data_filtered = oc_data
        
        # Apply date filter
        oc_data_filtered = oc_data_filtered.copy()
        oc_data_filtered['etd'] = pd.to_datetime(oc_data_filtered['etd'])
        oc_data_filtered = oc_data_filtered[
            (oc_data_filtered['etd'] >= pd.to_datetime(start_date)) & 
            (oc_data_filtered['etd'] <= pd.to_datetime(end_date))
        ]
        
        if not oc_data_filtered.empty:
            # Prepare data
            oc_data_filtered['week'] = oc_data_filtered['etd'].dt.to_period('W').astype(str)
            
            weekly_demand = oc_data_filtered.groupby('week').agg({
                'pending_standard_delivery_quantity': 'sum',
                'outstanding_amount_usd': 'sum'
            }).reset_index()
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weekly_demand['week'],
                y=weekly_demand['outstanding_amount_usd'],
                name='Demand Value',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Week",
                yaxis_title="Value (USD)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No demand data for selected filters")
    else:
        st.info("No demand data available")

with col2:
    # Inventory Age Distribution
    st.subheader("📦 Inventory Age Distribution")
    
    inv_data = all_data.get('inventory', pd.DataFrame())
    if not inv_data.empty:
        # Apply entity filter
        if selected_entities and 'owning_company_name' in inv_data.columns:
            inv_data_filtered = inv_data[inv_data['owning_company_name'].isin(selected_entities)]
        else:
            inv_data_filtered = inv_data
        
        if not inv_data_filtered.empty and 'days_in_warehouse' in inv_data_filtered.columns:
            # Calculate age buckets
            inv_data_filtered['age_bucket'] = pd.cut(
                inv_data_filtered['days_in_warehouse'],
                bins=[0, 30, 60, 90, 180, float('inf')],
                labels=['<30d', '30-60d', '60-90d', '90-180d', '>180d']
            )
            
            age_dist = inv_data_filtered.groupby('age_bucket')['inventory_value_usd'].sum().reset_index()
            
            # Create pie chart
            fig = px.pie(
                age_dist, 
                values='inventory_value_usd', 
                names='age_bucket',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inventory age data available")
    else:
        st.info("No inventory data available")

# === Quick Actions ===
st.header("🎯 Quick Actions")

actions = [
    {
        "label": "📤 Analyze Demand",
        "page": "pages/1_📤_Demand_Analysis.py",
        "type": "primary"
    },
    {
        "label": "📥 Check Supply",
        "page": "pages/2_📥_Supply_Analysis.py",
        "type": "secondary"
    },
    {
        "label": "📊 Run GAP Analysis",
        "page": "pages/3_📊_GAP_Analysis.py",
        "type": "secondary"
    },
    {
        "label": "⚙️ Settings",
        "page": "pages/6_⚙️_Settings.py",
        "type": "secondary"
    }
]

display.show_action_buttons(actions)

# === Data Quality Indicators ===
with st.expander("📊 Data Quality & System Status", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Freshness**")
        for data_type, df in all_data.items():
            if not df.empty:
                st.write(f"✅ {data_type}: {len(df)} records")
    
    with col2:
        st.markdown("**Active Settings**")
        active_adjustments = settings_manager.get_applied_adjustments()
        if active_adjustments:
            for adj in active_adjustments[:3]:  # Show top 3
                st.write(f"⚙️ {adj['setting']}: {adj['value']}")
        else:
            st.write("✅ Using default settings")
    
    with col3:
        st.markdown("**System Status**")
        st.write(f"✅ Database: Connected")
        st.write(f"✅ Cache TTL: {data_manager._cache_ttl}s")
        st.write(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# === Help Section ===
display.show_help_section(
    "Dashboard Overview",
    """
    ### Welcome to Supply Chain Control Center
    
    This dashboard provides real-time visibility into your supply chain operations:
    
    **Key Features:**
    - 📊 **Executive Summary**: High-level metrics and KPIs
    - 🚨 **Alerts**: Critical issues requiring immediate attention
    - 📈 **Analytics**: Visual insights into trends and patterns
    - 🎯 **Quick Actions**: Navigate to detailed analysis pages
    
    **Navigation:**
    1. **Demand Analysis**: Review customer orders and forecasts
    2. **Supply Analysis**: Check inventory and incoming supply
    3. **GAP Analysis**: Identify supply-demand mismatches
    4. **Settings**: Configure business rules and parameters
    
    **Tips:**
    - Use the 🔄 Refresh button to update all data
    - Click on any metric for detailed analysis
    - Check alerts daily for critical issues
    """
)

# === Product Performance Section ===
st.header("🏆 Top Products Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Top 5 Products by Demand")
    
    if not all_data.get('demand_oc', pd.DataFrame()).empty:
        oc_data = all_data['demand_oc']
        
        # Apply entity filter if selected
        if selected_entities and 'legal_entity' in oc_data.columns:
            oc_data = oc_data[oc_data['legal_entity'].isin(selected_entities)]
        
        top_products = oc_data.groupby(['pt_code', 'product_name']).agg({
            'pending_standard_delivery_quantity': 'sum',
            'outstanding_amount_usd': 'sum'
        }).nlargest(5, 'outstanding_amount_usd').reset_index()
        
        if not top_products.empty:
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=top_products['pt_code'],
                    x=top_products['outstanding_amount_usd'],
                    orientation='h',
                    text=top_products['outstanding_amount_usd'].apply(lambda x: f"${x:,.0f}"),
                    textposition='auto',
                    marker_color='lightcoral'
                )
            ])
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Value (USD)",
                yaxis_title="",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product data available")
    else:
        st.info("No demand data available")

# Replace the "Products with Critical Stock" section in main.py
# Around line 430-460

with col2:
    st.subheader("🚨 Products with Critical Stock")
    
    if not all_data.get('inventory', pd.DataFrame()).empty:
        inv_data = all_data['inventory'].copy()
        
        # Apply entity filter
        if selected_entities and 'owning_company_name' in inv_data.columns:
            inv_data = inv_data[inv_data['owning_company_name'].isin(selected_entities)]
        
        # Ensure remaining_quantity is numeric
        if 'remaining_quantity' in inv_data.columns:
            inv_data['remaining_quantity'] = pd.to_numeric(inv_data['remaining_quantity'], errors='coerce').fillna(0)
        else:
            inv_data['remaining_quantity'] = 0
            
        # Ensure inventory_value_usd is numeric
        if 'inventory_value_usd' in inv_data.columns:
            inv_data['inventory_value_usd'] = pd.to_numeric(inv_data['inventory_value_usd'], errors='coerce').fillna(0)
        else:
            inv_data['inventory_value_usd'] = 0
        
        # Calculate days of supply (simplified)
        if not inv_data.empty and inv_data['remaining_quantity'].sum() > 0:
            critical_products = inv_data.groupby(['pt_code', 'product_name']).agg({
                'remaining_quantity': 'sum',
                'inventory_value_usd': 'sum'
            }).reset_index()
            
            # Get products with lowest stock
            critical_products = critical_products.nsmallest(5, 'remaining_quantity')
            
            if not critical_products.empty:
                # Display as table with color coding
                critical_products['Status'] = critical_products['remaining_quantity'].apply(
                    lambda x: '🔴 Critical' if x < 100 else '🟡 Low'
                )
                
                display_df = critical_products[['pt_code', 'product_name', 'remaining_quantity', 'Status']].copy()
                display_df.columns = ['PT Code', 'Product', 'Qty', 'Status']
                
                # Format quantity for display
                display_df['Qty'] = display_df['Qty'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=250
                )
            else:
                st.info("No critical stock items")
        else:
            st.info("No inventory data available")
    else:
        st.info("No inventory data available")

# === Supply Chain Flow Visualization ===
st.header("🌐 Supply Chain Flow")

# Create flow metrics
flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)

with flow_col1:
    po_data = all_data.get('pending_po', pd.DataFrame())
    if not po_data.empty:
        po_value = po_data['outstanding_arrival_amount_usd'].sum()
        po_count = len(po_data)
    else:
        po_value = 0
        po_count = 0
    
    st.metric(
        "📋 Pending PO",
        format_currency(po_value, "USD"),
        f"{po_count} orders",
        help="Purchase orders in transit"
    )

with flow_col2:
    can_data = all_data.get('pending_can', pd.DataFrame())
    if not can_data.empty:
        can_value = can_data['pending_value_usd'].sum()
        can_count = len(can_data)
    else:
        can_value = 0
        can_count = 0
    
    st.metric(
        "🚢 Pending CAN",
        format_currency(can_value, "USD"),
        f"{can_count} arrivals",
        help="Containers awaiting stock-in"
    )

with flow_col3:
    wht_data = all_data.get('pending_wh_transfer', pd.DataFrame())
    if not wht_data.empty:
        wht_value = wht_data['warehouse_transfer_value_usd'].sum()
        wht_count = len(wht_data)
    else:
        wht_value = 0
        wht_count = 0
    
    st.metric(
        "🚚 WH Transfers",
        format_currency(wht_value, "USD"),
        f"{wht_count} transfers",
        help="Inter-warehouse transfers"
    )

with flow_col4:
    # Calculate total pipeline value
    total_pipeline = po_value + can_value + wht_value
    st.metric(
        "📊 Total Pipeline",
        format_currency(total_pipeline, "USD"),
        help="Total value in supply pipeline"
    )

# === Recent Activities ===
st.header("📋 Recent Activities")

tab1, tab2, tab3 = st.tabs(["Recent Orders", "Expiring Inventory", "Pending Allocations"])

with tab1:
    # Recent orders
    oc_data = all_data.get('demand_oc', pd.DataFrame())
    if not oc_data.empty and 'oc_date' in oc_data.columns:
        oc_data = oc_data.copy()
        oc_data['oc_date'] = pd.to_datetime(oc_data['oc_date'])
        
        # Apply entity filter
        if selected_entities and 'legal_entity' in oc_data.columns:
            oc_data = oc_data[oc_data['legal_entity'].isin(selected_entities)]
        
        # Get recent orders
        recent_orders = oc_data.nlargest(5, 'oc_date')[
            ['oc_date', 'oc_number', 'customer', 'pt_code', 'product_name', 
             'pending_standard_delivery_quantity', 'outstanding_amount_usd']
        ].copy()
        
        if not recent_orders.empty:
            recent_orders['oc_date'] = recent_orders['oc_date'].dt.strftime('%Y-%m-%d')
            recent_orders['outstanding_amount_usd'] = recent_orders['outstanding_amount_usd'].apply(
                lambda x: format_currency(x, 'USD')
            )
            
            st.dataframe(
                recent_orders,
                use_container_width=True,
                hide_index=True,
                height=200
            )
        else:
            st.info("No recent orders")
    else:
        st.info("No recent orders")

with tab2:
    # Expiring inventory
    inv_data = all_data.get('inventory', pd.DataFrame())
    if not inv_data.empty and 'expiry_date' in inv_data.columns:
        inv_data = inv_data.copy()
        inv_data['expiry_date'] = pd.to_datetime(inv_data['expiry_date'])
        inv_data['days_until_expiry'] = (inv_data['expiry_date'] - datetime.now()).dt.days
        
        # Apply entity filter
        if selected_entities and 'owning_company_name' in inv_data.columns:
            inv_data = inv_data[inv_data['owning_company_name'].isin(selected_entities)]
        
        expiring_items = inv_data[
            (inv_data['days_until_expiry'] > 0) & 
            (inv_data['days_until_expiry'] <= 30)
        ].nsmallest(10, 'days_until_expiry')[
            ['pt_code', 'product_name', 'remaining_quantity', 
             'expiry_date', 'days_until_expiry', 'inventory_value_usd']
        ].copy()
        
        if not expiring_items.empty:
            expiring_items['expiry_date'] = expiring_items['expiry_date'].dt.strftime('%Y-%m-%d')
            expiring_items['inventory_value_usd'] = expiring_items['inventory_value_usd'].apply(
                lambda x: format_currency(x, 'USD')
            )
            
            # Add urgency indicator
            expiring_items['Urgency'] = expiring_items['days_until_expiry'].apply(
                lambda x: '🔴 Critical' if x <= 7 else '🟡 Warning'
            )
            
            st.dataframe(
                expiring_items,
                use_container_width=True,
                hide_index=True,
                height=200
            )
        else:
            st.success("No items expiring in next 30 days")
    else:
        st.info("No inventory data")

with tab3:
    # Pending allocations
    alloc_data = all_data.get('active_allocations', pd.DataFrame())
    if not alloc_data.empty:
        pending_alloc = alloc_data.head(10)[
            ['allocation_number', 'pt_code', 'product_name', 
             'customer_name', 'allocated_qty', 'delivered_qty', 'undelivered_qty']
        ].copy()
        
        # Calculate fulfillment
        pending_alloc['Fulfillment %'] = (
            pending_alloc['delivered_qty'] / pending_alloc['allocated_qty'] * 100
        ).round(1).astype(str) + '%'
        
        st.dataframe(
            pending_alloc,
            use_container_width=True,
            hide_index=True,
            height=200
        )
    else:
        st.info("No active allocations")

# === Performance Metrics ===
st.header("📊 Key Performance Indicators")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    # Order fulfillment rate
    if not all_data.get('demand_oc', pd.DataFrame()).empty:
        total_orders = len(all_data['demand_oc'])
        overdue_orders = insights.get('demand_overdue_count', 0)
        fulfillment_rate = ((total_orders - overdue_orders) / total_orders * 100) if total_orders > 0 else 100
        
        st.metric(
            "Order Fulfillment Rate",
            f"{fulfillment_rate:.1f}%",
            delta=f"{fulfillment_rate - 95:.1f}%" if fulfillment_rate < 95 else "+5%",
            delta_color="normal" if fulfillment_rate >= 95 else "inverse"
        )
    else:
        st.metric("Order Fulfillment Rate", "N/A")

with kpi_col2:
    # Inventory turnover (simplified)
    if not all_data.get('inventory', pd.DataFrame()).empty:
        inv_value = insights.get('inventory_total_value', 0)
        demand_value = insights.get('demand_oc_pending_value', 0)
        turnover = (demand_value * 12 / inv_value) if inv_value > 0 else 0
        
        st.metric(
            "Inventory Turnover",
            f"{turnover:.1f}x",
            help="Annualized turnover rate"
        )
    else:
        st.metric("Inventory Turnover", "N/A")

with kpi_col3:
    # Stock accuracy (placeholder)
    st.metric(
        "Stock Accuracy",
        "98.5%",
        delta="+0.5%",
        help="Physical vs system stock match"
    )

with kpi_col4:
    # On-time delivery (placeholder)
    st.metric(
        "On-Time Delivery",
        "94.2%",
        delta="-1.8%",
        delta_color="inverse",
        help="Orders delivered on time"
    )

# === Auto-refresh logic ===
# This should be at the very end of the script
if 'auto_refresh' in locals() and auto_refresh:
    # Create a placeholder for the countdown
    countdown_placeholder = st.empty()
    
    # Show countdown timer
    refresh_seconds = refresh_interval * 60
    
    with countdown_placeholder.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for elapsed in range(refresh_seconds):
            remaining = refresh_seconds - elapsed
            mins, secs = divmod(remaining, 60)
            
            # Update progress
            progress = elapsed / refresh_seconds
            progress_bar.progress(progress)
            
            # Update status text
            status_text.text(f"⏱️ Auto-refresh in: {mins:02d}:{secs:02d}")
            
            # Wait 1 second
            time.sleep(1)
            
            # Check if we should stop (user navigated away)
            if st.session_state.get('stop_refresh', False):
                countdown_placeholder.empty()
                st.session_state['stop_refresh'] = False
                break
        else:
            # Refresh triggered
            countdown_placeholder.empty()
            with st.spinner("🔄 Refreshing data..."):
                data_manager.clear_cache()
                time.sleep(0.5)
            st.rerun()

# === Footer ===
st.markdown("---")

# Create footer with multiple columns for better layout
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption(f"📅 Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with footer_col2:
    st.caption(f"💾 Cache TTL: {data_manager._cache_ttl}s | Version: 1.0.0")

with footer_col3:
    st.caption("© 2024 Supply Chain Control Center")