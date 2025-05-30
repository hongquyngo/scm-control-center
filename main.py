import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Import existing modules
from utils.helpers import (
    get_from_session_state,
    save_to_session_state,
    format_number,
    format_currency,
    format_percentage,
    create_download_button
)
from utils.data_loader import load_allocation_history
from utils.db import get_db_engine
from sqlalchemy import text
from config import IS_RUNNING_ON_CLOUD

# Import new modules with error handling
try:
    from utils.settings_manager import SettingsManager
    from utils.data_preloader import DataPreloader
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.error("Please ensure settings_manager.py and data_preloader.py are in the utils folder")
    st.stop()

# === Page Config ===
st.set_page_config(
    page_title="SCM Control Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.status-critical {
    color: #ff4444;
    font-weight: bold;
}
.status-warning {
    color: #ff8800;
    font-weight: bold;
}
.status-ok {
    color: #00cc44;
    font-weight: bold;
}
.alert-box {
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border-left: 4px solid;
}
.alert-critical {
    background-color: #ffebee;
    border-left-color: #f44336;
}
.alert-warning {
    background-color: #fff8e1;
    border-left-color: #ff9800;
}
.alert-info {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
}
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 1rem 0;
    color: #1e3c72;
}
</style>
""", unsafe_allow_html=True)

# === Initialize Systems ===
@st.cache_resource
def init_systems():
    """Initialize singleton instances"""
    try:
        data_preloader = DataPreloader()
        settings_manager = SettingsManager()
        return data_preloader, settings_manager
    except Exception as e:
        st.error(f"Failed to initialize systems: {str(e)}")
        return None, None

# === Header Section ===
def render_header(data_preloader):
    """Render header with refresh controls"""
    col1, col2, col3 = st.columns([6, 2, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🏭 Supply Chain Control Center</h1>', unsafe_allow_html=True)
        env_indicator = "☁️ Cloud" if IS_RUNNING_ON_CLOUD else "💻 Local"
        st.caption(f"Real-time visibility and control | Environment: {env_indicator}")
    
    with col2:
        last_refresh = st.session_state.get('last_data_refresh', 'Never')
        st.caption(f"Last refresh: {last_refresh}")
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            # Force data reload
            data_preloader.preload_all_data(force_refresh=True)
            st.rerun()

# === Combined Dashboard Tab ===
def render_dashboard_tab(data_preloader):
    """Combined insights and overview dashboard"""
    
    try:
        # Get all data and insights
        insights = data_preloader.get_insights()
        critical_alerts = data_preloader.get_critical_alerts()
        warnings = data_preloader.get_warnings()
        info_metrics = data_preloader.get_info_metrics()
        
        # === Section 1: Critical Alerts ===
        st.markdown('<h2 class="section-header">🚨 Critical Alerts & Actions</h2>', unsafe_allow_html=True)
        
        # Display alerts in a grid
        alert_cols = st.columns(3)
        
        with alert_cols[0]:
            st.markdown("### 🔴 Immediate Action")
            if critical_alerts:
                for alert in critical_alerts:
                    alert_html = f"""
                    <div class="alert-box alert-critical">
                        <strong>{alert['icon']} {alert['message']}</strong><br>
                        {f"Value: {alert['value']}" if alert.get('value') else ""}
                        {f"({alert['action']})" if alert.get('action') else ""}
                    </div>
                    """
                    st.markdown(alert_html, unsafe_allow_html=True)
            else:
                st.success("✅ No critical issues")
        
        with alert_cols[1]:
            st.markdown("### 🟡 Warnings")
            if warnings:
                for warning in warnings[:5]:  # Show top 5
                    warning_html = f"""
                    <div class="alert-box alert-warning">
                        <strong>{warning['icon']} {warning['message']}</strong><br>
                        {f"{warning['value']}" if warning.get('value') else ""}
                    </div>
                    """
                    st.markdown(warning_html, unsafe_allow_html=True)
                if len(warnings) > 5:
                    st.caption(f"... and {len(warnings)-5} more warnings")
            else:
                st.info("ℹ️ No warnings")
        
        with alert_cols[2]:
            st.markdown("### 🔵 Key Insights")
            if info_metrics:
                for metric in info_metrics[:5]:  # Show top 5
                    info_html = f"""
                    <div class="alert-box alert-info">
                        <strong>{metric['icon']} {metric['message']}</strong><br>
                        {metric['value']}
                    </div>
                    """
                    st.markdown(info_html, unsafe_allow_html=True)
            else:
                st.info("📊 Run GAP Analysis for insights")
        
        # === Section 2: Key Performance Metrics ===
        st.markdown("---")
        st.markdown('<h2 class="section-header">📊 Key Performance Metrics</h2>', unsafe_allow_html=True)
        
        # Main metrics
        metrics_row1 = st.columns(5)
        
        with metrics_row1[0]:
            demand_count = insights.get('demand_oc_pending_count', 0)
            demand_value = insights.get('demand_oc_pending_value', 0)
            st.metric(
                "📤 Pending Orders",
                f"{demand_count:,}",
                delta=format_currency(demand_value, 'USD', 0),
                help="Total pending customer orders"
            )
        
        with metrics_row1[1]:
            inventory_value = insights.get('inventory_total_value', 0)
            st.metric(
                "📦 Inventory Value",
                format_currency(inventory_value, 'USD', 0),
                help="Current inventory value"
            )
        
        with metrics_row1[2]:
            # Product matching metrics
            matched = len(insights.get('matched_products', set()))
            total = len(set().union(
                insights.get('demand_only_products', set()),
                insights.get('supply_only_products', set()),
                insights.get('matched_products', set())
            ))
            coverage = (matched / total * 100) if total > 0 else 0
            st.metric(
                "🔗 Product Coverage",
                f"{coverage:.1f}%",
                delta=f"{matched}/{total} products",
                help="Products with both supply and demand"
            )
        
        with metrics_row1[3]:
            avg_fulfillment = insights.get('avg_fulfillment_rate', 0)
            st.metric(
                "📊 Fulfillment Rate",
                format_percentage(avg_fulfillment),
                help="Average fulfillment rate"
            )
        
        with metrics_row1[4]:
            # Risk value calculation
            risk_value = (
                insights.get('expired_items_value', 0) +
                insights.get('near_expiry_7d_value', 0) +
                insights.get('excess_inventory_value', 0)
            )
            st.metric(
                "⚠️ At-Risk Value",
                format_currency(risk_value, 'USD', 0),
                help="Value of expired + near expiry + excess inventory"
            )
        
        # === Section 3: Supply & Demand Analysis ===
        st.markdown("---")
        st.markdown('<h2 class="section-header">📈 Supply & Demand Quick Analysis</h2>', unsafe_allow_html=True)
        
        analysis_cols = st.columns(2)
        
        with analysis_cols[0]:
            # Demand analysis
            st.markdown("#### 📤 Demand Analysis")
            
            # Create demand summary
            demand_data = data_preloader.get_data('demand_oc')
            if not demand_data.empty:
                # Group by customer - top 5
                customer_demand = demand_data.groupby('customer')['outstanding_amount_usd'].sum().nlargest(5)
                
                fig = px.bar(
                    x=customer_demand.values,
                    y=customer_demand.index,
                    orientation='h',
                    title="Top 5 Customers by Pending Value",
                    labels={'x': 'Value (USD)', 'y': 'Customer'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No demand data available")
        
        with analysis_cols[1]:
            # Supply analysis
            st.markdown("#### 📥 Supply Analysis")
            
            # Supply breakdown by source
            supply_breakdown = {
                'Inventory': insights.get('inventory_total_value', 0),
                'Pending CAN': data_preloader.get_data('supply_can')['pending_value_usd'].sum() if not data_preloader.get_data('supply_can').empty else 0,
                'Pending PO': data_preloader.get_data('supply_po')['outstanding_arrival_amount_usd'].sum() if not data_preloader.get_data('supply_po').empty else 0,
            }
            
            if sum(supply_breakdown.values()) > 0:
                fig = px.pie(
                    values=list(supply_breakdown.values()),
                    names=list(supply_breakdown.keys()),
                    title="Supply Value by Source",
                    hole=0.3
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No supply data available")
        
        # === Section 4: Critical Products Analysis ===
        st.markdown("---")
        st.markdown('<h2 class="section-header">🎯 Critical Products Analysis</h2>', unsafe_allow_html=True)
        
        product_cols = st.columns(3)
        
        with product_cols[0]:
            # Products with no supply
            st.markdown("#### 📤 Demand-Only Products")
            demand_only = list(insights.get('demand_only_products', set()))[:5]
            if demand_only:
                for product in demand_only:
                    st.write(f"• {product}")
                if len(insights.get('demand_only_products', set())) > 5:
                    st.caption(f"... and {len(insights.get('demand_only_products', set()))-5} more")
            else:
                st.success("All demand products have supply")
        
        with product_cols[1]:
            # Expired/Near expiry
            st.markdown("#### 💀 Expiry Concerns")
            expired_count = insights.get('expired_items_count', 0)
            near_expiry_7d = insights.get('near_expiry_7d_count', 0)
            near_expiry_30d = insights.get('near_expiry_30d_count', 0)
            
            if expired_count > 0:
                st.error(f"🔴 {expired_count} expired items")
            if near_expiry_7d > 0:
                st.warning(f"🟡 {near_expiry_7d} items expiring in 7 days")
            if near_expiry_30d > 0:
                st.info(f"🔵 {near_expiry_30d} items expiring in 30 days")
            
            if expired_count == 0 and near_expiry_7d == 0 and near_expiry_30d == 0:
                st.success("No expiry concerns")
        
        with product_cols[2]:
            # Overdue deliveries
            st.markdown("#### 🕐 Delivery Status")
            overdue_count = insights.get('demand_overdue_count', 0)
            overdue_value = insights.get('demand_overdue_value', 0)
            
            if overdue_count > 0:
                st.error(f"🔴 {overdue_count} overdue orders")
                st.caption(f"Value: {format_currency(overdue_value, 'USD', 0)}")
            else:
                st.success("All deliveries on schedule")
        
        # === Section 5: Data Quality ===
        st.markdown("---")
        render_data_quality_section(insights)
        
        # === Section 6: Quick Actions ===
        st.markdown("---")
        st.markdown('<h2 class="section-header">⚡ Quick Actions</h2>', unsafe_allow_html=True)
        
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button("📊 Run GAP Analysis", type="primary", use_container_width=True):
                st.switch_page("pages/3_📊_GAP_Analysis.py")
        
        with action_cols[1]:
            if st.button("📤 View Demand Details", use_container_width=True):
                st.switch_page("pages/1_📤_Demand_Analysis.py")
        
        with action_cols[2]:
            if st.button("📥 View Supply Details", use_container_width=True):
                st.switch_page("pages/2_📥_Supply_Analysis.py")
        
        
    except Exception as e:
        st.error(f"Error rendering dashboard: {str(e)}")
        if st.checkbox("Show error details"):
            st.exception(e)

# === Data Quality Section ===
def render_data_quality_section(insights):
    """Render data quality indicators"""
    st.markdown('<h2 class="section-header">🔍 Data Quality Check</h2>', unsafe_allow_html=True)
    
    quality_cols = st.columns(4)
    
    # Missing dates
    with quality_cols[0]:
        missing_dates = insights.get('demand_missing_etd', 0) + insights.get('supply_missing_dates', 0)
        if missing_dates > 0:
            st.error(f"⚠️ {missing_dates} records with missing dates")
        else:
            st.success("✅ All dates complete")
    
    # Product matching
    with quality_cols[1]:
        total_products = len(set().union(
            insights.get('demand_only_products', set()),
            insights.get('supply_only_products', set()),
            insights.get('matched_products', set())
        ))
        matched = len(insights.get('matched_products', set()))
        match_rate = (matched / total_products * 100) if total_products > 0 else 0
        
        if match_rate >= 80:
            st.success(f"✅ {match_rate:.1f}% match rate")
        elif match_rate >= 60:
            st.warning(f"⚠️ {match_rate:.1f}% match rate")
        else:
            st.error(f"❌ {match_rate:.1f}% match rate")
    
    # Unmatched counts
    with quality_cols[2]:
        demand_only_count = len(insights.get('demand_only_products', set()))
        supply_only_count = len(insights.get('supply_only_products', set()))
        
        if demand_only_count > 0 or supply_only_count > 0:
            st.warning(f"📋 {demand_only_count + supply_only_count} unmatched products")
        else:
            st.success("✅ All products matched")
    
    # Overall score
    with quality_cols[3]:
        # Calculate overall data quality score
        score = 100
        if missing_dates > 0:
            score -= 20
        if match_rate < 80:
            score -= (80 - match_rate) * 0.5
        
        if score >= 90:
            st.success(f"✅ Quality Score: {score:.0f}%")
        elif score >= 70:
            st.warning(f"⚠️ Quality Score: {score:.0f}%")
        else:
            st.error(f"❌ Quality Score: {score:.0f}%")

# === Configuration Tab ===
def render_configuration_tab(settings_manager):
    """Render comprehensive configuration interface"""
    st.header("⚙️ Business Configuration Center")
    
    # Show current environment
    env_indicator = "☁️ Cloud" if IS_RUNNING_ON_CLOUD else "💻 Local"
    st.caption(f"Environment: {env_indicator}")
    
    # Import/Export functionality
    with st.expander("📤 Import/Export Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Current Settings",
                data=settings_manager.export_settings(),
                file_name=f"scm_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            uploaded_file = st.file_uploader("📤 Upload Settings", type=['json'])
            if uploaded_file is not None:
                settings_json = uploaded_file.read().decode('utf-8')
                if settings_manager.import_settings(settings_json):
                    st.success("✅ Settings imported successfully!")
                    st.rerun()
    
    # Show active adjustments
    adjustments = settings_manager.get_applied_adjustments()
    if adjustments:
        with st.expander("🔧 Active Adjustments", expanded=True):
            adj_df = pd.DataFrame(adjustments)
            st.dataframe(adj_df, use_container_width=True, hide_index=True)
    
    # Configuration tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⏰ Time Settings",
        "📊 Planning Parameters",
        "📦 Order Constraints",
        "🏢 Business Rules",
        "🚨 Alert Thresholds"
    ])
    
    with tab1:
        render_time_settings(settings_manager)
    
    with tab2:
        render_planning_parameters(settings_manager)
    
    with tab3:
        render_order_constraints(settings_manager)
    
    with tab4:
        render_business_rules(settings_manager)
    
    with tab5:
        render_alert_thresholds(settings_manager)
    
    # Save/Reset buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            st.success("✅ Settings saved successfully!")
            # Clear data cache to apply new settings
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            if st.checkbox("Confirm reset all settings to defaults"):
                settings_manager.reset_to_defaults()
                st.rerun()

# === Settings Sub-sections (same as before) ===
def render_time_settings(settings_manager):
    """Render time adjustment settings"""
    st.subheader("⏰ Time & Date Adjustments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        etd_offset = st.number_input(
            "ETD Offset (days)",
            min_value=-30,
            max_value=30,
            value=settings_manager.get_setting('time_adjustments.etd_offset_days', 0),
            help="Adjust ETD dates for demand analysis. Positive = delay, Negative = advance"
        )
        settings_manager.set_setting('time_adjustments.etd_offset_days', etd_offset)
        
        wh_transfer_time = st.number_input(
            "WH Transfer Lead Time (days)",
            min_value=0,
            max_value=30,
            value=settings_manager.get_setting('time_adjustments.wh_transfer_lead_time', 2),
            help="Expected time for warehouse transfers"
        )
        settings_manager.set_setting('time_adjustments.wh_transfer_lead_time', wh_transfer_time)
        
        buffer_days = st.number_input(
            "Planning Buffer (days)",
            min_value=0,
            max_value=30,
            value=settings_manager.get_setting('time_adjustments.buffer_days', 7),
            help="Safety buffer for planning"
        )
        settings_manager.set_setting('time_adjustments.buffer_days', buffer_days)
    
    with col2:
        supply_arrival_offset = st.number_input(
            "Supply Arrival Offset (days)",
            min_value=-30,
            max_value=30,
            value=settings_manager.get_setting('time_adjustments.supply_arrival_offset', 0),
            help="Adjust supply arrival dates"
        )
        settings_manager.set_setting('time_adjustments.supply_arrival_offset', supply_arrival_offset)
        
        transportation_time = st.number_input(
            "Transportation Time (days)",
            min_value=0,
            max_value=30,
            value=settings_manager.get_setting('time_adjustments.transportation_time', 3),
            help="Default transportation time"
        )
        settings_manager.set_setting('time_adjustments.transportation_time', transportation_time)
        
        working_days = st.number_input(
            "Working Days per Week",
            min_value=5,
            max_value=7,
            value=settings_manager.get_setting('time_adjustments.working_days_per_week', 5),
            help="For calculating lead times"
        )
        settings_manager.set_setting('time_adjustments.working_days_per_week', working_days)

def render_planning_parameters(settings_manager):
    """Render planning parameter settings"""
    st.subheader("📊 Planning Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        safety_stock = st.number_input(
            "Safety Stock Days",
            min_value=0,
            max_value=90,
            value=settings_manager.get_setting('planning_parameters.safety_stock_days', 14),
            help="Minimum stock to maintain"
        )
        settings_manager.set_setting('planning_parameters.safety_stock_days', safety_stock)
        
        reorder_point = st.number_input(
            "Reorder Point Days",
            min_value=0,
            max_value=90,
            value=settings_manager.get_setting('planning_parameters.reorder_point_days', 21),
            help="Trigger point for new orders"
        )
        settings_manager.set_setting('planning_parameters.reorder_point_days', reorder_point)
        
        min_coverage = st.number_input(
            "Min Order Coverage (days)",
            min_value=7,
            max_value=180,
            value=settings_manager.get_setting('planning_parameters.min_order_coverage_days', 30),
            help="Minimum days of coverage per order"
        )
        settings_manager.set_setting('planning_parameters.min_order_coverage_days', min_coverage)
    
    with col2:
        planning_horizon = st.number_input(
            "Planning Horizon (days)",
            min_value=30,
            max_value=365,
            value=settings_manager.get_setting('planning_parameters.planning_horizon_days', 90),
            help="How far ahead to plan"
        )
        settings_manager.set_setting('planning_parameters.planning_horizon_days', planning_horizon)
        
        forecast_confidence = st.slider(
            "Forecast Confidence Level",
            min_value=0.5,
            max_value=1.0,
            value=settings_manager.get_setting('planning_parameters.forecast_confidence', 0.8),
            step=0.05,
            help="Probability threshold for including forecasts"
        )
        settings_manager.set_setting('planning_parameters.forecast_confidence', forecast_confidence)
        
        max_coverage = st.number_input(
            "Max Order Coverage (days)",
            min_value=30,
            max_value=365,
            value=settings_manager.get_setting('planning_parameters.max_order_coverage_days', 180),
            help="Maximum days of coverage per order"
        )
        settings_manager.set_setting('planning_parameters.max_order_coverage_days', max_coverage)

def render_order_constraints(settings_manager):
    """Render order constraint settings"""
    st.subheader("📦 Order Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_moq = st.number_input(
            "Default MOQ",
            min_value=1,
            max_value=10000,
            value=settings_manager.get_setting('order_constraints.default_moq', 100),
            help="Default Minimum Order Quantity"
        )
        settings_manager.set_setting('order_constraints.default_moq', default_moq)
        
        default_spq = st.number_input(
            "Default SPQ",
            min_value=1,
            max_value=10000,
            value=settings_manager.get_setting('order_constraints.default_spq', 50),
            help="Default Standard Pack Quantity"
        )
        settings_manager.set_setting('order_constraints.default_spq', default_spq)
        
        enforce_spq = st.checkbox(
            "Enforce SPQ rounding",
            value=settings_manager.get_setting('order_constraints.enforce_spq', True),
            help="Always round to SPQ multiples"
        )
        settings_manager.set_setting('order_constraints.enforce_spq', enforce_spq)
    
    with col2:
        max_order_limit = st.number_input(
            "Max Order Limit",
            min_value=100,
            max_value=1000000,
            value=settings_manager.get_setting('order_constraints.max_order_limit', 10000),
            help="Maximum quantity per order"
        )
        settings_manager.set_setting('order_constraints.max_order_limit', max_order_limit)
        
        order_rounding = st.selectbox(
            "Order Rounding Method",
            options=['up', 'down', 'nearest'],
            index=['up', 'down', 'nearest'].index(
                settings_manager.get_setting('order_constraints.order_rounding', 'up')
            ),
            help="How to round order quantities"
        )
        settings_manager.set_setting('order_constraints.order_rounding', order_rounding)

def render_business_rules(settings_manager):
    """Render business rule settings"""
    st.subheader("🏢 Business Rules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        allocation_method = st.selectbox(
            "Default Allocation Method",
            options=['FIFO', 'Pro-rata', 'Priority', 'Fair Share'],
            index=['FIFO', 'Pro-rata', 'Priority', 'Fair Share'].index(
                settings_manager.get_setting('business_rules.allocation_method', 'FIFO')
            ),
            help="Default method for allocating scarce inventory"
        )
        settings_manager.set_setting('business_rules.allocation_method', allocation_method)
        
        customer_priority = st.checkbox(
            "Enable Customer Priority",
            value=settings_manager.get_setting('business_rules.customer_priority_enabled', True),
            help="Consider customer priority in allocation"
        )
        settings_manager.set_setting('business_rules.customer_priority_enabled', customer_priority)
        
        product_priority = st.checkbox(
            "Enable Product Priority",
            value=settings_manager.get_setting('business_rules.product_priority_enabled', False),
            help="Consider product priority in allocation"
        )
        settings_manager.set_setting('business_rules.product_priority_enabled', product_priority)
        
        auto_approve = st.checkbox(
            "Auto-approve Allocations",
            value=settings_manager.get_setting('business_rules.auto_approve_allocation', False),
            help="Automatically approve allocation plans"
        )
        settings_manager.set_setting('business_rules.auto_approve_allocation', auto_approve)
    
    with col2:
        shelf_life_threshold = st.number_input(
            "Shelf Life Threshold (days)",
            min_value=7,
            max_value=180,
            value=settings_manager.get_setting('business_rules.shelf_life_threshold_days', 30),
            help="Minimum acceptable shelf life"
        )
        settings_manager.set_setting('business_rules.shelf_life_threshold_days', shelf_life_threshold)
        
        shelf_life_percent = st.slider(
            "Max Shelf Life % for Allocation",
            min_value=0.5,
            max_value=1.0,
            value=settings_manager.get_setting('business_rules.shelf_life_allocation_percent', 0.75),
            step=0.05,
            help="Maximum % of shelf life consumed at delivery"
        )
        settings_manager.set_setting('business_rules.shelf_life_allocation_percent', shelf_life_percent)
        
        seasonality = st.checkbox(
            "Enable Seasonality Adjustments",
            value=settings_manager.get_setting('business_rules.seasonality_enabled', True),
            help="Consider seasonal patterns in planning"
        )
        settings_manager.set_setting('business_rules.seasonality_enabled', seasonality)

def render_alert_thresholds(settings_manager):
    """Render alert threshold settings"""
    st.subheader("🚨 Alert Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        critical_days = st.number_input(
            "Critical Shortage Days",
            min_value=1,
            max_value=14,
            value=settings_manager.get_setting('alert_thresholds.critical_shortage_days', 3),
            help="Days to consider shortage as critical"
        )
        settings_manager.set_setting('alert_thresholds.critical_shortage_days', critical_days)
        
        warning_days = st.number_input(
            "Warning Shortage Days",
            min_value=3,
            max_value=30,
            value=settings_manager.get_setting('alert_thresholds.warning_shortage_days', 7),
            help="Days to trigger shortage warning"
        )
        settings_manager.set_setting('alert_thresholds.warning_shortage_days', warning_days)
        
        slow_moving_months = st.number_input(
            "Slow Moving Threshold (months)",
            min_value=1,
            max_value=12,
            value=settings_manager.get_setting('alert_thresholds.slow_moving_months', 3),
            help="Months without movement to flag as slow"
        )
        settings_manager.set_setting('alert_thresholds.slow_moving_months', slow_moving_months)
    
    with col2:
        excess_months = st.number_input(
            "Excess Inventory Months",
            min_value=3,
            max_value=24,
            value=settings_manager.get_setting('alert_thresholds.excess_inventory_months', 6),
            help="Months of coverage to consider excess"
        )
        settings_manager.set_setting('alert_thresholds.excess_inventory_months', excess_months)
        
        min_fulfillment = st.slider(
            "Min Acceptable Fulfillment Rate",
            min_value=0.5,
            max_value=1.0,
            value=settings_manager.get_setting('alert_thresholds.min_fulfillment_rate', 0.85),
            step=0.05,
            help="Threshold for fulfillment alerts"
        )
        settings_manager.set_setting('alert_thresholds.min_fulfillment_rate', min_fulfillment)
        
        max_variance = st.slider(
            "Max Allocation Variance",
            min_value=0.05,
            max_value=0.5,
            value=settings_manager.get_setting('alert_thresholds.max_allocation_variance', 0.15),
            step=0.05,
            help="Maximum acceptable variance in allocation"
        )
        settings_manager.set_setting('alert_thresholds.max_allocation_variance', max_variance)

# === Recent Activities Section ===
def render_recent_activities():
    """Render recent system activities"""
    st.subheader("🕐 Recent Activities")
    
    activities = []
    
    # Collect activities from session state
    activity_keys = {
        'demand_analysis_timestamp': ('📤 Demand Analysis', 'Completed'),
        'supply_analysis_timestamp': ('📥 Supply Analysis', 'Completed'),
        'gap_analysis_result_timestamp': ('📊 GAP Analysis', 'Completed'),
        'final_allocation_plan_timestamp': ('🧩 Allocation Plan', 'Saved')
    }
    
    for key, (action, status) in activity_keys.items():
        timestamp = get_from_session_state(key)
        if timestamp:
            activities.append({
                'time': timestamp,
                'action': action,
                'status': status
            })
    
    if activities:
        # Sort by time descending
        activities_df = pd.DataFrame(activities)
        activities_df = activities_df.sort_values('time', ascending=False).head(5)
        activities_df['time'] = activities_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            activities_df,
            column_config={
                'time': 'Timestamp',
                'action': 'Activity',
                'status': 'Status'
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No recent activities recorded")

# === System Status Section ===
def render_system_status():
    """Render system health status"""
    st.subheader("🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Database**")
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.markdown('<span class="status-ok">● Connected</span>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<span class="status-critical">● Error</span>', unsafe_allow_html=True)
            st.caption(str(e)[:50])
    
    with col2:
        st.markdown("**Data Freshness**")
        last_refresh = st.session_state.get('last_data_refresh', None)
        if last_refresh:
            st.markdown('<span class="status-ok">● Up to date</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warning">● Needs refresh</span>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Cache Status**")
        cache_size = len(st.session_state.keys())
        st.markdown(f'<span class="status-ok">● Active ({cache_size})</span>', unsafe_allow_html=True)
    
    with col4:
        st.markdown("**Server Time**")
        st.markdown(f"{datetime.now().strftime('%H:%M:%S')}")

# === Main Function ===
def main():
    # Initialize systems
    data_preloader, settings_manager = init_systems()
    
    if data_preloader is None or settings_manager is None:
        st.error("Failed to initialize application. Please check system configuration.")
        return
    
    # Render header
    render_header(data_preloader)
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            data_preloader.preload_all_data()
            st.session_state['last_data_refresh'] = datetime.now().strftime("%H:%M:%S")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Some features may be limited.")
    
    # Main content - Only 2 tabs now
    tab1, tab2 = st.tabs([
        "📊 Dashboard",
        "⚙️ Configuration"
    ])
    
    with tab1:
        render_dashboard_tab(data_preloader)
    
    with tab2:
        render_configuration_tab(settings_manager)
    
    # Bottom sections
    st.markdown("---")
    
    # Recent activities and system status
    col1, col2 = st.columns(2)
    
    with col1:
        render_recent_activities()
    
    with col2:
        render_system_status()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("© 2024 Supply Chain Management System")
    with col2:
        st.caption("Version 2.0.0")
    with col3:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()