import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import (
   get_from_session_state,
   format_number,
   format_currency,
   format_percentage
)
from utils.data_loader import load_allocation_history
from utils.db import get_db_engine
from sqlalchemy import text

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
   .allocation-pending {
       background-color: #fff3cd;
       padding: 0.5rem;
       border-radius: 5px;
   }
   .allocation-approved {
       background-color: #d4edda;
       padding: 0.5rem;
       border-radius: 5px;
   }
   </style>
   """, unsafe_allow_html=True)

# === Header ===
st.markdown('<h1 class="main-header">🏭 Supply Chain Management Control Center</h1>', unsafe_allow_html=True)
st.markdown("**Real-time visibility and control over your supply chain operations**")
st.markdown("---")

# === Sidebar Configuration ===
with st.sidebar:
   st.markdown("### ⚙️ Settings")
   
   # Refresh Data button in sidebar
   if st.button("🔄 Refresh All Data", use_container_width=True):
       st.cache_data.clear()
       st.success("✅ All data caches cleared!")
       st.rerun()
   
   st.markdown("---")
   st.caption("💡 Tip: Refresh data clears all cached queries and reloads from database")

# === Quick Actions Bar ===
col1, col2, col3 = st.columns(3)
with col1:
   if st.button("📊 Run GAP Analysis", type="primary", use_container_width=True):
       st.switch_page("pages/3_📊_GAP_Analysis.py")
with col2:
   if st.button("📤 View Demand", use_container_width=True):
       st.switch_page("pages/1_📤_Demand_Analysis.py")
with col3:
   if st.button("📥 View Supply", use_container_width=True):
       st.switch_page("pages/2_📥_Supply_Analysis.py")

# === Load Allocation Data ===
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_allocation_summary():
   """Load allocation summary data"""
   engine = get_db_engine()
   
   query = """
   SELECT 
       ap.id as plan_id,
       ap.allocation_number,
       ap.allocation_date,
       ap.allocation_method,
       ap.status as plan_status,
       COUNT(DISTINCT ad.id) as line_count,
       COUNT(DISTINCT ad.pt_code) as product_count,
       COUNT(DISTINCT ad.customer_id) as customer_count,
       SUM(ad.allocated_qty) as total_allocated_qty,
       SUM(ad.delivered_qty) as total_delivered_qty,
       SUM(ad.allocated_qty - ad.delivered_qty) as total_remaining_qty,
       AVG(CASE WHEN ad.allocated_qty > 0 
           THEN ad.delivered_qty / ad.allocated_qty * 100 
           ELSE 0 END) as avg_fulfillment_rate
   FROM allocation_plans ap
   JOIN allocation_details ad ON ap.id = ad.allocation_plan_id
   WHERE ap.created_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
   GROUP BY ap.id
   ORDER BY ap.allocation_date DESC
   """
   
   return pd.read_sql(text(query), engine)

# === Check for Recent Analysis ===
last_gap_analysis = get_from_session_state('last_gap_analysis')
last_analysis_time = get_from_session_state('last_analysis_time', 'Never')

if last_gap_analysis is not None:
   # === Executive Summary with Analysis Data ===
   st.header("📈 Executive Summary")
   st.info(f"📅 Based on GAP Analysis from: {last_analysis_time}")
   
   # Calculate key metrics
   gap_df = last_gap_analysis
   shortage_products = gap_df[gap_df['gap_quantity'] < 0]['pt_code'].unique()
   surplus_products = gap_df[gap_df['gap_quantity'] > 0]['pt_code'].unique()
   balanced_products = gap_df[gap_df['gap_quantity'] == 0]['pt_code'].unique()
   
   total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
   total_surplus = gap_df[gap_df['gap_quantity'] > 0]['gap_quantity'].sum()
   avg_fulfillment = gap_df[gap_df['total_demand_qty'] > 0]['fulfillment_rate_percent'].mean()
   critical_items = len(gap_df[(gap_df['gap_quantity'] < 0) & (gap_df['fulfillment_rate_percent'] < 50)])
   
   # Display metrics
   metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
   
   with metrics_col1:
       st.metric(
           "⚠️ Products with Shortage",
           f"{len(shortage_products)}",
           delta=f"-{format_number(total_shortage)} units",
           delta_color="inverse"
       )
   
   with metrics_col2:
       st.metric(
           "📦 Products with Surplus",
           f"{len(surplus_products)}",
           delta=f"+{format_number(total_surplus)} units",
           delta_color="normal"
       )
   
   with metrics_col3:
       st.metric(
           "📊 Avg Fulfillment Rate",
           format_percentage(avg_fulfillment),
           delta=f"{100 - avg_fulfillment:.1f}% gap",
           delta_color="inverse" if avg_fulfillment < 90 else "normal"
       )
   
   with metrics_col4:
       st.metric(
           "🚨 Critical Items",
           f"{critical_items}",
           delta="<50% fulfillment",
           delta_color="inverse"
       )
   
   # === Visual Analytics ===
   st.markdown("---")
   
   col1, col2 = st.columns(2)
   
   with col1:
       st.subheader("🚨 Top 10 Critical Shortages")
       
       # Get shortage data
       shortage_detail = gap_df[gap_df['gap_quantity'] < 0].groupby(['pt_code', 'product_name']).agg({
           'gap_quantity': 'sum',
           'fulfillment_rate_percent': 'mean'
       }).reset_index()
       shortage_detail['shortage_qty'] = shortage_detail['gap_quantity'].abs()
       shortage_detail = shortage_detail.sort_values('shortage_qty', ascending=False).head(10)
       
       if not shortage_detail.empty:
           # Create horizontal bar chart
           fig_shortage = px.bar(
               shortage_detail, 
               x='shortage_qty', 
               y='product_name',
               orientation='h',
               text='shortage_qty',
               title='Shortage Quantity by Product',
               color='fulfillment_rate_percent',
               color_continuous_scale='Reds_r',
               labels={
                   'shortage_qty': 'Shortage Qty',
                   'product_name': 'Product',
                   'fulfillment_rate_percent': 'Fill Rate %'
               }
           )
           fig_shortage.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
           fig_shortage.update_layout(height=400, showlegend=False)
           st.plotly_chart(fig_shortage, use_container_width=True)
           
           # Action button
           if st.button("🧩 Create Allocation Plan", type="primary", key="shortage_action"):
               st.switch_page("pages/4_🧩_Allocation_Plan.py")
       else:
           st.success("✅ No critical shortages detected!")
   
   with col2:
       st.subheader("📈 Top 10 Surplus Items")
       
       # Get surplus data
       surplus_detail = gap_df[gap_df['gap_quantity'] > 0].groupby(['pt_code', 'product_name']).agg({
           'gap_quantity': 'sum'
       }).reset_index()
       surplus_detail = surplus_detail.sort_values('gap_quantity', ascending=False).head(10)
       
       if not surplus_detail.empty:
           # Create horizontal bar chart
           fig_surplus = px.bar(
               surplus_detail, 
               x='gap_quantity', 
               y='product_name',
               orientation='h',
               text='gap_quantity',
               title='Surplus Quantity by Product',
               color_discrete_sequence=['#4ECDC4'],
               labels={
                   'gap_quantity': 'Surplus Qty',
                   'product_name': 'Product'
               }
           )
           fig_surplus.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
           fig_surplus.update_layout(height=400, showlegend=False)
           st.plotly_chart(fig_surplus, use_container_width=True)
           
           # Action button
           if st.button("🔄 View Reallocation Options", key="surplus_action"):
               st.session_state['show_reallocation'] = True
               st.switch_page("pages/5_📌_PO_Suggestions.py")
       else:
           st.info("📊 No significant surplus inventory")
   
   # === Period Analysis ===
   st.markdown("---")
   st.subheader("📅 Supply-Demand Balance by Period")
   
   # Group by period for trend analysis
   period_summary = gap_df.groupby('period').agg({
       'total_demand_qty': 'sum',
       'total_available': 'sum',
       'gap_quantity': 'sum'
   }).reset_index()
   
   # Create line chart
   fig_trend = go.Figure()
   
   fig_trend.add_trace(go.Scatter(
       x=period_summary['period'],
       y=period_summary['total_demand_qty'],
       mode='lines+markers',
       name='Demand',
       line=dict(color='#FF6B6B', width=3),
       marker=dict(size=8)
   ))
   
   fig_trend.add_trace(go.Scatter(
       x=period_summary['period'],
       y=period_summary['total_available'],
       mode='lines+markers',
       name='Supply',
       line=dict(color='#4ECDC4', width=3),
       marker=dict(size=8)
   ))
   
   fig_trend.add_trace(go.Bar(
       x=period_summary['period'],
       y=period_summary['gap_quantity'],
       name='GAP',
       marker_color=period_summary['gap_quantity'].apply(
           lambda x: '#FF6B6B' if x < 0 else '#4ECDC4'
       ),
       opacity=0.6
   ))
   
   fig_trend.update_layout(
       title='Supply vs Demand Trend',
       xaxis_title='Period',
       yaxis_title='Quantity',
       hovermode='x unified',
       height=400,
       showlegend=True,
       legend=dict(
           yanchor="top",
           y=0.99,
           xanchor="left",
           x=0.01
       )
   )
   
   st.plotly_chart(fig_trend, use_container_width=True)

# === Allocation Status Section ===
st.markdown("---")
st.header("🧩 Allocation Management")

# Load allocation data
allocation_summary = load_allocation_summary()

if not allocation_summary.empty:
   # Allocation metrics
   alloc_col1, alloc_col2, alloc_col3, alloc_col4 = st.columns(4)
   
   with alloc_col1:
       total_plans = len(allocation_summary)
       draft_plans = len(allocation_summary[allocation_summary['plan_status'] == 'DRAFT'])
       st.metric(
           "Total Allocation Plans", 
           f"{total_plans}",
           delta=f"{draft_plans} pending" if draft_plans > 0 else None
       )
   
   with alloc_col2:
       active_plans = allocation_summary[allocation_summary['plan_status'].isin(['APPROVED', 'EXECUTED'])]
       total_allocated = active_plans['total_allocated_qty'].sum()
       st.metric("Total Allocated Qty", format_number(total_allocated))
   
   with alloc_col3:
       total_delivered = allocation_summary['total_delivered_qty'].sum()
       delivery_rate = (total_delivered / total_allocated * 100) if total_allocated > 0 else 0
       st.metric("Delivered", f"{format_number(total_delivered)} ({delivery_rate:.1f}%)")
   
   with alloc_col4:
       avg_fulfillment = allocation_summary['avg_fulfillment_rate'].mean()
       st.metric("Avg Fulfillment Rate", format_percentage(avg_fulfillment))
   
   # Recent allocations table
   st.markdown("#### 📋 Recent Allocation Plans")
   
   # Filter options
   col1, col2 = st.columns([3, 1])
   with col1:
       status_filter = st.multiselect(
           "Filter by Status",
           allocation_summary['plan_status'].unique(),
           default=allocation_summary['plan_status'].unique()
       )
   with col2:
       if st.button("🧩 View All Allocations"):
           st.switch_page("pages/4_🧩_Allocation_Plan.py")
   
   # Display filtered allocations
   filtered_allocations = allocation_summary[allocation_summary['plan_status'].isin(status_filter)]
   
   if not filtered_allocations.empty:
       display_allocations = filtered_allocations[[
           'allocation_number', 'allocation_date', 'allocation_method', 
           'plan_status', 'product_count', 'customer_count',
           'total_allocated_qty', 'avg_fulfillment_rate'
       ]].copy()
       
       # Format columns
       display_allocations['allocation_date'] = pd.to_datetime(display_allocations['allocation_date']).dt.strftime('%Y-%m-%d')
       display_allocations['total_allocated_qty'] = display_allocations['total_allocated_qty'].apply(format_number)
       display_allocations['avg_fulfillment_rate'] = display_allocations['avg_fulfillment_rate'].apply(lambda x: f"{x:.1f}%")
       
       # Apply status coloring
       def color_status(val):
           colors = {
               'DRAFT': 'background-color: #fff3cd',
               'APPROVED': 'background-color: #d4edda',
               'EXECUTED': 'background-color: #cce5ff',
               'CANCELLED': 'background-color: #f8d7da'
           }
           return colors.get(val, '')
       
       styled_df = display_allocations.style.applymap(color_status, subset=['plan_status'])
       st.dataframe(styled_df, use_container_width=True, height=300)
   else:
       st.info("No allocation plans found for selected filters")
else:
   st.info("No allocation plans created yet. Run GAP Analysis to identify shortages and create allocation plans.")
   
# === Recent Activities ===
st.markdown("---")
st.subheader("🕐 Recent Activities")

activities = []

# Check various timestamps
if get_from_session_state('demand_analysis_timestamp'):
   activities.append({
       'time': get_from_session_state('demand_analysis_timestamp'),
       'action': '📤 Demand Analysis',
       'status': 'Completed'
   })

if get_from_session_state('supply_analysis_timestamp'):
   activities.append({
       'time': get_from_session_state('supply_analysis_timestamp'),
       'action': '📥 Supply Analysis',
       'status': 'Completed'
   })

if get_from_session_state('gap_analysis_result_timestamp'):
   activities.append({
       'time': get_from_session_state('gap_analysis_result_timestamp'),
       'action': '📊 GAP Analysis',
       'status': 'Completed'
   })

if get_from_session_state('final_allocation_plan_timestamp'):
   activities.append({
       'time': get_from_session_state('final_allocation_plan_timestamp'),
       'action': '🧩 Allocation Plan',
       'status': 'Saved'
   })

if activities:
   # Sort by time
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

# === Welcome Section (if no analysis) ===
if last_gap_analysis is None:
   st.markdown("---")
   
   # Welcome message
   col1, col2, col3 = st.columns([1, 2, 1])
   
   with col2:
       st.markdown("## 👋 Welcome to SCM Control Center!")
       st.markdown("""
       This dashboard provides real-time visibility into your supply chain operations,
       helping you make data-driven decisions to optimize inventory and meet customer demand.
       
       ### 🚀 Getting Started
       
       1. **📤 Analyze Demand**: Review customer orders and forecasts
       2. **📥 Check Supply**: Monitor inventory and incoming shipments
       3. **📊 Run GAP Analysis**: Identify supply-demand mismatches
       4. **🧩 Create Allocation Plans**: Distribute limited supply fairly
       5. **📌 Generate PO Suggestions**: Replenish shortage items
       
       ### 📊 Key Features
       
       - **Real-time Analytics**: Up-to-date supply and demand visibility
       - **Smart Allocation**: Multiple methods to distribute scarce inventory
       - **Proactive Alerts**: Early warning for shortages and excess
       - **Export Reports**: Share insights with stakeholders
       - **Allocation Tracking**: Monitor allocation plan execution
       """)
       
       st.markdown("---")
       
       if st.button("🚀 Start with GAP Analysis", type="primary", use_container_width=True, key="start_gap"):
           st.switch_page("pages/3_📊_GAP_Analysis.py")
   
   # Feature cards
   st.markdown("---")
   st.markdown("### 🎯 Module Overview")
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       st.markdown("""
       #### 📤 Demand Analysis
       - View pending orders (OC)
       - Track customer forecasts
       - Monitor conversion rates
       - Identify demand patterns
       """)
       if st.button("Open Demand Analysis", use_container_width=True):
           st.switch_page("pages/1_📤_Demand_Analysis.py")
   
   with col2:
       st.markdown("""
       #### 📥 Supply Analysis
       - Current inventory levels
       - Pending arrivals (CAN)
       - Outstanding POs
       - Expiry tracking
       """)
       if st.button("Open Supply Analysis", use_container_width=True):
           st.switch_page("pages/2_📥_Supply_Analysis.py")
   
   with col3:
       st.markdown("""
       #### 📊 GAP Analysis
       - Compare supply vs demand
       - Period-wise analysis
       - Carry-forward logic
       - Shortage identification
       """)
       if st.button("Open GAP Analysis", use_container_width=True):
           st.switch_page("pages/3_📊_GAP_Analysis.py")

# === System Status ===
st.markdown("---")
st.subheader("🔧 System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
   st.markdown("**Database Connection**")
   try:
       engine = get_db_engine()
       with engine.connect() as conn:
           conn.execute(text("SELECT 1"))
       st.markdown('<span class="status-ok">● Connected</span>', unsafe_allow_html=True)
   except:
       st.markdown('<span class="status-critical">● Disconnected</span>', unsafe_allow_html=True)

with status_col2:
   st.markdown("**Data Freshness**")
   st.markdown('<span class="status-ok">● Real-time</span>', unsafe_allow_html=True)

with status_col3:
   st.markdown("**Cache Status**")
   cache_size = len(st.session_state.keys())
   st.markdown(f'<span class="status-ok">● Active ({cache_size} items)</span>', unsafe_allow_html=True)

with status_col4:
   st.markdown("**Last Refresh**")
   st.markdown(f"{datetime.now().strftime('%H:%M:%S')}")

# === Footer ===
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
   st.caption("© 2024 Supply Chain Management System")
with col2:
   st.caption("Version 1.1.0 (with Allocation)")
with col3:
   st.caption(f"Server Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")