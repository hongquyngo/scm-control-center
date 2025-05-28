import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import json
from sqlalchemy import text
from utils.db import get_db_engine
from utils.helpers import (
   convert_df_to_excel,
   export_multiple_sheets,
   format_number,
   format_currency,
   format_percentage,
   save_to_session_state,
   get_from_session_state
)

# === Page Config ===
st.set_page_config(
   page_title="Allocation Plan - SCM",
   page_icon="🧩",
   layout="wide"
)

# === Constants ===
ALLOCATION_METHODS = ["FIFO (First In First Out)", "Customer Priority", "Pro-rata", "Manual Override"]
PRIORITY_LEVELS = ["Critical", "High", "Medium", "Low"]
PRIORITY_COLORS = {
   'Critical': '#ff4444',
   'High': '#ff8800',
   'Medium': '#ffcc00',
   'Low': '#88cc00'
}

# === Check Prerequisites ===
gap_df = get_from_session_state('gap_analysis_result')
demand_df = get_from_session_state('demand_filtered')
supply_df = get_from_session_state('supply_filtered')

if gap_df is None or demand_df is None or supply_df is None:
   st.error("❌ No GAP Analysis data found!")
   st.warning("Please run GAP Analysis first before creating an allocation plan.")
   
   col1, col2, col3 = st.columns([1, 1, 1])
   with col2:
       if st.button("📊 Go to GAP Analysis", type="primary", use_container_width=True):
           st.switch_page("pages/3_📊_GAP_Analysis.py")
   
   st.stop()

# === Header with Navigation ===
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
   if st.button("← GAP Analysis"):
       st.switch_page("pages/3_📊_GAP_Analysis.py")
with col2:
   st.title("🧩 Allocation Plan")
with col3:
   if st.button("🏠 Dashboard"):
       st.switch_page("main.py")

st.markdown("---")

# === Analysis Info ===
analysis_time = get_from_session_state('last_analysis_time', 'Unknown')
st.info(f"📅 Using GAP Analysis from: {analysis_time}")

# === Check for Shortages ===
shortage_df = gap_df[gap_df['gap_quantity'] < 0].copy()

if shortage_df.empty:
   st.success("✅ Great news! No shortages detected in the GAP analysis.")
   st.info("All demand can be fulfilled with available supply. No allocation plan needed.")
   
   # Show surplus information if any
   surplus_df = gap_df[gap_df['gap_quantity'] > 0]
   if not surplus_df.empty:
       st.markdown("### 📦 Surplus Inventory Available")
       
       surplus_summary = surplus_df.groupby(['pt_code', 'product_name']).agg({
           'gap_quantity': 'sum'
       }).reset_index()
       surplus_summary = surplus_summary.sort_values('gap_quantity', ascending=False).head(10)
       
       st.dataframe(
           surplus_summary.style.format({
               'gap_quantity': '{:,.0f}'
           }),
           use_container_width=True
       )
       
       col1, col2 = st.columns(2)
       with col1:
           if st.button("🔄 View Reallocation Options", type="primary"):
               save_to_session_state('show_reallocation', True)
               st.switch_page("pages/5_📌_PO_Suggestions.py")
       with col2:
           if st.button("📊 Return to Analysis"):
               st.switch_page("pages/3_📊_GAP_Analysis.py")
   st.stop()

# === Shortage Analysis ===
def analyze_shortage():
   """Analyze shortage data and prepare for allocation"""
   # Get unique products with shortage
   shortage_products = shortage_df.groupby(['pt_code', 'product_name', 'package_size', 'standard_uom']).agg({
       'gap_quantity': 'sum',
       'total_demand_qty': 'sum',
       'total_available': 'sum'
   }).reset_index()
   
   shortage_products['shortage_qty'] = shortage_products['gap_quantity'].abs()
   shortage_products['fulfillment_rate'] = (
       shortage_products['total_available'] / shortage_products['total_demand_qty'] * 100
   ).round(1)
   
   return shortage_products

def show_shortage_summary(shortage_products):
   """Display shortage summary"""
   st.markdown("### 🚨 Shortage Summary")
   
   # Metrics
   col1, col2, col3, col4 = st.columns(4)
   with col1:
       total_products = len(shortage_products)
       st.metric("Products with Shortage", f"{total_products}")
   with col2:
       total_shortage = shortage_products['shortage_qty'].sum()
       st.metric("Total Shortage Quantity", format_number(total_shortage))
   with col3:
       avg_fulfillment = shortage_products['fulfillment_rate'].mean()
       st.metric("Average Fulfillment Rate", format_percentage(avg_fulfillment))
   with col4:
       critical_items = len(shortage_products[shortage_products['fulfillment_rate'] < 50])
       st.metric("Critical Items (<50%)", f"{critical_items}")
   
   # Detailed table
   st.markdown("#### 📋 Shortage Details by Product")
   display_df = shortage_products.copy()
   display_df = display_df.sort_values('shortage_qty', ascending=False)
   
   # Format columns
   display_df['total_demand_qty'] = display_df['total_demand_qty'].apply(format_number)
   display_df['total_available'] = display_df['total_available'].apply(format_number)
   display_df['shortage_qty'] = display_df['shortage_qty'].apply(format_number)
   display_df['fulfillment_rate'] = display_df['fulfillment_rate'].apply(lambda x: f"{x:.1f}%")
   
   # Rename columns for display
   display_df = display_df.rename(columns={
       'pt_code': 'PT Code',
       'product_name': 'Product',
       'package_size': 'Pack Size',
       'standard_uom': 'UOM',
       'total_demand_qty': 'Total Demand',
       'total_available': 'Available',
       'shortage_qty': 'Shortage',
       'fulfillment_rate': 'Fill Rate'
   })
   
   st.dataframe(display_df, use_container_width=True, height=300)

# === Allocation Configuration ===
def select_allocation_method():
   """Select allocation method and configuration"""
   st.markdown("### 📐 Allocation Configuration")
   
   col1, col2 = st.columns([1, 1])
   
   with col1:
       method = st.selectbox(
           "Select Allocation Method",
           ALLOCATION_METHODS,
           help="""
           - **FIFO**: Allocate to earliest orders first
           - **Customer Priority**: Allocate based on customer importance
           - **Pro-rata**: Distribute proportionally to demand
           - **Manual Override**: Manually set allocation
           """
       )
   
   with col2:
       if method == "Customer Priority":
           st.info("💡 Set customer priorities in the next section")
       elif method == "Pro-rata":
           st.info("💡 Each customer gets proportional allocation")
       elif method == "FIFO (First In First Out)":
           st.info("💡 Orders fulfilled by ETD date order")
       else:
           st.warning("⚠️ Manual allocation requires careful review")
   
   return method

def set_customer_priorities(demand_df):
   """Set customer priority levels"""
   st.markdown("### 🎯 Customer Priority Settings")
   
   # Get unique customers with their demand value
   customer_summary = demand_df.groupby('customer').agg({
       'value_in_usd': 'sum',
       'demand_quantity': 'sum'
   }).reset_index()
   customer_summary = customer_summary.sort_values('value_in_usd', ascending=False)
   
   # Initialize priority dict
   if 'customer_priorities' not in st.session_state:
       st.session_state.customer_priorities = {}
   
   # Quick actions
   col1, col2, col3 = st.columns(3)
   with col1:
       if st.button("Set All as Medium"):
           for customer in customer_summary['customer']:
               st.session_state.customer_priorities[customer] = "Medium"
           st.rerun()
   
   with col2:
       if st.button("Set Top 20% as High"):
           top_20_percent = int(len(customer_summary) * 0.2)
           for i, customer in enumerate(customer_summary['customer']):
               if i < top_20_percent:
                   st.session_state.customer_priorities[customer] = "High"
               else:
                   st.session_state.customer_priorities[customer] = "Medium"
           st.rerun()
   
   with col3:
       if st.button("Auto-assign by Value"):
           quartiles = customer_summary['value_in_usd'].quantile([0.25, 0.5, 0.75])
           for _, row in customer_summary.iterrows():
               if row['value_in_usd'] >= quartiles[0.75]:
                   st.session_state.customer_priorities[row['customer']] = "Critical"
               elif row['value_in_usd'] >= quartiles[0.5]:
                   st.session_state.customer_priorities[row['customer']] = "High"
               elif row['value_in_usd'] >= quartiles[0.25]:
                   st.session_state.customer_priorities[row['customer']] = "Medium"
               else:
                   st.session_state.customer_priorities[row['customer']] = "Low"
           st.rerun()
   
   # Manual priority setting with customer info
   st.markdown("#### Set Individual Priorities")
   
   # Create a more informative display
   priority_df = customer_summary.copy()
   priority_df['Priority'] = priority_df['customer'].map(
       lambda x: st.session_state.customer_priorities.get(x, "Medium")
   )
   priority_df['value_in_usd'] = priority_df['value_in_usd'].apply(lambda x: format_currency(x, "USD"))
   priority_df['demand_quantity'] = priority_df['demand_quantity'].apply(format_number)
   
   # Use columns for better layout
   cols_per_row = 3
   rows = (len(priority_df) + cols_per_row - 1) // cols_per_row
   
   for row in range(rows):
       cols = st.columns(cols_per_row)
       for col_idx in range(cols_per_row):
           idx = row * cols_per_row + col_idx
           if idx < len(priority_df):
               with cols[col_idx]:
                   customer_data = priority_df.iloc[idx]
                   st.markdown(f"**{customer_data['customer']}**")
                   st.caption(f"Value: {customer_data['value_in_usd']} | Qty: {customer_data['demand_quantity']}")
                   
                   priority = st.selectbox(
                       "Priority",
                       PRIORITY_LEVELS,
                       index=PRIORITY_LEVELS.index(customer_data['Priority']),
                       key=f"priority_{customer_data['customer']}",
                       label_visibility="collapsed"
                   )
                   st.session_state.customer_priorities[customer_data['customer']] = priority
   
   return st.session_state.customer_priorities

# === Allocation Engine ===
def prepare_demand_for_allocation(demand_df, shortage_products):
   """Prepare demand data for allocation"""
   # Filter demand for shortage products only
   shortage_pt_codes = shortage_products['pt_code'].unique()
   
   demand_allocation = demand_df[demand_df['pt_code'].isin(shortage_pt_codes)].copy()
   
   # Add priority if exists
   if 'customer_priorities' in st.session_state:
       demand_allocation['priority'] = demand_allocation['customer'].map(
           st.session_state.customer_priorities
       ).fillna('Medium')
       
       demand_allocation['priority_score'] = demand_allocation['priority'].map({
           'Critical': 1,
           'High': 2,
           'Medium': 3,
           'Low': 4
       })
   else:
       demand_allocation['priority'] = 'Medium'
       demand_allocation['priority_score'] = 3
   
   # Get product ID from product master if not available
   if 'product_id' not in demand_allocation.columns:
       # Simple approach - use hash of pt_code as temporary ID
       demand_allocation['product_id'] = demand_allocation['pt_code'].apply(
           lambda x: hash(x) % 1000000
       )
   
   # Get customer ID if not available
   if 'customer_id' not in demand_allocation.columns:
       demand_allocation['customer_id'] = demand_allocation['customer'].apply(
           lambda x: hash(x) % 1000000
       )
   
   return demand_allocation

def prepare_supply_for_allocation(supply_df, shortage_products):
   """Prepare supply data for allocation"""
   # Get supply for shortage products
   shortage_pt_codes = shortage_products['pt_code'].unique()
   
   supply_allocation = supply_df[supply_df['pt_code'].isin(shortage_pt_codes)].copy()
   
   # Group by product to get total available
   supply_summary = supply_allocation.groupby('pt_code').agg({
       'quantity': 'sum'
   }).reset_index()
   
   return supply_summary

def allocate_fifo(demand_allocation, supply_summary):
   """Allocate based on FIFO - earliest ETD first"""
   allocation_results = []
   
   # Sort demand by ETD
   demand_sorted = demand_allocation.sort_values('etd')
   
   # Track available supply for each product
   available_supply = supply_summary.set_index('pt_code')['quantity'].to_dict()
   
   # Allocate to each demand line
   for idx, row in demand_sorted.iterrows():
       pt_code = row['pt_code']
       requested_qty = row['demand_quantity']
       
       if pt_code in available_supply and available_supply[pt_code] > 0:
           allocated_qty = min(requested_qty, available_supply[pt_code])
           available_supply[pt_code] -= allocated_qty
       else:
           allocated_qty = 0
       
       allocation_results.append({
           'demand_number': row['demand_number'],
           'demand_reference_id': row.get('ocd_id', None),
           'customer': row['customer'],
           'customer_id': row.get('customer_id', None),
           'pt_code': pt_code,
           'product_id': row.get('product_id', None),
           'product_name': row['product_name'],
           'package_size': row.get('package_size', ''),
           'standard_uom': row.get('standard_uom', ''),
           'etd': row['etd'],
           'allocated_etd': row['etd'],  # Same as original ETD for FIFO
           'requested_qty': requested_qty,
           'allocated_qty': allocated_qty,
           'shortage_qty': requested_qty - allocated_qty,
           'fulfillment_rate': (allocated_qty / requested_qty * 100) if requested_qty > 0 else 0,
           'priority': row['priority'],
           'allocation_method': 'FIFO'
       })
   
   return pd.DataFrame(allocation_results)

def allocate_by_priority(demand_allocation, supply_summary):
   """Allocate based on customer priority"""
   allocation_results = []
   
   # Sort by priority score (ascending) and ETD
   demand_sorted = demand_allocation.sort_values(['priority_score', 'etd'])
   
   # Track available supply
   available_supply = supply_summary.set_index('pt_code')['quantity'].to_dict()
   
   # Allocate based on priority
   for idx, row in demand_sorted.iterrows():
       pt_code = row['pt_code']
       requested_qty = row['demand_quantity']
       
       if pt_code in available_supply and available_supply[pt_code] > 0:
           allocated_qty = min(requested_qty, available_supply[pt_code])
           available_supply[pt_code] -= allocated_qty
       else:
           allocated_qty = 0
       
       allocation_results.append({
           'demand_number': row['demand_number'],
           'demand_reference_id': row.get('ocd_id', None),
           'customer': row['customer'],
           'customer_id': row.get('customer_id', None),
           'pt_code': pt_code,
           'product_id': row.get('product_id', None),
           'product_name': row['product_name'],
           'package_size': row.get('package_size', ''),
           'standard_uom': row.get('standard_uom', ''),
           'etd': row['etd'],
           'allocated_etd': row['etd'],
           'requested_qty': requested_qty,
           'allocated_qty': allocated_qty,
           'shortage_qty': requested_qty - allocated_qty,
           'fulfillment_rate': (allocated_qty / requested_qty * 100) if requested_qty > 0 else 0,
           'priority': row['priority'],
           'allocation_method': 'Priority'
       })
   
   return pd.DataFrame(allocation_results)

def allocate_pro_rata(demand_allocation, supply_summary):
   """Allocate proportionally based on demand"""
   allocation_results = []
   
   # Group demand by product
   for pt_code in demand_allocation['pt_code'].unique():
       product_demand = demand_allocation[demand_allocation['pt_code'] == pt_code]
       
       # Get available supply
       available_qty = supply_summary[supply_summary['pt_code'] == pt_code]['quantity'].sum()
       total_demand = product_demand['demand_quantity'].sum()
       
       # Calculate allocation ratio
       if total_demand > 0:
           allocation_ratio = min(1.0, available_qty / total_demand)
       else:
           allocation_ratio = 0
       
       # Allocate to each demand line
       for idx, row in product_demand.iterrows():
           requested_qty = row['demand_quantity']
           allocated_qty = round(requested_qty * allocation_ratio)
           
           allocation_results.append({
               'demand_number': row['demand_number'],
               'demand_reference_id': row.get('ocd_id', None),
               'customer': row['customer'],
               'customer_id': row.get('customer_id', None),
               'pt_code': pt_code,
               'product_id': row.get('product_id', None),
               'product_name': row['product_name'],
               'package_size': row.get('package_size', ''),
               'standard_uom': row.get('standard_uom', ''),
               'etd': row['etd'],
               'allocated_etd': row['etd'],
               'requested_qty': requested_qty,
               'allocated_qty': allocated_qty,
               'shortage_qty': requested_qty - allocated_qty,
               'fulfillment_rate': (allocated_qty / requested_qty * 100) if requested_qty > 0 else 0,
               'priority': row['priority'],
               'allocation_method': 'Pro-rata'
           })
   
   return pd.DataFrame(allocation_results)

def prepare_manual_allocation(demand_allocation, supply_summary):
   """Prepare template for manual allocation"""
   allocation_results = []
   
   for idx, row in demand_allocation.iterrows():
       allocation_results.append({
           'demand_number': row['demand_number'],
           'demand_reference_id': row.get('ocd_id', None),
           'customer': row['customer'],
           'customer_id': row.get('customer_id', None),
           'pt_code': row['pt_code'],
           'product_id': row.get('product_id', None),
           'product_name': row['product_name'],
           'package_size': row.get('package_size', ''),
           'standard_uom': row.get('standard_uom', ''),
           'etd': row['etd'],
           'allocated_etd': row['etd'],
           'requested_qty': row['demand_quantity'],
           'allocated_qty': 0,  # To be filled manually
           'shortage_qty': row['demand_quantity'],
           'fulfillment_rate': 0,
           'priority': row['priority'],
           'allocation_method': 'Manual'
       })
   
   return pd.DataFrame(allocation_results)

# === Save to Database ===
def save_allocation_to_database(allocation_df, method, notes=None):
   """Save allocation plan to database"""
   
   engine = get_db_engine()
   
   with engine.begin() as conn:
       try:
           # Create allocation plan header
           allocation_number = f"ALLOC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
           
           # Get user info from session
           creator_id = st.session_state.get('user_id', 1)  # Default to 1 if not set
           
           # Prepare header data
           plan_query = text("""
               INSERT INTO allocation_plans (
                   allocation_number, allocation_date, allocation_method, 
                   status, creator_id, notes
               ) VALUES (
                   :allocation_number, :allocation_date, :allocation_method,
                   :status, :creator_id, :notes
               )
           """)
           
           conn.execute(plan_query, {
               'allocation_number': allocation_number,
               'allocation_date': datetime.now(),
               'allocation_method': method.split()[0].upper(),
               'status': 'DRAFT',
               'creator_id': creator_id,
               'notes': notes
           })
           
           # Get plan ID
           plan_id_result = conn.execute(text(
               "SELECT id FROM allocation_plans WHERE allocation_number = :alloc_num"
           ), {'alloc_num': allocation_number})
           plan_id = plan_id_result.fetchone()[0]
           
           # Prepare allocation details
           for _, row in allocation_df.iterrows():
               detail_query = text("""
                   INSERT INTO allocation_details (
                       allocation_plan_id, demand_type, demand_reference_id, demand_number,
                       product_id, pt_code, customer_id, customer_name, 
                       legal_entity_name, requested_qty, allocated_qty,
                       etd, allocated_etd, status, snapshot_demand_json
                   ) VALUES (
                       :allocation_plan_id, :demand_type, :demand_reference_id, :demand_number,
                       :product_id, :pt_code, :customer_id, :customer_name,
                       :legal_entity_name, :requested_qty, :allocated_qty,
                       :etd, :allocated_etd, :status, :snapshot_demand_json
                   )
               """)
               
               # Get legal entity from demand_df
               legal_entity = demand_df[
                   demand_df['demand_number'] == row['demand_number']
               ]['legal_entity'].iloc[0] if len(demand_df[demand_df['demand_number'] == row['demand_number']]) > 0 else None
               
               # Create snapshot
               snapshot = json.dumps({
                   'requested_qty': float(row['requested_qty']),
                   'etd': str(row['etd']) if pd.notna(row['etd']) else None,
                   'customer': row['customer'],
                   'priority': row['priority']
               })
               
               conn.execute(detail_query, {
                   'allocation_plan_id': plan_id,
                   'demand_type': 'OC',
                   'demand_reference_id': row.get('demand_reference_id'),
                   'demand_number': row['demand_number'],
                   'product_id': row.get('product_id'),
                   'pt_code': row['pt_code'],
                   'customer_id': row.get('customer_id'),
                   'customer_name': row['customer'],
                   'legal_entity_name': legal_entity,
                   'requested_qty': float(row['requested_qty']),
                   'allocated_qty': float(row['allocated_qty']),
                   'etd': row['etd'] if pd.notna(row['etd']) else None,
                   'allocated_etd': row['allocated_etd'] if pd.notna(row['allocated_etd']) else None,
                   'status': 'ALLOCATED',
                   'snapshot_demand_json': snapshot
               })
           
           st.success(f"✅ Allocation plan {allocation_number} saved successfully!")
           save_to_session_state('last_allocation_number', allocation_number)
           save_to_session_state('last_allocation_plan_id', plan_id)
           
           # Clear cache to reflect new allocation
           st.cache_data.clear()
           
           return allocation_number, plan_id
           
       except Exception as e:
           st.error(f"❌ Error saving allocation: {str(e)}")
           return None, None

# === Display Allocation Results ===
def show_allocation_results(allocation_df):
   """Display allocation results with summary metrics"""
   st.markdown("### 📊 Allocation Results")
   
   # Summary metrics
   col1, col2, col3, col4 = st.columns(4)
   
   with col1:
       total_allocated = allocation_df['allocated_qty'].sum()
       st.metric("Total Allocated", format_number(total_allocated))
   
   with col2:
       total_shortage = allocation_df['shortage_qty'].sum()
       st.metric("Remaining Shortage", format_number(total_shortage))
   
   with col3:
       avg_fulfillment = allocation_df['fulfillment_rate'].mean()
       st.metric("Avg Fulfillment Rate", format_percentage(avg_fulfillment))
   
   with col4:
       fully_fulfilled = len(allocation_df[allocation_df['fulfillment_rate'] == 100])
       st.metric("Fully Fulfilled Orders", f"{fully_fulfilled}/{len(allocation_df)}")
   
   # Customer-level summary
   st.markdown("#### 📈 Allocation by Customer")
   customer_summary = allocation_df.groupby(['customer', 'priority']).agg({
       'requested_qty': 'sum',
       'allocated_qty': 'sum',
       'shortage_qty': 'sum'
   }).reset_index()
   
   customer_summary['fulfillment_rate'] = (
       customer_summary['allocated_qty'] / customer_summary['requested_qty'] * 100
   ).round(1)
   
   # Sort by priority
   priority_order = {'Critical': 1, 'High': 2, 'Medium': 3, 'Low': 4}
   customer_summary['priority_order'] = customer_summary['priority'].map(priority_order)
   customer_summary = customer_summary.sort_values('priority_order').drop('priority_order', axis=1)
   
   # Format for display
   display_summary = customer_summary.copy()
   for col in ['requested_qty', 'allocated_qty', 'shortage_qty']:
       display_summary[col] = display_summary[col].apply(format_number)
   display_summary['fulfillment_rate'] = display_summary['fulfillment_rate'].apply(
       lambda x: format_percentage(x)
   )
   
   # Apply priority colors
   def color_priority(val):
       if val in PRIORITY_COLORS:
           return f'background-color: {PRIORITY_COLORS[val]}; color: white;'
       return ''
   
   styled_summary = display_summary.style.applymap(
       color_priority, subset=['priority']
   )
   
   st.dataframe(styled_summary, use_container_width=True)
   
   # Detailed allocation table
   st.markdown("#### 📋 Detailed Allocation Plan")
   
   # Filters
   col1, col2, col3 = st.columns(3)
   with col1:
       filter_customer = st.multiselect(
           "Filter by Customer",
           allocation_df['customer'].unique(),
           key="allocation_filter_customer"
       )
   with col2:
       filter_product = st.multiselect(
           "Filter by Product",
           allocation_df['pt_code'].unique(),
           key="allocation_filter_product"
       )
   with col3:
       show_shortage_only = st.checkbox(
           "Show only items with shortage",
           value=False,
           key="allocation_show_shortage"
       )
   
   # Apply filters
   filtered_df = allocation_df.copy()
   if filter_customer:
       filtered_df = filtered_df[filtered_df['customer'].isin(filter_customer)]
   if filter_product:
       filtered_df = filtered_df[filtered_df['pt_code'].isin(filter_product)]
   if show_shortage_only:
       filtered_df = filtered_df[filtered_df['shortage_qty'] > 0]
   
   # Format for display
   display_df = format_allocation_display(filtered_df)
   
   st.dataframe(display_df, use_container_width=True, height=400)

def format_allocation_display(df):
   """Format allocation dataframe for display"""
   display_df = df.copy()
   
   # Format dates
   display_df['etd'] = pd.to_datetime(display_df['etd']).dt.strftime('%Y-%m-%d')
   
   # Format numbers
   for col in ['requested_qty', 'allocated_qty', 'shortage_qty']:
       display_df[col] = display_df[col].apply(format_number)
   display_df['fulfillment_rate'] = display_df['fulfillment_rate'].apply(
       lambda x: format_percentage(x)
   )
   
   # Select and rename columns
   display_df = display_df[[
       'demand_number', 'customer', 'priority', 'pt_code', 'product_name',
       'etd', 'requested_qty', 'allocated_qty', 'shortage_qty', 'fulfillment_rate'
   ]]
   
   display_df.columns = [
       'Order #', 'Customer', 'Priority', 'PT Code', 'Product',
       'ETD', 'Requested', 'Allocated', 'Shortage', 'Fill Rate'
   ]
   
   return display_df

# === Manual Adjustment ===
def show_manual_adjustment(allocation_df):
   """Allow manual adjustment of allocation"""
   st.markdown("### ✏️ Manual Adjustment")
   
   with st.expander("Adjust Allocation Manually", expanded=False):
       st.info("💡 Adjust allocated quantities below. The system will recalculate fulfillment rates automatically.")
       
       # Prepare data for editing
       edit_df = allocation_df[['demand_number', 'customer', 'pt_code', 'product_name', 
                               'requested_qty', 'allocated_qty']].copy()
       
       # Create editable dataframe
       edited_df = st.data_editor(
           edit_df,
           column_config={
               "allocated_qty": st.column_config.NumberColumn(
                   "Allocated Qty",
                   help="Adjust the allocated quantity",
                   min_value=0,
                   step=1,
               )
           },
           disabled=['demand_number', 'customer', 'pt_code', 'product_name', 'requested_qty'],
           use_container_width=True,
           key="allocation_editor"
       )
       
       if st.button("Apply Changes", type="primary"):
           # Update allocation dataframe
           allocation_df['allocated_qty'] = edited_df['allocated_qty']
           allocation_df['shortage_qty'] = allocation_df['requested_qty'] - allocation_df['allocated_qty']
           allocation_df['fulfillment_rate'] = (
               allocation_df['allocated_qty'] / allocation_df['requested_qty'] * 100
           ).fillna(0)
           
           save_to_session_state('final_allocation_plan', allocation_df)
           st.success("✅ Changes applied successfully!")
           st.rerun()
   
   return allocation_df

# === Approval Section ===
def show_allocation_approval_section(allocation_plan_id):
   """Show approval section for allocation plan"""
   
   st.markdown("### ✅ Allocation Approval")
   
   # Load plan details
   engine = get_db_engine()
   plan_query = """
   SELECT ap.*, e.name as creator_name
   FROM allocation_plans ap
   LEFT JOIN employees e ON ap.creator_id = e.id
   WHERE ap.id = :plan_id
   """
   
   plan_df = pd.read_sql(text(plan_query), engine, params={'plan_id': allocation_plan_id})
   
   if plan_df.empty:
       st.error("Allocation plan not found!")
       return
   
   plan = plan_df.iloc[0]
   
   if plan['status'] == 'DRAFT':
       col1, col2, col3 = st.columns(3)
       
       with col1:
           st.info(f"Status: {plan['status']}")
           st.caption(f"Created by: {plan.get('creator_name', 'Unknown')}")
       
       with col2:
           if st.button("✅ Approve Allocation", type="primary"):
               approve_allocation_plan(allocation_plan_id)
               st.rerun()
       
       with col3:
           rejection_reason = st.text_input("Rejection reason (if rejecting)")
           if st.button("❌ Reject Allocation"):
               if rejection_reason:
                   reject_allocation_plan(allocation_plan_id, rejection_reason)
                   st.rerun()
               else:
                   st.error("Please provide a rejection reason")
   else:
       st.success(f"Status: {plan['status']}")
       if plan['approved_by']:
           st.caption(f"Approved by: {plan['approved_by']} on {plan['approved_date']}")

def approve_allocation_plan(plan_id):
   """Approve allocation plan"""
   engine = get_db_engine()
   
   query = text("""
   UPDATE allocation_plans 
   SET status = 'APPROVED',
       approved_by = :approver,
       approved_date = NOW()
   WHERE id = :plan_id
   """)
   
   with engine.begin() as conn:
       conn.execute(query, {
           'approver': st.session_state.get('user_email', 'system'),
           'plan_id': plan_id
       })
   
   st.success("✅ Allocation plan approved!")

def reject_allocation_plan(plan_id, reason):
   """Reject allocation plan"""
   engine = get_db_engine()
   
   query = text("""
   UPDATE allocation_plans 
   SET status = 'CANCELLED',
       notes = CONCAT(IFNULL(notes, ''), ' | Rejected: ', :reason)
   WHERE id = :plan_id
   """)
   
   with engine.begin() as conn:
       conn.execute(query, {
           'reason': reason,
           'plan_id': plan_id
       })
   
   st.warning("❌ Allocation plan rejected!")

# === Export Functions ===
def show_export_options(allocation_df):
   """Show export options for allocation plan"""
   st.markdown("### 📤 Export Allocation Plan")
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       # Export detailed allocation
       excel_data = convert_df_to_excel(allocation_df, "Allocation Plan")
       st.download_button(
           "📊 Export Detailed Plan",
           data=excel_data,
           file_name=f"allocation_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
       )
   
   with col2:
       # Export customer summary
       customer_summary = allocation_df.groupby(['customer', 'priority']).agg({
           'requested_qty': 'sum',
           'allocated_qty': 'sum',
           'shortage_qty': 'sum'
       }).reset_index()
       
       summary_excel = convert_df_to_excel(customer_summary, "Customer Summary")
       st.download_button(
           "📈 Export Customer Summary",
           data=summary_excel,
           file_name=f"allocation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
       )
   
   with col3:
       # Export shortage notification
       shortage_df = allocation_df[allocation_df['shortage_qty'] > 0]
       
       if not shortage_df.empty:
           notification_excel = convert_df_to_excel(shortage_df, "Shortage Notice")
           st.download_button(
               "📧 Export Shortage Notice",
               data=notification_excel,
               file_name=f"shortage_notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
           )
       else:
           st.info("No shortage items to export")

# === Main Allocation Flow ===

# Step 1: Analyze shortage
shortage_products = analyze_shortage()
show_shortage_summary(shortage_products)

st.markdown("---")

# Step 2: Select allocation method
allocation_method = select_allocation_method()

st.markdown("---")

# Step 3: Set customer priorities if needed
if allocation_method == "Customer Priority" or st.checkbox("Configure Customer Priorities", value=False):
   customer_priorities = set_customer_priorities(demand_df)
else:
   # Default all to Medium
   customer_priorities = {customer: "Medium" for customer in demand_df['customer'].unique()}
   save_to_session_state('customer_priorities', customer_priorities)

st.markdown("---")

# Step 4: Prepare data for allocation
demand_allocation = prepare_demand_for_allocation(demand_df, shortage_products)
supply_summary = prepare_supply_for_allocation(supply_df, shortage_products)

# Step 5: Generate allocation plan
with st.spinner("Generating allocation plan..."):
   if allocation_method == "FIFO (First In First Out)":
       allocation_df = allocate_fifo(demand_allocation, supply_summary)
   elif allocation_method == "Customer Priority":
       allocation_df = allocate_by_priority(demand_allocation, supply_summary)
   elif allocation_method == "Pro-rata":
       allocation_df = allocate_pro_rata(demand_allocation, supply_summary)
   else:  # Manual Override
       allocation_df = prepare_manual_allocation(demand_allocation, supply_summary)

# Save initial allocation
save_to_session_state('initial_allocation_plan', allocation_df)

# Step 6: Display results
show_allocation_results(allocation_df)

st.markdown("---")

# Step 7: Manual adjustment option
if allocation_method != "Manual Override":
   allocation_df = show_manual_adjustment(allocation_df)

# Save final allocation
save_to_session_state('final_allocation_plan', allocation_df)

st.markdown("---")

# Step 8: Export options
show_export_options(allocation_df)

# === Action Buttons ===
st.markdown("---")
st.header("🎯 Next Steps")

col1, col2, col3 = st.columns(3)

with col1:
   notes = st.text_area("Notes for this allocation plan", key="allocation_notes")
   if st.button("💾 Save to Database", type="primary", use_container_width=True):
       allocation_number, plan_id = save_allocation_to_database(
           allocation_df, 
           allocation_method,
           notes
       )
       if plan_id:
           save_to_session_state('allocation_saved_time', datetime.now().strftime('%Y-%m-%d %H:%M'))
           # Show approval section
           show_allocation_approval_section(plan_id)

with col2:
   if allocation_df['shortage_qty'].sum() > 0:
       if st.button("📌 Generate PO Suggestions", use_container_width=True):
           st.switch_page("pages/5_📌_PO_Suggestions.py")
   else:
       st.info("No shortage remaining")

with col3:
   if st.button("📊 Back to GAP Analysis", use_container_width=True):
       st.switch_page("pages/3_📊_GAP_Analysis.py")

# === Help Section ===
with st.expander("ℹ️ Allocation Plan Guide", expanded=False):
   st.markdown("""
   ### Understanding Allocation Methods
   
   **1. FIFO (First In First Out)**
   - Orders with earliest ETD get priority
   - Fair for customers who ordered first
   - Good for perishable products
   
   **2. Customer Priority**
   - Critical > High > Medium > Low
   - Protects key customer relationships
   - Consider contracts and SLAs
   
   **3. Pro-rata**
   - Everyone gets same % of their order
   - Fair distribution across all customers
   - No one gets 100% if shortage exists
   
   **4. Manual Override**
   - Full control over allocation
   - Use when special circumstances apply
   - Requires careful documentation
   
   ### Best Practices
   
   **Setting Priorities:**
   - Consider customer contract terms
   - Review historical order volumes
   - Factor in payment terms
   - Account for strategic importance
   
   **Manual Adjustments:**
   - Document reasons for changes
   - Consider minimum order quantities
   - Check customer communication
   - Verify inventory accuracy
   
   **After Allocation:**
   1. Save and get approval
   2. Export and share with sales team
   3. Notify affected customers
   4. Create PO for shortage items
   5. Schedule follow-up review
   
   ### Database Integration
   
   **Saved Allocations Include:**
   - Allocation plan header with method and status
   - Detailed allocation by product and customer
   - Snapshot of demand at allocation time
   - Audit trail with creator and approver
   
   **Status Flow:**
   - DRAFT → APPROVED → EXECUTED → COMPLETED
   - Only approved allocations affect future GAP analysis
   - Cancelled allocations are kept for audit
   """)

# Footer
st.markdown("---")
st.caption(f"Allocation plan generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")