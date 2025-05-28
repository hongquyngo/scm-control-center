import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from utils.data_loader import load_product_master, load_vendor_master
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
    page_title="PO Suggestions & Reallocation - SCM",
    page_icon="📌",
    layout="wide"
)

# === Constants ===
LEAD_TIME_CATEGORIES = {
    'Urgent': 7,
    'Standard': 30,
    'Economic': 60
}

SAFETY_STOCK_METHODS = [
    'Fixed Days of Supply',
    'Percentage of Demand',
    'Statistical (Z-score)',
    'Custom per Product'
]

# === Header with Navigation ===
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("← Back"):
        # Determine where to go back
        if get_from_session_state('final_allocation_plan') is not None:
            st.switch_page("pages/4_🧩_Allocation_Plan.py")
        else:
            st.switch_page("pages/3_📊_GAP_Analysis.py")
with col2:
    st.title("📌 PO Suggestions & Reallocation")
with col3:
    if st.button("🏠 Dashboard"):
        st.switch_page("main.py")

st.markdown("---")

# === Check Data Availability ===
gap_df = get_from_session_state('gap_analysis_result')
allocation_df = get_from_session_state('final_allocation_plan')
show_reallocation = get_from_session_state('show_reallocation', False)

if gap_df is None:
    st.error("❌ No GAP Analysis data found!")
    st.warning("Please run GAP Analysis first.")
    if st.button("📊 Go to GAP Analysis", type="primary"):
        st.switch_page("pages/3_📊_GAP_Analysis.py")
    st.stop()

# === Determine Mode ===
# Check for shortages after allocation
if allocation_df is not None:
    # Use allocation shortage
    shortage_data = allocation_df[allocation_df['shortage_qty'] > 0].copy()
    shortage_source = "post-allocation"
else:
    # Use GAP analysis shortage
    shortage_data = gap_df[gap_df['gap_quantity'] < 0].copy()
    shortage_data['shortage_qty'] = shortage_data['gap_quantity'].abs()
    shortage_source = "gap-analysis"

# Get surplus data
surplus_data = gap_df[gap_df['gap_quantity'] > 0].copy()

# === Mode Selection ===
if not shortage_data.empty and not surplus_data.empty:
    mode = st.radio(
        "Select Action Mode",
        ["📦 PO Suggestions", "🔄 Reallocation Analysis", "🎯 Combined Strategy"],
        index=2 if show_reallocation else 0,
        horizontal=True
    )
elif not shortage_data.empty:
    mode = "📦 PO Suggestions"
    st.info("📦 Showing PO suggestions for shortage items")
elif not surplus_data.empty:
    mode = "🔄 Reallocation Analysis"
    st.info("🔄 Showing reallocation options for surplus items")
else:
    st.success("✅ No shortage or surplus detected! Supply and demand are balanced.")
    st.stop()

# === Load Master Data ===
@st.cache_data
def load_master_data():
    """Load product and vendor master data"""
    with st.spinner("Loading master data..."):
        products_df = load_product_master()
        vendors_df = load_vendor_master()
    return products_df, vendors_df

products_master, vendors_master = load_master_data()

# === PO Suggestions Functions ===
def calculate_po_suggestions(shortage_data, lead_time_days, safety_stock_days):
    """Calculate PO suggestions based on shortage and parameters"""
    
    # Group shortage by product
    po_suggestions = shortage_data.groupby(['pt_code', 'product_name', 'standard_uom']).agg({
        'shortage_qty': 'sum'
    }).reset_index()
    
    # Add lead time buffer
    po_suggestions['lead_time_days'] = lead_time_days
    po_suggestions['safety_stock_days'] = safety_stock_days
    
    # Calculate suggested order quantity
    # Basic formula: Shortage + Safety Stock
    # In real implementation, would consider MOQ, SPQ, economic order quantity
    po_suggestions['safety_stock_qty'] = po_suggestions['shortage_qty'] * (safety_stock_days / 30)
    po_suggestions['suggested_order_qty'] = po_suggestions['shortage_qty'] + po_suggestions['safety_stock_qty']
    
    # Round to reasonable quantities
    po_suggestions['suggested_order_qty'] = po_suggestions['suggested_order_qty'].apply(
        lambda x: round(x, -1) if x > 100 else round(x)
    )
    
    # Add dates
    po_suggestions['suggested_po_date'] = datetime.now().date()
    po_suggestions['expected_arrival_date'] = datetime.now().date() + timedelta(days=lead_time_days)
    
    # Add mock vendor and cost info (in real implementation, would join with vendor master)
    po_suggestions['suggested_vendor'] = 'Primary Vendor'  # Placeholder
    po_suggestions['estimated_unit_cost'] = 10.0  # Placeholder
    po_suggestions['estimated_total_cost'] = (
        po_suggestions['suggested_order_qty'] * po_suggestions['estimated_unit_cost']
    )
    
    return po_suggestions

def show_po_configuration():
    """Show PO suggestion configuration options"""
    st.markdown("### ⚙️ PO Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lead_time_category = st.selectbox(
            "Lead Time Category",
            list(LEAD_TIME_CATEGORIES.keys()),
            index=1,
            help="Select based on urgency and shipping method"
        )
        lead_time_days = LEAD_TIME_CATEGORIES[lead_time_category]
        st.caption(f"📅 {lead_time_days} days")
    
    with col2:
        safety_stock_method = st.selectbox(
            "Safety Stock Method",
            SAFETY_STOCK_METHODS,
            index=0
        )
        
        if safety_stock_method == 'Fixed Days of Supply':
            safety_stock_days = st.number_input(
                "Days of Safety Stock",
                min_value=0,
                max_value=90,
                value=30,
                step=5
            )
        else:
            safety_stock_days = 30  # Default
    
    with col3:
        consolidate_orders = st.checkbox(
            "Consolidate Orders",
            value=True,
            help="Combine multiple products to same vendor"
        )
        
        consider_moq = st.checkbox(
            "Apply MOQ/SPQ",
            value=True,
            help="Consider minimum order quantities"
        )
    
    return lead_time_days, safety_stock_days, consolidate_orders, consider_moq

def show_po_suggestions(po_suggestions):
    """Display PO suggestions"""
    st.markdown("### 📦 Purchase Order Suggestions")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(po_suggestions)
        st.metric("Products to Order", f"{total_products}")
    
    with col2:
        total_quantity = po_suggestions['suggested_order_qty'].sum()
        st.metric("Total Order Quantity", format_number(total_quantity))
    
    with col3:
        total_cost = po_suggestions['estimated_total_cost'].sum()
        st.metric("Estimated Total Cost", format_currency(total_cost, "USD"))
    
    with col4:
        avg_lead_time = po_suggestions['lead_time_days'].mean()
        st.metric("Avg Lead Time", f"{avg_lead_time:.0f} days")
    
    # Detailed suggestions table
    st.markdown("#### 📋 Detailed PO Suggestions")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        min_value = st.number_input(
            "Minimum Order Value (USD)",
            min_value=0,
            value=0,
            step=1000,
            help="Filter out small orders"
        )
    
    with col2:
        selected_products = st.multiselect(
            "Filter by Product",
            po_suggestions['pt_code'].unique()
        )
    
    # Apply filters
    filtered_po = po_suggestions.copy()
    if min_value > 0:
        filtered_po = filtered_po[filtered_po['estimated_total_cost'] >= min_value]
    if selected_products:
        filtered_po = filtered_po[filtered_po['pt_code'].isin(selected_products)]
    
    # Format for display
    display_po = filtered_po.copy()
    display_po['shortage_qty'] = display_po['shortage_qty'].apply(format_number)
    display_po['safety_stock_qty'] = display_po['safety_stock_qty'].apply(format_number)
    display_po['suggested_order_qty'] = display_po['suggested_order_qty'].apply(format_number)
    display_po['estimated_unit_cost'] = display_po['estimated_unit_cost'].apply(
        lambda x: format_currency(x, "USD")
    )
    display_po['estimated_total_cost'] = display_po['estimated_total_cost'].apply(
        lambda x: format_currency(x, "USD")
    )
    
    # Select columns for display
    display_columns = [
        'pt_code', 'product_name', 'shortage_qty', 'safety_stock_qty',
        'suggested_order_qty', 'suggested_vendor', 'estimated_unit_cost',
        'estimated_total_cost', 'expected_arrival_date'
    ]
    
    st.dataframe(
        display_po[display_columns],
        use_container_width=True,
        height=400
    )
    
    return filtered_po

# === Reallocation Functions ===
def analyze_reallocation_opportunities(surplus_data, shortage_data):
    """Analyze reallocation opportunities between locations/entities"""
    
    # Group surplus by product and entity
    surplus_summary = surplus_data.groupby(['pt_code', 'product_name', 'legal_entity']).agg({
        'gap_quantity': 'sum'
    }).reset_index()
    surplus_summary = surplus_summary.rename(columns={'gap_quantity': 'surplus_qty'})
    
    # Group shortage by product
    if not shortage_data.empty:
        shortage_summary = shortage_data.groupby(['pt_code', 'product_name']).agg({
            'shortage_qty': 'sum'
        }).reset_index()
    else:
        shortage_summary = pd.DataFrame()
    
    # Find matching opportunities
    reallocation_opportunities = []
    
    for _, surplus_row in surplus_summary.iterrows():
        pt_code = surplus_row['pt_code']
        
        # Check if this product has shortage elsewhere
        if not shortage_summary.empty:
            shortage_match = shortage_summary[shortage_summary['pt_code'] == pt_code]
            
            if not shortage_match.empty:
                shortage_qty = shortage_match.iloc[0]['shortage_qty']
                reallocate_qty = min(surplus_row['surplus_qty'], shortage_qty)
                
                reallocation_opportunities.append({
                    'pt_code': pt_code,
                    'product_name': surplus_row['product_name'],
                    'from_entity': surplus_row['legal_entity'],
                    'surplus_available': surplus_row['surplus_qty'],
                    'shortage_elsewhere': shortage_qty,
                    'suggested_reallocation_qty': reallocate_qty,
                    'reallocation_percentage': (reallocate_qty / shortage_qty * 100)
                })
        else:
            # No shortage, but still show surplus for awareness
            reallocation_opportunities.append({
                'pt_code': pt_code,
                'product_name': surplus_row['product_name'],
                'from_entity': surplus_row['legal_entity'],
                'surplus_available': surplus_row['surplus_qty'],
                'shortage_elsewhere': 0,
                'suggested_reallocation_qty': 0,
                'reallocation_percentage': 0
            })
    
    return pd.DataFrame(reallocation_opportunities)

def show_reallocation_analysis(reallocation_df):
    """Display reallocation analysis"""
    st.markdown("### 🔄 Reallocation Opportunities")
    
    # Filter for actual opportunities
    actual_opportunities = reallocation_df[reallocation_df['suggested_reallocation_qty'] > 0]
    
    if not actual_opportunities.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = len(actual_opportunities)
            st.metric("Products for Reallocation", f"{total_products}")
        
        with col2:
            total_reallocation = actual_opportunities['suggested_reallocation_qty'].sum()
            st.metric("Total Reallocation Qty", format_number(total_reallocation))
        
        with col3:
            avg_coverage = actual_opportunities['reallocation_percentage'].mean()
            st.metric("Avg Shortage Coverage", format_percentage(avg_coverage))
        
        with col4:
            entities_involved = actual_opportunities['from_entity'].nunique()
            st.metric("Entities Involved", f"{entities_involved}")
        
        # Detailed table
        st.markdown("#### 📋 Reallocation Details")
        
        display_realloc = actual_opportunities.copy()
        for col in ['surplus_available', 'shortage_elsewhere', 'suggested_reallocation_qty']:
            display_realloc[col] = display_realloc[col].apply(format_number)
        display_realloc['reallocation_percentage'] = display_realloc['reallocation_percentage'].apply(
            lambda x: format_percentage(x)
        )
        
        st.dataframe(display_realloc, use_container_width=True)
        
        # Cost-benefit analysis
        st.markdown("#### 💰 Cost-Benefit Analysis")
        st.info("""
        **Reallocation Benefits:**
        - Faster than new PO (immediate availability)
        - Lower cost (only transportation)
        - Reduces excess inventory
        - Improves overall inventory efficiency
        
        **Consider:**
        - Transportation costs between entities
        - Handling and documentation
        - Tax implications for inter-company transfers
        """)
    else:
        st.warning("No direct reallocation opportunities found (no matching products between surplus and shortage)")
    
    # Show all surplus even if no shortage match
    st.markdown("#### 📦 All Surplus Inventory")
    all_surplus = reallocation_df.copy()
    for col in ['surplus_available', 'shortage_elsewhere', 'suggested_reallocation_qty']:
        all_surplus[col] = all_surplus[col].apply(format_number)
    
    st.dataframe(
        all_surplus[['pt_code', 'product_name', 'from_entity', 'surplus_available']],
        use_container_width=True
    )

# === Combined Strategy ===
def show_combined_strategy(po_suggestions, reallocation_opportunities):
    """Show combined PO + Reallocation strategy"""
    st.markdown("### 🎯 Combined Strategy: Reallocation + PO")
    
    # Calculate net shortage after reallocation
    products_with_realloc = reallocation_opportunities[
        reallocation_opportunities['suggested_reallocation_qty'] > 0
    ]['pt_code'].unique()
    
    # Adjust PO suggestions
    adjusted_po = po_suggestions.copy()
    
    for pt_code in products_with_realloc:
        realloc_qty = reallocation_opportunities[
            reallocation_opportunities['pt_code'] == pt_code
        ]['suggested_reallocation_qty'].sum()
        
        # Reduce PO quantity by reallocation amount
        mask = adjusted_po['pt_code'] == pt_code
        if mask.any():
            adjusted_po.loc[mask, 'shortage_qty'] -= realloc_qty
            adjusted_po.loc[mask, 'suggested_order_qty'] = (
                adjusted_po.loc[mask, 'shortage_qty'] + 
                adjusted_po.loc[mask, 'safety_stock_qty']
            )
    
    # Remove products that no longer need PO
    adjusted_po = adjusted_po[adjusted_po['suggested_order_qty'] > 0]
    
    # Summary comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Original Plan (PO Only)")
        st.metric("Products to Order", len(po_suggestions))
        st.metric("Total Order Qty", format_number(po_suggestions['suggested_order_qty'].sum()))
        st.metric("Total Cost", format_currency(po_suggestions['estimated_total_cost'].sum(), "USD"))
    
    with col2:
        st.markdown("#### ✨ Optimized Plan (Realloc + PO)")
        st.metric("Products to Order", len(adjusted_po))
        st.metric("Total Order Qty", format_number(adjusted_po['suggested_order_qty'].sum()))
        st.metric("Total Cost", format_currency(adjusted_po['estimated_total_cost'].sum(), "USD"))
    
    # Savings
    st.markdown("#### 💰 Savings from Reallocation")
    cost_savings = (
        po_suggestions['estimated_total_cost'].sum() - 
        adjusted_po['estimated_total_cost'].sum()
    )
    qty_savings = (
        po_suggestions['suggested_order_qty'].sum() - 
        adjusted_po['suggested_order_qty'].sum()
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cost Savings", format_currency(cost_savings, "USD"))
    with col2:
        st.metric("Quantity Reduction", format_number(qty_savings))
    with col3:
        savings_pct = (cost_savings / po_suggestions['estimated_total_cost'].sum() * 100)
        st.metric("Savings %", format_percentage(savings_pct))
    
    return adjusted_po

# === Export Functions ===
def show_export_section(po_suggestions, reallocation_df=None, combined_po=None):
    """Show export options"""
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if po_suggestions is not None and not po_suggestions.empty:
            excel_data = convert_df_to_excel(po_suggestions, "PO Suggestions")
            st.download_button(
                "📦 Export PO Suggestions",
                data=excel_data,
                file_name=f"po_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if reallocation_df is not None and not reallocation_df.empty:
            excel_data = convert_df_to_excel(reallocation_df, "Reallocation")
            st.download_button(
                "🔄 Export Reallocation Plan",
                data=excel_data,
                file_name=f"reallocation_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📑 Export Complete Report"):
            sheets_dict = {}
            
            if po_suggestions is not None and not po_suggestions.empty:
                sheets_dict["PO Suggestions"] = po_suggestions
            
            if reallocation_df is not None and not reallocation_df.empty:
                sheets_dict["Reallocation"] = reallocation_df
            
            if combined_po is not None and not combined_po.empty:
                sheets_dict["Combined Strategy"] = combined_po
            
            if sheets_dict:
                excel_data = export_multiple_sheets(sheets_dict)
                st.download_button(
                    "Download Complete Report",
                    data=excel_data,
                    file_name=f"supply_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# === Main Logic Based on Mode ===
st.markdown("---")

if mode == "📦 PO Suggestions" or mode == "🎯 Combined Strategy":
    # PO Configuration
    lead_time_days, safety_stock_days, consolidate_orders, consider_moq = show_po_configuration()
    
    st.markdown("---")
    
    # Calculate PO suggestions
    po_suggestions = calculate_po_suggestions(shortage_data, lead_time_days, safety_stock_days)
    
    if mode == "📦 PO Suggestions":
        # Show PO suggestions only
        filtered_po = show_po_suggestions(po_suggestions)
        
        st.markdown("---")
        show_export_section(filtered_po)
    
    else:  # Combined Strategy
        # Also calculate reallocation
        reallocation_opportunities = analyze_reallocation_opportunities(surplus_data, shortage_data)
        
        # Show combined strategy
        adjusted_po = show_combined_strategy(po_suggestions, reallocation_opportunities)
        
        st.markdown("---")
        
        # Show both tables
        tab1, tab2, tab3 = st.tabs(["🔄 Reallocation First", "📦 Then PO", "📊 Summary"])
        
        with tab1:
            show_reallocation_analysis(reallocation_opportunities)
        
        with tab2:
            st.markdown("#### 📦 Adjusted PO Suggestions (After Reallocation)")
            show_po_suggestions(adjusted_po)
        
        with tab3:
            st.markdown("#### 📊 Implementation Summary")
            st.markdown("""
            **Step 1: Execute Reallocations**
            - Transfer surplus inventory between entities
            - Typically 1-3 days for internal transfers
            - Lower cost than new purchases
            
            **Step 2: Place Purchase Orders**
            - Order remaining shortage quantities
            - Include safety stock buffer
            - Consider lead times for planning
            
            **Benefits:**
            - Optimized inventory levels
            - Reduced procurement costs
            - Faster shortage resolution
            - Better cash flow management
            """)
        
        st.markdown("---")
        show_export_section(po_suggestions, reallocation_opportunities, adjusted_po)

else:  # Reallocation Analysis only
    reallocation_opportunities = analyze_reallocation_opportunities(surplus_data, shortage_data)
    show_reallocation_analysis(reallocation_opportunities)
    
    st.markdown("---")
    show_export_section(None, reallocation_opportunities)

# === Action Summary ===
st.markdown("---")
st.header("📋 Action Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Immediate Actions")
    if 'reallocation_opportunities' in locals() and not reallocation_opportunities.empty:
        st.markdown("""
        1. **Review reallocation opportunities**
           - Verify surplus inventory availability
           - Calculate transfer costs
           - Get approvals for inter-company transfers
        """)
    
    if 'po_suggestions' in locals() and not po_suggestions.empty:
        st.markdown("""
        2. **Prepare Purchase Orders**
           - Confirm vendor availability
           - Negotiate prices for large orders
           - Set delivery priorities
        """)

with col2:
    st.markdown("### 📅 Follow-up Actions")
    st.markdown("""
    1. **Monitor execution**
       - Track reallocation transfers
       - Confirm PO acknowledgments
       - Update system with new ETAs
    
    2. **Communicate updates**
       - Inform sales of resolution timeline
       - Update customers on availability
       - Adjust allocation if needed
    
    3. **Review and optimize**
       - Analyze root causes
       - Adjust safety stock levels
       - Improve forecast accuracy
    """)

# === Help Section ===
with st.expander("ℹ️ Understanding PO Suggestions & Reallocation", expanded=False):
    st.markdown("""
    ### PO Suggestions Logic
    
    **Order Quantity Calculation:**
    - Base: Current shortage quantity
    - Plus: Safety stock buffer
    - Rounded: To practical order sizes
    - Future: Consider MOQ, SPQ, EOQ
    
    **Lead Time Categories:**
    - **Urgent**: Air freight, 7 days
    - **Standard**: Regular shipping, 30 days
    - **Economic**: Sea freight, 60 days
    
    **Safety Stock Methods:**
    - **Fixed Days**: X days of average demand
    - **Percentage**: X% above shortage
    - **Statistical**: Based on demand variability
    - **Custom**: Product-specific rules
    
    ### Reallocation Benefits
    
    **When to Reallocate:**
    - Surplus in one location, shortage in another
    - Transfer cost < new purchase cost
    - Time critical situations
    - Expiring inventory
    
    **Considerations:**
    - Transportation costs
    - Handling complexity
    - Tax implications
    - System updates
    
    ### Combined Strategy
    
    **Optimization Approach:**
    1. First: Use available surplus (reallocation)
    2. Then: Order remaining shortage (PO)
    3. Result: Minimum cost and time
    
    **Success Factors:**
    - Accurate inventory data
    - Good inter-entity coordination
    - Reliable transportation
    - Clear communication
    """)

# Footer
st.markdown("---")
st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")