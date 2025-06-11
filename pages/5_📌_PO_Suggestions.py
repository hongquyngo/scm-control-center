import streamlit as st
from datetime import datetime

# === Page Config ===
st.set_page_config(
    page_title="PO Suggestions & Reallocation - SCM",
    page_icon="📌",
    layout="wide"
)


# === Help Section ===
with st.expander("ℹ️ Understanding PO Suggestions & Reallocation", expanded=True):
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