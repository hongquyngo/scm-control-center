import streamlit as st
import uuid

# Page config MUST be first
st.set_page_config(
    page_title="Data Adjustment Settings",
    page_icon="⚙️",
    layout="wide"
)

# Now import other modules
from utils.display_components import DisplayComponents
from utils.session_state import initialize_session_state
from utils.settings_manager import SettingsManager
from utils.adjustments.time_adjustments import TimeAdjustmentManager
from utils.adjustments.business_rules import BusinessRulesManager

# Initialize
initialize_session_state()
settings_manager = SettingsManager()

# Header
DisplayComponents.show_page_header(
    title="Data Adjustment Settings",
    icon="⚙️",
    prev_page="pages/5_📌_PO_Suggestions.py",
    next_page=None
)

# Navigation in sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    # Navigation options
    nav_option = st.radio(
        "Select Category",
        ["⏱️ Time Adjustments", "📋 Allocation Rules", "📦 PO Rules"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    
    if nav_option == "⏱️ Time Adjustments":
        rules_count = len(st.session_state.get('time_adjustment_rules', []))
        st.metric("Active Rules", rules_count)
        
        if rules_count > 0:
            st.caption(f"✅ {rules_count} time adjustment rules configured")
        else:
            st.caption("📝 No rules configured yet")
    
    elif nav_option == "📋 Allocation Rules":
        st.caption("📋 Configure allocation priorities")
    
    elif nav_option == "📦 PO Rules":
        st.caption("📦 Set purchase order parameters")

# Main content area (full width)
# Display content based on selection
if nav_option == "⏱️ Time Adjustments":
    st.markdown("### ⏱️ Time Adjustments")
    st.markdown("Adjust dates for more accurate analysis")
    
    # Render time adjustments UI
    TimeAdjustmentManager.render_time_adjustments()
    
elif nav_option == "📋 Allocation Rules":
    st.markdown("### 📋 Allocation Rules")
    st.markdown("Configure how inventory is allocated to orders")
    
    # Render allocation rules UI
    BusinessRulesManager.render_allocation_rules()
    
elif nav_option == "📦 PO Rules":
    st.markdown("### 📦 PO Suggestion Rules")
    st.markdown("Configure purchase order generation parameters")
    
    # Render PO rules UI
    BusinessRulesManager.render_po_rules()

# Footer
st.markdown("---")
st.caption("💡 Settings are applied to current session only. They will be saved when creating Allocation Plans or PO Suggestions.")