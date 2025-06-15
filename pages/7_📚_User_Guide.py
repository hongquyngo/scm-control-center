# pages/7_📚_User_Guide.py
import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import AuthManager
from utils.display_components import DisplayComponents

# === Page Config ===
st.set_page_config(
    page_title="User Guide - SCM",
    page_icon="📚",
    layout="wide"
)

# === Authentication Check ===
auth_manager = AuthManager()
if not auth_manager.check_session():
    st.switch_page("pages/0_🔐_Login.py")
    st.stop()

# === Custom CSS ===
st.markdown("""
<style>
    .guide-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .term-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #366092;
    }
    .workflow-step {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .example-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .module-card {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        cursor: pointer;
    }
    .module-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# === Header ===
st.title("📚 User Guide")
st.markdown("---")

# === Navigation Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", 
    "📖 Business Terms", 
    "🔄 Module Workflows", 
    "💡 Use Cases", 
    "⚡ Quick Reference"
])

# === Tab 1: Overview ===
with tab1:
    st.markdown("## Welcome to SCM Control Center")
    
    st.markdown("""
    <div class="guide-section">
    <h3>🎯 System Purpose</h3>
    <p>The Supply Chain Management (SCM) Control Center is designed to help you:</p>
    <ul>
        <li>Monitor and analyze customer demand</li>
        <li>Track inventory and supply sources</li>
        <li>Identify gaps between supply and demand</li>
        <li>Create optimal allocation plans</li>
        <li>Generate purchase order recommendations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Module Overview
    st.markdown("### 📋 System Modules")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <h4>📊 Demand Analysis</h4>
            <p>View and analyze customer orders, forecasts, and demand patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h4>📥 Supply Analysis</h4>
            <p>Monitor inventory levels, incoming supply, and expiry management</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <h4>📊 GAP Analysis</h4>
            <p>Compare demand vs supply to identify shortages and surpluses</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h4>🧩 Allocation Plan</h4>
            <p>Distribute available inventory to customer orders optimally</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <h4>📌 PO Suggestions</h4>
            <p>AI-powered purchase order recommendations (Coming Soon)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h4>⚙️ Settings</h4>
            <p>Configure time adjustments and business rules</p>
        </div>
        """, unsafe_allow_html=True)
    
    # User Information
    st.markdown("### 👤 User Information")
    st.info("""
    The system tracks user login information including:
    - Username and full name
    - User role (stored in database)
    - Login time and session duration
    
    All authenticated users currently have the same access level to all features.
    """)

# === Tab 2: Business Terms ===
with tab2:
    st.markdown("## Business Terminology")
    
    # Supply Chain Terms
    st.markdown("### 📦 Supply Chain Terms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="term-card">
            <h4>OC (Order Confirmation)</h4>
            <p>Sales order confirmations from customers. These are confirmed orders with agreed delivery dates (ETD) and quantities.</p>
            <p><b>Example:</b> Customer ABC confirmed order for 1000 units to be delivered on March 15</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="term-card">
            <h4>Forecast</h4>
            <p>Predicted future demand based on historical data or customer projections. Less certain than OC.</p>
            <p><b>Example:</b> Expected demand of 5000 units in April based on last year's sales</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="term-card">
            <h4>MOQ (Minimum Order Quantity)</h4>
            <p>The smallest amount a supplier will accept for an order.</p>
            <p><b>Example:</b> Supplier requires minimum 500 units per order</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="term-card">
            <h4>ETD (Expected Time of Delivery)</h4>
            <p>The date when goods should be delivered to the customer.</p>
            <p><b>Example:</b> Customer expects delivery by March 20, 2025</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="term-card">
            <h4>ETA (Estimated Time of Arrival)</h4>
            <p>When incoming supply is expected to arrive at your warehouse.</p>
            <p><b>Example:</b> Purchase order expected to arrive on March 10, 2025</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="term-card">
            <h4>SPQ (Standard Package Quantity)</h4>
            <p>The standard packaging size from suppliers.</p>
            <p><b>Example:</b> Products come in boxes of 50 units each</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Status Terms
    st.markdown("### 📊 System Status Terms")
    
    status_df = pd.DataFrame({
        'Status': ['Shortage', 'Surplus', 'Balanced', 'Critical', 'Overdue'],
        'Meaning': [
            'Demand exceeds available supply',
            'Supply exceeds demand',
            'Supply matches demand perfectly',
            'Urgent shortage requiring immediate action',
            'Orders past their ETD date'
        ],
        'Visual Indicator': ['🔴 Red', '🟢 Green', '⚪ Gray', '🚨 Red Alert', '⚠️ Yellow'],
        'Action Required': [
            'Create PO or allocate available stock',
            'Consider promotions or redistribution',
            'Monitor for changes',
            'Immediate allocation or expedite supply',
            'Contact customer or expedite delivery'
        ]
    })
    
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    # Allocation Terms
    st.markdown("### 🧩 Allocation Terms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="term-card">
            <h4>SOFT Allocation (90% cases)</h4>
            <p>Flexible quantity allocation without locking specific batches. System automatically selects best supply at delivery time.</p>
            <p><b>Use when:</b> Standard orders without specific batch requirements</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="term-card">
            <h4>FCFS (First Come First Served)</h4>
            <p>Allocation method that prioritizes orders with earliest ETD.</p>
            <p><b>Best for:</b> Time-sensitive products or fair distribution</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="term-card">
            <h4>HARD Allocation (10% cases)</h4>
            <p>Locks specific supply batches to customer orders. Cannot be changed after approval.</p>
            <p><b>Use when:</b> Customer needs specific batch/origin/quality</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="term-card">
            <h4>Pro-rata Allocation</h4>
            <p>Distributes available supply proportionally based on demand quantities.</p>
            <p><b>Example:</b> If supply is 50% of demand, each customer gets 50% of their order</p>
        </div>
        """, unsafe_allow_html=True)

# === Tab 3: Module Workflows ===
with tab3:
    st.markdown("## Module Workflows")
    
    # Demand Analysis Workflow
    with st.expander("📊 Demand Analysis Workflow", expanded=True):
        st.markdown("""
        <div class="workflow-step">
            <h4>Step 1: Access Demand Analysis</h4>
            <p>Navigate to Demand Analysis from the sidebar or main dashboard</p>
        </div>
        
        <div class="workflow-step">
            <h4>Step 2: Apply Filters</h4>
            <ul>
                <li>Select date range for analysis</li>
                <li>Filter by products, customers, or entities</li>
                <li>Choose demand types (OC/Forecast)</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 3: Review Demand Data</h4>
            <ul>
                <li>Check total demand quantities</li>
                <li>Identify overdue orders (past ETD)</li>
                <li>Analyze demand distribution by product/customer</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 4: Take Action</h4>
            <ul>
                <li>Export demand report for planning</li>
                <li>Navigate to GAP Analysis for supply comparison</li>
                <li>Create allocation plan for urgent orders</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # GAP Analysis Workflow
    with st.expander("📊 GAP Analysis Workflow"):
        st.markdown("""
        <div class="workflow-step">
            <h4>Step 1: Configure Analysis Parameters</h4>
            <ul>
                <li>Set period type (Daily/Weekly/Monthly)</li>
                <li>Choose date modes for demand and supply</li>
                <li>Apply product/customer filters</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 2: Run GAP Analysis</h4>
            <ul>
                <li>Click "Run GAP Analysis" button</li>
                <li>System calculates supply vs demand by period</li>
                <li>Identifies shortages and surpluses</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 3: Review Results</h4>
            <ul>
                <li>Check shortage summary for critical items</li>
                <li>Review fulfillment rates by product</li>
                <li>Analyze time-phased supply/demand balance</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 4: Next Actions</h4>
            <ul>
                <li>Create Allocation Plan for products with shortage</li>
                <li>Generate PO for items below safety stock</li>
                <li>Export GAP report for management review</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Allocation Plan Workflow
    with st.expander("🧩 Allocation Plan Workflow"):
        st.markdown("""
        <div class="workflow-step">
            <h4>Step 1: Create New Allocation</h4>
            <ul>
                <li>Click "Create New" from Allocation Plan page</li>
                <li>System loads current GAP analysis data</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 2: Select Products</h4>
            <ul>
                <li>Choose products for allocation</li>
                <li>Can select shortage items, available items, or all</li>
                <li>Review demand and supply for selected products</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 3: Choose Allocation Method</h4>
            <ul>
                <li><b>FCFS:</b> For time-sensitive orders</li>
                <li><b>Priority:</b> For VIP customers</li>
                <li><b>Pro-rata:</b> For fair distribution</li>
                <li><b>Manual:</b> For custom scenarios</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 4: Configure Parameters</h4>
            <ul>
                <li>Set allocation type (SOFT/HARD)</li>
                <li>Configure method-specific settings</li>
                <li>Add notes and comments</li>
            </ul>
        </div>
        
        <div class="workflow-step">
            <h4>Step 5: Review and Approve</h4>
            <ul>
                <li>Review allocation results</li>
                <li>Make manual adjustments if needed</li>
                <li>Save as draft or approve immediately</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# === Tab 4: Use Cases ===
with tab4:
    st.markdown("## Common Use Cases & Examples")
    
    # Use Case 1: Shortage Handling
    st.markdown("### 🔴 Use Case 1: Handling Product Shortage")
    st.markdown("""
    <div class="example-box">
        <h4>Scenario:</h4>
        <p>Product ABC has 500 units in stock but customer orders total 800 units for next week.</p>
        
        <h4>Steps to resolve:</h4>
        <ol>
            <li><b>Run GAP Analysis:</b> Identify the 300-unit shortage</li>
            <li><b>Check Supply Pipeline:</b> Look for incoming POs or transfers</li>
            <li><b>Create Allocation Plan:</b>
                <ul>
                    <li>Use Priority method if you have VIP customers</li>
                    <li>Use FCFS for fair distribution</li>
                    <li>Use Pro-rata to give each customer partial quantity</li>
                </ul>
            </li>
            <li><b>Communicate:</b> Inform customers about partial fulfillment</li>
            <li><b>Create PO:</b> Order additional 300+ units for future</li>
        </ol>
        
        <h4>Result:</h4>
        <p>All customers receive fair allocation based on chosen method, and future supply is secured.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Case 2: Surplus Management
    st.markdown("### 🟢 Use Case 2: Managing Surplus Inventory")
    st.markdown("""
    <div class="example-box">
        <h4>Scenario:</h4>
        <p>Product XYZ has 2000 units in stock but only 500 units of demand for next month.</p>
        
        <h4>Steps to resolve:</h4>
        <ol>
            <li><b>Run GAP Analysis:</b> Identify 1500-unit surplus</li>
            <li><b>Check Expiry Dates:</b> Priority for near-expiry items</li>
            <li><b>Actions to consider:</b>
                <ul>
                    <li>Offer promotions to increase demand</li>
                    <li>Transfer to other locations with shortage</li>
                    <li>Negotiate with customers for early delivery</li>
                    <li>Adjust future PO quantities</li>
                </ul>
            </li>
            <li><b>Monitor:</b> Track inventory aging and adjust strategy</li>
        </ol>
        
        <h4>Result:</h4>
        <p>Reduced holding costs and minimized expiry risk.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Case 3: VIP Customer Priority
    st.markdown("### ⭐ Use Case 3: VIP Customer Priority Allocation")
    st.markdown("""
    <div class="example-box">
        <h4>Scenario:</h4>
        <p>Limited stock of 1000 units with orders from both VIP and regular customers totaling 1500 units.</p>
        
        <h4>Steps to resolve:</h4>
        <ol>
            <li><b>Set Customer Priorities:</b>
                <ul>
                    <li>VIP Customers: Priority 10</li>
                    <li>Regular Customers: Priority 5</li>
                    <li>New Customers: Priority 1</li>
                </ul>
            </li>
            <li><b>Create Allocation Plan:</b>
                <ul>
                    <li>Choose "Priority Based" method</li>
                    <li>Set minimum allocation % if needed</li>
                </ul>
            </li>
            <li><b>Review Results:</b>
                <ul>
                    <li>VIP customers get 100% fulfillment</li>
                    <li>Regular customers get partial based on remaining</li>
                </ul>
            </li>
        </ol>
        
        <h4>Result:</h4>
        <p>Maintained VIP customer satisfaction while fairly distributing remaining stock.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Case 4: Time-Sensitive Orders
    st.markdown("### ⏰ Use Case 4: Time-Sensitive Order Management")
    st.markdown("""
    <div class="example-box">
        <h4>Scenario:</h4>
        <p>Multiple orders with different ETDs but limited supply arriving in batches.</p>
        
        <h4>Steps to resolve:</h4>
        <ol>
            <li><b>Apply Time Adjustments:</b>
                <ul>
                    <li>Set transportation lead time</li>
                    <li>Add buffer days for safety</li>
                </ul>
            </li>
            <li><b>Run GAP Analysis:</b> With daily period type</li>
            <li><b>Create FCFS Allocation:</b>
                <ul>
                    <li>System prioritizes earliest ETD</li>
                    <li>Ensures on-time delivery for urgent orders</li>
                </ul>
            </li>
            <li><b>Monitor Execution:</b>
                <ul>
                    <li>Track allocation vs actual delivery</li>
                    <li>Adjust future plans based on performance</li>
                </ul>
            </li>
        </ol>
        
        <h4>Result:</h4>
        <p>Minimized late deliveries and improved customer satisfaction.</p>
    </div>
    """, unsafe_allow_html=True)

# === Tab 5: Quick Reference ===
with tab5:
    st.markdown("## Quick Reference Guide")
    
    # Common Tasks
    st.markdown("### ✅ Common Tasks Checklist")
    
    tasks_df = pd.DataFrame({
        'Task': [
            'Check daily demand',
            'Review inventory levels',
            'Identify shortages',
            'Create allocation plan',
            'Export reports',
            'Adjust time settings'
        ],
        'Module': [
            'Demand Analysis',
            'Supply Analysis',
            'GAP Analysis',
            'Allocation Plan',
            'Any module',
            'Settings'
        ],
        'Quick Steps': [
            '1. Go to Demand Analysis → 2. Set date range → 3. Review summary',
            '1. Go to Supply Analysis → 2. Check "Current Stock" metric',
            '1. Run GAP Analysis → 2. Check shortage summary → 3. Sort by gap quantity',
            '1. GAP Analysis → 2. Click "Create Allocation" → 3. Follow wizard',
            '1. Look for 📥 export button → 2. Choose format → 3. Download',
            '1. Go to Settings → 2. Adjust offsets → 3. Save settings'
        ]
    })
    
    st.dataframe(tasks_df, use_container_width=True, hide_index=True)
    
    # Keyboard Shortcuts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⌨️ Navigation Tips")
        st.markdown("""
        - Use sidebar to switch between modules
        - Press 'R' to refresh data (when available)
        - Use browser back button to return to previous page
        - Bookmark frequently used pages
        """)
    
    with col2:
        st.markdown("### 💡 Best Practices")
        st.markdown("""
        - Run GAP Analysis daily for critical products
        - Review allocation performance weekly
        - Export reports for offline analysis
        - Document special cases in notes
        - Keep time adjustments updated
        """)
    
    # Troubleshooting
    st.markdown("### 🔧 Troubleshooting")
    
    trouble_df = pd.DataFrame({
        'Issue': [
            'No data showing',
            'GAP Analysis not running',
            'Cannot create allocation',
            'Export not working',
            'Login issues'
        ],
        'Possible Cause': [
            'Filters too restrictive',
            'No demand/supply data',
            'Insufficient permissions',
            'Large dataset',
            'Session expired'
        ],
        'Solution': [
            'Check date range and remove filters',
            'Ensure data is loaded in Demand/Supply first',
            'Contact admin for role upgrade',
            'Try filtering data before export',
            'Log out and log in again'
        ]
    })
    
    st.dataframe(trouble_df, use_container_width=True, hide_index=True)
    
    # Contact Support
    st.markdown("### 📞 Need Help?")
    st.info("""
    **For additional support:**
    - 📧 Email: scm-support@company.com
    - 📱 Hotline: +84 xxx xxx xxx
    - 💬 Teams: SCM Support Channel
    - 📋 Submit ticket: helpdesk.company.com
    """)

# === Footer ===
st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.caption(f"SCM Control Center v1.0 - User Guide")

with col2:
    current_user = st.session_state.get('username', 'Guest')
    user_role = st.session_state.get('user_role', 'user')
    st.caption(f"Logged in as: {current_user} ({user_role})")

with col3:
    st.caption("Last updated: Jan 2025")