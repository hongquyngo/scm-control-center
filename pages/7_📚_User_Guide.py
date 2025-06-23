# pages/7_📖_User_Guide.py

import streamlit as st
from datetime import datetime
import base64
from pathlib import Path

# Page config
st.set_page_config(
    page_title="User Guide - SCM Control Center",
    page_icon="📖",
    layout="wide"
)

# Initialize session state for navigation
if 'guide_section' not in st.session_state:
    st.session_state.guide_section = 'overview'

# Header
st.title("📖 User Guide - SCM Control Center")
st.markdown("---")

# Create navigation sidebar
with st.sidebar:
    st.markdown("### 📑 Navigation")
    
    # Overview
    if st.button("🏠 Overview", use_container_width=True):
        st.session_state.guide_section = 'overview'
    
    st.markdown("**Modules:**")
    
    # Module buttons
    if st.button("📤 1. Demand Analysis", use_container_width=True):
        st.session_state.guide_section = 'demand'
        
    if st.button("📦 2. Supply Analysis", use_container_width=True):
        st.session_state.guide_section = 'supply'
        
    if st.button("📊 3. GAP Analysis", use_container_width=True):
        st.session_state.guide_section = 'gap'
        
    if st.button("🧩 4. Allocation Plan", use_container_width=True):
        st.session_state.guide_section = 'allocation'
    
    st.markdown("---")
    
    # Quick links
    st.markdown("**Quick Links:**")
    if st.button("⚡ Quick Start", use_container_width=True):
        st.session_state.guide_section = 'quickstart'
        
    if st.button("❓ FAQs", use_container_width=True):
        st.session_state.guide_section = 'faqs'
    
    st.markdown("---")
    
    # Download section
    st.markdown("### 📥 Download Guide")
    st.caption("Get offline PDF version")
    
    # Create download button for PDF (placeholder)
    if st.button("📄 Download PDF Guide", use_container_width=True):
        st.info("PDF download will be available soon")

# === Section Functions ===

def show_overview_section():
    """Show overview section"""
    st.header("🏠 System Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to SCM Control Center! 👋
        
        SCM Control Center là hệ thống quản lý chuỗi cung ứng giúp bạn:
        - 📊 **Phân tích** nhu cầu (Demand) và nguồn cung (Supply)
        - 🔍 **Tìm ra** chênh lệch (GAP) giữa cung và cầu
        - 📋 **Lập kế hoạch** phân bổ hàng hóa (Allocation)
        - 📌 **Đề xuất** đơn hàng mua (PO Suggestions)
        """)
        
        st.info("""
        💡 **Quick Navigation Tips:**
        - Use sidebar để chuyển giữa các modules
        - Click vào các expanders để xem chi tiết
        - Các emoji 🔴🟡🟢 chỉ mức độ ưu tiên
        """)
        
    with col2:
        # Workflow diagram
        st.markdown("### 🔄 Basic Workflow")
        st.markdown("""
        ```
        1. Load Demand Data
              ↓
        2. Load Supply Data
              ↓
        3. Run GAP Analysis
              ↓
        4. Create Allocation
              ↓
        5. Generate Reports
        ```
        """)
    
    # Key features
    st.markdown("---")
    st.markdown("### ⭐ Key Features")
    
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        st.markdown("""
        **📤 Demand Analysis**
        - Order tracking
        - Forecast management
        - Customer analytics
        """)
        
    with feature_cols[1]:
        st.markdown("""
        **📦 Supply Analysis**
        - Inventory status
        - Pending orders
        - Expiry tracking
        """)
        
    with feature_cols[2]:
        st.markdown("""
        **📊 GAP Analysis**
        - Shortage detection
        - Surplus identification
        - Period planning
        """)
        
    with feature_cols[3]:
        st.markdown("""
        **🧩 Allocation Plan**
        - Smart distribution
        - Multiple methods
        - Credit control
        """)

def show_demand_section():
    """Show demand analysis guide"""
    st.header("📤 Demand Analysis Guide")
    
    # Quick intro
    st.info("**Mục đích:** Theo dõi và phân tích nhu cầu từ khách hàng thông qua Order Confirmation (OC) và Forecast")
    
    # Data sources
    with st.expander("📋 Data Sources", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔵 Order Confirmation (OC)**
            - Đơn hàng đã xác nhận
            - Chưa giao hoàn toàn
            - Có ETD cụ thể
            """)
            
        with col2:
            st.markdown("""
            **🟣 Forecast**
            - Dự báo từ khách hàng
            - Chưa chuyển thành OC
            - ETD dự kiến
            """)
    
    # Step by step guide
    st.markdown("### 📝 How to Use")
    
    steps = [
        ("1️⃣ Choose Data Source", 
         "- **OC Only:** Chỉ xem đơn đã confirm\n- **Forecast Only:** Chỉ xem dự báo\n- **Both:** Xem cả hai (⚠️ careful với double counting!)"),
        
        ("2️⃣ Select Date Mode",
         "- **Original:** Ngày gốc từ system\n- **Adjusted:** Ngày đã điều chỉnh theo rules"),
        
        ("3️⃣ Apply Filters",
         "- **Smart Filters:** Interactive, tự động update\n- **Standard:** Traditional độc lập"),
        
        ("4️⃣ View Results",
         "- **Summary:** Overview metrics\n- **Details:** Chi tiết từng dòng\n- **Grouped:** Theo Daily/Weekly/Monthly")
    ]
    
    for title, content in steps:
        with st.expander(title):
            st.markdown(content)
    
    # Common issues
    with st.expander("⚠️ Common Issues & Solutions"):
        st.markdown("""
        **🔴 Past ETD Orders**
        - Đơn hàng quá hạn
        - **Action:** Xử lý gấp hoặc update ETD
        
        **❌ Missing ETD**
        - Thiếu ngày giao hàng
        - **Action:** Liên hệ customer để bổ sung
        
        **⚠️ Converted Forecast**
        - Forecast đã chuyển thành OC
        - **Action:** Uncheck "Include Converted" để tránh tính 2 lần
        """)
    
    # Pro tips
    st.success("""
    💡 **Pro Tips:**
    - Export Grouped View để làm báo cáo tuần/tháng
    - Ưu tiên xử lý: Past ETD → Missing ETD → Future orders
    - Regular check converted forecasts để avoid duplication
    """)

def show_supply_section():
    """Show supply analysis guide"""
    st.header("📦 Supply Analysis Guide")
    
    st.info("**Mục đích:** Theo dõi toàn bộ nguồn hàng có thể đáp ứng nhu cầu từ 4 nguồn khác nhau")
    
    # Supply sources
    st.markdown("### 📋 Supply Sources")
    
    supply_tabs = st.tabs(["📦 Inventory", "📥 Pending CAN", "📄 Pending PO", "🚚 WH Transfer"])
    
    with supply_tabs[0]:
        st.markdown("""
        **Inventory - Tồn kho hiện tại**
        - ✅ Available ngay (TODAY)
        - 📍 Có vị trí kho cụ thể
        - 📅 Track expiry date
        - 💰 Có giá trị USD
        
        **Key Info:**
        - Batch number & Expiry
        - Zone-Rack-Bin location
        - Days in warehouse
        - Owner matching check
        """)
        
    with supply_tabs[1]:
        st.markdown("""
        **Pending CAN - Hàng đã đến chờ nhập**
        - 📦 Đã về kho nhưng chưa stock-in
        - 🔗 Linked với PO number
        - ⏱️ Track days since arrival
        
        **Warnings:**
        - 🔴 > 7 days: Chậm nhập kho
        - Check với warehouse team
        """)
        
    with supply_tabs[2]:
        st.markdown("""
        **Pending PO - Đơn đặt hàng**
        - 📄 Hàng đã đặt chưa về
        - 📅 Có ETA dự kiến
        - 💼 Track vendor & terms
        
        **Important:**
        - MOQ & SPQ requirements
        - Payment terms
        - Lead time tracking
        """)
        
    with supply_tabs[3]:
        st.markdown("""
        **WH Transfer - Chuyển kho**
        - 🚚 Hàng đang di chuyển
        - 🏭 From → To warehouse
        - ⏱️ Transfer duration
        
        **Alerts:**
        - 🔴 > 3 days: Check delay reason
        - Update transfer status
        """)
    
    # Expiry management
    with st.expander("💀 Expiry Management"):
        st.markdown("""
        **Color Coding:**
        - 💀 **Expired:** Hàng hết hạn → Xử lý ngay
        - 🔴 **≤7 days:** Sắp hết hạn → Ưu tiên xuất
        - 🟡 **≤30 days:** Cần theo dõi
        - 🟢 **>30 days:** An toàn
        
        **Settings:**
        - "Exclude Expired" = ON cho planning thực tế
        - OFF để kiểm tra toàn bộ inventory
        """)
    
    # Priority guide
    st.success("""
    💡 **Supply Priority Order:**
    1. Inventory (sẵn có)
    2. Pending CAN (sắp nhập)  
    3. WH Transfer (đang chuyển)
    4. Pending PO (chờ về)
    """)

def show_gap_section():
    """Show GAP analysis guide"""
    st.header("📊 GAP Analysis Guide")
    
    st.info("**Mục đích:** So sánh Supply vs Demand để tìm shortage/surplus và đưa ra action plans")
    
    # Key concepts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📐 Calculation Options")
        st.markdown("""
        **Ảnh hưởng cách tính:**
        - **Period Type:** Daily/Weekly/Monthly
        - **Exclude Missing:** Bỏ records không có date
        - **Track Backlog:** 
          - ON: Shortage carry forward
          - OFF: Each period độc lập
        """)
        
    with col2:
        st.markdown("### 🎯 Display Filters")
        st.markdown("""
        **Chỉ ảnh hưởng hiển thị:**
        - **Matched:** Có cả D&S
        - **Demand Only:** Cần PO
        - **Supply Only:** Dead stock risk
        - **Period filters:** Past/Future/Critical
        """)
    
    # Workflow
    st.markdown("### 🔄 GAP Analysis Workflow")
    
    workflow_steps = [
        ("1️⃣ Select Data Sources",
         "- Demand: OC/Forecast/Both\n- Supply: All 4 sources\n- Customer filter if needed"),
        
        ("2️⃣ Configure Settings",
         "- Date modes (Original/Adjusted)\n- Period type & Backlog tracking\n- Filters (Entity/Product/Brand)"),
        
        ("3️⃣ Run Analysis",
         "- Click 'Run GAP Analysis'\n- Wait for calculation\n- Check completion message"),
        
        ("4️⃣ Review Results",
         "- Summary tab: Key metrics\n- Details tab: Product breakdown\n- Pivot view: Period summary"),
        
        ("5️⃣ Take Actions",
         "- 🧩 Create Allocation (if products available)\n- 📌 Generate PO (if shortage)\n- 📤 Export reports")
    ]
    
    for step, details in workflow_steps:
        with st.expander(step):
            st.markdown(details)
    
    # Backlog logic explanation
    with st.expander("📚 Understanding Backlog Logic"):
        tab1, tab2 = st.tabs(["Backlog OFF", "Backlog ON"])
        
        with tab1:
            st.markdown("""
            **Simple Mode - Each period independent:**
            ```
            Period 1: D=100, S=80 → GAP=-20 ❌
            Period 2: D=50, S=100 → GAP=+50 ✅  
            Period 3: D=60, S=20 → GAP=-40 ❌
            ```
            - Shortage không carry forward
            - Surplus carry forward bình thường
            """)
            
        with tab2:
            st.markdown("""
            **Enhanced Mode - Shortage accumulates:**
            ```
            Period 1: D=100, S=80 → GAP=-20 ❌ (Backlog=20)
            Period 2: D=50+20, S=100 → GAP=+30 ✅
            Period 3: D=40, S=20+30 → GAP=+10 ✅
            ```
            - Shortage chuyển sang period sau
            - More realistic view
            """)
    
    # Result interpretation
    st.success("""
    💡 **Action Priority:**
    1. **Demand Only** → Create PO immediately
    2. **Critical Shortage** (FR<50%) → Priority allocation
    3. **Past Period** shortage → Urgent handling
    4. **Future shortage** → Plan ahead
    """)

def show_allocation_section():
    """Show allocation plan guide"""
    st.header("🧩 Allocation Plan Guide")
    
    st.info("**Mục đích:** Phân bổ nguồn hàng cho đơn hàng dựa trên GAP Analysis results")
    
    # Allocation types
    st.markdown("### 📋 Allocation Types")
    
    type_col1, type_col2 = st.columns(2)
    
    with type_col1:
        st.markdown("""
        **🌊 SOFT Allocation (90% cases)**
        - Chỉ phân bổ số lượng
        - System tự chọn nguồn khi giao
        - Flexible điều chỉnh
        - ✅ **Recommended mặc định**
        """)
        
    with type_col2:
        st.markdown("""
        **🔒 HARD Allocation (10% special)**
        - Lock specific batch/lot
        - Cho yêu cầu xuất xứ/quality
        - Không thể đổi sau approve
        - ⚠️ **Chỉ khi thực sự cần**
        """)
    
    # Allocation methods
    st.markdown("### 🎯 Allocation Methods")
    
    method_tabs = st.tabs(["📅 FCFS", "⭐ Priority", "⚖️ Pro Rata", "✋ Manual"])
    
    with method_tabs[0]:
        st.markdown("""
        **First Come First Served**
        - Ưu tiên ETD sớm nhất
        - Fair & transparent
        - Good for time-sensitive
        
        Example: ETD Jan 1 → Jan 5 → Jan 10
        """)
        
    with method_tabs[1]:
        st.markdown("""
        **Priority Based**
        - Set score 1-10 per customer
        - VIP gets preference
        - Strategic accounts
        
        Example: Priority 9 → 7 → 5
        """)
        
    with method_tabs[2]:
        st.markdown("""
        **Pro Rata Distribution**
        - Proportional to demand
        - Equal treatment
        - Can set minimum %
        
        Example: All get 50% if shortage
        """)
        
    with method_tabs[3]:
        st.markdown("""
        **Manual Adjustment**
        - Full control
        - Start with pro-rata
        - Adjust as needed
        
        Best for complex scenarios
        """)
    
    # Creation workflow
    st.markdown("### 📝 Create Allocation - Step by Step")
    
    steps_data = [
        ("Step 1: Select Products", 
         "- Use filters to find products\n- Can allocate ALL products\n- Not limited to shortage only"),
        
        ("Step 2: Choose Method & Type",
         "- Pick allocation method\n- Select SOFT (default) or HARD\n- Consider business needs"),
        
        ("Step 3: Set Parameters",
         "- Configure method settings\n- Credit limit check ON/OFF\n- Allow partial YES/NO"),
        
        ("Step 4: Preview & Adjust",
         "- Review auto-calculation\n- Adjust quantities manually\n- Check warnings"),
        
        ("Step 5: [HARD only] Map Supply",
         "- Select specific batches\n- Link to customer orders\n- Validate availability"),
        
        ("Step 6: Confirm & Save",
         "- Final validation\n- Save as Draft or Approve\n- Export preview if needed")
    ]
    
    for i, (title, content) in enumerate(steps_data, 1):
        with st.expander(f"{title}"):
            st.markdown(content)
            if i == 4:
                st.warning("⚠️ Lines with 0 quantity will be auto-excluded")
    
    # Best practices
    with st.expander("💡 Best Practices & Tips"):
        st.markdown("""
        **Do's:**
        - ✅ Use SOFT allocation by default
        - ✅ Check credit limits warning
        - ✅ Review excluded lines before save
        - ✅ Document reasons in notes
        
        **Don'ts:**
        - ❌ Use HARD without specific need
        - ❌ Ignore validation warnings
        - ❌ Forget to check zero allocations
        - ❌ Edit after approval (can only cancel)
        
        **Remember:**
        - Draft → Allocated is ONE-WAY
        - Snapshot saves GAP context
        - Partial cancel keeps plan active
        - Full cancel releases all supply
        """)

def show_quickstart_section():
    """Show quick start guide"""
    st.header("⚡ Quick Start Guide")
    
    st.success("Follow these steps to get started quickly!")
    
    # Quick workflow
    steps = [
        {
            "step": "1. Load Demand Data",
            "action": "Go to Demand Analysis → Load OC + Forecast",
            "time": "2 mins",
            "icon": "📤"
        },
        {
            "step": "2. Load Supply Data", 
            "action": "Go to Supply Analysis → Select All Sources",
            "time": "2 mins",
            "icon": "📦"
        },
        {
            "step": "3. Run GAP Analysis",
            "action": "Go to GAP Analysis → Configure → Run",
            "time": "3 mins",
            "icon": "📊"
        },
        {
            "step": "4. Review Results",
            "action": "Check shortage/surplus → Identify actions",
            "time": "5 mins",
            "icon": "🔍"
        },
        {
            "step": "5. Create Allocation",
            "action": "If products available → Create allocation plan",
            "time": "5 mins",
            "icon": "🧩"
        },
        {
            "step": "6. Export Reports",
            "action": "Export results for team/management",
            "time": "2 mins",
            "icon": "📤"
        }
    ]
    
    # Display steps
    total_time = 0
    for step_info in steps:
        col1, col2, col3 = st.columns([0.5, 3, 1])
        
        with col1:
            st.markdown(f"### {step_info['icon']}")
            
        with col2:
            st.markdown(f"**{step_info['step']}**")
            st.caption(step_info['action'])
            
        with col3:
            st.metric("Time", step_info['time'])
            
        total_time += int(step_info['time'].split()[0])
        st.markdown("---")
    
    # Summary
    st.info(f"⏱️ **Total estimated time: {total_time} minutes** for complete workflow")
    
    # Quick tips
    st.markdown("### 🎯 Quick Tips")
    
    tip_cols = st.columns(3)
    
    with tip_cols[0]:
        st.markdown("""
        **🔴 Urgent Actions**
        - Past ETD orders
        - Expired inventory
        - Credit limit breach
        """)
        
    with tip_cols[1]:
        st.markdown("""
        **🟡 Important Checks**
        - Converted forecasts
        - Expiring products
        - Pending allocations
        """)
        
    with tip_cols[2]:
        st.markdown("""
        **🟢 Regular Tasks**
        - Weekly GAP run
        - Allocation review
        - Report generation
        """)

def show_faqs_section():
    """Show FAQs section"""
    st.header("❓ Frequently Asked Questions")
    
    # Categorize FAQs
    faq_categories = {
        "General": [
            ("What is SCM Control Center?",
             "A supply chain management system that helps analyze demand vs supply, identify gaps, and create allocation plans."),
            
            ("Do I need to run all modules?",
             "No, but recommended workflow is: Demand → Supply → GAP → Allocation for best results."),
            
            ("How often should I run GAP Analysis?",
             "Weekly for regular planning, daily for critical products or urgent situations.")
        ],
        
        "Demand Analysis": [
            ("What's the difference between OC and Forecast?",
             "OC = Confirmed orders, Forecast = Predictions. OC has higher priority."),
            
            ("How to avoid double counting?",
             "When viewing Both, uncheck 'Include Converted Forecasts' or filter Forecast status = 'No'."),
            
            ("What if ETD is missing?",
             "Contact customer to get ETD. System flags these as high priority issues.")
        ],
        
        "Supply Analysis": [
            ("Which supply source is most reliable?",
             "Priority order: Inventory > Pending CAN > WH Transfer > Pending PO"),
            
            ("How to handle expired products?",
             "Set 'Exclude Expired' = ON for planning. Review expired separately for disposal."),
            
            ("Why is my CAN not showing in inventory?",
             "Check if stock-in completed. CAN > 7 days needs warehouse follow-up.")
        ],
        
        "GAP Analysis": [
            ("What's Track Backlog?",
             "ON = Shortage accumulates to next period (realistic). OFF = Each period independent (simple)."),
            
            ("When to use different period types?",
             "Daily = Short-term ops, Weekly = Regular planning, Monthly = Long-term strategy"),
            
            ("What does negative GAP mean?",
             "Shortage - demand exceeds supply. Positive = Surplus.")
        ],
        
        "Allocation Plan": [
            ("SOFT vs HARD allocation?",
             "SOFT (90%) = Flexible quantity only. HARD (10%) = Lock specific batch/lot."),
            
            ("Can I edit after approval?",
             "No, only cancel allowed. Plan carefully before approving."),
            
            ("What happens to zero allocations?",
             "Auto-excluded from plan. Review before saving."),
            
            ("How does credit limit work?",
             "System checks customer credit vs allocation value. Can override with reason.")
        ]
    }
    
    # Display FAQs by category
    for category, questions in faq_categories.items():
        with st.expander(f"📌 {category}", expanded=True):
            for question, answer in questions:
                st.markdown(f"**Q: {question}**")
                st.info(f"A: {answer}")
                st.markdown("")


# Main content area
if st.session_state.guide_section == 'overview':
    show_overview_section()
elif st.session_state.guide_section == 'demand':
    show_demand_section()
elif st.session_state.guide_section == 'supply':
    show_supply_section()
elif st.session_state.guide_section == 'gap':
    show_gap_section()
elif st.session_state.guide_section == 'allocation':
    show_allocation_section()
elif st.session_state.guide_section == 'quickstart':
    show_quickstart_section()
elif st.session_state.guide_section == 'faqs':
    show_faqs_section()


# Footer
st.markdown("---")
footer_cols = st.columns([2, 1, 1])

with footer_cols[0]:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')} | Version 1.0")
    
with footer_cols[1]:
    st.caption("Need help? Contact IT Support")
    
with footer_cols[2]:
    if st.button("🔝 Back to Top"):
        st.markdown('<a href="#user-guide-scm-control-center"></a>', unsafe_allow_html=True)