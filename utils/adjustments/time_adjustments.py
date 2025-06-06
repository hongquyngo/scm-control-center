import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid
import json


class TimeAdjustmentManager:
    """Manage time offset adjustments for data sources"""
    
    @staticmethod
    def render_time_adjustments():
        """Render time adjustment UI"""
        
        # Initialize rules in session state if not exists
        if 'time_adjustment_rules' not in st.session_state:
            st.session_state.time_adjustment_rules = []
        
        # Configuration Management Section
        st.markdown("#### ⚙️ Configuration Management")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("💾 Save Config", use_container_width=True):
                if st.session_state.time_adjustment_rules:
                    TimeAdjustmentManager._save_configuration()
                else:
                    st.warning("No rules to save. Please add at least one rule first.")

        with col2:
            # Use a button to trigger file upload instead
            if st.button("📂 Load Config", use_container_width=True):
                st.session_state.show_file_uploader = True
            
            # Show file uploader in a modal-like way
            if st.session_state.get('show_file_uploader', False):
                with st.container():
                    st.markdown("---")
                    uploaded_file = st.file_uploader(
                        "Select JSON configuration file",
                        type=['json'],
                        key="rule_upload_file"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if uploaded_file is not None:
                            if st.button("✅ Import", type="primary", use_container_width=True):
                                TimeAdjustmentManager._load_configuration(uploaded_file)
                                st.session_state.show_file_uploader = False
                                st.rerun()
                    
                    with col_b:
                        if st.button("❌ Cancel", use_container_width=True):
                            st.session_state.show_file_uploader = False
                            st.rerun()
                    st.markdown("---")

        with col3:
            if st.button("⚠️ Analyze Conflicts", use_container_width=True):
                if st.session_state.time_adjustment_rules:
                    st.session_state.show_conflict_analysis = True
                else:
                    st.info("No rules to analyze")

        with col4:
            if st.button("🗑️ Clear All", use_container_width=True):
                if st.session_state.time_adjustment_rules:
                    if st.checkbox("Confirm clear all rules", key="confirm_clear"):
                        st.session_state.time_adjustment_rules = []
                        st.success("✅ All rules cleared")
                        st.rerun()
                else:
                    st.info("No rules to clear")
        
        # Show conflict analysis if requested
        if st.session_state.get('show_conflict_analysis', False):
            st.markdown("---")
            try:
                from utils.adjustments.preview_manager import PreviewManager
                from utils.adjustments.conflict_manager import TimeAdjustmentConflictManager
                preview_manager = PreviewManager()
                
                TimeAdjustmentConflictManager.show_conflict_analysis_ui(
                    st.session_state.time_adjustment_rules,
                    preview_manager
                )
                
                if st.button("❌ Close Conflict Analysis", key="close_conflict_analysis"):
                    st.session_state.show_conflict_analysis = False
                    st.rerun()
                    
            except ImportError:
                st.error("Conflict analysis module not available")
                st.session_state.show_conflict_analysis = False
        
        st.markdown("---")
        
        # Display existing rules
        TimeAdjustmentManager._display_existing_rules()
        
        # Add new rule section
        st.markdown("---")
        st.markdown("##### ➕ Add New Rule")
        
        # Help text
        with st.expander("ℹ️ How to create a rule", expanded=False):
            st.markdown("""
            **1. Adjustment Type**:
            - **Relative (Days)**: Move dates forward or backward by a number of days
            - **Absolute (Date)**: Change all matching dates to a specific date
            
            **2. Offset Days** (for Relative adjustment): 
            - **Negative values** (-): Move dates to the **past** (earlier)
            - **Positive values** (+): Move dates to the **future** (later)
            - **Range**: -365 to +365 days
            - Example: `-7` = 1 week earlier, `+14` = 2 weeks later
            - **Note**: Offset cannot be 0
            
            **3. Priority**: 
            - Controls which rule wins when multiple rules match the same record
            - **Higher number = Higher priority** (100 is highest, 1 is lowest)
            - Example: Rule with priority 80 will override rule with priority 20
            
            **4. Filters**: 
            - Leave empty to apply to ALL records
            - Select specific values to target specific records
            - Multiple filters work with AND logic
            """)
        
        # Import preview manager
        preview_manager = None
        use_dynamic_filters = False
        
        try:
            from utils.adjustments.preview_manager import PreviewManager
            preview_manager = PreviewManager()
            use_dynamic_filters = True
        except ImportError:
            st.warning("Preview manager not available. Using simple filters.")
        
        # Section 1: Data Source & Adjustment Type
        st.markdown("**📊 Data Source & Adjustment Type**")
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            data_source = st.selectbox(
                "Data Source",
                ["OC", "Forecast", "Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"],
                help="Select which data source this rule applies to",
                key="new_rule_data_source"
            )
        
        with col2:
            adjustment_type = st.radio(
                "Adjustment Type",
                ["Relative (Days)", "Absolute (Date)"],
                key="adjustment_type"
            )
        
        if adjustment_type == "Relative (Days)":
            with col3:
                offset_days = st.number_input(
                    "Offset Days",
                    min_value=-365,
                    max_value=365,
                    value=0,
                    help="Adjust dates by this many days. Range: -365 to +365.",
                    key="new_rule_offset"
                )
            absolute_date = None
        else:
            with col3:
                absolute_date = st.date_input(
                    "Target Date",
                    value=datetime.now().date(),
                    min_value=datetime.now().date() - timedelta(days=365),
                    max_value=datetime.now().date() + timedelta(days=365),
                    help="All matching records will have their dates changed to this specific date",
                    key="new_rule_absolute_date"
                )
            offset_days = None
        
        with col4:
            priority = st.number_input(
                "Rule Priority",
                min_value=1,
                max_value=100,
                value=50,
                help="Priority when multiple rules match same record. Higher number = Higher priority",
                key="new_rule_priority"
            )
        
        with col5:
            st.markdown("**Impact Preview**")
            if adjustment_type == "Relative (Days)":
                if offset_days and offset_days > 0:
                    st.info(f"📅 +{offset_days} days later")
                elif offset_days and offset_days < 0:
                    st.warning(f"📅 {offset_days} days earlier")
                else:
                    st.caption("📅 No change")
            else:
                st.success(f"📅 → {absolute_date.strftime('%Y-%m-%d')}")
        
        # Clear filters when data source changes
        if "prev_data_source" not in st.session_state:
            st.session_state.prev_data_source = data_source
        
        if st.session_state.prev_data_source != data_source:
            # Data source changed, clear all filter selections
            st.session_state.temp_entity_selection = []
            st.session_state.temp_customer_selection = []
            st.session_state.temp_product_selection = []
            st.session_state.temp_number_selection = []
            st.session_state.temp_brand_selection = []
            st.session_state.prev_data_source = data_source
            st.rerun()
        
        # Section 2: Filter Conditions
        st.markdown("---")
        st.markdown("**🎯 Filter Conditions**")
        st.caption("Leave empty to apply to all records")
        
        if use_dynamic_filters and preview_manager:
            # Initialize session state for filter selections to enable interactivity
            if "temp_entity_selection" not in st.session_state:
                st.session_state.temp_entity_selection = []
            if "temp_customer_selection" not in st.session_state:
                st.session_state.temp_customer_selection = []
            if "temp_product_selection" not in st.session_state:
                st.session_state.temp_product_selection = []
            if "temp_number_selection" not in st.session_state:
                st.session_state.temp_number_selection = []
            if "temp_brand_selection" not in st.session_state:
                st.session_state.temp_brand_selection = []
            
            # Get current filter options based on selections (interactive)
            try:
                with st.spinner("Loading filter options..."):
                    filter_options = preview_manager.get_interactive_filter_options(
                        data_source,
                        selected_entities=st.session_state.temp_entity_selection,
                        selected_customers=st.session_state.temp_customer_selection,
                        selected_products=st.session_state.temp_product_selection,
                        selected_numbers=st.session_state.temp_number_selection,
                        selected_brands=st.session_state.temp_brand_selection
                    )
            except Exception as e:
                st.error(f"Error loading filter options: {str(e)}")
                filter_options = {
                    'entities': [],
                    'customers': [],
                    'products': [],
                    'numbers': [],
                    'brands': []
                }
            
            # Create filter columns based on data source
            if data_source in ["OC", "Forecast"]:
                col1, col2, col3 = st.columns(3)
                col4, col5 = st.columns(2)
            else:
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
            
            # Entity filter
            with col1:
                entity_selection = st.multiselect(
                    "Legal Entity",
                    options=filter_options['entities'],
                    default=st.session_state.temp_entity_selection,
                    key="new_rule_entity",
                    help="Select specific entities or leave empty for all"
                )
                
                # Update session state if changed
                if sorted(entity_selection) != sorted(st.session_state.temp_entity_selection):
                    st.session_state.temp_entity_selection = entity_selection
                    st.rerun()
            
            # Brand filter
            with col2:
                brand_selection = st.multiselect(
                    "Brand",
                    options=filter_options['brands'],
                    default=st.session_state.temp_brand_selection,
                    key="new_rule_brand",
                    help="Select specific brands or leave empty for all"
                )
                
                # Update session state if changed
                if sorted(brand_selection) != sorted(st.session_state.temp_brand_selection):
                    st.session_state.temp_brand_selection = brand_selection
                    st.rerun()
            
            # Customer filter (only for demand sources)
            customer_selection = []
            if data_source in ["OC", "Forecast"]:
                with col3:
                    customer_selection = st.multiselect(
                        "Customer",
                        options=filter_options['customers'],
                        default=st.session_state.temp_customer_selection,
                        key="new_rule_customer",
                        help="Select specific customers or leave empty for all"
                    )
                    
                    # Update session state if changed
                    if sorted(customer_selection) != sorted(st.session_state.temp_customer_selection):
                        st.session_state.temp_customer_selection = customer_selection
                        st.rerun()
                
                product_col = col4
            else:
                product_col = col3
            
            # Product filter
            with product_col:
                product_selection = st.multiselect(
                    "Product",
                    options=filter_options['products'],
                    default=st.session_state.temp_product_selection,
                    key="new_rule_product", 
                    help="Search and select products or leave empty for all"
                )
                
                # Update session state if changed
                if sorted(product_selection) != sorted(st.session_state.temp_product_selection):
                    st.session_state.temp_product_selection = product_selection
                    st.rerun()
            
            # Number filter
            number_col = col5 if data_source in ["OC", "Forecast"] else col4
            with number_col:
                number_label = {
                    "OC": "OC Number",
                    "Forecast": "Forecast Number",
                    "Inventory": "Inventory ID", 
                    "Pending CAN": "CAN Number",
                    "Pending PO": "PO Number",
                    "Pending WH Transfer": "Transfer ID"
                }.get(data_source, "Reference Number")
                
                number_selection = st.multiselect(
                    number_label,
                    options=filter_options['numbers'],
                    default=st.session_state.temp_number_selection,
                    key="new_rule_number",
                    help=f"Select specific {number_label.lower()}s or leave empty for all"
                )
                
                # Update session state if changed
                if sorted(number_selection) != sorted(st.session_state.temp_number_selection):
                    st.session_state.temp_number_selection = number_selection
                    st.rerun()
            
            # Set final filter values
            entity_filter = entity_selection if entity_selection else ["All"]
            customer_filter = customer_selection if customer_selection else ["All"] 
            product_filter = product_selection if product_selection else ["All"]
            number_filter = number_selection if number_selection else ["All"]
            brand_filter = brand_selection if brand_selection else ["All"]
            
            # Show data availability indicator
            if entity_selection or customer_selection or product_selection or number_selection or brand_selection:
                total_available = (len(filter_options['entities']) + 
                                 len(filter_options.get('customers', [])) + 
                                 len(filter_options['products']) + 
                                 len(filter_options['numbers']) +
                                 len(filter_options['brands']))
                
                # Status indicator
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                with col1:
                    if total_available > 0:
                        st.success(f"✅ Valid filter combination - {total_available} options available across all filters")
                    else:
                        st.warning("⚠️ No data matches current filter combination")
                with col2:
                    st.metric("Total Options", total_available, label_visibility="visible")
            
            # Section 3: Preview Section
            if preview_manager and (
                (adjustment_type == "Relative (Days)" and offset_days != 0) or 
                adjustment_type == "Absolute (Date)"
            ):
                st.markdown("---")
                st.markdown("### 👁️ Preview Rule Impact")
                
                if st.button("🔍 Show Preview", use_container_width=True, type="primary"):
                    # Create temporary rule for preview
                    temp_rule = {
                        'data_source': data_source,
                        'filters': {
                            'entity': entity_filter,
                            'customer': customer_filter,
                            'product': product_filter,
                            'number': number_filter,
                            'brand': brand_filter
                        },
                        'adjustment_type': adjustment_type,
                        'offset_days': offset_days,
                        'absolute_date': absolute_date.isoformat() if absolute_date else None
                    }
                    
                    # Show styled preview
                    preview_manager.show_rule_preview_ui(temp_rule)
            
            elif adjustment_type == "Relative (Days)" and offset_days == 0:
                st.info("💡 Set an offset (non-zero) to preview the impact.")
        
        else:
            # Fallback to text input
            entity_filter_text = st.text_input(
                "Entity Filter",
                placeholder="Leave empty for all, or enter entity names (comma separated)",
                help="Leave empty to apply to all entities"
            )
            entity_filter = [e.strip() for e in entity_filter_text.split(',')] if entity_filter_text else ["All"]
            
            if data_source in ["OC", "Forecast"]:
                customer_filter_text = st.text_input(
                    "Customer Filter", 
                    placeholder="Leave empty for all, or enter customer names (comma separated)",
                    help="Leave empty to apply to all customers"
                )
                customer_filter = [c.strip() for c in customer_filter_text.split(',')] if customer_filter_text else ["All"]
            else:
                customer_filter = ["All"]
            
            product_filter = ["All"]
            number_filter = ["All"]
            brand_filter = ["All"]
        
        # Section 4: Action Buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("➕ Add Rule", type="primary", use_container_width=True):
                # Validate
                if adjustment_type == "Relative (Days)" and offset_days == 0:
                    st.error("❌ Offset days cannot be 0. Please set a non-zero offset or remove this rule.")
                else:
                    # Create new rule
                    new_rule = {
                        'id': str(uuid.uuid4()),
                        'data_source': data_source,
                        'filters': {
                            'entity': entity_filter,
                            'customer': customer_filter,
                            'product': product_filter,
                            'number': number_filter,
                            'brand': brand_filter
                        },
                        'adjustment_type': adjustment_type,
                        'offset_days': offset_days if adjustment_type == "Relative (Days)" else None,
                        'absolute_date': absolute_date.isoformat() if adjustment_type == "Absolute (Date)" else None,
                        'priority': priority
                    }
                    
                    st.session_state.time_adjustment_rules.append(new_rule)
                    
                    # Clear temporary filter selections
                    st.session_state.temp_entity_selection = []
                    st.session_state.temp_customer_selection = []
                    st.session_state.temp_product_selection = []
                    st.session_state.temp_number_selection = []
                    st.session_state.temp_brand_selection = []
                    
                    if adjustment_type == "Relative (Days)":
                        st.success(f"✅ Rule added: {data_source} {offset_days:+d} days")
                    else:
                        st.success(f"✅ Rule added: {data_source} → {absolute_date.strftime('%Y-%m-%d')}")
                    st.rerun()

        with col2:
            if st.button("💾 Apply to Session", use_container_width=True):
                # Save rules to session state
                if 'business_settings' not in st.session_state:
                    st.session_state.business_settings = {}
                st.session_state.business_settings['time_adjustment_mode'] = 'advanced'
                st.session_state.business_settings['time_adjustment_rules'] = st.session_state.time_adjustment_rules
                st.success("✅ Rules applied to current session!")

        with col3:
            if st.button("🔄 Reset Form", use_container_width=True):
                # Clear temporary selections
                st.session_state.temp_entity_selection = []
                st.session_state.temp_customer_selection = []
                st.session_state.temp_product_selection = []
                st.session_state.temp_number_selection = []
                st.session_state.temp_brand_selection = []
                if "new_rule_offset" in st.session_state:
                    del st.session_state["new_rule_offset"]
                st.rerun()
    
    @staticmethod
    def _display_existing_rules():
        """Display existing rules with management options"""
        if st.session_state.time_adjustment_rules:
            st.markdown("#### 📋 Active Rules")
            st.caption(f"Total: {len(st.session_state.time_adjustment_rules)} rules configured")
            
            # Import conflict manager
            try:
                from utils.adjustments.conflict_manager import TimeAdjustmentConflictManager
                
                # Check for conflicts and show warnings
                TimeAdjustmentConflictManager.show_conflict_warnings_full(st.session_state.time_adjustment_rules)
                
            except ImportError:
                st.warning("Conflict manager not available")
            
            for idx, rule in enumerate(st.session_state.time_adjustment_rules):
                # Format header with clear information
                adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
                if adjustment_type == "Relative (Days)":
                    offset_display = f"{rule['offset_days']:+d} days" if rule.get('offset_days') is not None else "No offset"
                else:
                    absolute_date = rule.get('absolute_date', 'Unknown')
                    offset_display = f"→ {absolute_date}"
                
                priority_display = f"Priority: {rule.get('priority', idx + 1)}"
                header = f"Rule {idx + 1}: {rule['data_source']} | {offset_display} | {priority_display}"
                
                with st.expander(header, expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Rule details with better formatting
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.markdown(f"**Data Source:** {rule['data_source']}")
                            
                            adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
                            st.markdown(f"**Type:** {adjustment_type}")
                            
                            if adjustment_type == "Relative (Days)":
                                offset_days = rule.get('offset_days', 0)
                                st.markdown(f"**Offset:** {offset_days:+d} days")
                                if offset_days > 0:
                                    st.caption("→ Dates move to future")
                                elif offset_days < 0:
                                    st.caption("← Dates move to past")
                            else:
                                absolute_date = rule.get('absolute_date', 'Unknown')
                                st.markdown(f"**Target Date:** {absolute_date}")
                                st.caption("→ All dates change to this date")
                        
                        with col_detail2:
                            priority_val = rule.get('priority', idx + 1)
                            st.markdown(f"**Priority:** {priority_val}")
                            
                            # Priority visual indicator
                            if priority_val >= 80:
                                st.caption("🔴 Very High Priority")
                            elif priority_val >= 60:
                                st.caption("🟠 High Priority")
                            elif priority_val >= 40:
                                st.caption("🟡 Normal Priority")
                            elif priority_val >= 20:
                                st.caption("🔵 Low Priority")
                            else:
                                st.caption("⚪ Very Low Priority")
                        
                        # Display filters
                        st.markdown("**Filters Applied:**")
                        for filter_type, filter_value in rule['filters'].items():
                            if filter_value != ['All']:
                                display_name = {
                                    'entity': 'Entity',
                                    'customer': 'Customer', 
                                    'product': 'Product',
                                    'number': f"{rule['data_source']} Number",
                                    'brand': 'Brand'
                                }.get(filter_type, filter_type.title())
                                st.markdown(f"- {display_name}: {', '.join(filter_value)}")
                        
                        # Preview button for existing rules
                        if st.button(f"👁️ Preview Rule {idx + 1}", key=f"preview_rule_{rule['id']}"):
                            st.markdown("---")
                            try:
                                from utils.adjustments.preview_manager import PreviewManager
                                preview_manager = PreviewManager()
                                preview_manager.show_rule_preview_ui(rule)
                            except ImportError:
                                st.error("Preview manager not available")
                    
                    with col2:
                        st.markdown("**Actions**")
                        if st.button("🗑️ Delete", key=f"delete_{rule['id']}"):
                            st.session_state.time_adjustment_rules.remove(rule)
                            st.rerun()
                        
                        # Move up/down buttons
                        col_up, col_down = st.columns(2)
                        with col_up:
                            if idx > 0 and st.button("⬆️", key=f"up_{rule['id']}"):
                                # Swap with previous rule
                                st.session_state.time_adjustment_rules[idx], st.session_state.time_adjustment_rules[idx-1] = \
                                    st.session_state.time_adjustment_rules[idx-1], st.session_state.time_adjustment_rules[idx]
                                st.rerun()
                        
                        with col_down:
                            if idx < len(st.session_state.time_adjustment_rules) - 1 and st.button("⬇️", key=f"down_{rule['id']}"):
                                # Swap with next rule
                                st.session_state.time_adjustment_rules[idx], st.session_state.time_adjustment_rules[idx+1] = \
                                    st.session_state.time_adjustment_rules[idx+1], st.session_state.time_adjustment_rules[idx]
                                st.rerun()
        else:
            st.info("📝 No time adjustment rules configured yet. Add your first rule below.")
    
    @staticmethod
    def _save_configuration():
        """Save complete configuration with rules and context"""
        try:
            # Get current business settings
            business_settings = st.session_state.get('business_settings', {})
            
            # Prepare comprehensive configuration data
            config_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "description": "Time Adjustment Rules Configuration",
                "rules_count": len(st.session_state.time_adjustment_rules),
                "rules": st.session_state.time_adjustment_rules,
                "metadata": {
                    "session_id": st.session_state.get('session_id', 'unknown'),
                    "adjustment_mode": business_settings.get('time_adjustment_mode', 'advanced'),
                    "last_updated": st.session_state.get('settings_last_updated', datetime.now().isoformat()),
                    "exported_by": "Time Adjustment Manager v1.0"
                }
            }
            
            # Convert to JSON string with proper formatting
            json_str = json.dumps(config_data, indent=2, ensure_ascii=False)
            
            # Create download button
            st.download_button(
                label="📥 Download Configuration",
                data=json_str,
                file_name=f"time_adjustment_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_config"
            )
            
            # Show success message
            st.success(f"✅ Configuration ready to download ({len(st.session_state.time_adjustment_rules)} rules)")
            
        except Exception as e:
            st.error(f"❌ Error saving configuration: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)
    
    @staticmethod
    def _load_configuration(uploaded_file):
        """Load configuration from uploaded JSON file with enhanced validation"""
        try:
            # Read and parse JSON file
            content = uploaded_file.read()
            data = json.loads(content)
            
            # Support both old and new format
            if isinstance(data, list):
                # Old format: just a list of rules
                rules = data
                file_version = "0.0"
                metadata = {}
            elif isinstance(data, dict):
                # New format with metadata
                if 'rules' not in data:
                    st.error("❌ Invalid configuration file: Missing 'rules' section")
                    return
                rules = data.get('rules', [])
                file_version = data.get('version', '0.0')
                metadata = data.get('metadata', {})
            else:
                st.error("❌ Invalid configuration file format: Expected JSON object or array")
                return
            
            # Validate version compatibility
            current_version = "1.0"
            if file_version != current_version and file_version != "0.0":
                st.warning(f"⚠️ File version ({file_version}) differs from current version ({current_version}). Some features may not work correctly.")
            
            # Validate each rule structure
            required_fields = ['data_source', 'filters']
            valid_data_sources = ["OC", "Forecast", "Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"]
            validated_rules = []
            
            for idx, rule in enumerate(rules):
                # Add default id if missing
                if 'id' not in rule:
                    rule['id'] = str(uuid.uuid4())
                else:
                    # Regenerate ID to avoid conflicts
                    rule['id'] = str(uuid.uuid4())
                
                # Check required fields
                missing_fields = [field for field in required_fields if field not in rule]
                if missing_fields:
                    st.warning(f"⚠️ Rule {idx + 1}: Missing fields {missing_fields}, skipping...")
                    continue
                
                # Validate data source
                if rule['data_source'] not in valid_data_sources:
                    st.warning(f"⚠️ Rule {idx + 1}: Invalid data source '{rule['data_source']}', skipping...")
                    continue
                
                # Validate filters structure
                if not isinstance(rule.get('filters'), dict):
                    st.warning(f"⚠️ Rule {idx + 1}: Invalid filters format, skipping...")
                    continue
                
                # Add missing filter fields with defaults
                default_filters = {
                    'entity': ['All'],
                    'customer': ['All'],
                    'product': ['All'],
                    'number': ['All'],
                    'brand': ['All']
                }
                for filter_key, default_value in default_filters.items():
                    if filter_key not in rule['filters']:
                        rule['filters'][filter_key] = default_value
                
                # Ensure adjustment type exists
                if 'adjustment_type' not in rule:
                    # Try to infer from other fields
                    if 'absolute_date' in rule and rule['absolute_date']:
                        rule['adjustment_type'] = 'Absolute (Date)'
                    else:
                        rule['adjustment_type'] = 'Relative (Days)'
                
                # Ensure priority exists
                if 'priority' not in rule:
                    rule['priority'] = 50  # Default middle priority
                else:
                    # Ensure priority is within valid range
                    rule['priority'] = max(1, min(100, int(rule['priority'])))
                
                # Validate offset days for relative adjustments
                if rule['adjustment_type'] == 'Relative (Days)':
                    if 'offset_days' not in rule or rule['offset_days'] is None:
                        st.warning(f"⚠️ Rule {idx + 1}: Missing offset_days for relative adjustment, defaulting to 7")
                        rule['offset_days'] = 7
                    else:
                        # Ensure offset is not zero
                        if rule['offset_days'] == 0:
                            st.warning(f"⚠️ Rule {idx + 1}: Offset cannot be 0, changing to 1")
                            rule['offset_days'] = 1
                
                validated_rules.append(rule)
            
            if not validated_rules:
                st.error("❌ No valid rules found in the file")
                return
            
            # Load rules into session state
            st.session_state.time_adjustment_rules = validated_rules
            
            # Apply metadata if available
            if metadata:
                if metadata.get('adjustment_mode'):
                    if 'business_settings' not in st.session_state:
                        st.session_state.business_settings = {}
                    st.session_state.business_settings['time_adjustment_mode'] = metadata['adjustment_mode']
            
            # Show success message with details
            st.success(f"✅ Successfully loaded {len(validated_rules)} rules from '{uploaded_file.name}'")
            
            if len(validated_rules) < len(rules):
                st.info(f"ℹ️ {len(rules) - len(validated_rules)} rules were skipped due to validation errors")
            
            # Show configuration info
            info_parts = []
            if data.get('created_at'):
                info_parts.append(f"Created: {data['created_at']}")
            if data.get('description'):
                info_parts.append(f"Description: {data['description']}")
            if metadata.get('session_id'):
                info_parts.append(f"Session: {metadata['session_id']}")
            
            if info_parts:
                st.info(" | ".join(info_parts))
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON file: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error loading configuration: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)