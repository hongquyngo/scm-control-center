import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import uuid


class ConflictResolutionStrategy(Enum):
    PRIORITY_BASED = "priority_based"
    CUMULATIVE = "cumulative"
    FIRST_MATCH = "first_match"
    LAST_MATCH = "last_match"
    MOST_SPECIFIC = "most_specific"


@dataclass
class RuleMatch:
    rule_id: str
    rule_index: int
    offset_days: int
    specificity_score: int
    filters_matched: Dict[str, str]
    priority: int = 50  # Add priority field with default


class TimeAdjustmentConflictManager:
    """Manage conflicts between time adjustment rules"""
    
    def __init__(self, resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.PRIORITY_BASED):
        self.resolution_strategy = resolution_strategy
    
    def detect_conflicts(self, rules: List[Dict[str, Any]], data_df: pd.DataFrame) -> Dict[str, List[RuleMatch]]:
        """Detect which records are affected by multiple rules"""
        conflicts = {}
        
        # Limit analysis to a reasonable sample size for performance
        sample_size = min(1000, len(data_df))
        if len(data_df) > sample_size:
            # Sample randomly to get representative data
            data_df = data_df.sample(n=sample_size, random_state=42)
        
        for idx, row in data_df.iterrows():
            matches = []
            
            for rule_idx, rule in enumerate(rules):
                if self._record_matches_rule(row, rule):
                    specificity = self._calculate_specificity(rule)
                    
                    # Handle both relative and absolute adjustments
                    adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
                    if adjustment_type == 'Absolute (Date)':
                        # For absolute date, we use a special offset value
                        offset_days = 999999  # Special marker for absolute date
                    else:
                        offset_days = rule.get('offset_days', 0)
                    
                    match = RuleMatch(
                        rule_id=rule['id'],
                        rule_index=rule_idx,
                        offset_days=offset_days,
                        specificity_score=specificity,
                        filters_matched=self._get_matched_filters(row, rule),
                        priority=rule.get('priority', rule_idx + 1)  # Use explicit priority
                    )
                    matches.append(match)
            
            if len(matches) > 1:
                # Create a unique key for the record
                record_key = self._create_record_key(row)
                conflicts[record_key] = matches
        
        return conflicts
    
    def resolve_conflicts(self, conflicts: Dict[str, List[RuleMatch]]) -> Dict[str, RuleMatch]:
        """Resolve conflicts based on selected strategy"""
        resolved = {}
        
        for record_key, matches in conflicts.items():
            if self.resolution_strategy == ConflictResolutionStrategy.PRIORITY_BASED:
                # Sort by explicit priority (higher number = higher priority), then by specificity
                winner = max(matches, key=lambda m: (m.priority, m.specificity_score))
            
            elif self.resolution_strategy == ConflictResolutionStrategy.MOST_SPECIFIC:
                # Most specific rule wins
                winner = max(matches, key=lambda m: m.specificity_score)
            
            elif self.resolution_strategy == ConflictResolutionStrategy.FIRST_MATCH:
                winner = min(matches, key=lambda m: m.rule_index)
            
            elif self.resolution_strategy == ConflictResolutionStrategy.LAST_MATCH:
                winner = max(matches, key=lambda m: m.rule_index)
            
            elif self.resolution_strategy == ConflictResolutionStrategy.CUMULATIVE:
                # Create a synthetic match with cumulative offset
                # Skip absolute date adjustments in cumulative mode
                valid_offsets = [m.offset_days for m in matches if m.offset_days != 999999]
                
                if valid_offsets:
                    total_offset = sum(valid_offsets)
                    winner = RuleMatch(
                        rule_id="CUMULATIVE",
                        rule_index=-1,
                        offset_days=total_offset,
                        specificity_score=0,
                        filters_matched={"type": "cumulative", "rules_count": str(len(matches))},
                        priority=0
                    )
                else:
                    # If all are absolute dates, use priority-based resolution
                    winner = max(matches, key=lambda m: (m.priority, m.specificity_score))
            
            resolved[record_key] = winner
        
        return resolved
    
    def _record_matches_rule(self, record: pd.Series, rule: Dict[str, Any]) -> bool:
        """Check if a record matches a rule's filters"""
        filters = rule.get('filters', {})
        
        # Entity filter
        if filters.get('entity') and filters['entity'] != ['All']:
            entity_col = self._get_entity_column(rule['data_source'])
            if entity_col in record and record[entity_col] not in filters['entity']:
                return False
        
        # Customer filter (only for demand sources)
        if rule['data_source'] in ['OC', 'Forecast']:
            if filters.get('customer') and filters['customer'] != ['All']:
                if 'customer' in record and record['customer'] not in filters['customer']:
                    return False
        
        # Product filter
        if filters.get('product') and filters['product'] != ['All']:
            # Extract PT codes from filter
            pt_codes = [p.split(' - ')[0] for p in filters['product'] if ' - ' in p]
            if pt_codes and 'pt_code' in record and record['pt_code'] not in pt_codes:
                return False
        
        # Number filter
        if filters.get('number') and filters['number'] != ['All']:
            number_col = self._get_number_column(rule['data_source'])
            if number_col in record and str(record[number_col]) not in filters['number']:
                return False
        
        # Brand filter
        if filters.get('brand') and filters['brand'] != ['All']:
            if 'brand' in record and record['brand'] not in filters['brand']:
                return False
        
        return True
    
    def _calculate_specificity(self, rule: Dict[str, Any]) -> int:
        """Calculate how specific a rule is (more filters = more specific)"""
        score = 0
        filters = rule.get('filters', {})
        
        for filter_type, filter_values in filters.items():
            if filter_values != ['All']:
                # Add points based on filter type and number of values
                if filter_type == 'number':
                    score += 100  # Most specific
                elif filter_type == 'product':
                    score += 50
                elif filter_type in ['entity', 'customer']:
                    score += 30
                elif filter_type == 'brand':
                    score += 20
                
                # Bonus for fewer selected values (more specific)
                if len(filter_values) == 1:
                    score += 10
                elif len(filter_values) <= 3:
                    score += 5
        
        return score
    
    def _get_matched_filters(self, record: pd.Series, rule: Dict[str, Any]) -> Dict[str, str]:
        """Get which filters matched for this record"""
        matched = {}
        filters = rule.get('filters', {})
        
        for filter_type, filter_values in filters.items():
            if filter_values != ['All']:
                matched[filter_type] = ', '.join(filter_values[:3])  # Show first 3
                if len(filter_values) > 3:
                    matched[filter_type] += f" (+{len(filter_values)-3} more)"
        
        return matched
    
    def _create_record_key(self, record: pd.Series) -> str:
        """Create unique key for a record"""
        key_parts = []
        
        # Add key fields based on what's available
        if 'pt_code' in record:
            key_parts.append(f"PT:{record['pt_code']}")
        
        if 'legal_entity' in record:
            key_parts.append(f"E:{record['legal_entity']}")
        
        if 'customer' in record:
            key_parts.append(f"C:{record['customer']}")
        
        # Add ID field if available
        for id_field in ['oc_number', 'forecast_number', 'po_number', 'inventory_history_id']:
            if id_field in record:
                key_parts.append(f"ID:{record[id_field]}")
                break
        
        return "|".join(key_parts)
    
    def _get_entity_column(self, data_source: str) -> str:
        """Get entity column name for data source"""
        mapping = {
            "OC": "legal_entity",
            "Forecast": "legal_entity",
            "Inventory": "legal_entity",
            "Pending CAN": "consignee",
            "Pending PO": "legal_entity",
            "Pending WH Transfer": "owning_company_name"
        }
        return mapping.get(data_source, "legal_entity")
    
    def _get_number_column(self, data_source: str) -> str:
        """Get number column name for data source"""
        mapping = {
            "OC": "oc_number",
            "Forecast": "forecast_number",
            "Inventory": "inventory_history_id",
            "Pending CAN": "arrival_note_number",
            "Pending PO": "po_number",
            "Pending WH Transfer": "warehouse_transfer_line_id"
        }
        return mapping.get(data_source, "id")
    
    def _get_filter_summary(self, filters: Dict[str, List[str]]) -> str:
        """Get human readable filter summary"""
        summary_parts = []
        
        for filter_type, filter_values in filters.items():
            if filter_values != ['All'] and filter_values:
                count = len(filter_values)
                display_name = {
                    'entity': 'Entity',
                    'customer': 'Customer',
                    'product': 'Product', 
                    'number': 'Number',
                    'brand': 'Brand'
                }.get(filter_type, filter_type.title())
                
                if count == 1:
                    value = filter_values[0]
                    if filter_type == 'product' and len(value) > 30:
                        value = value[:30] + "..."
                    summary_parts.append(f"{display_name}: {value}")
                else:
                    summary_parts.append(f"{display_name}: {count} selected")
        
        return "; ".join(summary_parts) if summary_parts else "All records"
    
    @staticmethod
    def show_conflict_analysis_ui(rules: List[Dict[str, Any]], preview_manager):
        """Show UI for detailed conflict analysis with sample records"""
        st.markdown("### ⚠️ Detailed Conflict Analysis")
        
        # Skip the strategy selector if we came from the auto-analysis
        if not st.session_state.get('skip_strategy_selector', False):
            # This path is for when conflict analysis is opened directly (shouldn't happen with auto-analysis)
            st.info("Conflicts have been automatically analyzed. Please return to the main view.")
            return
        
        # Clear the skip flag
        st.session_state.skip_strategy_selector = False
        
        # Get the configured strategy order
        strategy_order = st.session_state.get('conflict_resolution_order', [
            ConflictResolutionStrategy.PRIORITY_BASED,
            ConflictResolutionStrategy.MOST_SPECIFIC,
            ConflictResolutionStrategy.FIRST_MATCH,
            ConflictResolutionStrategy.LAST_MATCH,
            ConflictResolutionStrategy.CUMULATIVE
        ])
        
        # Show current resolution strategy order
        st.markdown("#### 🎯 Current Resolution Strategy Order")
        strategy_names = []
        for i, strategy in enumerate(strategy_order):
            name = {
                ConflictResolutionStrategy.PRIORITY_BASED: "Priority Based",
                ConflictResolutionStrategy.MOST_SPECIFIC: "Most Specific",
                ConflictResolutionStrategy.CUMULATIVE: "Cumulative",
                ConflictResolutionStrategy.FIRST_MATCH: "First Match",
                ConflictResolutionStrategy.LAST_MATCH: "Last Match"
            }.get(strategy, strategy.value)
            strategy_names.append(f"{i+1}. {name}")
        
        st.caption(" → ".join(strategy_names))
        
        with st.spinner("Loading detailed conflict analysis..."):
            # Group rules by data source
            rules_by_source = {}
            for rule in rules:
                source = rule['data_source']
                if source not in rules_by_source:
                    rules_by_source[source] = []
                rules_by_source[source].append(rule)
            
            # Analyze conflicts for each data source
            all_conflicts = {}
            total_conflicts = 0
            total_affected_records = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (data_source, source_rules) in enumerate(rules_by_source.items()):
                progress = (idx + 1) / len(rules_by_source)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {data_source}...")
                
                if len(source_rules) > 1:  # Only check if multiple rules
                    # Load data for this source
                    df = preview_manager._load_data_by_source(data_source)
                    
                    if not df.empty:
                        # Create a manager for each strategy to test resolution
                        conflicts_by_strategy = {}
                        
                        # First detect conflicts
                        manager = TimeAdjustmentConflictManager(ConflictResolutionStrategy.PRIORITY_BASED)
                        conflicts = manager.detect_conflicts(source_rules, df)
                        
                        if conflicts:
                            # Test each strategy in order
                            final_resolution = {}
                            resolution_details = []
                            
                            for strategy in strategy_order:
                                strategy_manager = TimeAdjustmentConflictManager(strategy)
                                resolved = strategy_manager.resolve_conflicts(conflicts)
                                
                                # Track which conflicts each strategy resolves
                                for record_key, matches in conflicts.items():
                                    if record_key not in final_resolution:
                                        winner = resolved[record_key]
                                        
                                        # Check if this strategy actually resolved the conflict
                                        if strategy == ConflictResolutionStrategy.CUMULATIVE:
                                            # Cumulative always resolves
                                            final_resolution[record_key] = {
                                                'winner': winner,
                                                'strategy': strategy,
                                                'matches': matches
                                            }
                                        elif len(set(m.priority for m in matches)) > 1 and strategy == ConflictResolutionStrategy.PRIORITY_BASED:
                                            # Priority based can resolve if priorities differ
                                            final_resolution[record_key] = {
                                                'winner': winner,
                                                'strategy': strategy,
                                                'matches': matches
                                            }
                                        elif len(set(m.specificity_score for m in matches)) > 1 and strategy == ConflictResolutionStrategy.MOST_SPECIFIC:
                                            # Most specific can resolve if specificities differ
                                            final_resolution[record_key] = {
                                                'winner': winner,
                                                'strategy': strategy,
                                                'matches': matches
                                            }
                                        elif strategy in [ConflictResolutionStrategy.FIRST_MATCH, ConflictResolutionStrategy.LAST_MATCH]:
                                            # First/Last always resolve
                                            final_resolution[record_key] = {
                                                'winner': winner,
                                                'strategy': strategy,
                                                'matches': matches
                                            }
                            
                            all_conflicts[data_source] = {
                                'conflicts': conflicts,
                                'final_resolution': final_resolution,
                                'rules': source_rules,
                                'total_records': len(df),
                                'sample_size': min(1000, len(df))
                            }
                            total_conflicts += len(conflicts)
                            total_affected_records += len(conflicts)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            
            if all_conflicts:
                # Summary metrics
                st.markdown("#### 📊 Detailed Conflict Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Conflicts", f"{total_conflicts:,}")
                
                with col2:
                    st.metric("Affected Records", f"{total_affected_records:,}")
                
                with col3:
                    st.metric("Data Sources", len(all_conflicts))
                
                with col4:
                    # Count resolutions by strategy
                    strategy_counts = {}
                    for source_data in all_conflicts.values():
                        for resolution in source_data['final_resolution'].values():
                            strategy = resolution['strategy']
                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    
                    most_used = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else None
                    if most_used:
                        st.metric("Most Used Strategy", most_used.value.replace('_', ' ').title())
                
                # Detailed conflicts by data source
                st.markdown("---")
                st.markdown("#### 📋 Detailed Analysis by Data Source")
                
                # Create tabs for each data source with conflicts
                if len(all_conflicts) > 1:
                    tabs = st.tabs([f"{source} ({len(data['conflicts'])})" for source, data in all_conflicts.items()])
                    
                    for tab, (data_source, conflict_data) in zip(tabs, all_conflicts.items()):
                        with tab:
                            TimeAdjustmentConflictManager._display_detailed_source_conflicts(
                                data_source, conflict_data, preview_manager
                            )
                else:
                    # Single source, no tabs needed
                    for data_source, conflict_data in all_conflicts.items():
                        TimeAdjustmentConflictManager._display_detailed_source_conflicts(
                            data_source, conflict_data, preview_manager
                        )
                
            else:
                # No conflicts found
                st.success("✅ No conflicts detected in detailed analysis!")
                st.markdown("""
                **Good news!** Each record in your data is affected by at most one rule. 
                There are no overlapping rules that would create conflicts.
                """)
    
    @staticmethod
    def _display_detailed_source_conflicts(data_source: str, conflict_data: Dict, preview_manager):
        """Display detailed conflicts for a specific data source with sample records"""
        # Summary info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conflicting Records", f"{len(conflict_data['conflicts']):,}")
        
        with col2:
            st.metric("Rules Involved", len(conflict_data['rules']))
        
        with col3:
            if conflict_data['sample_size'] < conflict_data['total_records']:
                st.metric("Analysis Sample", f"{conflict_data['sample_size']:,} of {conflict_data['total_records']:,}")
            else:
                st.metric("Records Analyzed", f"{conflict_data['total_records']:,}")
        
        # Resolution summary by strategy
        st.markdown("##### 📊 Resolution Summary")
        
        strategy_counts = {}
        for resolution in conflict_data['final_resolution'].values():
            strategy = resolution['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        if strategy_counts:
            resolution_df = pd.DataFrame([
                {
                    "Strategy": strategy.value.replace('_', ' ').title(),
                    "Conflicts Resolved": count,
                    "Percentage": f"{(count / len(conflict_data['conflicts']) * 100):.1f}%"
                }
                for strategy, count in strategy_counts.items()
            ])
            
            st.dataframe(resolution_df, use_container_width=True, hide_index=True)
        
        # Sample conflicts with resolution details
        st.markdown("##### 🔍 Sample Conflict Details")
        
        # Show up to 5 sample conflicts
        sample_size = min(5, len(conflict_data['conflicts']))
        sample_conflicts = list(conflict_data['conflicts'].items())[:sample_size]
        
        for i, (record_key, matches) in enumerate(sample_conflicts):
            with st.expander(f"Conflict {i + 1}: {record_key}", expanded=(i == 0)):
                resolution = conflict_data['final_resolution'].get(record_key)
                
                if resolution:
                    # Show the conflicting rules
                    st.markdown("**Conflicting Rules:**")
                    
                    for match in matches:
                        rule = next((r for r in conflict_data['rules'] if r['id'] == match.rule_id), None)
                        if rule:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"- **Rule {match.rule_index + 1}**: {match.offset_days:+d} days")
                                st.caption(f"  Priority: {match.priority} | Specificity: {match.specificity_score}")
                                
                                # Show filters
                                if match.filters_matched:
                                    filter_str = ", ".join([f"{k}: {v}" for k, v in match.filters_matched.items()])
                                    st.caption(f"  Filters: {filter_str}")
                    
                    # Show resolution
                    st.markdown("---")
                    st.markdown("**Resolution:**")
                    
                    winner = resolution['winner']
                    strategy = resolution['strategy']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Winning Rule", f"Rule {winner.rule_index + 1}")
                    
                    with col2:
                        st.metric("Resolution Strategy", strategy.value.replace('_', ' ').title())
                    
                    with col3:
                        if winner.offset_days == 999999:
                            st.metric("Final Adjustment", "→ Absolute Date")
                        else:
                            st.metric("Final Adjustment", f"{winner.offset_days:+d} days")
                    
                    # Try to show sample affected record
                    if preview_manager:
                        try:
                            # Parse record key to reconstruct a filter
                            key_parts = record_key.split('|')
                            sample_filter = {}
                            
                            for part in key_parts:
                                if part.startswith('PT:'):
                                    sample_filter['pt_code'] = part[3:]
                                elif part.startswith('C:'):
                                    sample_filter['customer'] = part[2:]
                                elif part.startswith('E:'):
                                    sample_filter['legal_entity'] = part[2:]
                            
                            if sample_filter:
                                st.markdown("---")
                                st.markdown("**Sample Affected Record:**")
                                st.caption("This record matches multiple rules, creating the conflict")
                                
                                # Create a simple display of the key info
                                display_data = []
                                for k, v in sample_filter.items():
                                    display_data.append({
                                        "Field": k.replace('_', ' ').title(),
                                        "Value": v
                                    })
                                
                                if display_data:
                                    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
                        except:
                            pass
        
        if len(conflict_data['conflicts']) > sample_size:
            st.caption(f"📝 Showing {sample_size} of {len(conflict_data['conflicts']):,} conflicts. ")
        
        # Visual conflict resolution flow
        st.markdown("##### 🎯 Resolution Flow Example")
        st.caption("How the multi-strategy resolution works for conflicts in this data source")
        
        # Take first conflict as example
        if conflict_data['conflicts']:
            example_key = list(conflict_data['conflicts'].keys())[0]
            example_matches = conflict_data['conflicts'][example_key]
            example_resolution = conflict_data['final_resolution'][example_key]
            
            # Create a flow diagram
            st.markdown("```")
            st.text("Conflict Detected → Try Priority Based → Try Most Specific → Try First Match → ... → Resolved!")
            st.markdown("```")
            
            st.caption(f"In this example, the conflict was resolved using **{example_resolution['strategy'].value.replace('_', ' ').title()}** strategy.")
    
    @staticmethod
    def _display_source_conflicts(data_source: str, conflict_data: Dict, manager):
        """Display conflicts for a specific data source"""
        # Summary info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conflicting Records", f"{len(conflict_data['conflicts']):,}")
        
        with col2:
            st.metric("Rules Involved", len(conflict_data['rules']))
        
        with col3:
            if conflict_data['sample_size'] < conflict_data['total_records']:
                st.metric("Analysis Sample", f"{conflict_data['sample_size']:,} of {conflict_data['total_records']:,}")
            else:
                st.metric("Records Analyzed", f"{conflict_data['total_records']:,}")
        
        # Rules involved
        with st.expander("📜 Rules Causing Conflicts", expanded=True):
            for idx, rule in enumerate(conflict_data['rules']):
                priority = rule.get('priority', idx + 1)
                filters_summary = manager._get_filter_summary(rule['filters'])
                
                adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
                if adjustment_type == 'Absolute (Date)':
                    offset_display = f"→ {rule.get('absolute_date', 'Unknown')}"
                else:
                    offset_display = f"{rule.get('offset_days', 0):+d} days"
                
                st.markdown(f"""
                **Rule {idx + 1}:** {offset_display} | Priority: {priority} | Filters: {filters_summary}
                """)
        
        # Sample conflicts with better formatting
        st.markdown("##### 🔍 Sample Conflict Details")
        
        # Prepare conflict data for display
        conflict_display_data = []
        
        sample_size = min(10, len(conflict_data['conflicts']))
        for record_key, matches in list(conflict_data['conflicts'].items())[:sample_size]:
            resolved = conflict_data['resolved'][record_key]
            
            # Parse record key for display
            key_parts = record_key.split('|')
            display_key = ', '.join(key_parts[:3])  # Show first 3 parts
            
            # Format offsets
            rule_offsets = []
            for m in matches:
                if m.offset_days == 999999:
                    rule_offsets.append("→Date")
                else:
                    rule_offsets.append(f"{m.offset_days:+d}d")
            
            # Format final offset
            final_offset = ""
            if resolved.offset_days == 999999:
                final_offset = "→ Absolute Date"
            else:
                final_offset = f"{resolved.offset_days:+d} days"
            
            conflict_display_data.append({
                "Record": display_key,
                "Matched Rules": len(matches),
                "Rule Offsets": ', '.join(rule_offsets),
                "Final Offset": final_offset,
                "Winning Rule": f"Rule {resolved.rule_index + 1}" if resolved.rule_id != "CUMULATIVE" else "Cumulative"
            })
        
        # Display as a nice table
        if conflict_display_data:
            df_display = pd.DataFrame(conflict_display_data)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Record": st.column_config.TextColumn("Record", width="large"),
                    "Matched Rules": st.column_config.NumberColumn("Matches", width="small"),
                    "Rule Offsets": st.column_config.TextColumn("All Offsets", width="medium"),
                    "Final Offset": st.column_config.TextColumn("Final Result", width="small"),
                    "Winning Rule": st.column_config.TextColumn("Winner", width="small")
                }
            )
            
            if len(conflict_data['conflicts']) > sample_size:
                st.caption(f"📝 Showing {sample_size} of {len(conflict_data['conflicts']):,} conflicts. "
                          f"{'Analysis based on sample data.' if conflict_data['sample_size'] < conflict_data['total_records'] else ''}")
        
        # Visual conflict resolution example
        if sample_size > 0:
            st.markdown("##### 🎯 Conflict Resolution Example")
            
            # Take first conflict as example
            example_key = list(conflict_data['conflicts'].keys())[0]
            example_matches = conflict_data['conflicts'][example_key]
            example_resolved = conflict_data['resolved'][example_key]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Before Resolution:**")
                for match in example_matches:
                    rule_num = match.rule_index + 1
                    if match.offset_days == 999999:
                        st.markdown(f"- Rule {rule_num}: → Absolute Date (Priority: {match.priority})")
                    else:
                        st.markdown(f"- Rule {rule_num}: {match.offset_days:+d} days (Priority: {match.priority})")
            
            with col2:
                st.markdown("**After Resolution:**")
                if example_resolved.rule_id == "CUMULATIVE":
                    st.success(f"✅ Final: {example_resolved.offset_days:+d} days (Sum of all)")
                else:
                    winning_rule = example_resolved.rule_index + 1
                    if example_resolved.offset_days == 999999:
                        st.success(f"✅ Final: → Absolute Date (Rule {winning_rule} wins)")
                    else:
                        st.success(f"✅ Final: {example_resolved.offset_days:+d} days (Rule {winning_rule} wins)")
    
    @staticmethod
    def show_conflict_warnings_full(rules: List[Dict[str, Any]]):
        """Show full-width conflict warnings with automatic analysis"""
        # Group rules by data source
        rules_by_source = {}
        for idx, rule in enumerate(rules):
            source = rule['data_source']
            if source not in rules_by_source:
                rules_by_source[source] = []
            rules_by_source[source].append((idx, rule))
        
        # Check for potential conflicts
        warnings = []
        
        for data_source, source_rules in rules_by_source.items():
            if len(source_rules) > 1:
                # Check for overlapping filters
                for i in range(len(source_rules)):
                    for j in range(i + 1, len(source_rules)):
                        idx1, rule1 = source_rules[i]
                        idx2, rule2 = source_rules[j]
                        
                        if TimeAdjustmentConflictManager._rules_may_overlap(rule1, rule2):
                            # Get adjustment info for display
                            adj1_type = rule1.get('adjustment_type', 'Relative (Days)')
                            adj2_type = rule2.get('adjustment_type', 'Relative (Days)')
                            
                            offset1_display = ""
                            offset2_display = ""
                            
                            if adj1_type == 'Absolute (Date)':
                                offset1_display = f"→ {rule1.get('absolute_date', 'Unknown')}"
                            else:
                                offset1_display = f"{rule1.get('offset_days', 0):+d} days"
                            
                            if adj2_type == 'Absolute (Date)':
                                offset2_display = f"→ {rule2.get('absolute_date', 'Unknown')}"
                            else:
                                offset2_display = f"{rule2.get('offset_days', 0):+d} days"
                            
                            warnings.append({
                                'source': data_source,
                                'rules': [idx1 + 1, idx2 + 1],
                                'rule_ids': [rule1['id'], rule2['id']],
                                'offsets': [rule1.get('offset_days', 0), rule2.get('offset_days', 0)],
                                'offset_displays': [offset1_display, offset2_display],
                                'priorities': [rule1.get('priority', idx1 + 1), rule2.get('priority', idx2 + 1)],
                                'specificities': [
                                    TimeAdjustmentConflictManager._calculate_rule_specificity(rule1),
                                    TimeAdjustmentConflictManager._calculate_rule_specificity(rule2)
                                ]
                            })
        
        # Display warnings with automatic analysis
        if warnings:
            # Initialize default resolution strategy order if not exists
            if 'conflict_resolution_order' not in st.session_state:
                st.session_state.conflict_resolution_order = [
                    ConflictResolutionStrategy.PRIORITY_BASED,
                    ConflictResolutionStrategy.MOST_SPECIFIC,
                    ConflictResolutionStrategy.FIRST_MATCH,
                    ConflictResolutionStrategy.LAST_MATCH,
                    ConflictResolutionStrategy.CUMULATIVE
                ]
            
            # Create a warning container
            with st.container():
                warning_container = st.warning("", icon="⚠️")
                
                with warning_container.container():
                    # Header
                    st.markdown(f"### ⚠️ {len(warnings)} Potential Conflicts Detected")
                    st.caption("Multiple rules may affect the same records. Configure resolution strategy order below.")
                    
                    # Resolution Strategy Configuration
                    st.markdown("---")
                    st.markdown("#### 🎯 Conflict Resolution Strategy Order")
                    st.caption("Drag strategies to reorder. Conflicts will be resolved using strategies in order until a winner is found.")
                    
                    # Help text for strategies
                    with st.expander("ℹ️ Understanding Resolution Strategies", expanded=False):
                        st.markdown("""
                        ### 📊 **Priority Based**
                        Rules with higher priority value win (100 > 50 > 1)
                        - **When to use:** When you have clear rule importance hierarchy
                        - **Example:** Company-wide rule (priority 90) overrides department rule (priority 50)
                        - **Tie breaker:** If priorities are equal, moves to next strategy
                        
                        ### 🎯 **Most Specific**
                        Rules with more specific filters win over general rules
                        - **When to use:** When detailed rules should override general ones
                        - **Specificity scoring:**
                          - Number filter: +100 points (most specific)
                          - Product filter: +50 points
                          - Entity/Customer: +30 points
                          - Brand filter: +20 points
                          - Single value selected: +10 bonus points
                        - **Example:** Rule for specific product wins over rule for all products
                        
                        ### 1️⃣ **First Match**
                        The rule that appears first in the list wins
                        - **When to use:** When rule order represents precedence
                        - **Example:** Rule 1 wins over Rule 5 for same record
                        - **Note:** Simple but predictable
                        
                        ### 🔚 **Last Match**
                        The rule that appears last in the list wins
                        - **When to use:** When newer rules should override older ones
                        - **Example:** Rule 5 wins over Rule 1 for same record
                        - **Note:** Last rule has final say
                        
                        ### ➕ **Cumulative**
                        All matching rules are applied additively
                        - **When to use:** When adjustments should stack
                        - **Example:** Rule 1 (+5 days) + Rule 2 (+3 days) = +8 days total
                        - **⚠️ Limitation:** Cannot combine with Absolute Date adjustments
                        - **Note:** Only works with Relative (Days) adjustments
                        
                        ---
                        
                        💡 **Strategy Order Tips:**
                        - Place your preferred strategy first
                        - Strategies are tried in order until one produces a clear winner
                        - If all strategies result in ties, First Match is used as final fallback
                        """)
                    
                    # Strategy order configuration
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Show current order with reorder controls
                        new_order = []
                        strategy_names = {
                            ConflictResolutionStrategy.PRIORITY_BASED: "📊 Priority Based",
                            ConflictResolutionStrategy.MOST_SPECIFIC: "🎯 Most Specific", 
                            ConflictResolutionStrategy.CUMULATIVE: "➕ Cumulative",
                            ConflictResolutionStrategy.FIRST_MATCH: "1️⃣ First Match",
                            ConflictResolutionStrategy.LAST_MATCH: "🔚 Last Match"
                        }
                        
                        strategy_descriptions = {
                            ConflictResolutionStrategy.PRIORITY_BASED: "Higher priority value wins (100 > 50)",
                            ConflictResolutionStrategy.MOST_SPECIFIC: "More filters = more specific = wins",
                            ConflictResolutionStrategy.CUMULATIVE: "Add all adjustments together",
                            ConflictResolutionStrategy.FIRST_MATCH: "First rule in list wins",
                            ConflictResolutionStrategy.LAST_MATCH: "Last rule in list wins"
                        }
                        
                        for i, strategy in enumerate(st.session_state.conflict_resolution_order):
                            cols = st.columns([1, 3, 1, 1])
                            
                            with cols[0]:
                                st.markdown(f"**{i + 1}.**")
                            
                            with cols[1]:
                                st.markdown(
                                    f"{strategy_names[strategy]}",
                                    help=strategy_descriptions[strategy]
                                )
                            
                            with cols[2]:
                                if i > 0 and st.button("⬆️", key=f"up_strategy_{strategy.value}"):
                                    # Move up
                                    st.session_state.conflict_resolution_order[i], st.session_state.conflict_resolution_order[i-1] = \
                                        st.session_state.conflict_resolution_order[i-1], st.session_state.conflict_resolution_order[i]
                                    st.rerun()
                            
                            with cols[3]:
                                if i < len(st.session_state.conflict_resolution_order) - 1 and st.button("⬇️", key=f"down_strategy_{strategy.value}"):
                                    # Move down
                                    st.session_state.conflict_resolution_order[i], st.session_state.conflict_resolution_order[i+1] = \
                                        st.session_state.conflict_resolution_order[i+1], st.session_state.conflict_resolution_order[i]
                                    st.rerun()
                    
                    with col2:
                        if st.button("🔄 Reset Order", use_container_width=True):
                            st.session_state.conflict_resolution_order = [
                                ConflictResolutionStrategy.PRIORITY_BASED,
                                ConflictResolutionStrategy.MOST_SPECIFIC,
                                ConflictResolutionStrategy.FIRST_MATCH,
                                ConflictResolutionStrategy.LAST_MATCH,
                                ConflictResolutionStrategy.CUMULATIVE
                            ]
                            st.rerun()
                    
                    # Show conflict analysis results
                    st.markdown("---")
                    st.markdown("#### 📋 Conflict Analysis Results")
                    
                    # Add legend for understanding results
                    with st.expander("📖 How to read the results", expanded=False):
                        st.markdown("""
                        **Columns explained:**
                        - **Rule X**: Shows the rule number, adjustment, priority, and specificity
                        - **Resolution**: Which rule wins and the strategy used
                        - **Final Offset**: The actual adjustment that will be applied
                        
                        **Resolution methods:**
                        - 🏆 The winning rule is determined by the first strategy that can decide
                        - 🔄 If a strategy results in a tie, the next strategy is used
                        - ✅ The final result shows which strategy made the decision
                        """)
                    
                    # Group warnings by data source
                    warnings_by_source = {}
                    for warning in warnings:
                        source = warning['source']
                        if source not in warnings_by_source:
                            warnings_by_source[source] = []
                        warnings_by_source[source].append(warning)
                    
                    # Display by source
                    for source, source_warnings in warnings_by_source.items():
                        st.markdown(f"##### {source}")
                        st.caption(f"{len(source_warnings)} conflicts")
                        
                        for warning in source_warnings:
                            # Resolve conflict using strategy order
                            resolution = TimeAdjustmentConflictManager._resolve_conflict_with_strategy_order(
                                warning, st.session_state.conflict_resolution_order
                            )
                            
                            # Display conflict details
                            cols = st.columns([2, 2, 2, 3])
                            
                            with cols[0]:
                                st.metric(
                                    f"Rule {warning['rules'][0]}",
                                    warning['offset_displays'][0],
                                    f"Priority: {warning['priorities'][0]} | Specificity: {warning['specificities'][0]}",
                                    label_visibility="visible"
                                )
                            
                            with cols[1]:
                                st.metric(
                                    f"Rule {warning['rules'][1]}",
                                    warning['offset_displays'][1],
                                    f"Priority: {warning['priorities'][1]} | Specificity: {warning['specificities'][1]}",
                                    label_visibility="visible"
                                )
                            
                            with cols[2]:
                                st.metric(
                                    "Resolution",
                                    resolution['winner'],
                                    resolution['method'],
                                    label_visibility="visible"
                                )
                            
                            with cols[3]:
                                st.metric(
                                    "Final Offset",
                                    resolution['final_offset'],
                                    resolution['explanation'],
                                    label_visibility="visible"
                                )
                        
                        st.markdown("---")
                    
                    # Full analysis button
                    if st.button("🔍 View Detailed Analysis", type="primary", use_container_width=True):
                        st.session_state.show_conflict_analysis = True
                        st.session_state.skip_strategy_selector = True  # Skip the strategy selector in detailed view
                        st.rerun()
                    
                    # Footer with example
                    st.caption(
                        "💡 **How it works**: Conflicts are resolved by applying strategies in order. "
                        "If the first strategy results in a tie, the next strategy is used, and so on."
                    )
                    
                    # Add visual example
                    with st.expander("🎓 Example: How conflict resolution works", expanded=False):
                        st.markdown("""
                        **Scenario:** Two rules match the same record:
                        - Rule A: +5 days, Priority 50, 2 filters
                        - Rule B: +10 days, Priority 50, 4 filters
                        
                        **Resolution process with current order:**
                        
                        1. **📊 Priority Based** → Both have priority 50 → **TIE**
                        2. **🎯 Most Specific** → Rule B has more filters → **Rule B WINS!** ✅
                        3. ~~First Match~~ (not needed, already resolved)
                        4. ~~Last Match~~ (not needed, already resolved)
                        5. ~~Cumulative~~ (not needed, already resolved)
                        
                        **Result:** Record will be adjusted by +10 days (Rule B)
                        """)
        else:
            # No conflicts
            with st.container():
                st.success("✅ No potential conflicts detected between rules", icon="✅")
    
    @staticmethod
    def _calculate_rule_specificity(rule: Dict[str, Any]) -> int:
        """Calculate specificity score for a rule"""
        score = 0
        filters = rule.get('filters', {})
        
        for filter_type, filter_values in filters.items():
            if filter_values != ['All']:
                # Add points based on filter type and number of values
                if filter_type == 'number':
                    score += 100  # Most specific
                elif filter_type == 'product':
                    score += 50
                elif filter_type in ['entity', 'customer']:
                    score += 30
                elif filter_type == 'brand':
                    score += 20
                
                # Bonus for fewer selected values (more specific)
                if len(filter_values) == 1:
                    score += 10
                elif len(filter_values) <= 3:
                    score += 5
        
        return score
    
    @staticmethod
    def _resolve_conflict_with_strategy_order(warning: Dict, strategy_order: List[ConflictResolutionStrategy]) -> Dict[str, str]:
        """Resolve a conflict using the ordered list of strategies"""
        rule1_idx = warning['rules'][0] - 1
        rule2_idx = warning['rules'][1] - 1
        
        for strategy in strategy_order:
            if strategy == ConflictResolutionStrategy.PRIORITY_BASED:
                if warning['priorities'][0] > warning['priorities'][1]:
                    return {
                        'winner': f"Rule {warning['rules'][0]}",
                        'method': "Priority Based",
                        'final_offset': warning['offset_displays'][0],
                        'explanation': f"Higher priority ({warning['priorities'][0]} > {warning['priorities'][1]})"
                    }
                elif warning['priorities'][0] < warning['priorities'][1]:
                    return {
                        'winner': f"Rule {warning['rules'][1]}",
                        'method': "Priority Based",
                        'final_offset': warning['offset_displays'][1],
                        'explanation': f"Higher priority ({warning['priorities'][1]} > {warning['priorities'][0]})"
                    }
                # Tie, continue to next strategy
                
            elif strategy == ConflictResolutionStrategy.MOST_SPECIFIC:
                if warning['specificities'][0] > warning['specificities'][1]:
                    return {
                        'winner': f"Rule {warning['rules'][0]}",
                        'method': "Most Specific",
                        'final_offset': warning['offset_displays'][0],
                        'explanation': f"Higher specificity ({warning['specificities'][0]} > {warning['specificities'][1]})"
                    }
                elif warning['specificities'][0] < warning['specificities'][1]:
                    return {
                        'winner': f"Rule {warning['rules'][1]}",
                        'method': "Most Specific",
                        'final_offset': warning['offset_displays'][1],
                        'explanation': f"Higher specificity ({warning['specificities'][1]} > {warning['specificities'][0]})"
                    }
                # Tie, continue to next strategy
                
            elif strategy == ConflictResolutionStrategy.FIRST_MATCH:
                if rule1_idx < rule2_idx:
                    return {
                        'winner': f"Rule {warning['rules'][0]}",
                        'method': "First Match",
                        'final_offset': warning['offset_displays'][0],
                        'explanation': "First rule in order"
                    }
                else:
                    return {
                        'winner': f"Rule {warning['rules'][1]}",
                        'method': "First Match",
                        'final_offset': warning['offset_displays'][1],
                        'explanation': "First rule in order"
                    }
                    
            elif strategy == ConflictResolutionStrategy.LAST_MATCH:
                if rule1_idx > rule2_idx:
                    return {
                        'winner': f"Rule {warning['rules'][0]}",
                        'method': "Last Match",
                        'final_offset': warning['offset_displays'][0],
                        'explanation': "Last rule in order"
                    }
                else:
                    return {
                        'winner': f"Rule {warning['rules'][1]}",
                        'method': "Last Match",
                        'final_offset': warning['offset_displays'][1],
                        'explanation': "Last rule in order"
                    }
                    
            elif strategy == ConflictResolutionStrategy.CUMULATIVE:
                # For cumulative, both rules apply
                if '→' in warning['offset_displays'][0] or '→' in warning['offset_displays'][1]:
                    # Can't cumulate absolute dates
                    continue
                    
                total_offset = warning['offsets'][0] + warning['offsets'][1]
                return {
                    'winner': "Both",
                    'method': "Cumulative",
                    'final_offset': f"{total_offset:+d} days",
                    'explanation': f"Sum of {warning['offset_displays'][0]} and {warning['offset_displays'][1]}"
                }
        
        # If all strategies result in ties (shouldn't happen normally)
        return {
            'winner': "Unresolved",
            'method': "No Resolution",
            'final_offset': "N/A",
            'explanation': "All strategies resulted in ties"
        }
    
    @staticmethod
    def _rules_may_overlap(rule1: Dict[str, Any], rule2: Dict[str, Any]) -> bool:
        """Check if two rules may overlap"""
        filters1 = rule1.get('filters', {})
        filters2 = rule2.get('filters', {})
        
        # If both rules have no filters (apply to all), they definitely overlap
        all_filters1 = all(v == ['All'] for v in filters1.values())
        all_filters2 = all(v == ['All'] for v in filters2.values())
        
        if all_filters1 or all_filters2:
            return True
        
        # Check if any filter is completely different (no overlap possible)
        for filter_type in ['entity', 'customer', 'product', 'number', 'brand']:
            vals1 = set(filters1.get(filter_type, ['All']))
            vals2 = set(filters2.get(filter_type, ['All']))
            
            # If both have specific values and they don't intersect, no overlap
            if 'All' not in vals1 and 'All' not in vals2:
                if not vals1.intersection(vals2):
                    return False
        
        # If we get here, rules may overlap
        return True