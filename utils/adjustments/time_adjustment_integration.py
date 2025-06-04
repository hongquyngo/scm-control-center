# utils/adjustments/time_adjustment_integration.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TimeAdjustmentIntegration:
    """Centralized time adjustment handling for supply-demand analysis"""
    
    def __init__(self):
        # Lazy loading to avoid circular imports
        self._time_manager = None
        self._conflict_manager = None
        self._adjustment_cache = {}
        self._metadata = {}
        
    def _get_time_manager(self):
        """Lazy load TimeAdjustmentManager"""
        if self._time_manager is None:
            try:
                from utils.adjustments.time_adjustments import TimeAdjustmentManager
                self._time_manager = TimeAdjustmentManager
            except ImportError:
                logger.warning("TimeAdjustmentManager not available")
        return self._time_manager
    
    def _get_conflict_manager(self):
        """Lazy load ConflictManager"""
        if self._conflict_manager is None:
            try:
                from utils.adjustments.conflict_manager import TimeAdjustmentConflictManager
                self._conflict_manager = TimeAdjustmentConflictManager
            except ImportError:
                logger.warning("ConflictManager not available")
        return self._conflict_manager

    def apply_adjustments(self, df: pd.DataFrame, data_source: str) -> pd.DataFrame:
        """Apply time adjustments to dataframe and preserve original dates"""
        if df.empty:
            return df
        
        try:
            # Get date column for this source
            date_column = self._get_date_column(data_source)
            
            if not date_column or date_column not in df.columns:
                logger.info(f"No date column to adjust for {data_source}")
                return df
            
            # Create a copy
            result_df = df.copy()
            
            # Store original dates FIRST
            original_column = f"{date_column}_original"
            
            # Insert original column next to the date column
            try:
                col_index = result_df.columns.get_loc(date_column)
                result_df.insert(col_index + 1, original_column, result_df[date_column].copy())
            except:
                result_df[original_column] = result_df[date_column].copy()
            
            # Ensure datetime type
            result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
            result_df[original_column] = pd.to_datetime(result_df[original_column], errors='coerce')
            
            # Check for rules
            rules = st.session_state.get('time_adjustment_rules', [])
            
            if rules:
                result_df = self._apply_advanced_adjustments(result_df, data_source, date_column)
            else:
                result_df = self._apply_simple_adjustments(result_df, data_source, date_column)
            
            # Create adjusted column with correct naming
            adjusted_column = f"{date_column}_adjusted"  # Keep this format
            
            # Insert adjusted column after original
            try:
                col_index = result_df.columns.get_loc(original_column)
                result_df.insert(col_index + 1, adjusted_column, result_df[date_column].copy())
            except:
                result_df[adjusted_column] = result_df[date_column].copy()
            
            # Store metadata
            self._store_adjustment_metadata(data_source, result_df, date_column)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error applying adjustments to {data_source}: {str(e)}")
            return df

    def _apply_advanced_adjustments(self, df: pd.DataFrame, data_source: str, date_column: str) -> pd.DataFrame:
        """Apply advanced time adjustment rules"""
        rules = st.session_state.get('time_adjustment_rules', [])
        
        if not rules:
            return df
        
        # Filter rules for this data source
        source_rules = [rule for rule in rules if rule.get('data_source') == data_source]
        
        if not source_rules:
            logger.info(f"No rules found for {data_source}")
            return df
        
        # Log rules being applied
        logger.info(f"Applying {len(source_rules)} rules to {data_source}")
        
        # Check for conflicts
        if len(source_rules) > 1:
            df = self._apply_with_conflict_resolution(df, source_rules, data_source, date_column)
        else:
            # Single rule, apply directly
            df = self._apply_single_rule(df, source_rules[0], date_column)
        
        return df
    
    def _apply_simple_adjustments(self, df: pd.DataFrame, data_source: str, date_column: str) -> pd.DataFrame:
        """Apply simple offset adjustments from settings"""
        # Map data sources to setting keys
        offset_mapping = {
            'OC': 'time_adjustments.oc_etd_offset',
            'Forecast': 'time_adjustments.forecast_etd_offset',
            'Pending CAN': 'time_adjustments.can_arrival_offset',
            'Pending PO': 'time_adjustments.po_crd_offset',
            'Pending WH Transfer': 'time_adjustments.wh_transfer_lead_time'
        }
        
        setting_key = offset_mapping.get(data_source)
        if not setting_key:
            return df
        
        # Get offset from settings
        key_parts = setting_key.split('.')
        offset_days = st.session_state.get('business_settings', {}).get(
            key_parts[0], {}
        ).get(key_parts[1], 0)
        
        if offset_days != 0:
            # Apply offset with safe date handling
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce') + timedelta(days=offset_days)
                logger.info(f"Applied simple offset of {offset_days} days to {data_source} {date_column}")
            except Exception as e:
                logger.error(f"Error applying simple offset: {str(e)}")
        
        return df
    
    def _apply_single_rule(self, df: pd.DataFrame, rule: Dict[str, Any], date_column: str) -> pd.DataFrame:
        """Apply a single adjustment rule"""
        try:
            # Filter dataframe based on rule filters
            mask = self._create_filter_mask(df, rule)
            
            if mask.sum() == 0:
                logger.info(f"No matching records for rule {rule.get('name', 'unnamed')}")
                return df  # No matching records
            
            logger.info(f"Applying rule to {mask.sum()} records")
            
            adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
            
            if adjustment_type == 'Absolute (Date)':
                # Set to specific date - this will replace even NULL/missing dates
                absolute_date = pd.to_datetime(rule.get('absolute_date'))
                df.loc[mask, date_column] = absolute_date
                logger.info(f"Set {mask.sum()} records to absolute date {absolute_date}")
            else:
                # Apply relative offset only to non-null dates
                offset_days = rule.get('offset_days', 0)
                if offset_days != 0:
                    # Create a sub-mask for records that have non-null dates
                    date_series = pd.to_datetime(df.loc[mask, date_column], errors='coerce')
                    non_null_mask = mask & date_series.notna()
                    
                    if non_null_mask.sum() > 0:
                        # Apply offset only to non-null dates
                        df.loc[non_null_mask, date_column] = date_series[non_null_mask] + timedelta(days=offset_days)
                        logger.info(f"Applied {offset_days} days offset to {non_null_mask.sum()} non-null dates")
                    
                    # Log if some records were skipped due to null dates
                    null_count = mask.sum() - non_null_mask.sum()
                    if null_count > 0:
                        logger.info(f"Skipped {null_count} records with missing dates (relative adjustment cannot be applied to null dates)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying single rule: {str(e)}")
            return df
    
    def _apply_with_conflict_resolution(self, df: pd.DataFrame, rules: List[Dict], 
                                      data_source: str, date_column: str) -> pd.DataFrame:
        """Apply multiple rules with conflict resolution using vectorized operations"""
        ConflictManager = self._get_conflict_manager()
        if not ConflictManager:
            # Fallback to first rule if conflict manager not available
            return self._apply_single_rule(df, rules[0], date_column)
        
        # Get resolution strategy order
        strategy_order = st.session_state.get('conflict_resolution_order', [
            'PRIORITY_BASED',
            'MOST_SPECIFIC',
            'FIRST_MATCH',
            'LAST_MATCH',
            'CUMULATIVE'
        ])
        
        # Create masks for all rules
        rule_masks = []
        for rule in rules:
            mask = self._create_filter_mask(df, rule)
            rule_masks.append(mask)
        
        # Find conflicts (rows matching multiple rules)
        conflict_mask = pd.Series(False, index=df.index)
        for i in range(len(rule_masks)):
            for j in range(i + 1, len(rule_masks)):
                conflict_mask |= (rule_masks[i] & rule_masks[j])
        
        # Apply non-conflicting rules first (vectorized)
        for idx, (rule, mask) in enumerate(zip(rules, rule_masks)):
            # Apply only to non-conflicting rows
            apply_mask = mask & ~conflict_mask
            
            if apply_mask.sum() > 0:
                adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
                
                if adjustment_type == 'Absolute (Date)':
                    df.loc[apply_mask, date_column] = pd.to_datetime(rule.get('absolute_date'))
                else:
                    offset_days = rule.get('offset_days', 0)
                    if offset_days != 0:
                        # Only apply to non-null dates for relative adjustments
                        date_series = pd.to_datetime(df.loc[apply_mask, date_column], errors='coerce')
                        non_null_mask = apply_mask & date_series.notna()
                        
                        if non_null_mask.sum() > 0:
                            df.loc[non_null_mask, date_column] = date_series[non_null_mask] + timedelta(days=offset_days)
        
        # Handle conflicting rows (if any)
        if conflict_mask.sum() > 0:
            logger.info(f"Resolving conflicts for {conflict_mask.sum()} rows")
            
            # For now, use priority-based resolution (highest priority wins)
            conflict_indices = df[conflict_mask].index
            
            for idx in conflict_indices:
                # Find all matching rules for this row
                matching_rules = []
                for rule_idx, (rule, mask) in enumerate(zip(rules, rule_masks)):
                    if mask[idx]:
                        priority = rule.get('priority', rule_idx + 1)
                        matching_rules.append((priority, rule))
                
                if matching_rules:
                    # Sort by priority (lower number = higher priority)
                    matching_rules.sort(key=lambda x: x[0])
                    winning_rule = matching_rules[0][1]
                    
                    # Apply winning rule
                    adjustment_type = winning_rule.get('adjustment_type', 'Relative (Days)')
                    
                    if adjustment_type == 'Absolute (Date)':
                        df.at[idx, date_column] = pd.to_datetime(winning_rule.get('absolute_date'))
                    else:
                        offset_days = winning_rule.get('offset_days', 0)
                        if offset_days != 0:
                            current_date = pd.to_datetime(df.at[idx, date_column], errors='coerce')
                            if pd.notna(current_date):
                                df.at[idx, date_column] = current_date + timedelta(days=offset_days)
                            # If current_date is NaT/null, skip for relative adjustment
        
        return df
    
    def _create_filter_mask(self, df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
        """Create boolean mask based on rule filters"""
        mask = pd.Series(True, index=df.index)
        filters = rule.get('filters', {})
        
        # Entity filter
        if filters.get('entity') and filters['entity'] != ['All']:
            entity_col = self._get_entity_column(rule['data_source'])
            if entity_col in df.columns:
                mask &= df[entity_col].isin(filters['entity'])
            else:
                logger.warning(f"Entity column '{entity_col}' not found for {rule['data_source']}")
        
        # Customer filter
        if filters.get('customer') and filters['customer'] != ['All']:
            if 'customer' in df.columns:
                mask &= df['customer'].isin(filters['customer'])
            else:
                logger.warning(f"Customer column not found")
        
        # Product filter
        if filters.get('product') and filters['product'] != ['All']:
            pt_codes = [p.split(' - ')[0] for p in filters['product'] if ' - ' in p]
            if pt_codes and 'pt_code' in df.columns:
                mask &= df['pt_code'].isin(pt_codes)
            elif not pt_codes:
                logger.warning("No valid PT codes found in product filter")
            else:
                logger.warning("PT code column not found")
        
        # Brand filter
        if filters.get('brand') and filters['brand'] != ['All']:
            if 'brand' in df.columns:
                mask &= df['brand'].isin(filters['brand'])
            else:
                logger.warning("Brand column not found")
        
        # Number filter
        if filters.get('number') and filters['number'] != ['All']:
            number_col = self._get_number_column(rule['data_source'])
            if number_col in df.columns:
                mask &= df[number_col].astype(str).isin(filters['number'])
            else:
                logger.warning(f"Number column '{number_col}' not found")
        
        return mask
    
    def _row_matches_rule(self, row: Dict, rule: Dict) -> bool:
        """Check if a single row matches rule filters"""
        filters = rule.get('filters', {})
        
        try:
            # Check each filter type
            if filters.get('entity') and filters['entity'] != ['All']:
                entity_col = self._get_entity_column(rule['data_source'])
                if entity_col in row and row[entity_col] not in filters['entity']:
                    return False
            
            if filters.get('customer') and filters['customer'] != ['All']:
                if 'customer' in row and row['customer'] not in filters['customer']:
                    return False
            
            if filters.get('product') and filters['product'] != ['All']:
                pt_codes = [p.split(' - ')[0] for p in filters['product'] if ' - ' in p]
                if pt_codes and 'pt_code' in row and row['pt_code'] not in pt_codes:
                    return False
            
            if filters.get('brand') and filters['brand'] != ['All']:
                if 'brand' in row and row['brand'] not in filters['brand']:
                    return False
            
            if filters.get('number') and filters['number'] != ['All']:
                number_col = self._get_number_column(rule['data_source'])
                if number_col in row and str(row[number_col]) not in filters['number']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching row to rule: {str(e)}")
            return False
    
    def _resolve_conflict(self, matching_rules: List[Tuple[int, Dict]], 
                         strategy_order: List) -> Optional[Dict]:
        """Resolve conflict between multiple matching rules"""
        if not matching_rules:
            return None
        
        # For now, use priority-based resolution
        # Get rule with highest priority (lowest number)
        best_rule = min(matching_rules, key=lambda x: x[1].get('priority', x[0] + 1))
        return best_rule[1]
    
    def _get_date_column(self, data_source: str) -> Optional[str]:
        """Get date column name for data source"""
        mapping = {
            "OC": "etd",
            "Forecast": "etd",
            "Inventory": "date_ref",  # CHANGED FROM None TO date_ref
            "Pending CAN": "arrival_date",
            "Pending PO": "eta",
            "Pending WH Transfer": "transfer_date"
        }
        return mapping.get(data_source)
    
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
    
    def _get_adjustment_mode(self) -> str:
        """Get current adjustment mode"""
        # Check if we have advanced rules
        if st.session_state.get('time_adjustment_rules'):
            return 'advanced'
        # Check if we have simple adjustments
        elif self._has_simple_adjustments():
            return 'simple'
        else:
            return 'none'  # No adjustments configured
    
    def _has_simple_adjustments(self) -> bool:
        """Check if any simple adjustments are configured"""
        settings = st.session_state.get('business_settings', {}).get('time_adjustments', {})
        
        offset_keys = [
            'oc_etd_offset',
            'forecast_etd_offset',
            'can_arrival_offset',
            'po_crd_offset',
            'wh_transfer_lead_time'
        ]
        
        for key in offset_keys:
            if settings.get(key, 0) != 0:
                return True
        
        return False
    
    def _store_adjustment_metadata(self, data_source: str, df: pd.DataFrame, date_column: str):
        """Store metadata about adjustments applied"""
        try:
            original_col = f"{date_column}_original"
            adjusted_col = f"{date_column}_adjusted"
            
            if original_col in df.columns and adjusted_col in df.columns:
                # Ensure datetime type for comparison
                original_dates = pd.to_datetime(df[original_col], errors='coerce')
                adjusted_dates = pd.to_datetime(df[adjusted_col], errors='coerce')
                
                # Calculate records affected (where dates differ)
                date_diff = (adjusted_dates - original_dates).dt.days
                affected = (date_diff.notna() & (date_diff != 0)).sum()
                
                self._metadata[data_source] = {
                    'last_applied': datetime.now(),
                    'records_total': len(df),
                    'records_affected': int(affected),
                    'mode': self._get_adjustment_mode(),
                    'date_column': date_column
                }
                
                logger.info(f"Stored metadata for {data_source}: {affected} records affected")
                
        except Exception as e:
            logger.error(f"Error storing adjustment metadata: {str(e)}")
    
    def get_adjustment_status(self) -> Dict[str, Any]:
        """Get current adjustment status for UI display"""
        mode = self._get_adjustment_mode()
        
        status = {
            'active': mode != 'none',
            'mode': mode,
            'rules_count': 0,
            'last_applied': None,
            'affected_records': 0
        }
        
        if mode == 'advanced':
            rules = st.session_state.get('time_adjustment_rules', [])
            status['rules_count'] = len(rules)
        
        # Aggregate metadata from all sources
        if self._metadata:
            status['last_applied'] = max(
                (m['last_applied'] for m in self._metadata.values()),
                default=None
            )
            status['affected_records'] = sum(
                m['records_affected'] for m in self._metadata.values()
            )
        
        return status
    
    def get_adjustment_summary(self, data_source: str = None) -> pd.DataFrame:
        """Get detailed summary of adjustments applied"""
        if data_source and data_source in self._metadata:
            # Single source summary
            meta = self._metadata[data_source]
            summary_data = [{
                'Data Source': data_source,
                'Mode': meta['mode'],
                'Date Column': meta['date_column'],
                'Total Records': meta['records_total'],
                'Records Adjusted': meta['records_affected'],
                'Last Applied': meta['last_applied'].strftime('%Y-%m-%d %H:%M:%S')
            }]
        else:
            # All sources summary
            summary_data = []
            for source, meta in self._metadata.items():
                summary_data.append({
                    'Data Source': source,
                    'Mode': meta['mode'],
                    'Date Column': meta['date_column'],
                    'Total Records': meta['records_total'],
                    'Records Adjusted': meta['records_affected'],
                    'Last Applied': meta['last_applied'].strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return pd.DataFrame(summary_data)
    
    def clear_cache(self):
        """Clear adjustment cache and metadata"""
        self._adjustment_cache.clear()
        self._metadata.clear()
        logger.info("Cleared adjustment cache and metadata")
    
    def validate_adjustments(self, df: pd.DataFrame, data_source: str) -> Dict[str, Any]:
        """Validate adjustments applied to dataframe"""
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        date_column = self._get_date_column(data_source)
        if not date_column:
            return validation
        
        original_col = f"{date_column}_original"
        adjusted_col = f"{date_column}_adjusted"
        
        # Check if adjustment columns exist
        if original_col not in df.columns or adjusted_col not in df.columns:
            validation['warnings'].append("Adjustment columns not found")
            return validation
        
        try:
            # Validate date types
            original_dates = pd.to_datetime(df[original_col], errors='coerce')
            adjusted_dates = pd.to_datetime(df[adjusted_col], errors='coerce')
            
            # Check for invalid dates
            invalid_original = original_dates.isna().sum()
            invalid_adjusted = adjusted_dates.isna().sum()
            
            if invalid_original > 0:
                validation['warnings'].append(f"{invalid_original} invalid original dates found")
            
            if invalid_adjusted > 0:
                validation['warnings'].append(f"{invalid_adjusted} invalid adjusted dates found")
            
            # Check for extreme adjustments (more than 365 days)
            date_diff = (adjusted_dates - original_dates).dt.days
            extreme_adjustments = (date_diff.abs() > 365).sum()
            
            if extreme_adjustments > 0:
                validation['warnings'].append(f"{extreme_adjustments} records with adjustments > 365 days")
            
            # Check for future dates too far out (more than 2 years)
            future_threshold = pd.Timestamp.now() + pd.Timedelta(days=730)
            far_future = (adjusted_dates > future_threshold).sum()
            
            if far_future > 0:
                validation['warnings'].append(f"{far_future} records with dates > 2 years in future")
            
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Validation error: {str(e)}")
        
        return validation