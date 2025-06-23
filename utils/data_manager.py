# utils/data_manager.py - Consolidated Data Management

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from sqlalchemy import text

from .db import get_db_engine
from utils.adjustments.time_adjustment_integration import TimeAdjustmentIntegration

logger = logging.getLogger(__name__)

class DataManager:
    """Unified data management singleton - combines functionality from all data modules"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._cache = {}
            self._metadata = {}
            self._last_refresh = {}
            self._insights_cache = {}
            # Initialize settings manager separately to avoid circular dependency
            self._settings_manager = None
            self._cache_ttl = 300  # 5 minutes default
            # Add time adjustment integration
            self._time_adjustment_integration = TimeAdjustmentIntegration()
            self._initialized = True
    
    def _get_settings_manager(self):
        """Lazy load settings manager to avoid initialization issues"""
        if self._settings_manager is None:
            from .settings_manager import SettingsManager
            self._settings_manager = SettingsManager()
        return self._settings_manager
    
    # === Core Data Loading Methods ===
    
    @st.cache_data(ttl=1800)
    def load_demand_oc(_self):
        """
        Load OC (Order Confirmation) pending delivery data
        
        Hàm này load dữ liệu các đơn hàng đang chờ giao (pending delivery) từ view outbound_oc_pending_delivery_view
        
        Các trường dữ liệu được load:
        
        === THÔNG TIN SẢN PHẨM ===
        1. product_name: Tên sản phẩm
        2. pt_code: Mã PT của sản phẩm
        3. brand: Tên thương hiệu (brand_name)
        4. package_size: Kích thước đóng gói
        
        === THÔNG TIN DÒNG ĐƠN HÀNG (OC Line) ===
        5. ocd_id: ID của chi tiết đơn hàng (order_confirmation_details.id)
        6. oc_number: Số đơn hàng xác nhận (OC number)
        7. customer_po_number: Số PO của khách hàng
        
        === THÔNG TIN KHÁCH HÀNG & PHÁP NHÂN ===
        8. customer: Tên khách hàng tiếng Anh (buyer.english_name)
        9. customer_code: Mã khách hàng (buyer.company_code)
        10. legal_entity: Tên pháp nhân bán hàng (seller.english_name)
        
        === THÔNG TIN NGÀY THÁNG ===
        11. oc_date: Ngày tạo đơn hàng xác nhận
        12. etd: Ngày dự kiến giao hàng (Expected Time of Delivery)
        
        === THÔNG TIN ĐƠN VỊ & SỐ LƯỢNG ===
        13. standard_uom: Đơn vị tính chuẩn (Unit of Measurement)
        14. selling_uom: Đơn vị bán hàng
        15. uom_conversion: Tỷ lệ quy đổi giữa selling_uom và standard_uom
        16. selling_quantity: Số lượng bán (theo selling_uom) sau khi trừ hủy
        17. standard_quantity: Số lượng chuẩn (theo standard_uom) sau khi trừ hủy
        18. total_delivered_selling_quantity: Tổng số lượng đã giao (theo selling_uom)
        19. total_delivered_standard_quantity: Tổng số lượng đã giao (theo standard_uom)
        20. pending_selling_delivery_quantity: Số lượng chờ giao (theo selling_uom)
        21. pending_standard_delivery_quantity: Số lượng chờ giao (theo standard_uom)
        
        === TRẠNG THÁI GIAO HÀNG ===
        22. delivery_status: Trạng thái giao hàng ('Not Delivered', 'Partially Delivered', 'Unknown')
        
        === THÔNG TIN GIÁ & TIỀN ===
        23. selling_unit_price: Đơn giá bán (theo selling_uom)
        24. total_amount_usd: Tổng giá trị đơn hàng tính bằng USD (sau khi trừ hủy)
        25. delivered_amount_usd: Giá trị đã giao tính bằng USD
        26. outstanding_amount_usd: Giá trị còn lại chưa giao tính bằng USD
        
        Lưu ý:
        - View chỉ lấy các đơn hàng có pending delivery (chưa giao hoàn toàn)
        - Số lượng đã được điều chỉnh sau khi trừ các phần hủy (cancellation)
        - Dữ liệu được sắp xếp theo outstanding_amount_usd giảm dần và oc_date mới nhất
        """
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.outbound_oc_pending_delivery_view;"
        df = pd.read_sql(text(query), engine)
        
        print("Loaded OC pending delivery data successfully ")
        print(df.info())
        
        return df


    @st.cache_data(ttl=1800)
    def load_demand_forecast(_self):
        """
        Load customer demand forecast data
        
        Hàm này load dữ liệu dự báo nhu cầu từ khách hàng từ view customer_demand_forecast_full_view
        
        Các trường dữ liệu được load:
        
        === THÔNG TIN ĐỊNH DANH & CƠ BẢN ===
        1. forecast_line_id: ID của dòng dự báo (demand_forecast_details.id)
        2. forecast_number: Số dự báo (fc_number)
        3. customer_po_number: Số PO của khách hàng
        4. creator: Email người tạo dự báo
        5. forecast_date: Ngày tạo dự báo (fc_date)
        
        === THÔNG TIN KHÁCH HÀNG ===
        6. customer: Tên khách hàng tiếng Anh (buyer.english_name)
        7. customer_code: Mã khách hàng (buyer.company_code)
        8. registration_code: Mã đăng ký doanh nghiệp
        9. local_name: Tên khách hàng bằng tiếng địa phương
        10. customer_id: ID khách hàng (buyer.id)
        
        === THÔNG TIN NGƯỜI BÁN ===
        11. legal_entity: Tên pháp nhân bán hàng (seller.english_name)
        12. entity_id: ID pháp nhân bán (seller.id)
        
        === THÔNG TIN SẢN PHẨM ===
        13. product_name: Tên sản phẩm
        14. product_id: ID sản phẩm
        15. brand: Tên thương hiệu (brand_name)
        16. pt_code: Mã PT của sản phẩm
        17. package_size: Kích thước đóng gói
        18. standard_uom: Đơn vị tính chuẩn của sản phẩm
        19. hs_code: Mã HS (Harmonized System) cho hải quan
        20. shelf_life: Hạn sử dụng (kết hợp số và đơn vị thời gian)
        21. storage_condition: Điều kiện bảo quản
        22. vietnamese_name: Tên sản phẩm tiếng Việt (vn_custom_name)
        23. legacy_code: Mã sản phẩm cũ (legacy_pt_code)
        24. customer_product_code: Mã sản phẩm của khách hàng
        
        === ĐƠN VỊ & GIÁ CẢ ===
        25. selling_uom: Đơn vị bán hàng
        26. uom_conversion: Tỷ lệ quy đổi giữa selling_uom và standard_uom
        27. selling_unit_price: Đơn giá bán (theo selling_uom)
        28. currency: Mã tiền tệ (currency.code)
        
        === TỶ GIÁ ===
        29. usd_exchange_rate: Tỷ giá trung bình USD sang currency
        30. standard_unit_price_usd: Đơn giá chuẩn tính bằng USD
        
        === SỐ LƯỢNG & THUẾ ===
        31. vat_percent: Phần trăm thuế VAT
        32. standard_quantity: Số lượng theo đơn vị chuẩn
        33. selling_quantity: Số lượng theo đơn vị bán
        
        === TÍNH TOÁN GIÁ TRỊ ===
        34. total_amount: Tổng giá trị (selling_unit_price × selling_quantity)
        35. total_amount_usd: Tổng giá trị tính bằng USD
        
        === ĐIỀU KHOẢN GIAO HÀNG & THANH TOÁN ===
        36. etd: Ngày dự kiến giao hàng (Expected Time of Delivery)
        37. payment_term: Điều khoản thanh toán (payment_terms.name)
        38. delivery_term: Điều khoản giao hàng (trade_terms.name)
        
        === TRẠNG THÁI CHUYỂN ĐỔI ===
        39. is_converted_to_oc: Đã chuyển thành Order Confirmation chưa ('Yes'/'No')
        
        Lưu ý:
        - View lấy tất cả dòng forecast không bị xóa (delete_flag = 0)
        - Tỷ giá USD được tính trung bình từ bảng exchange_rates
        - Trường is_converted_to_oc giúp phân biệt forecast đã được chuyển thành đơn hàng thực tế
        - standard_unit_price_usd được tính từ unit_price chia cho tỷ giá
        """
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.customer_demand_forecast_full_view;"
        df = pd.read_sql(text(query), engine)
        
        print("Loaded customer demand forecast data successfully ")
        print(df.info())

        return df
    

    @st.cache_data(ttl=1800)
    def load_inventory(_self):
        """
        Load current inventory data
        
        Hàm này load dữ liệu tồn kho hiện tại từ view inventory_detailed_view
        Chỉ lấy các dòng còn tồn kho (remain > 0) với type là stockIn, stockInOpeningBalance hoặc stockInProduction
        
        Các trường dữ liệu được load:
        
        === THÔNG TIN INVENTORY ===
        1. inventory_history_id: ID của lịch sử tồn kho (inventory_histories.id)
        2. product_id: ID sản phẩm
        
        === THÔNG TIN SẢN PHẨM ===
        3. product_name: Tên sản phẩm (products.name)
        4. description: Mô tả sản phẩm
        5. package_size: Kích thước đóng gói
        6. standard_uom: Đơn vị tính chuẩn (products.uom)
        7. pt_code: Mã PT của sản phẩm
        8. legacy_code: Mã sản phẩm cũ (legacy_pt_code)
        9. brand: Tên thương hiệu (brand_name)
        
        === THÔNG TIN LÔ HÀNG ===
        10. batch_number: Số lô (batch_no)
        11. expiry_date: Ngày hết hạn (định dạng YYYY-MM-DD)
        
        === THÔNG TIN KHO & VỊ TRÍ ===
        12. warehouse_name: Tên kho (warehouses.name)
        13. warehouse_owner_id: ID công ty sở hữu kho (warehouses.company_id)
        14. warehouse_owner_name: Tên công ty sở hữu kho
        15. location: Vị trí lưu trữ (Zone-Rack-Bin hoặc legacy location)
        
        === THÔNG TIN SỐ LƯỢNG ===
        16. initial_stock_in_quantity: Số lượng nhập kho ban đầu (quantity)
        17. remaining_quantity: Số lượng còn lại hiện tại (remain)
            
        === THÔNG TIN CÔNG TY SỞ HỮU HÀNG ===
        18. owning_company_id: ID công ty sở hữu hàng hóa
                            - stockInOpeningBalance: luôn = 1
                            - stockIn: lấy từ stock_in.company_id (có thể NULL)
                            - stockInProduction: lấy từ manufacturing_orders.entity_id (có thể NULL)
                            - NULL: Cần kiểm tra và bổ sung thông tin
        19. owning_company_name: Tên công ty sở hữu hàng hóa (NULL nếu owning_company_id là NULL)
        20. same_owner_flag: Cờ kiểm tra sở hữu
                            - '✔ Same': Cùng chủ sở hữu kho và hàng
                            - '⚠ Mismatch': Khác chủ sở hữu
                            - NULL: Thiếu thông tin (khi owning_company_id hoặc warehouse_owner_id là NULL)
        
        === THÔNG TIN GIÁ TRỊ ===
        21. average_landed_cost_usd: Chi phí trung bình đã bao gồm các chi phí phụ (USD/đơn vị)
        22. inventory_value_usd: Giá trị tồn kho (remaining_quantity × average_landed_cost_usd)
        
        === THÔNG TIN THỜI GIAN ===
        23. days_in_warehouse: Số ngày lưu kho (tính từ created_date đến hiện tại)
        
        Lưu ý:
        - View chỉ lấy các dòng có tồn kho thực tế (remain > 0)
        - Xét ba loại nhập kho:
        + stockIn: Nhập kho thông thường từ mua hàng
        + stockInOpeningBalance: Tồn đầu kỳ
        + stockInProduction: Nhập kho từ sản xuất
        - Giá trị tồn kho được tính theo average landed cost từ view avg_landed_cost_looker_view
        - Location có thể là format mới (Zone-Rack-Bin) hoặc legacy location cho dữ liệu cũ
        - Dữ liệu được sắp xếp theo tên sản phẩm (product_name ASC)
        """
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.inventory_detailed_view"
        df = pd.read_sql(text(query), engine)

        print("Loaded inventory data successfully ")
        print(df.info())

        return df


    @st.cache_data(ttl=1800)
    def load_pending_can(_self):
        """
        Load pending CAN (Container Arrival Note) data
        
        Hàm này load dữ liệu các mặt hàng đang chờ nhập kho từ Container Arrival Notes
        Chỉ hiển thị các items chưa được nhập kho hoàn toàn (arrival_quantity > total_stocked_in)
        
        Các trường dữ liệu được load:
        
        === THÔNG TIN CAN (Container Arrival Note) ===
        1. arrival_note_number: Số phiếu nhận hàng
        2. creator: Email người tạo CAN
        3. can_line_id: ID chi tiết dòng CAN (arrival_details.id)
        4. arrival_date: Ngày hàng đến (định dạng DATE)
        
        === THÔNG TIN ĐƠN MUA HÀNG (PO) ===
        5. po_number: Số đơn mua hàng
        6. vendor: Nhà cung cấp (shipper.english_name)
        7. consignee: Bên nhận hàng (consignee.english_name)
        
        === THÔNG TIN SẢN PHẨM ===
        8. product_name: Tên sản phẩm
        9. brand: Tên thương hiệu (brand_name)
        10. package_size: Kích thước đóng gói
        11. pt_code: Mã PT của sản phẩm
        12. hs_code: Mã HS cho hải quan
        13. shelf_life: Hạn sử dụng (format: số + đơn vị + '(s)')
        14. standard_uom: Đơn vị tính chuẩn
        
        === SỐ LƯỢNG & ĐƠN VỊ ===
        15. buying_uom: Đơn vị mua hàng (purchaseuom)
        16. uom_conversion: Tỷ lệ quy đổi giữa buying_uom và standard_uom
        17. buying_quantity: Số lượng mua (theo buying_uom)
        18. standard_quantity: Số lượng chuẩn (theo standard_uom)
        
        === THÔNG TIN CHI PHÍ ===
        19. buying_unit_cost: Đơn giá mua (format: giá + currency code)
        20. standard_unit_cost: Đơn giá chuẩn (format: giá + currency code)
        21. landed_cost: Chi phí đã đến kho (format: giá + currency code hoặc 'N/A')
        22. usd_landed_cost_currency_exchange_rate: Tỷ giá USD sang landed cost currency
        23. landed_cost_usd: Chi phí đã đến kho tính bằng USD (NULL nếu không có tỷ giá hợp lệ)
        
        === LUỒNG SỐ LƯỢNG ===
        24. arrival_quantity: Số lượng đã đến kho
        25. total_stocked_in: Tổng số lượng đã nhập kho
        
        === PHÂN TÍCH HÀNG CHỜ NHẬP ===
        26. pending_quantity: Số lượng chờ nhập kho (arrival_quantity - total_stocked_in)
        27. pending_value_usd: Giá trị hàng chờ nhập tính bằng USD (NULL nếu không có tỷ giá)
        28. pending_percent: Phần trăm hàng chờ nhập (pending_quantity / arrival_quantity * 100)
        29. days_since_arrival: Số ngày kể từ khi hàng đến
        
        === TRẠNG THÁI ===
        30. stocked_in_status: Trạng thái nhập kho (luôn là 'pending' trong view này)
        31. can_status: Trạng thái tổng thể của CAN
                    - 'stocked_in': Đã nhập kho hoàn toàn
                    - 'pending': Đang chờ xử lý
                    - 'partially_stocked_in': Đã nhập kho một phần
                    - 'warehouse_arrival': Đã đến kho
                    - 'on_delivery': Đang vận chuyển
        
        Lưu ý:
        - View chỉ hiển thị các items chưa nhập kho hoàn toàn (có pending quantity > 0)
        - Sử dụng CTEs để tối ưu performance: pre-aggregate stocked quantities và pre-filter pending items
        - An toàn xử lý NULL và division by zero cho các phép tính
        - Tỷ giá USD được kiểm tra > 0 trước khi tính toán
        - Dữ liệu sắp xếp theo arrival_note_number DESC, can_line_id ASC
        """
        engine = get_db_engine()
        query = "SELECT * FROM prostechvn.can_pending_stockin_view"
        df = pd.read_sql(text(query), engine)

        print("Loaded pending can data successfully ")
        print(df.info())

        return df
            

    @st.cache_data(ttl=1800)
    def load_pending_po(_self):
        """
        Hàm này load dữ liệu các dòng PO chưa nhận hàng đầy đủ từ view purchase_order_full_view
        Chỉ lấy các dòng có pending_standard_arrival_quantity > 0 (còn hàng chưa nhận)
        
        Các trường dữ liệu được load:
        
        === THÔNG TIN ĐỊNH DANH PO ===
        1. po_line_id: ID dòng PO (product_purchase_orders.id)
        2. po_number: Số PO nội bộ
        3. external_ref_number: Số PO từ hệ thống khác (nếu có)
        4. po_date: Ngày tạo PO (format YYYY-MM-DD)
        5. created_by: Email người tạo PO
        
        === THÔNG TIN NHÀ CUNG CẤP & PHÁP NHÂN ===
        6. vendor_name: Tên nhà cung cấp (seller.english_name)
        7. legal_entity: Tên pháp nhân mua hàng (buyer.english_name)
        
        === THÔNG TIN SẢN PHẨM ===
        8. product_name: Tên sản phẩm
        9. pt_code: Mã PT của sản phẩm
        10. brand: Tên thương hiệu (brand_name)
        11. package_size: Kích thước đóng gói
        12. hs_code: Mã HS cho hải quan
        13. vn_custom_name: Tên sản phẩm cho hải quan VN
        14. legacy_pt_code: Mã sản phẩm cũ
        15. vendor_product_code: Mã sản phẩm của nhà cung cấp
        16. shelf_life: Hạn sử dụng (format: số + đơn vị + '(s)')
        17. storage_condition: Điều kiện bảo quản
        
        === ĐƠN VỊ & CHUYỂN ĐỔI ===
        18. standard_uom: Đơn vị tính chuẩn
        19. buying_uom: Đơn vị mua hàng (purchaseuom)
        20. uom_conversion: Tỷ lệ chuyển đổi giữa buying_uom và standard_uom
        
        === SỐ LƯỢNG & GIÁ ===
        21. moq: Số lượng đặt hàng tối thiểu (minimum_order_quantity)
        22. spq: Số lượng đóng gói chuẩn (standard_pack_quantity)
        23. buying_quantity: Số lượng mua (theo buying_uom)
        24. standard_quantity: Số lượng chuẩn (theo standard_uom)
        25. purchase_unit_cost: Đơn giá mua (theo buying_uom)
        26. standard_unit_cost: Đơn giá chuẩn (theo standard_uom)
        
        === TỔNG GIÁ TRỊ PO ===
        27. total_amount: Tổng giá trị PO (purchase_unit_cost × buying_quantity)
        28. currency: Mã tiền tệ
        29. usd_exchange_rate: Tỷ giá USD
        30. total_amount_usd: Tổng giá trị PO tính bằng USD
        
        === THÔNG TIN NHẬN HÀNG & HÓA ĐƠN ===
        31. total_standard_arrived_quantity: Tổng số lượng đã nhận (standard UOM)
        32. total_buying_invoiced_quantity: Tổng số lượng đã xuất hóa đơn (buying UOM)
        33. last_invoice_date: Ngày xuất hóa đơn gần nhất
        34. ci_numbers: Danh sách số Commercial Invoice (cách nhau bởi dấu phẩy)
        
        === SỐ LƯỢNG PENDING ===
        35. pending_standard_arrival_quantity: Số lượng chờ nhận hàng (standard UOM)
        36. pending_buying_invoiced_quantity: Số lượng chờ xuất hóa đơn (buying UOM)
        
        === GIÁ TRỊ ĐÃ XỬ LÝ & CÒN LẠI ===
        37. invoiced_amount_usd: Giá trị đã xuất hóa đơn (USD)
        38. outstanding_invoiced_amount_usd: Giá trị chưa xuất hóa đơn (USD)
        39. arrival_amount_usd: Giá trị hàng đã nhận (USD)
        40. outstanding_arrival_amount_usd: Giá trị hàng chưa nhận (USD)
        
        === NGÀY DỰ KIẾN ===
        41. etd: Ngày giao hàng dự kiến (Expected Time of Delivery)
        42. eta: Ngày hàng đến dự kiến (Expected Time of Arrival)
        
        === ĐIỀU KHOẢN ===
        43. payment_term: Điều khoản thanh toán
        44. trade_term: Điều khoản thương mại
        45. vat_gst_percent: Phần trăm thuế VAT/GST
        
        === TRẠNG THÁI ===
        46. status: Trạng thái tổng thể của dòng PO
                    - 'COMPLETED': Đã nhận và xuất hóa đơn đầy đủ
                    - 'PENDING': Chưa nhận và chưa xuất hóa đơn
                    - 'PENDING_INVOICING': Đã nhận nhưng chưa xuất hóa đơn
                    - 'PENDING_RECEIPT': Đã xuất hóa đơn nhưng chưa nhận
                    - 'IN_PROCESS': Đang xử lý (nhận hoặc xuất hóa đơn một phần)
                    - 'PARTIAL_STATUS': Trạng thái khác
        
        Lưu ý:
        - Chỉ lấy các dòng PO có pending_standard_arrival_quantity > 0 (còn hàng chưa nhận)
        - Sử dụng CTE để tối ưu performance khi aggregate arrival và invoice data
        - Tránh duplicate khi JOIN nhiều bảng bằng cách aggregate riêng
        - Dữ liệu được sắp xếp theo thời gian tạo mới nhất (created_date DESC)
        - Không aggregate các dòng PO - giữ nguyên từng dòng chi tiết
        """
        engine = get_db_engine()
        query = """
        SELECT * FROM prostechvn.purchase_order_full_view
        WHERE pending_standard_arrival_quantity > 0
        """
        df = pd.read_sql(text(query), engine)

        print("Loaded pending po data successfully ")
        print(df.info())

        return df


    @st.cache_data(ttl=1800)
    def load_pending_wh_transfer(_self):
        """
        Load pending Warehouse Transfer data
        
        Hàm này load dữ liệu các lệnh chuyển kho đang chờ xử lý từ view warehouse_transfer_details_view
        Chỉ lấy các lệnh chuyển kho chưa hoàn thành (is_completed = 0)
        
        Các trường dữ liệu được load:
        
        === THÔNG TIN CHUYỂN KHO ===
        1. warehouse_transfer_line_id: ID chi tiết dòng chuyển kho (stock_out_warehouse_transfer_details.id)
        2. transfer_date: Ngày tạo lệnh chuyển kho (stock_out_warehouse_transfer.created_date)
        
        === THÔNG TIN SẢN PHẨM ===
        3. product_id: ID sản phẩm
        4. product_name: Tên sản phẩm
        5. product_description: Mô tả sản phẩm
        6. package_size: Kích thước đóng gói
        7. standard_uom: Đơn vị tính chuẩn
        8. pt_code: Mã PT của sản phẩm
        9. legacy_code: Mã sản phẩm cũ (legacy_pt_code)
        10. brand: Tên thương hiệu (brand_name)
        11. batch_number: Số lô hàng (inventory_histories.batch_no)
        12. expiry_date: Ngày hết hạn (format YYYY-MM-DD)
        
        === THÔNG TIN CÔNG TY SỞ HỮU ===
        13. owning_company_id: ID công ty sở hữu hàng hóa
        14. owning_company_name: Tên công ty sở hữu hàng hóa
        
        === SỐ LƯỢNG & GIÁ TRỊ ===
        15. transfer_quantity: Số lượng chuyển kho
        16. average_landed_cost_usd: Chi phí trung bình đã bao gồm các chi phí phụ (USD/đơn vị)
        17. warehouse_transfer_value_usd: Giá trị hàng chuyển kho (transfer_quantity × average_landed_cost_usd)
        
        === THÔNG TIN KHO ===
        18. from_warehouse: Tên kho chuyển đi (từ inventory_histories.warehouse_id)
        19. to_warehouse: Tên kho nhận (từ stock_out_warehouse_transfer_details.to_warehouse_id)
        
        === TRẠNG THÁI ===
        20. is_completed: Trạng thái hoàn thành
                        - 1: Đã hoàn thành chuyển kho
                        - 0: Đang trong quá trình chuyển kho
        
        Lưu ý:
        - View chỉ lấy các lệnh chuyển kho chưa hoàn thành (is_completed = 0)
        - Giá trị chuyển kho được tính theo average landed cost từ view avg_landed_cost_looker_view
        - Thông tin batch và expiry date được lấy từ inventory_histories của hàng được chuyển
        - Mỗi dòng đại diện cho một chi tiết chuyển kho (một sản phẩm trong lệnh chuyển)
        - Chỉ lấy các inventory histories không bị xóa (delete_flag = 0)
        """
        engine = get_db_engine()
        query = """
        SELECT * FROM prostechvn.warehouse_transfer_details_view wtdv
        WHERE wtdv.is_completed = 0
        """
        df = pd.read_sql(text(query), engine)
        
        print("Loaded pending wh transfer data successfully ")
        print(df.info())

        return df


    @st.cache_data(ttl=3600)
    def load_product_master(_self):
        """Load product master data"""
        engine = get_db_engine()
        query = """
        SELECT 
            p.id as product_id,
            p.pt_code,
            p.name as product_name,
            p.description,
            b.brand_name as brand,
            p.package_size,
            p.uom as standard_uom,
            p.shelf_life,
            p.shelf_life_uom,
            p.storage_condition,
            p.hs_code,
            p.legacy_pt_code
        FROM products p
        LEFT JOIN brands b ON p.brand_id = b.id
        WHERE p.delete_flag = 0
        """
        df = pd.read_sql(text(query), engine)

        print("Loaded product master data successfully ")
        print(df.info())

        return df


    @st.cache_data(ttl=3600)
    def load_customer_master(_self):
        """
        Load customer master data
        
        Hàm này load dữ liệu tổng hợp khách hàng từ view customer_master_view
        Chỉ lấy các companies có type là 'Customer' và không bị xóa (delete_flag = 0)
        
        Các trường dữ liệu được load:
        
        === THÔNG TIN KHÁCH HÀNG CƠ BẢN ===
        1. customer_id: ID khách hàng (companies.id)
        2. customer_name: Tên khách hàng tiếng Anh (english_name)
        3. customer_code: Mã khách hàng (company_code)
        4. registration_code: Mã đăng ký doanh nghiệp
        5. local_name: Tên khách hàng bằng tiếng địa phương
        
        === THÔNG TIN HẠN MỨC TÍN DỤNG ===
        6. credit_limit: Hạn mức tín dụng (từ term_and_conditions.limit_credit)
        7. credit_limit_currency: Mã tiền tệ của hạn mức tín dụng
        8. payment_term_days: Điều khoản thanh toán (payment_terms.name)
        
        === THÔNG TIN TỶ GIÁ & QUY ĐỔI ===
        9. usd_exchange_rate: Tỷ giá trung bình USD sang credit_limit_currency
                            - Lấy từ bảng exchange_rates với from_currency_code = 'USD'
                            - Mặc định = 1 nếu không có tỷ giá
        10. credit_limit_usd: Hạn mức tín dụng quy đổi sang USD
                            - Nếu currency là USD: giữ nguyên credit_limit
                            - Nếu currency khác: credit_limit / usd_exchange_rate
        
        === THÔNG TIN HỆ THỐNG ===
        11. created_date: Ngày tạo khách hàng
        12. modified_date: Ngày cập nhật gần nhất
        13. delete_flag: Cờ xóa (luôn = 0 trong view này)
        
        Lưu ý:
        - View chỉ lấy companies có company_type = 'Customer'
        - JOIN với term_and_conditions để lấy thông tin credit limit và payment terms
        - Tỷ giá USD được tính trung bình từ tất cả exchange rates không bị xóa
        - Credit limit USD được tính toán tự động dựa trên tỷ giá
        - Cache time dài hơn (3600s = 1 giờ) vì dữ liệu master ít thay đổi
        """
        engine = get_db_engine()
        query = """SELECT * FROM prostechvn.customer_master_view;
        """
        df = pd.read_sql(text(query), engine)
        
        print("Loaded customer master data successfully ")
        print(df.info())


        return df


    def _is_bulk_cache_valid(self) -> bool:
        """Check if bulk cache is still valid"""
        if not self._cache:
            return False
        
        # Check age of oldest cache entry
        if self._last_refresh:
            oldest = min(self._last_refresh.values())
            elapsed = (datetime.now() - oldest).seconds
            return elapsed < self._cache_ttl
        
        return False

    def _standardize_demand_df(self, df: pd.DataFrame, is_forecast: bool) -> pd.DataFrame:
        """Standardize demand dataframe"""
        df = df.copy()
        
        # Date columns
        if 'etd' in df.columns:
            df["etd"] = pd.to_datetime(df["etd"], errors="coerce")
            
        
        if 'oc_date' in df.columns:
            df["oc_date"] = pd.to_datetime(df["oc_date"], errors="coerce")
        
        # Quantity and value columns - check existence first
        if is_forecast:
            if 'standard_quantity' in df.columns:
                df['demand_quantity'] = pd.to_numeric(df['standard_quantity'], errors='coerce').fillna(0)
            else:
                df['demand_quantity'] = 0
                
            if 'total_amount_usd' in df.columns:
                df['value_in_usd'] = pd.to_numeric(df['total_amount_usd'], errors='coerce').fillna(0)
            else:
                df['value_in_usd'] = 0
                
            df['demand_number'] = df.get('forecast_number', '')
            
            # Handle is_converted_to_oc properly - preserve original value
            if 'is_converted_to_oc' in df.columns:
                # Just ensure it's string type for consistent comparison
                df['is_converted_to_oc'] = df['is_converted_to_oc'].astype(str).str.strip()
            else:
                df['is_converted_to_oc'] = 'No'
            
            if 'forecast_line_id' in df.columns:
                df['demand_line_id'] = df['forecast_line_id'].astype(str) + '_FC'
            else:
                df['demand_line_id'] = ''
        else:
            if 'pending_standard_delivery_quantity' in df.columns:
                df['demand_quantity'] = pd.to_numeric(df['pending_standard_delivery_quantity'], errors='coerce').fillna(0)
            else:
                df['demand_quantity'] = 0
                
            if 'outstanding_amount_usd' in df.columns:
                df['value_in_usd'] = pd.to_numeric(df['outstanding_amount_usd'], errors='coerce').fillna(0)
            else:
                df['value_in_usd'] = 0
                
            df['demand_number'] = df.get('oc_number', '')
            df['is_converted_to_oc'] = 'N/A'
            
            if 'ocd_id' in df.columns:
                df['demand_line_id'] = df['ocd_id'].astype(str) + '_OC'
            else:
                df['demand_line_id'] = ''
        
        # Clean string columns - only if they exist
        string_cols = ['product_name', 'pt_code', 'brand', 'legal_entity', 'customer']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            else:
                df[col] = ''  # Create column with empty string if not exists
        
        # UOM and package size
        if 'standard_uom' in df.columns:
            df['standard_uom'] = df['standard_uom'].astype(str).str.strip().str.upper()
        else:
            df['standard_uom'] = ''
            
        if 'package_size' in df.columns:
            df['package_size'] = df['package_size'].astype(str).str.strip()
        else:
            df['package_size'] = ''
        
        return df


    def _prepare_inventory_data(self, inv_df: pd.DataFrame, today: pd.Timestamp, exclude_expired: bool) -> pd.DataFrame:
        """Prepare inventory data - inventory is available NOW"""
        inv_df = inv_df.copy()
        inv_df["source_type"] = "Inventory"
        
        # IMPORTANT: Inventory is available NOW - use today as date_ref
        inv_df["date_ref"] = today
        
        # Keep created_date for reference (this is the only date from SQL view)
        if 'created_date' in inv_df.columns:
            inv_df["created_date"] = pd.to_datetime(inv_df["created_date"], errors='coerce')
        
        # Map columns with existence check
        if 'remaining_quantity' in inv_df.columns:
            inv_df["quantity"] = pd.to_numeric(inv_df["remaining_quantity"], errors="coerce").fillna(0)
        else:
            inv_df["quantity"] = 0
            
        if 'inventory_value_usd' in inv_df.columns:
            inv_df["value_in_usd"] = pd.to_numeric(inv_df["inventory_value_usd"], errors="coerce").fillna(0)
        else:
            inv_df["value_in_usd"] = 0
            
        if 'owning_company_name' in inv_df.columns:
            inv_df["legal_entity"] = inv_df["owning_company_name"]
        else:
            inv_df["legal_entity"] = ''
            
        if 'expiry_date' in inv_df.columns:
            inv_df["expiry_date"] = pd.to_datetime(inv_df["expiry_date"], errors="coerce")
            inv_df["days_until_expiry"] = (inv_df["expiry_date"] - today).dt.days
            
            if exclude_expired:
                inv_df = inv_df[(inv_df["expiry_date"].isna()) | (inv_df["expiry_date"] >= today)]
        
        # Add supply number for tracking
        if 'inventory_history_id' in inv_df.columns:
            inv_df["supply_number"] = inv_df["inventory_history_id"].astype(str)
        else:
            inv_df["supply_number"] = ''
        
        return inv_df


    def _prepare_can_data(self, can_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare CAN data"""
        can_df = can_df.copy()
        can_df["source_type"] = "Pending CAN"
        
        # arrival_date exists in SQL view
        if 'arrival_date' in can_df.columns:
            can_df["arrival_date"] = pd.to_datetime(can_df["arrival_date"], errors="coerce")
            can_df["date_ref"] = can_df["arrival_date"]
        else:
            can_df["date_ref"] = pd.NaT
            logger.warning("No arrival_date column in CAN data")
            
        # days_since_arrival already calculated in SQL
        
        if 'pending_quantity' in can_df.columns:
            can_df["quantity"] = pd.to_numeric(can_df["pending_quantity"], errors="coerce").fillna(0)
        else:
            can_df["quantity"] = 0
            
        if 'pending_value_usd' in can_df.columns:
            can_df["value_in_usd"] = pd.to_numeric(can_df["pending_value_usd"], errors="coerce").fillna(0)
        else:
            can_df["value_in_usd"] = 0
            
        if 'consignee' in can_df.columns:
            can_df["legal_entity"] = can_df["consignee"]
        else:
            can_df["legal_entity"] = ''
            
        if 'arrival_note_number' in can_df.columns:
            can_df["supply_number"] = can_df["arrival_note_number"].astype(str)
        else:
            can_df["supply_number"] = ''
            
        return can_df

    def _prepare_po_data(self, po_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare PO data"""
        po_df = po_df.copy()
        po_df["source_type"] = "Pending PO"
        
        # eta exists in SQL view
        if 'eta' in po_df.columns:
            po_df["eta"] = pd.to_datetime(po_df["eta"], errors="coerce")
            po_df["date_ref"] = po_df["eta"]
        else:
            po_df["date_ref"] = pd.NaT
            logger.warning("No eta column in PO data")
            
        if 'pending_standard_arrival_quantity' in po_df.columns:
            po_df["quantity"] = pd.to_numeric(po_df["pending_standard_arrival_quantity"], errors="coerce").fillna(0)
        else:
            po_df["quantity"] = 0
            
        if 'outstanding_arrival_amount_usd' in po_df.columns:
            po_df["value_in_usd"] = pd.to_numeric(po_df["outstanding_arrival_amount_usd"], errors="coerce").fillna(0)
        else:
            po_df["value_in_usd"] = 0
            
        if 'legal_entity' not in po_df.columns:
            po_df["legal_entity"] = ''
            
        # Create unique supply_number combining po_number and line_id
        if 'po_line_id' in po_df.columns and 'po_number' in po_df.columns:
            po_df["supply_number"] = po_df["po_number"].astype(str) + "_L" + po_df["po_line_id"].astype(str)
        elif 'po_number' in po_df.columns:
            po_df["supply_number"] = po_df["po_number"].astype(str) + "_" + po_df.index.astype(str)
        else:
            po_df["supply_number"] = ''
            
        # Add vendor info if available
        if 'vendor_name' in po_df.columns:
            po_df["vendor"] = po_df["vendor_name"]
            
        return po_df

    def _prepare_wh_transfer_data(self, wht_df: pd.DataFrame, today: pd.Timestamp, exclude_expired: bool) -> pd.DataFrame:
        """Prepare warehouse transfer data"""
        wht_df = wht_df.copy()
        wht_df["source_type"] = "Pending WH Transfer"
        
        # transfer_date exists in SQL view
        if 'transfer_date' in wht_df.columns:
            wht_df["transfer_date"] = pd.to_datetime(wht_df["transfer_date"], errors="coerce")
            wht_df["date_ref"] = wht_df["transfer_date"]
            
            # Calculate days in transfer
            wht_df["days_in_transfer"] = (today - wht_df["transfer_date"]).dt.days
            wht_df["days_in_transfer"] = wht_df["days_in_transfer"].fillna(0)
        else:
            wht_df["date_ref"] = pd.NaT
            logger.warning("No transfer_date column in WH Transfer data")
            
        if 'transfer_quantity' in wht_df.columns:
            wht_df["quantity"] = pd.to_numeric(wht_df["transfer_quantity"], errors="coerce").fillna(0)
        else:
            wht_df["quantity"] = 0
            
        if 'warehouse_transfer_value_usd' in wht_df.columns:
            wht_df["value_in_usd"] = pd.to_numeric(wht_df["warehouse_transfer_value_usd"], errors="coerce").fillna(0)
        else:
            wht_df["value_in_usd"] = 0
            
        if 'owning_company_name' in wht_df.columns:
            wht_df["legal_entity"] = wht_df["owning_company_name"]
        else:
            wht_df["legal_entity"] = ''
            
        if 'warehouse_transfer_line_id' in wht_df.columns:
            wht_df["supply_number"] = wht_df["warehouse_transfer_line_id"].astype(str)
        else:
            wht_df["supply_number"] = ''
            
        # Handle expiry date
        if 'expiry_date' in wht_df.columns:
            wht_df["expiry_date"] = pd.to_datetime(wht_df["expiry_date"], errors="coerce")
            
            if exclude_expired:
                wht_df = wht_df[(wht_df["expiry_date"].isna()) | (wht_df["expiry_date"] >= today)]
        
        # Add transfer route info
        if 'from_warehouse' in wht_df.columns and 'to_warehouse' in wht_df.columns:
            wht_df["transfer_route"] = wht_df["from_warehouse"] + " → " + wht_df["to_warehouse"]
            
        return wht_df


  # utils/data_manager.py - Add this method to DataManager class

    def _standardize_supply_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize supply dataframe - preserve source-specific date columns"""
        df = df.copy()
        
        # Clean string columns - only if they exist
        string_cols = ["pt_code", "product_name", "brand"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            else:
                df[col] = ''
        
        # UOM and package size
        if 'standard_uom' in df.columns:
            df["standard_uom"] = df["standard_uom"].astype(str).str.strip().str.upper()
        else:
            df["standard_uom"] = ''
            
        if 'package_size' in df.columns:
            df["package_size"] = df["package_size"].astype(str).str.strip()
        else:
            df["package_size"] = ''
        
        # Ensure required columns exist
        if "value_in_usd" not in df.columns:
            df["value_in_usd"] = 0
            
        if "quantity" not in df.columns:
            df["quantity"] = 0
            
        if "legal_entity" not in df.columns:
            df["legal_entity"] = ''
            
        if "date_ref" not in df.columns:
            df["date_ref"] = pd.NaT
            
        if "source_type" not in df.columns:
            df["source_type"] = 'Unknown'
        
        # Select standard columns - ensure they all exist
        standard_cols = [
            "source_type", "pt_code", "product_name", "brand", 
            "package_size", "standard_uom", "legal_entity", "date_ref", 
            "quantity", "value_in_usd"
        ]
        
        # Add optional columns if they exist - INCLUDING unified_date columns
        optional_cols = [
            "supply_number", "expiry_date", "days_until_expiry", 
            "days_since_arrival", "vendor", "transfer_route", 
            "days_in_transfer", "from_warehouse", "to_warehouse",
            # Source-specific date columns with original and adjusted format
            "arrival_date", "arrival_date_original", "arrival_date_adjusted",
            "eta", "eta_original", "eta_adjusted", 
            "transfer_date", "transfer_date_original", "transfer_date_adjusted",
            "date_ref_original", "date_ref_adjusted",
            # Unified date columns for All view - IMPORTANT!
            "unified_date", "unified_date_adjusted",
            # Keep these for inventory date selection logic
            "receipt_date", "stockin_date", "created_date",
            # PO specific columns
            "po_number", "po_line_id", "buying_quantity", "buying_uom", 
            "purchase_unit_cost", "vendor_name"
        ]
        
        # Create final columns list without duplicates
        final_cols = standard_cols.copy()
        for col in optional_cols:
            if col in df.columns and col not in final_cols:
                final_cols.append(col)
        
        return df[final_cols]


    def _calculate_insights(self):
        """Calculate key insights from loaded data"""
        insights = {}
        settings = self._get_settings_manager()
        
        # Get thresholds from settings
        critical_days = settings.get_setting('alert_thresholds.critical_shortage_days', 3)
        warning_days = settings.get_setting('alert_thresholds.warning_shortage_days', 7)
        excess_months = settings.get_setting('alert_thresholds.excess_inventory_months', 6)
        shelf_life_threshold = settings.get_setting('business_rules.shelf_life_threshold_days', 30)
        
        # Demand insights
        if 'demand_oc' in self._cache and not self._cache['demand_oc'].empty:
            oc_df = self._cache['demand_oc'].copy()
            
            # Check if required columns exist
            if 'etd' in oc_df.columns:
                oc_df['etd'] = pd.to_datetime(oc_df['etd'], errors='coerce')
            
            insights['demand_oc_pending_count'] = len(oc_df)
            
            if 'outstanding_amount_usd' in oc_df.columns:
                insights['demand_oc_pending_value'] = oc_df['outstanding_amount_usd'].sum()
            else:
                insights['demand_oc_pending_value'] = 0
            
            if 'etd' in oc_df.columns:
                insights['demand_missing_etd'] = len(oc_df[oc_df['etd'].isna()])
                
                # Overdue analysis - use normalized dates for comparison
                today = pd.Timestamp.now().normalize()
                overdue_mask = oc_df['etd'] < today
                insights['demand_overdue_count'] = len(oc_df[overdue_mask])
                
                if 'outstanding_amount_usd' in oc_df.columns:
                    insights['demand_overdue_value'] = oc_df[overdue_mask]['outstanding_amount_usd'].sum()
                else:
                    insights['demand_overdue_value'] = 0
                
                # Critical shortage analysis
                critical_date = today + timedelta(days=critical_days)
                critical_mask = (oc_df['etd'] <= critical_date) & (~overdue_mask)
                insights['critical_shortage_count'] = len(oc_df[critical_mask])
                
                if 'outstanding_amount_usd' in oc_df.columns:
                    insights['critical_shortage_value'] = oc_df[critical_mask]['outstanding_amount_usd'].sum()
                else:
                    insights['critical_shortage_value'] = 0
            else:
                insights['demand_missing_etd'] = 0
                insights['demand_overdue_count'] = 0
                insights['demand_overdue_value'] = 0
                insights['critical_shortage_count'] = 0
                insights['critical_shortage_value'] = 0
        
        # Supply insights
        if 'supply_inventory' in self._cache and not self._cache['supply_inventory'].empty:
            inv_df = self._cache['supply_inventory'].copy()
            
            if 'expiry_date' in inv_df.columns:
                inv_df['expiry_date'] = pd.to_datetime(inv_df['expiry_date'], errors='coerce')
            
            if 'inventory_value_usd' in inv_df.columns:
                insights['inventory_total_value'] = inv_df['inventory_value_usd'].sum()
            else:
                insights['inventory_total_value'] = 0
            
            # Expiry analysis
            if 'expiry_date' in inv_df.columns:
                today = pd.Timestamp.now().normalize()
                expired_mask = inv_df['expiry_date'] < today
                insights['expired_items_count'] = len(inv_df[expired_mask])
                
                if 'inventory_value_usd' in inv_df.columns:
                    insights['expired_items_value'] = inv_df[expired_mask]['inventory_value_usd'].sum()
                else:
                    insights['expired_items_value'] = 0
                
                # Near expiry analysis
                near_expiry_7d_mask = (
                    (inv_df['expiry_date'] >= today) & 
                    (inv_df['expiry_date'] <= today + timedelta(days=7))
                )
                insights['near_expiry_7d_count'] = len(inv_df[near_expiry_7d_mask])
                
                if 'inventory_value_usd' in inv_df.columns:
                    insights['near_expiry_7d_value'] = inv_df[near_expiry_7d_mask]['inventory_value_usd'].sum()
                else:
                    insights['near_expiry_7d_value'] = 0
            else:
                insights['expired_items_count'] = 0
                insights['expired_items_value'] = 0
                insights['near_expiry_7d_count'] = 0
                insights['near_expiry_7d_value'] = 0
            
            # Excess inventory analysis
            if 'days_in_warehouse' in inv_df.columns:
                excess_days = excess_months * 30
                excess_mask = inv_df['days_in_warehouse'] > excess_days
                insights['excess_inventory_count'] = len(inv_df[excess_mask])
                
                if 'inventory_value_usd' in inv_df.columns:
                    insights['excess_inventory_value'] = inv_df[excess_mask]['inventory_value_usd'].sum()
                else:
                    insights['excess_inventory_value'] = 0
            else:
                insights['excess_inventory_count'] = 0
                insights['excess_inventory_value'] = 0
        
        # Product matching insights - ONLY OC vs Inventory for dashboard
        if ('demand_oc' in self._cache and not self._cache['demand_oc'].empty and
            'supply_inventory' in self._cache and not self._cache['supply_inventory'].empty):
            
            demand_df = self._cache['demand_oc']
            supply_df = self._cache['supply_inventory']
            
            # Check if pt_code exists in both dataframes
            if 'pt_code' in demand_df.columns and 'pt_code' in supply_df.columns:
                demand_products = set(demand_df['pt_code'].dropna().unique())
                supply_products = set(supply_df['pt_code'].dropna().unique())
                
                insights['demand_only_products'] = demand_products - supply_products
                insights['supply_only_products'] = supply_products - demand_products
                insights['matched_products'] = demand_products & supply_products
                
                # Calculate values for unmatched products
                if insights['demand_only_products'] and 'outstanding_amount_usd' in demand_df.columns:
                    insights['demand_only_value'] = demand_df[
                        demand_df['pt_code'].isin(insights['demand_only_products'])
                    ]['outstanding_amount_usd'].sum()
                else:
                    insights['demand_only_value'] = 0
                
                if insights['supply_only_products'] and 'inventory_value_usd' in supply_df.columns:
                    insights['supply_only_value'] = supply_df[
                        supply_df['pt_code'].isin(insights['supply_only_products'])
                    ]['inventory_value_usd'].sum()
                else:
                    insights['supply_only_value'] = 0
            else:
                # If pt_code doesn't exist, set empty values
                insights['demand_only_products'] = set()
                insights['supply_only_products'] = set()
                insights['matched_products'] = set()
                insights['demand_only_value'] = 0
                insights['supply_only_value'] = 0
        else:
            # Set default empty values
            insights['demand_only_products'] = set()
            insights['supply_only_products'] = set()
            insights['matched_products'] = set()
            insights['demand_only_value'] = 0
            insights['supply_only_value'] = 0
        
        self._insights_cache = insights
        return insights

    # === Public Data Access Methods ===
    def get_demand_data(self, sources: List[str], include_converted: bool = False) -> pd.DataFrame:
        """Get combined demand data with standardization - SIMPLIFIED"""
        df_parts = []
        
        if "OC" in sources:
            df_oc = self.load_demand_oc()
            if not df_oc.empty:
                df_oc["source_type"] = "OC"
                df_oc = self._standardize_demand_df(df_oc, is_forecast=False)
                df_oc = self._time_adjustment_integration.apply_adjustments(df_oc, "OC")
                df_parts.append(df_oc)
        
        if "Forecast" in sources:
            df_fc = self.load_demand_forecast()
            if not df_fc.empty:
                df_fc["source_type"] = "Forecast"
                standardized_fc = self._standardize_demand_df(df_fc, is_forecast=True)
                standardized_fc = self._time_adjustment_integration.apply_adjustments(standardized_fc, "Forecast")
                
                if not include_converted and 'is_converted_to_oc' in standardized_fc.columns:
                    converted_values = ['Yes', 'yes', 'Y', 'y', '1', 1, True, 'True', 'true']
                    standardized_fc = standardized_fc[~standardized_fc["is_converted_to_oc"].isin(converted_values)]
                
                df_parts.append(standardized_fc)
        
        return pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()

    # Add this fix to get_supply_data method in DataManager
    # This handles the case when some sources return empty dataframes

    def get_supply_data(self, sources: List[str], exclude_expired: bool = True) -> pd.DataFrame:
        """Get combined supply data with standardization"""
        today = pd.to_datetime("today").normalize()
        df_parts = []
        
        if "Inventory" in sources:
            inv_df = self.load_inventory()
            if not inv_df.empty:
                inv_df = self._prepare_inventory_data(inv_df, today, exclude_expired)
                inv_df = self._time_adjustment_integration.apply_adjustments(inv_df, "Inventory")
                df_parts.append(inv_df)
        
        if "Pending CAN" in sources:
            can_df = self.load_pending_can()
            if not can_df.empty:
                can_df = self._prepare_can_data(can_df)
                can_df = self._time_adjustment_integration.apply_adjustments(can_df, "Pending CAN")
                df_parts.append(can_df)
        
        if "Pending PO" in sources:
            po_df = self.load_pending_po()
            if not po_df.empty:
                po_df = self._prepare_po_data(po_df)
                po_df = self._time_adjustment_integration.apply_adjustments(po_df, "Pending PO")
                df_parts.append(po_df)
        
        if "Pending WH Transfer" in sources:
            wht_df = self.load_pending_wh_transfer()
            if not wht_df.empty:
                wht_df = self._prepare_wh_transfer_data(wht_df, today, exclude_expired)
                wht_df = self._time_adjustment_integration.apply_adjustments(wht_df, "Pending WH Transfer")
                df_parts.append(wht_df)
        
        if not df_parts:
            return pd.DataFrame()
        
        # Standardize each part BEFORE concatenating
        standardized_parts = []
        for df in df_parts:
            if not df.empty and len(df.columns) > 0:
                standardized_df = self._standardize_supply_df(df)
                standardized_df = standardized_df.reset_index(drop=True)
                standardized_parts.append(standardized_df)
        
        if not standardized_parts:
            return pd.DataFrame()
        
        # Combine all parts
        try:
            combined_df = pd.concat(standardized_parts, ignore_index=True, sort=False)
            combined_df = combined_df.reset_index(drop=True)
            
            # Create unified date columns if multiple sources
            if len(sources) > 1:
                combined_df = self.create_unified_date_columns(combined_df)
            
            return combined_df
        except Exception as e:
            logger.error(f"Error concatenating supply data: {str(e)}")
            if standardized_parts:
                result = standardized_parts[0].copy()
                for df in standardized_parts[1:]:
                    try:
                        result = pd.concat([result, df], ignore_index=True, sort=False)
                    except:
                        continue
                # Create unified date columns for combined result
                if len(sources) > 1:
                    result = self.create_unified_date_columns(result)
                return result.reset_index(drop=True)
            else:
                return pd.DataFrame()

    def get_adjustment_status(self) -> Dict[str, Any]:
        """Get adjustment status from integration layer"""
        return self._time_adjustment_integration.get_adjustment_status()


    def preload_all_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Parallel loading of all data types - SIMPLIFIED"""
        if not force_refresh and self._is_bulk_cache_valid():
            return self._cache
        
        loading_tasks = {
            'demand_oc': self.load_demand_oc,
            'demand_forecast': self.load_demand_forecast,
            'supply_inventory': self.load_inventory,
            'supply_can': self.load_pending_can,
            'supply_po': self.load_pending_po,
            'supply_wh_transfer': self.load_pending_wh_transfer,
            'master_products': self.load_product_master,
            'master_customers': self.load_customer_master,
        }
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_key = {
                executor.submit(func): key 
                for key, func in loading_tasks.items()
            }
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    self._cache[key] = future.result()
                    self._last_refresh[key] = datetime.now()
                except Exception as e:
                    logger.error(f"Error loading {key}: {str(e)}")
                    self._cache[key] = pd.DataFrame()
        
        self._calculate_insights()
        return self._cache


    def get_insights(self) -> Dict[str, Any]:
        """Get calculated insights across all data"""
        if not self._insights_cache:
            self._calculate_insights()
        return self._insights_cache

    def get_critical_alerts(self) -> List[Dict[str, Any]]:
        """Get critical alerts requiring immediate action"""
        alerts = []
        insights = self.get_insights()
        
        # Demand-only products alert
        if insights.get('demand_only_products'):
            alerts.append({
                'level': 'critical',
                'icon': '📤',
                'message': f"{len(insights.get('demand_only_products', set()))} Demand-Only products",
                'value': f"${insights.get('demand_only_value', 0):,.0f}",
                'action': 'no supply'
            })
        
        # Overdue orders alert
        if insights.get('demand_overdue_count', 0) > 0:
            alerts.append({
                'level': 'critical',
                'icon': '🕐',
                'message': f"{insights['demand_overdue_count']} Past ETD orders",
                'value': f"${insights['demand_overdue_value']:,.0f}",
                'action': 'overdue delivery'
            })
        
        # Expired items alert
        if insights.get('expired_items_count', 0) > 0:
            alerts.append({
                'level': 'critical',
                'icon': '💀',
                'message': f"{insights['expired_items_count']} Expired items",
                'value': f"${insights['expired_items_value']:,.0f}",
                'action': 'immediate disposal'
            })
        
        return alerts

    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get warning level insights"""
        warnings = []
        insights = self.get_insights()
        
        # Supply-only products warning
        if insights.get('supply_only_products'):
            warnings.append({
                'level': 'warning',
                'icon': '📦',
                'message': f"{len(insights.get('supply_only_products', set()))} Supply-Only products",
                'value': f"${insights.get('supply_only_value', 0):,.0f}",
                'action': 'potential dead stock'
            })
        
        # Missing dates warnings
        if insights.get('demand_missing_etd', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '⚠️',
                'message': f"{insights['demand_missing_etd']} records missing ETD",
                'action': 'demand side'
            })
        
        # Near expiry warnings
        if insights.get('near_expiry_7d_count', 0) > 0:
            warnings.append({
                'level': 'warning',
                'icon': '📅',
                'message': f"{insights['near_expiry_7d_count']} items expiring in 7 days",
                'value': f"${insights['near_expiry_7d_value']:,.0f}"
            })
        
        return warnings


    def clear_cache(self, data_type: str = None):
        """Clear cache for specific data type or all"""
        if data_type:
            keys_to_clear = [k for k in self._cache.keys() if k.startswith(data_type)]
            for key in keys_to_clear:
                if key in self._cache:
                    del self._cache[key]
                if key in self._last_refresh:
                    del self._last_refresh[key]
        else:
            self._cache.clear()
            self._last_refresh.clear()
            self._insights_cache.clear()
            st.cache_data.clear()

    def set_cache_ttl(self, seconds: int):
        """Set cache time-to-live"""
        self._cache_ttl = seconds


    def create_unified_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create unified date columns for All tab display"""
        if df.empty or 'source_type' not in df.columns:
            return df
        
        # Create unified date columns
        df['unified_date'] = pd.NaT
        df['unified_date_adjusted'] = pd.NaT
        
        # Map each source to its specific date columns
        for source_type in df['source_type'].unique():
            mask = df['source_type'] == source_type
            
            if source_type == 'Inventory':
                if 'date_ref' in df.columns:
                    df.loc[mask, 'unified_date'] = df.loc[mask, 'date_ref']
                if 'date_ref_adjusted' in df.columns:
                    df.loc[mask, 'unified_date_adjusted'] = df.loc[mask, 'date_ref_adjusted']
                    
            elif source_type == 'Pending CAN':
                if 'arrival_date' in df.columns:
                    df.loc[mask, 'unified_date'] = df.loc[mask, 'arrival_date']
                if 'arrival_date_adjusted' in df.columns:
                    df.loc[mask, 'unified_date_adjusted'] = df.loc[mask, 'arrival_date_adjusted']
                    
            elif source_type == 'Pending PO':
                if 'eta' in df.columns:
                    df.loc[mask, 'unified_date'] = df.loc[mask, 'eta']
                if 'eta_adjusted' in df.columns:
                    df.loc[mask, 'unified_date_adjusted'] = df.loc[mask, 'eta_adjusted']
                    
            elif source_type == 'Pending WH Transfer':
                if 'transfer_date' in df.columns:
                    df.loc[mask, 'unified_date'] = df.loc[mask, 'transfer_date']
                if 'transfer_date_adjusted' in df.columns:
                    df.loc[mask, 'unified_date_adjusted'] = df.loc[mask, 'transfer_date_adjusted']
        
        # Ensure datetime type
        df['unified_date'] = pd.to_datetime(df['unified_date'], errors='coerce')
        df['unified_date_adjusted'] = pd.to_datetime(df['unified_date_adjusted'], errors='coerce')
        
        logger.info(f"Created unified date columns: {df['unified_date'].notna().sum()} non-null unified_date, "
                    f"{df['unified_date_adjusted'].notna().sum()} non-null unified_date_adjusted")
        
        return df
    