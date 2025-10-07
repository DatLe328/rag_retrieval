from rag_retrieval.db.weaviate_db import WeaviateManager
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

# ==============================================================================
# DỮ LIỆU SẢN PHẨM TÀI CHÍNH ĐÃ ĐƯỢC CHUẨN BỊ
# ==============================================================================

products_data = [
    {
        "title": "Gói tín dụng doanh nghiệp GROW500",
        "abstract": "Gói tín dụng GROW500 cung cấp hạn mức lên đến 500 triệu VNĐ cho các doanh nghiệp đã hoạt động trên 1 năm và có doanh thu thuế từ 2 tỷ VNĐ. Hỗ trợ vay vốn lưu động với lãi suất cạnh tranh.",
        "keywords": ["vay kinh doanh", "vốn lưu động", "hạn mức tín dụng", "doanh nghiệp vừa và nhỏ", "doanh thu cao", "lãi suất cạnh tranh", "grow500"],
        "text": """# ** GROW500**
- HẠN MỨC TÍN DỤNG TỐI ĐA
    - 500 triệu VNĐ (tối đa 10% doanh thu thuế năm gần nhất) Bao gồm Vay, Thấu chi, Thẻ tín dụng (hạn mức tối đa 200tr)
- LÃI SUẤT VAY
    - Lãi hằng tháng, gốc cuối kì (vay hạn mức): lãi cơ sở 6.4%, biên độ 9.5%, tổng cộng 15.9% (có thể thay đổi theo từng thời kỳ)
    - Gốc và lãi hằng tháng (vay trả góp): lãi cơ sở 7.6%, biên độ 9.5%, tổng cộng 17.1%.
- THỜI GIAN VAY
    - Hợp đồng cho vay 12 tháng
    - Khế ước nhận nợ: vay hạn mức (6 tháng), vay trả góp (12 tháng)
- MỤC ĐÍCH SỬ DỤNG VỐN
    - Vay kinh doanh, vốn lưu động cho ngành nghề kinh doanh chính của Khách hàng
- LĨNH VỰC KINH DOANH
    - Tất cả loại trừ các trường hợp: Doanh nghiệp hợp tác xã hoặc thuộc các ngành nghề hạn chế hoặc có vốn 100% nước ngoài
- THỜI GIAN THÀNH LẬP
    - Thành lập và hoạt động liên tục trong 1 năm gần nhất
- DOANH THU THUẾ NĂM N-1
    - Doanh thu ≥ 2 tỷ VND
- HỒ SƠ YÊU CẦU:
    - Các chứng từ cơ bản: GPDKKD, CCCD của DDPL và các thành viên góp vốn, điều lệ, giấy phép con (nếu có), BCTC năm gần nhất, - Tờ khai thuế 12 tháng gần nhất  và 3 hóa đơn đầu ra, 3 hóa đơn đầu vào
        """,
        "created_date": datetime.now(timezone.utc)
    },
    {
        "title": "Sản phẩm vay linh hoạt MINIFLEX cho doanh nghiệp mới",
        "abstract": "MINIFLEX là giải pháp tài chính linh hoạt cho các doanh nghiệp mới thành lập dưới 12 tháng, cung cấp hạn mức 300 triệu VNĐ mà không yêu cầu doanh thu. Điều kiện đi kèm là sử dụng phần mềm kế toán của đối tác 3T-Smartkey.",
        "keywords": ["doanh nghiệp mới thành lập", "startup", "vay vốn startup", "thẻ tín dụng", "phần mềm kế toán", "3T-Smartkey", "miniflex"],
        "text": """# **MINIFLEX:**
- HẠN MỨC TÍN DỤNG TỐI ĐA
  - 300 triệu VNĐ (tối đa 50% vốn điều lệ) (bao gồm 2 sản phẩm vay + thẻ tín dụng, hạn mức thẻ từ 50-200 triệu)
- LÃI SUẤT VAY
  - Sản phẩm vay: Gốc và lãi hằng tháng (vay trả góp): lãi cơ sở 7.6%, biên độ 9.5%, tổng cộng 17.1%.
  - Thẻ: Lãi suất khi KH sử dụng mà không thanh toán 34%; lãi suất chuyển đổi trả góp 0.9%/tháng
- THỜI GIAN VAY
  - Hợp đồng cho vay 12 tháng
  - Khế ước nhận nợ: vay hạn mức (6 tháng), vay trả góp (12 tháng)
- MỤC ĐÍCH SỬ DỤNG VỐN
  - Vay kinh doanh, vốn lưu động cho ngành nghề kinh doanh chính của Khách hàng
- LĨNH VỰC KINH DOANH
  - Tất cả loại trừ các trường hợp: Doanh nghiệp hợp tác xã hoặc có vốn 100% nước ngoài
- THỜI GIAN THÀNH LẬP
  - Tối đa 12 tháng
- DOANH THU THUẾ NĂM N-1
  - Không yêu cầu
- HỒ SƠ YÊU CẦU:
  - Các chứng từ cơ bản: GPDKKD, CCCD của DDPL và các thành viên góp vốn, điều lệ, giấy phép con (nếu có)
    Điều kiện:
  - KH cần đăng ký sử dụng và cài đặt thành công Phần mềm kế toán của Đối tác 3T-Smartkey (hỗ trợ 3 user sử dụng, miễn phí tối đa 12 tháng kể từ ngày đăng ký) để được hỗ trợ lên hồ sơ vay
  - Đăng ký chữ ký số của đối tác Hilo đc giảm 1% lãi suất
        """,
        "created_date": datetime.now(timezone.utc)
    },
    {
        "title": "Thẻ tín dụng doanh nghiệp FAST CARD",
        "abstract": "FAST CARD là sản phẩm thẻ tín dụng doanh nghiệp với hạn mức lên đến 400 triệu VNĐ, phù hợp cho các doanh nghiệp hoạt động trên 1 năm có nhu cầu chi tiêu linh hoạt. Lãi suất chuyển đổi trả góp hấp dẫn chỉ 0.9%/tháng.",
        "keywords": ["thẻ tín dụng doanh nghiệp", "fast card", "chi tiêu doanh nghiệp", "trả góp 0.9%", "hạn mức cao", "thanh toán linh hoạt"],
        "text": """# **FAST CARD (Thẻ tín dụng)**
- HẠN MỨC TÍN DỤNG TỐI ĐA
  - 400 triệu VNĐ (tối đa 10% doanh thu thuế năm gần nhất)
- LÃI SUẤT VAY
  - Lãi suất khi KH sử dụng mà không thanh toán: 34%/năm
  - Lãi suất chuyển đổi trả góp: 0.9%/tháng
- THỜI GIAN VAY
  - Không quy định
- MỤC ĐÍCH SỬ DỤNG VỐN
  - Không quy định
- LĨNH VỰC KINH DOANH
  - Tất cả loại trừ các trường hợp: Doanh nghiệp hợp tác xã hoặc thuộc các ngành nghề hạn chế hoặc có vốn 100% nước ngoài
- THỜI GIAN THÀNH LẬP
  - Thành lập hoạt động liên tục trong 1 năm gần nhất
- DOANH THU THUẾ NĂM N-1
  - Hạn mức thẻ ≤ 100tr: doanh thu ≥ 1 tỷ VND
  - Hạn mức thẻ > 100tr: doanh thu ≥ 2 tỷ VND
- HỒ SƠ YÊU CẦU:
  - Các chứng từ cơ bản: GPDKKD, CCCD của DDPL và các thành viên góp vốn, điều lệ, giấy phép con (nếu có), BCTC năm gần nhất, - Tờ khai thuế 12 tháng gần nhất  và 3 hóa đơn đầu ra, 3 hóa đơn đầu vào
        """,
        "created_date": datetime.now(timezone.utc)
    }
]

# ==============================================================================
# LOGIC THÊM DỮ LIỆU VÀO WEAVIATE
# ==============================================================================
if __name__ == "__main__":
    load_dotenv()
    print("--- Bắt đầu quá trình thêm dữ liệu sản phẩm vào Weaviate ---")

    try:
        # Sử dụng WeaviateManager mà không cần truyền embedder, vì Weaviate sẽ tự xử lý
        with WeaviateManager() as manager:
            # LƯU Ý: Đảm bảo collection này đã tồn tại và có schema phù hợp
            # (title, abstract, keywords, text, created_date)
            # Nếu chưa, bạn có thể chạy hàm gen() trong weaviate_db.py để tạo nó.
            collection_name = "Papers"
            print(f"Sẽ thêm dữ liệu vào collection: '{collection_name}'")

            for product in products_data:
                product_title = product.get("title", "Không có tiêu đề")
                print(f"  -> Đang chuẩn bị thêm: {product_title}")
                try:
                    # Gọi hàm add với dictionary chứa toàn bộ thuộc tính của sản phẩm
                    manager.add(
                        collection_name=collection_name,
                        properties=product
                    )
                    print(f"     ✅ Thêm thành công: {product_title}")
                except Exception as e:
                    print(f"     ❌ Lỗi khi thêm '{product_title}': {e}")
    
    except ConnectionError as ce:
        print(f"\nLỖI KẾT NỐI: Không thể kết nối tới Weaviate. Vui lòng kiểm tra lại dịch vụ.")
        print(f"Chi tiết: {ce}")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")

    print("\n--- ✅ Quá trình thêm dữ liệu đã hoàn tất. ---")