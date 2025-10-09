import weaviate
import weaviate.classes as wvc

# --- CẤU HÌNH KẾT NỐI ---
WEAVIATE_HOST = "localhost"
# WEAVIATE_HOST = "10.1.1.237"
WEAVIATE_PORT = 8080  # Bạn cũng có thể đổi thành 3000

# Tên collection bạn muốn truy vấn (dựa trên các lần trao đổi trước)
COLLECTION_NAME = "VERBA_Embedding_nomic_embed_text_latest"

client = None
try:
    # Sử dụng connect_to_http để chỉ kết nối qua HTTP
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        skip_init_checks=True
    )
    
    print(f"✅ Kết nối tới Weaviate tại http://{WEAVIATE_HOST}:{WEAVIATE_PORT} thành công!")
    print("-" * 40)

    # 1. LIỆT KÊ TẤT CẢ CÁC COLLECTION HIỆN CÓ
    print("Các collection đang có trên Weaviate:")
    collections = client.collections.list_all()
    for collection in collections:
        print(f"- {collection}")
    print("-" * 40)

    # 2. THỬ TRUY VẤN (QUERY) ĐỂ LẤY MỘT VÀI OBJECT
    if client.collections.exists(COLLECTION_NAME):
        print(f"Thử lấy 5 object đầu tiên từ collection '{COLLECTION_NAME}':")
        
        # Lấy collection object
        my_collection = client.collections.get(COLLECTION_NAME)

        # Thực hiện một truy vấn đơn giản để lấy dữ liệu
        response = my_collection.query.fetch_objects(
            limit=5
        )

        # In kết quả
        if not response.objects:
            print("Collection trống hoặc không có dữ liệu.")
        else:
            for i, obj in enumerate(response.objects):
                print(f"\n--- Object {i+1} ---")
                # obj.properties là một dictionary chứa dữ liệu của bạn
                print(obj.properties) 
    else:
        print(f"Collection '{COLLECTION_NAME}' không tồn tại để truy vấn.")

except Exception as e:
    print(f"❌ Đã xảy ra lỗi: {e}")

finally:
    if client:
        client.close()
        print("\nℹ️ Đã đóng kết nối.")