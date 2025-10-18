import sys
from pathlib import Path
from src.rag_system.qa_handler import QAHandler
from src.rag_system.retriever import HybridRetriever
from src.vectordb.store import VectorStore

def main():
    print("🚀 Khởi tạo hệ thống RAG với HybridRetriever...")
    
    # 1. Khởi tạo embedding model
    from src.embedding.model import EmbeddingModel
    embedding_model = EmbeddingModel()
    
    # 2. Khởi tạo vector store
    collection_name = "collection_public-test-input"
    vector_store = VectorStore(collection_name, embedding_model)
    
    # 3. Load corpus data từ file đã extract
    print("📚 Đang tải corpus data...")
    corpus_path = Path("output/public_test_output/corpus.json")
    
    if not corpus_path.exists():
        print(f"❌ Không tìm thấy file corpus: {corpus_path}")
        print("Vui lòng chạy task extract trước để tạo corpus data.")
        return
    
    try:
        import json
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        print(f"✅ Đã load {len(corpus_data)} documents từ corpus.")
    except Exception as e:
        print(f"❌ Lỗi khi load corpus: {e}")
        return
    
    # 4. Khởi tạo HybridRetriever
    print("🔍 Đang khởi tạo HybridRetriever...")
    hybrid_retriever = HybridRetriever(vector_store, corpus_data)
    
    # 5. Khởi tạo QAHandler với HybridRetriever
    print("🤖 Đang khởi tạo QAHandler...")
    qa_handler = QAHandler(hybrid_retriever)
    
    # 6. Test với câu hỏi
    test_question = "Nội dung chính của tài liệu Public 103?"
    
    print(f"\n🔍 Testing với câu hỏi: {test_question}")
    print("="*60)
    
    # Gọi hàm test_rag_qa
    result = qa_handler.test_rag_qa(test_question)
    
    print("\n" + "="*60)
    print(f"📊 Kết quả: {result}")

if __name__ == "__main__":
    main()