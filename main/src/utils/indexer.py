from main.src.utils._utils import get_files_in_directory, load_file_as_markdown
from main.src.utils.collections import COLLECTIONS
from main.src.utils._utils import chunking
from main.src.vectordb.qdrant import VectorStore

def preprocess_file(path):
    chunked_data = []

    # Xử lý file .docx, .txt và .pdf
    if path.endswith(".docx") or path.endswith(".pdf") or path.endswith(".txt"):
        print(f"Processing file: {path}")
        file_name = path.split("/")[-1]
        markdown_text = load_file_as_markdown(path)
        chunk_contents = chunking(markdown_text)
        chunked_data = [[chunk, file_name] for chunk in chunk_contents]
    else:
        pass

    return chunked_data

def load_and_index_data(vector_store, path):
    vector_store.recreate_collection()

    print(f"Indexing data from {path}...")
    files = get_files_in_directory(path)
    
    for path in files:
        chunked_data = preprocess_file(path)
        vector_store.insert_data(["content", "source"], chunked_data, [0, 1])


def index_extracted_data(extracted_data: dict, vector_store: VectorStore):
    """
    Chunk và index dữ liệu đã được trích xuất vào vector database.
    """
    print("🔄 Đang chunk và index dữ liệu...")
    vector_store.recreate_collection()
    
    all_chunks_with_source = []
    for pdf_name, content in extracted_data.items():
        chunks = chunking(content)
        for chunk in chunks:
            # Payload bao gồm nội dung chunk và tên file nguồn
            all_chunks_with_source.append([chunk, pdf_name])
    
    if all_chunks_with_source:
        # Keys của payload phải khớp với lúc insert
        vector_store.insert_data(["content", "source"], all_chunks_with_source)
    
    print(f"✅ Đã index {len(all_chunks_with_source)} chunks từ {len(extracted_data)} PDF.")
    