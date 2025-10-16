# main/src/pipeline/extract.py
import traceback
from pathlib import Path
from main.extract_pdf import PDFToMarkdownConverter
from main.src.embedding.model import DenseEmbedding
from main.src.vectordb.qdrant import VectorStore
from main.src.utils.indexer import index_extracted_data


def run_task_extract(paths: dict) -> bool:
    print("\n" + "="*25 + " BẮT ĐẦU TÁC VỤ EXTRACT " + "="*25)
    input_dir = Path(paths["pdf_dir"])
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = PDFToMarkdownConverter()
    extracted_data = {}

    for pdf in input_dir.glob("*.pdf"):
        pdf_output_dir = output_dir / pdf.stem / "images"
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            md_path = converter.convert_pdf_to_markdown(str(pdf), str(pdf_output_dir))
            if md_path and Path(md_path).exists():
                extracted_data[pdf.stem] = Path(md_path).read_text(encoding="utf-8")
            print(f"✅ Trích xuất thành công: {pdf.name}")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {pdf.name}: {e}")
            traceback.print_exc()

    if not extracted_data:
        print("❌ Không có file PDF nào được xử lý thành công.")
        return False

    embedding_model = DenseEmbedding()
    collection_name = f"collection_{paths['pdf_dir'].name}"
    vector_db = VectorStore(collection_name, embedding_model)
    index_extracted_data(extracted_data, vector_db)

    print("\n" + "="*24 + " HOÀN THÀNH TÁC VỤ EXTRACT " + "="*24)
    return True
