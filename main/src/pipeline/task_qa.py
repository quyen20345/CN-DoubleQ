# main/src/pipeline/qa.py
import shutil
from pathlib import Path
from main.src.embedding.model import DenseEmbedding
from main.src.vectordb.qdrant import VectorStore
from main.src.services.handler import QAHandler
from main.src.services.make_output import AnswerGenerator


def run_task_qa(paths: dict):
    print("\n" + "="*28 + " BẮT ĐẦU TÁC VỤ QA " + "="*28)
    output_dir = Path(paths["output_dir"])

    extracted_data = {}
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            main_md = subdir / "main.md"
            images_md = subdir / "images" / "main.md"
            md_file = main_md if main_md.exists() else images_md if images_md.exists() else None
            if md_file:
                extracted_data[subdir.name] = md_file.read_text(encoding="utf-8")
                print(f"✅ Tìm thấy dữ liệu: {md_file.relative_to(output_dir)}")

    if not extracted_data:
        print("❌ Không tìm thấy dữ liệu đã trích xuất. Vui lòng chạy tác vụ 'extract' trước.")
        return

    embedding_model = DenseEmbedding()
    collection_name = f"collection_{paths['pdf_dir'].name}"
    vector_db = VectorStore(collection_name, embedding_model)
    qa_handler = QAHandler(vector_db)
    qa_results = qa_handler.process_questions_csv(paths["question_csv"])
    if qa_results is None:
        return

    generator = AnswerGenerator(output_dir)
    generator.generate_answer_md(extracted_data, qa_results)

    main_script = paths["project_root"] / "main" / "src" / "main.py"
    shutil.copy(main_script, output_dir / "main.py")

    generator.create_zip(paths["zip_name"])
    print("\n" + "="*27 + " HOÀN THÀNH TÁC VỤ QA " + "="*27)
