from pathlib import Path
import zipfile


class AnswerGenerator:
    """Tạo file answer.md và zip."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.answer_md_path = self.output_dir / "answer.md"

    def generate_answer_md(self, extracted_data: dict, qa_results: list):
        """Generate answer.md với format chuẩn."""
        print(f"\n📝 Tạo {self.answer_md_path}")

        with self.answer_md_path.open("w", encoding="utf-8") as f:
            # TASK EXTRACT
            f.write("### TASK EXTRACT\n")
            for pdf_name in sorted(extracted_data.keys()):
                # pdf_title = Path(pdf_name).stem
                # f.write(f"# {pdf_title}\n\n")
                f.write(extracted_data[pdf_name].strip() + "\n\n")

            # TASK QA
            f.write("### TASK QA\n")
            f.write("num_correct,answers\n")
            for count, answers in qa_results:
                if not answers:
                    count, answers = 1, ["A"]

                if len(answers) > 1:
                    answers_str = f"\"{','.join(answers)}\""
                else:
                    answers_str = answers[0]

                f.write(f"{count},{answers_str}\n")

        print("✅ Tạo answer.md thành công")

    def create_zip(self, zip_name: str):
        """Tạo zip file."""
        project_root = self.output_dir.parent
        zip_path = project_root / zip_name

        print(f"\n📦 Đang nén thành '{zip_path}'...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.answer_md_path, arcname="answer.md")

            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = Path(self.output_dir.name) / file_path.relative_to(self.output_dir)
                    zipf.write(file_path, arcname=arcname)

        print(f"✅ Tạo zip thành công: {zip_path}")