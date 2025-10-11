import os
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_file_as_markdown(path: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(path)
    return result.document.export_to_markdown()

def load_txt_file_as_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def load_file_as_markdown(path: str) -> str:
    if path.endswith(".pdf"):
        return load_pdf_file_as_markdown(path)
    elif path.endswith(".docx"):
        return load_pdf_file_as_markdown(path)
    elif path.endswith(".txt"):
        return load_txt_file_as_markdown(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def get_files_in_directory(path):
    files = []
    for f in os.listdir(path):
        path_f = os.path.join(path, f)
        if os.path.isfile(path_f):
            files.append(path_f)
        elif os.path.isdir(path_f): 
            files.extend(get_files_in_directory(path_f))

    return files

# def chunking(text):
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#                 chunk_size=512, chunk_overlap=126
#             )
#     chunks = text_splitter.split_text(text)
#     return [chunk for chunk in chunks if len(chunk) >= 40]