from main.src.llm.chat import ask_llm, ask_rag
from main.src.utils.indexer import load_and_index_data
from main.src.utils.collections import COLLECTIONS

question = "Xin chào, bạn là mô hình gì?"
answer = ask_llm(question)
print(answer)
print("--------------------------------------------------------")

question = "Con vịt có mấy chân?"
docs = ["Con vịt có 4 chân."]
answer = ask_rag(question, docs)
print(answer)
print("--------------------------------------------------------")

load_and_index_data(COLLECTIONS["books"], "main/data/books")  # index data xong thì comment lại.

question = "phụ nữ muốn gì ở đàn ông?"
docs = COLLECTIONS["books"].search(question, top_k=3, threshold=0.3)
print(docs)
print("--------------------------------------------------------")

answer = ask_rag(question, docs)
print(answer)