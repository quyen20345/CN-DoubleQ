# -*- coding: utf-8 -*-
import re
from typing import List
from main.src.llm.llm_integrations import get_llm

# =================================================================
# ✅ CÁC THAM SỐ TỐI ƯU
# =================================================================

# Ngưỡng để gộp các điểm ngắt gần nhau (số câu)
DEBOUNCE_THRESHOLD = 2
# Ngưỡng số từ tối thiểu cho một chunk
MIN_CHUNK_WORDS = 40

def call_ollama_for_chunking(prompt: str) -> str | None:
    """
    Calls the configured Ollama LLM and returns the response.
    This function is adapted to use the project's LLM integration.
    """
    try:
        llm = get_llm(temperature=0.0)
        # We set a low max_tokens for this specific task, as we only expect a number.
        # Note: Langchain Ollama wrapper might not directly support num_predict.
        # The model behavior should still be constrained by the prompt.
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"\n⚠️ Lỗi khi gọi API Ollama: {e}")
        return None

def post_process_chunks(chunks: list[str], min_words: int) -> list[str]:
    """
    Hậu xử lý các chunk: gộp các chunk quá nhỏ vào chunk trước đó.
    """
    print(f"\n⚙️ Hậu xử lý: Gộp các chunk có ít hơn {min_words} từ...")
    processed_chunks = []
    for chunk in chunks:
        if not processed_chunks:
            processed_chunks.append(chunk)
            continue

        # Nếu chunk hiện tại quá nhỏ, gộp nó vào chunk trước đó
        if len(chunk.split()) < min_words:
            print(f"  - Merging small chunk (words: {len(chunk.split())}) into previous one.")
            processed_chunks[-1] += " " + chunk
        else:
            processed_chunks.append(chunk)
    return processed_chunks

def advanced_llm_chunker(
    text: str,
    window_size: int = 15,
    step_size: int = 5
) -> list[str]:
    """
    Phiên bản chunker nâng cao với logic lọc điểm ngắt và hậu xử lý chunk.
    Sử dụng LLM được cấu hình trong dự án.
    """
    text_cleaned = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text_cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    split_indices = set()
    print(f"\n🚀 Bắt đầu phân tích {len(sentences)} câu bằng cửa sổ trượt...")

    for i in range(0, len(sentences), step_size):
        window_end = min(i + window_size, len(sentences))
        window_sentences = sentences[i:window_end]

        if len(window_sentences) < 5: continue

        numbered_sentences_str = "\n".join(
            f"{idx + 1}. {sent}" for idx, sent in enumerate(window_sentences)
        )
        prompt = f"""You are an expert text segmentation engine. Your task is to find the single most significant topic change in the following numbered list of sentences. A good split point is usually a heading or the start of a new, distinct topic.

Analyze the text below. Return ONLY the integer number of the sentence where the main topic changes. For example, if the topic changes at sentence 5, your response must be exactly "5". Do not add any explanation, punctuation, or text.

TEXT TO ANALYZE:
---
{numbered_sentences_str}
---

The number of the best sentence to split at is:"""

        response_text = call_ollama_for_chunking(prompt)
        if not response_text: continue

        try:
            extracted_numbers = re.findall(r'\d+', response_text)
            if not extracted_numbers: continue

            split_num_in_window = int(extracted_numbers[0])
            if 1 < split_num_in_window < len(window_sentences):
                global_index = i + split_num_in_window - 1
                split_indices.add(global_index)
                print(f"  - 🧩 Cửa sổ [{i}-{window_end}]: Đề xuất ngắt tại câu toàn cục số {global_index}")
        except (ValueError, IndexError):
            print(f"  - ⚠️ Cửa sổ [{i}-{window_end}]: Phản hồi không hợp lệ: '{response_text}'")

    # ================================================================
    # TỐI ƯU 1: LỌC VÀ GỘP CÁC ĐIỂM NGẮT GẦN NHAU (DEBOUNCING)
    # ================================================================
    print(f"\n⚙️ Lọc điểm ngắt: Gộp các điểm cách nhau dưới {DEBOUNCE_THRESHOLD} câu...")
    if not split_indices:
        print("  - ❕ Không tìm thấy điểm ngắt nào. Trả về toàn bộ văn bản.")
        return [text]

    sorted_indices = sorted(list(split_indices))
    debounced_indices = []
    if sorted_indices:
        last_index = sorted_indices[0]
        debounced_indices.append(last_index)
        for index in sorted_indices[1:]:
            if index - last_index > DEBOUNCE_THRESHOLD:
                debounced_indices.append(index)
                last_index = index
            else:
                print(f"  - Ignoring split at {index} (too close to {last_index})")

    # ================================================================
    # TỐI ƯU 2: SỬA LOGIC CẮT CHUNK
    # ================================================================
    print("\n📦 Đang ghép lại thành các đoạn cuối cùng với logic mới...")
    final_chunks = []
    start_idx = 0
    for split_point in debounced_indices:
        chunk_sentences = sentences[start_idx:split_point]
        if chunk_sentences:
            final_chunks.append(" ".join(chunk_sentences))
        start_idx = split_point

    last_chunk_sentences = sentences[start_idx:]
    if last_chunk_sentences:
        final_chunks.append(" ".join(last_chunk_sentences))

    # ================================================================
    # TỐI ƯU 3: HẬU XỬ LÝ GỘP CHUNK QUÁ NHỎ
    # ================================================================
    final_chunks = post_process_chunks(final_chunks, min_words=MIN_CHUNK_WORDS)

    return [chunk for chunk in final_chunks if chunk.strip()]
