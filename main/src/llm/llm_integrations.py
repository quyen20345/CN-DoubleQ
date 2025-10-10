import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()
LLM_TYPE = os.getenv("LLM_TYPE", "ollama")

def init_ollama_chat(temperature):
    return OllamaLLM(
        model=os.getenv("CHAT_MODEL"),
        streaming=True,
        temperature=temperature,
    )

MAP_LLM_TYPE_TO_CHAT_MODEL = {
    "ollama": init_ollama_chat,
}

def get_llm(temperature=0):
    if LLM_TYPE not in MAP_LLM_TYPE_TO_CHAT_MODEL:
        raise Exception(
            "LLM type not found. Please set LLM_TYPE to one of: "
            + ", ".join(MAP_LLM_TYPE_TO_CHAT_MODEL.keys())
            + "."
        )

    return MAP_LLM_TYPE_TO_CHAT_MODEL[LLM_TYPE](temperature=temperature)


if __name__ == "__main__":
    llm = get_llm(temperature=0.3)
    prompt = "Viết một đoạn thơ ngắn về mùa thu Hà Nội"
    
    # Gọi mô hình để sinh nội dung
    response = llm.invoke(prompt)
    
    print("💬 LLM Output:")
    print(response)
