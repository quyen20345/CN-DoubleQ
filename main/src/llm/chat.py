import os
from main.src.llm.llm_integrations import get_llm
from main.src.llm._utils import render_prompt

def ask_llm(prompt):
    response = get_llm().invoke(prompt)
    if isinstance(response, str):
        return response
    return response.content

def ask_rag(prompt, documents):
    prompt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompts", "prompt.txt")
    prompt = render_prompt(prompt_path, documents, prompt)
    return ask_llm(prompt)