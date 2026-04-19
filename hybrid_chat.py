from hybrid_search import hybrid_search
from app.services.llm import generate_text_from_prompt
from config import *

PROMPT_TEMPLATE = """Ты — помощник, отвечающий на вопросы по документу.
    Используй ТОЛЬКО информацию из контекста ниже.
    Если ответа нет в контексте — скажи "В документе нет информации по этому вопросу."
    Отвечай на том же языке, на котором задан вопрос.

    Контекст:
    {context}

    Вопрос: {question}

    Ответ:"""


def ask_hybrid(question: str):
    results = hybrid_search(question, top_k=TOP_K)
    context = "\n\n".join([r["text"] for r in results])
    filled_prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generate_text_from_prompt(filled_prompt, temperature=LLM_TEMPERATURE)

    print(f"\nОтвет:\n{response}")

    print(f"\nИсточники (RRF scores):")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r['rrf_score']}] {r['text'][:100]}...")
    print()


def chat():
    print("Гибридный чат-бот запущен (BM25 + векторный + RRF)")
    print("Введи 'выход' для остановки.\n")

    while True:
        question = input(" Вопрос: ").strip()
        if question.lower() in ["выход", "exit", "quit"]:
            break
        if not question:
            continue
        ask_hybrid(question)


if __name__ == "__main__":
    chat()
