from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from chat import build_chain
from hybrid_chat import hybrid_search
from config import *


TEST_QUESTIONS = [
    {
        "question": "Что такое конечно-разностный метод Эйлера?",
        "ground_truth": "Метод Эйлера заключается в аппроксимации непрерывно-дифференцируемой функции кусочно-линейной, её производной конечно-разностным выражением, а интеграла кубатурной формулой."
    },
    {
        "question": "В чём суть метода Ритца?",
        "ground_truth": "Метод Ритца заключается в том, что значения функционала рассматриваются на всевозможных линейных комбинациях координатных функций, коэффициенты которых определяются из условия экстремума."
    },
    {
        "question": "Что такое собственные значения в задаче Штурма-Лиувилля?",
        "ground_truth": "Собственные значения — это те значения параметра лямбда, при которых краевая задача Штурма-Лиувилля имеет нетривиальные решения, называемые собственными функциями."
    },
    {
        "question": "Чем метод Галеркина отличается от метода Ритца?",
        "ground_truth": "В методе Галеркина приближённое решение ищется в том же виде что и в методе Ритца, но коэффициенты определяются из условия ортогональности невязки координатным функциям."
    },
]


def collect_rag_results(use_hybrid: bool = False) -> Dataset:
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    if use_hybrid:
        print("Используем гибридный поиск (BM25 + векторный + RRF)")
    else:
        print("Используем обычный векторный поиск")

    chain = None if use_hybrid else build_chain()

    PROMPT = """Используй ТОЛЬКО контекст ниже для ответа.
            Контекст: {context}
            Вопрос: {question}
            Ответ:"""

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    for item in TEST_QUESTIONS:
        question = item["question"]
        print(f"   Обрабатываем: '{question[:50]}...'")

        if use_hybrid:
            results = hybrid_search(question, top_k=TOP_K)
            context_texts = [r["text"] for r in results]

            prompt = PromptTemplate(
                template=PROMPT,
                input_variables=["context", "question"]
            )
            filled = prompt.format(
                context="\n\n".join(context_texts),
                question=question
            )
            answer = llm.invoke(filled).content

        else:
            result = chain.invoke({"query": question})
            answer = result["result"]
            context_texts = [
                doc.page_content
                for doc in result["source_documents"]
            ]

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation(dataset: Dataset, label: str) -> dict:
    print(f"Оцениваем: {label}")

    ragas_llm = LangchainLLMWrapper(
        ChatOllama(model=OLLAMA_MODEL, temperature=0)
    )

    ragas_embeddings = RagasHuggingFaceEmbeddings(
        model=EMBEDDING_MODEL
    )

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextRecall(llm=ragas_llm),
    ]

    run_config = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=3
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config
    )

    return result


def main():
    print("=" * 60)
    print("RAGAS оценка качества RAG системы")
    print("=" * 60)

    print("\n[ 1/2 ] Собираем результаты векторного поиска...")
    vector_dataset = collect_rag_results(use_hybrid=False)

    print("\n[ 2/2 ] Собираем результаты гибридного поиска...")
    hybrid_dataset = collect_rag_results(use_hybrid=True)

    vector_scores = run_evaluation(vector_dataset, "Векторный поиск")
    hybrid_scores = run_evaluation(hybrid_dataset, "Гибридный поиск")

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)
    print(f"{'Метрика':<25} {'Векторный':>12} {'Гибридный':>12} {'Разница':>10}")
    print("-" * 60)

    metric_names = ["faithfulness", "answer_relevancy", "context_recall"]
    for metric in metric_names:
        v_raw = vector_scores[metric]
        h_raw = hybrid_scores[metric]

        v = round(float(sum(v_raw) / len(v_raw)) if isinstance(v_raw, list) else float(v_raw), 3)
        h = round(float(sum(h_raw) / len(h_raw)) if isinstance(h_raw, list) else float(h_raw), 3)

        diff = round(h - v, 3)
        arrow = "^" if diff > 0 else "v" if diff < 0 else "="
        print(f"{metric:<25} {v:>12} {h:>12} {arrow} {abs(diff):>7}")

    print("\nИнтерпретация:")
    print("   faithfulness     — не галлюцинирует ли модель")
    print("   answer_relevancy — релевантен ли ответ вопросу")
    print("   context_recall   — полно ли найден нужный контекст")


if __name__ == "__main__":
    main()