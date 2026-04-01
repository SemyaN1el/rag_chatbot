import argparse
import json
import math
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextRecall,
    Faithfulness,
)
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import PromptValue
from chat import build_chain
from hybrid_search import hybrid_search
from config import *


DEFAULT_EVAL_DATASET_PATH = Path("data/eval_questions.json")
DEFAULT_QUICK_EVAL_DATASET_PATH = Path("data/eval_questions_quick.json")
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка качества RAG-системы через RAGAS."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        help="Путь к JSON-файлу с eval-вопросами.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Использовать короткий eval-набор: {DEFAULT_QUICK_EVAL_DATASET_PATH}",
    )
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset is not None:
        return args.dataset
    if args.quick:
        return DEFAULT_QUICK_EVAL_DATASET_PATH
    return DEFAULT_EVAL_DATASET_PATH


def build_answer_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        seed=42,
    )


class SchemaAwareOllamaRagasLLM(BaseRagasLLM):
    schema_start_marker = "following schema as specified in JSON Schema:\n"
    schema_end_marker = "Do not use single quotes in your response but double quotes,"

    def __init__(self) -> None:
        super().__init__()

    def _extract_schema(self, prompt_text: str) -> dict | None:
        if self.schema_start_marker not in prompt_text or self.schema_end_marker not in prompt_text:
            return None

        start_index = prompt_text.index(self.schema_start_marker) + len(self.schema_start_marker)
        end_index = prompt_text.index(self.schema_end_marker, start_index)
        raw_schema = prompt_text[start_index:end_index].strip()

        try:
            return json.loads(raw_schema)
        except json.JSONDecodeError:
            return None

    def _build_llm(self, prompt: PromptValue, temperature: float) -> ChatOllama:
        schema = self._extract_schema(prompt.to_string())
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=temperature,
            seed=42,
            disable_streaming=True,
            num_ctx=8192,
            format=schema,
        )

    def _normalize_generations(self, result: LLMResult, n: int) -> LLMResult:
        if n == 1:
            return result

        generations = [[generation[0] for generation in result.generations]]
        result.generations = generations
        return result

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        llm = self._build_llm(prompt, temperature)
        result = llm.generate_prompt(
            prompts=[prompt] * n,
            stop=stop,
            callbacks=callbacks,
        )
        return self._normalize_generations(result, n)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        llm = self._build_llm(prompt, temperature or 0.01)
        result = await llm.agenerate_prompt(
            prompts=[prompt] * n,
            stop=stop,
            callbacks=callbacks,
        )
        return self._normalize_generations(result, n)


def build_metric_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_test_questions(dataset_path: str | Path) -> list[dict]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            "Файл с eval-вопросами не найден. Создай JSON-файл со структурой "
            '[{"question": "...", "ground_truth": "..."}] '
            f"и запусти: python evaluate.py {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Eval-набор должен быть непустым JSON-массивом.")

    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Элемент #{index} должен быть объектом JSON.")

        question = item.get("question")
        ground_truth = item.get("ground_truth")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"У элемента #{index} поле 'question' должно быть непустой строкой.")
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError(f"У элемента #{index} поле 'ground_truth' должно быть непустой строкой.")

    return payload


def collect_rag_results(test_questions: list[dict], use_hybrid: bool = False) -> Dataset:
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

    llm = build_answer_llm()

    for item in test_questions:
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

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRecall(),
    ]

    run_config = RunConfig(
        max_workers=1,
        timeout=90,
        max_retries=2,
        seed=42,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=SchemaAwareOllamaRagasLLM(),
        embeddings=build_metric_embeddings(),
        run_config=run_config,
        raise_exceptions=False,
        batch_size=1,
    )

    return result


def summarize_metric(raw_values) -> tuple[float | None, int, int]:
    values = raw_values if isinstance(raw_values, list) else [raw_values]
    valid_values = []
    invalid_count = 0

    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            invalid_count += 1
            continue

        if math.isnan(numeric_value):
            invalid_count += 1
            continue

        valid_values.append(numeric_value)

    if not valid_values:
        return None, 0, invalid_count

    average = round(sum(valid_values) / len(valid_values), 3)
    return average, len(valid_values), invalid_count


def format_score(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def main():
    args = parse_args()
    dataset_path = resolve_dataset_path(args)
    test_questions = load_test_questions(dataset_path)

    print("=" * 60)
    print("RAGAS оценка качества RAG системы")
    print("=" * 60)
    print(f"Eval-набор: {dataset_path}")

    print("\n[ 1/2 ] Собираем результаты векторного поиска...")
    vector_dataset = collect_rag_results(test_questions, use_hybrid=False)

    print("\n[ 2/2 ] Собираем результаты гибридного поиска...")
    hybrid_dataset = collect_rag_results(test_questions, use_hybrid=True)

    vector_scores = run_evaluation(vector_dataset, "Векторный поиск")
    hybrid_scores = run_evaluation(hybrid_dataset, "Гибридный поиск")

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)
    print(f"{'Метрика':<25} {'Векторный':>12} {'Гибридный':>12} {'Разница':>10}")
    print("-" * 60)

    invalid_metrics = []
    for metric in METRIC_NAMES:
        v_mean, v_valid, v_invalid = summarize_metric(vector_scores[metric])
        h_mean, h_valid, h_invalid = summarize_metric(hybrid_scores[metric])

        if v_mean is not None and h_mean is not None:
            diff = round(h_mean - v_mean, 3)
            arrow = "^" if diff > 0 else "v" if diff < 0 else "="
            diff_text = f"{arrow} {abs(diff):>6.3f}"
        else:
            diff_text = "n/a"

        print(
            f"{metric:<25} "
            f"{format_score(v_mean):>12} "
            f"{format_score(h_mean):>12} "
            f"{diff_text:>10}"
        )

        if v_invalid or h_invalid:
            invalid_metrics.append(
                (
                    metric,
                    v_valid,
                    v_invalid,
                    h_valid,
                    h_invalid,
                )
            )

    print("\nИнтерпретация:")
    print("   faithfulness     — не галлюцинирует ли модель")
    print("   answer_relevancy — релевантен ли ответ вопросу")
    print("   context_recall   — полно ли найден нужный контекст")

    if invalid_metrics:
        print("\nНеполные метрики:")
        for metric, v_valid, v_invalid, h_valid, h_invalid in invalid_metrics:
            print(
                f"   {metric}: "
                f"vector valid={v_valid}, invalid={v_invalid}; "
                f"hybrid valid={h_valid}, invalid={h_invalid}"
            )


if __name__ == "__main__":
    main()
