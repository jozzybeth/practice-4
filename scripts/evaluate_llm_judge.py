import glob
import json
import os
import random
import time
import click
import pandas as pd
import yaml
from groq import Groq
from dataset import load_raw_data
from utils import calculate_metrics


def run_llm_judge(data_cfg: dict, model_cfg: dict):
    """Оценка качества всех найденных предсказаний."""
    proc_dir = data_cfg["processed_data_dir"]
    _, _, test = load_raw_data(data_cfg)

    prediction_files = glob.glob(os.path.join(proc_dir, "*.csv"))
    prediction_files = [
        f for f in prediction_files if not f.endswith("llm_judge_summary.csv")
    ]

    model_predictions = {}
    results_metrics = []
    references = test["summary"].tolist()

    for path in prediction_files:
        filename = os.path.basename(path)

        model_name = (
            filename.replace(".csv", "")
            .replace("qwen3_5_0_8b", "Qwen 3.5 0.8B")
            .replace("qwen3_5_27b", "Qwen 3.5 27B")
            .replace("baseline_predictions", "Baseline")
            .replace("_", " ")
            .title()
        )

        df = pd.read_csv(path)
        if "prediction" in df.columns:
            preds = df["prediction"].fillna("").tolist()
            model_predictions[model_name] = preds

            m = calculate_metrics(preds, references)
            results_metrics.append({"Model": model_name, **m})

    print("\n=== АВТОМАТИЧЕСКИЕ МЕТРИКИ (BLEU & ROUGE) ===")
    if results_metrics:
        print(pd.DataFrame(results_metrics).to_string(index=False))
    else:
        print("Предсказания не найдены в data/processed/")
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n[Внимание] GROQ_API_KEY не задан. Оценка LLM-as-a-Judge пропущена.")
        return

    client = Groq(api_key=api_key)
    judge_cfg = model_cfg["judge"]
    rng = random.Random(judge_cfg["seed"])

    selected_indices = rng.sample(
        list(range(len(test))),
        min(judge_cfg["num_examples"], len(test)),
    )

    judge_results = []
    total_tokens = 0

    print("\nСтарт оценки LLM-as-a-Judge...")
    for idx in selected_indices:
        row = test.iloc[idx]
        shuffled_names = list(model_predictions.keys())
        rng.shuffle(shuffled_names)

        mapping = {chr(65 + i): name for i, name in enumerate(shuffled_names)}
        candidates = {
            letter: model_predictions[m_name][idx]
            for letter, m_name in mapping.items()
        }

        prompt = f"""Evaluate summaries using dialogue, reference and topic. Return JSON {{ "A": [F,R,C], ... }}.
F=faithfulness, R=relevance, C=conciseness (1-5).

DIALOGUE: {row['dialogue']}
REFERENCE: {row['summary']}
TOPIC: {row['topic']}
""" + "\n".join(f"{k}: {v}" for k, v in candidates.items())

        try:
            res = client.chat.completions.create(
                model=judge_cfg["model"],
                messages=[
                    {"role": "system", "content": "You are a strict evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            scores = json.loads(res.choices[0].message.content)
            total_tokens += res.usage.total_tokens

            for letter, model_name in mapping.items():
                f, r, c = scores[letter]
                judge_results.append(
                    {
                        "test_index": idx,
                        "model": model_name,
                        "Faithfulness": f,
                        "Relevance": r,
                        "Conciseness": c,
                        "Total": f + r + c,
                    }
                )

            if total_tokens >= judge_cfg["daily_token_limit"]:
                print("Достигнут дневной лимит токенов Groq.")
                break
        except Exception as e:
            print(f"Ошибка на объекте idx {idx}: {e}")
            time.sleep(2)

    if judge_results:
        j_df = (
            pd.DataFrame(judge_results)
            .groupby("model")[
                ["Faithfulness", "Relevance", "Conciseness", "Total"]
            ]
            .mean()
        )
        summary_out = os.path.join(proc_dir, "llm_judge_summary.csv")
        j_df.to_csv(summary_out)
        print("\n=== ИТОГИ LLM-AS-A-JUDGE ===")
        print(j_df)


@click.command()
@click.option(
    "--data-config",
    default="configs/data_config.yaml",
    help="Путь к конфигу данных",
)
@click.option(
    "--model-config",
    default="configs/model_qwen08b.yaml",
    help="Путь к конфигу моделей",
)
def main(data_config: str, model_config: str):
    with open(data_config, "r") as f:
        data_cfg = yaml.safe_load(f)
    with open(model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    run_llm_judge(data_cfg, model_cfg)


if __name__ == "__main__":
    main()