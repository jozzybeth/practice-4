import os
import click
import pandas as pd
import yaml
from openai import OpenAI
from dataset import load_raw_data
from utils import get_model_slug
from tqdm.auto import tqdm


def run_vllm_inference(
    model_identifier: str,
    output_csv_name: str,
    data_cfg: dict,
    model_cfg: dict,
):
    """Инференс через vLLM и сохранение CSV."""
    _, _, test = load_raw_data(data_cfg)

    client = OpenAI(base_url=model_cfg["vllm_endpoint"], api_key="EMPTY")

    predictions = []
    out_dir = data_cfg["processed_data_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_csv_name)

    print(f"\n[vLLM Inference] Запуск инференса для модели: {model_identifier}")

    for i, row in tqdm(
        test.iterrows(), total=len(test), desc=f"Inference {output_csv_name}"
    ):
        prompt = data_cfg["prompt_template"].format(dialogue=row["dialogue"],topic=row["topic"])

        response = client.chat.completions.create(
            model=model_identifier,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128,
        )

        pred = response.choices[0].message.content.strip()
        predictions.append(pred)

        if (i + 1) % 50 == 0 or (i + 1) == len(test):
            pd.DataFrame(
                {
                    "id": test.iloc[: i + 1]["id"],
                    "reference": test.iloc[: i + 1]["summary"],
                    "prediction": predictions,
                }
            ).to_csv(out_path, index=False)

    print(f"Предикты сохранены в: {out_path}")


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
@click.option(
    "--model-type",
    type=click.Choice(["zero_shot", "lora"], case_sensitive=False),
    default="zero_shot",
    help="Какую модель запускать в vLLM: zero_shot или lora",
)
def main(data_config: str, model_config: str, model_type: str):
    with open(data_config, "r") as f:
        data_cfg = yaml.safe_load(f)
    with open(model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    slug = get_model_slug(model_cfg["base_model_name"])

    if model_type == "zero_shot":
        run_vllm_inference(
            model_cfg["base_model_name"],
            f"{slug}_zero_shot.csv",
            data_cfg,
            model_cfg,
        )
    elif model_type == "lora":
        run_vllm_inference(
            model_cfg["merged_dir"],
            f"{slug}_lora.csv",
            data_cfg,
            model_cfg,
        )


if __name__ == "__main__":
    main()