import os
import click
import yaml
from dataset import load_raw_data
from utils import first_last_sentences, setup_nltk


def generate_baseline(data_config_path: str):
    """First & Last Sentences"""
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)

    setup_nltk()
    _, _, test = load_raw_data(data_config)

    print("Генерация предиктов Baseline (First & Last sentences)...")
    test["prediction"] = test["dialogue"].apply(first_last_sentences)

    out_dir = data_config["processed_data_dir"]
    os.makedirs(out_dir, exist_ok=True)
    baseline_path = os.path.join(out_dir, "baseline_predictions.csv")
    test.to_csv(baseline_path, index=False)
    print(f"Предикты бейзлайна сохранены в: {baseline_path}")


@click.command()
@click.option(
    "--data-config",
    default="configs/data_config.yaml",
    help="Путь к конфигу данных",
)
def main(data_config: str):
    generate_baseline(data_config)


if __name__ == "__main__":
    main()