import os
import pandas as pd
from datasets import Dataset


def load_raw_data(config: dict):
    """Загрузка исходных CSV датасетов."""
    raw_dir = config["raw_data_dir"]
    train = pd.read_csv(os.path.join(raw_dir, config["train_filename"]))
    val = pd.read_csv(os.path.join(raw_dir, config["validation_filename"]))
    test = pd.read_csv(os.path.join(raw_dir, config["test_filename"]))
    return train, val, test


def build_sft_dataset(
    train_df: pd.DataFrame, val_df: pd.DataFrame, data_config: dict
):
    """Форматирование датасета под Hugging Face SFT Trainer."""
    prompt_template = data_config["prompt_template"]
    limit = data_config["train_sample_size"]

    train_data = train_df[["dialogue", "summary"]].iloc[:limit].copy()
    val_data = val_df[["dialogue", "summary"]].copy()

    def format_row(row):
        prompt = prompt_template.format(dialogue=row["dialogue"], topic="topic")
        return f"{prompt}\n Summary: {row['summary']}"

    train_data["text"] = train_data.apply(format_row, axis=1)
    val_data["text"] = val_data.apply(format_row, axis=1)

    train_ds = Dataset.from_pandas(
        train_data[["text"]], preserve_index=False
    )
    val_ds = Dataset.from_pandas(val_data[["text"]], preserve_index=False)

    return train_ds, val_ds