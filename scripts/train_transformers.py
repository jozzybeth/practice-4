import click
import torch
import yaml
from peft import LoraConfig, PeftModel
from dataset import build_sft_dataset, load_raw_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def train_lora(data_cfg: dict, model_cfg: dict):
    """Обучение LoRA адаптера."""
    train_df, val_df, _ = load_raw_data(data_cfg)
    train_ds, val_ds = build_sft_dataset(train_df, val_df, data_cfg)

    model_name = model_cfg["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if model_cfg["training"].get("bf16", False) else torch.float16

    print(f"Загрузка модели {model_name} в {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=model_cfg["lora"]["r"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        bias=model_cfg["lora"]["bias"],
        task_type=model_cfg["lora"]["task_type"],
        target_modules=model_cfg["lora"]["target_modules"],
    )

    training_args = SFTConfig(
        output_dir=model_cfg["lora_dir"],
        num_train_epochs=model_cfg["training"]["num_epochs"],
        per_device_train_batch_size=model_cfg["training"][
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=model_cfg["training"][
            "per_device_eval_batch_size"
        ],
        gradient_accumulation_steps=model_cfg["training"][
            "gradient_accumulation_steps"
        ],
        learning_rate=model_cfg["training"]["learning_rate"],
        bf16=model_cfg["training"].get("bf16", False),
        fp16=model_cfg["training"].get("fp16", False),
        gradient_checkpointing=model_cfg["training"].get(
            "gradient_checkpointing", False
        ),
        logging_steps=model_cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=model_cfg["training"]["eval_steps"],
        save_strategy="steps",
        save_steps=model_cfg["training"]["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_length=data_cfg["max_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    print("Запуск LoRA обучения...")
    trainer.train()
    trainer.save_model(model_cfg["lora_dir"])
    tokenizer.save_pretrained(model_cfg["lora_dir"])


def merge_weights(model_cfg: dict):
    """Слияние весов LoRA с базовой моделью."""
    print("Объединение весов LoRA и базовой модели...")
    dtype = torch.bfloat16 if model_cfg["training"].get("bf16", False) else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_name"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, model_cfg["lora_dir"])
    model = model.merge_and_unload()

    model.save_pretrained(model_cfg["merged_dir"], safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_name"], trust_remote_code=True
    )
    tokenizer.save_pretrained(model_cfg["merged_dir"])
    print(f"Объединённая модель успешно сохранена в: {model_cfg['merged_dir']}")


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

    train_lora(data_cfg, model_cfg)
    merge_weights(model_cfg)


if __name__ == "__main__":
    main()