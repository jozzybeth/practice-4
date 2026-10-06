import click
import yaml
from utils import get_model_slug


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
    "--stage",
    type=click.Choice(
        ["prepare", "baseline", "zero_shot", "train", "lora", "evaluate", "all"],
        case_sensitive=False,
    ),
    default="all",
    help="Шаг запуска: prepare, baseline, zero_shot, train, lora, evaluate, all",
)
def main(data_config: str, model_config: str, stage: str):
    with open(data_config, "r") as f:
        data_cfg = yaml.safe_load(f)
    with open(model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    model_slug = get_model_slug(model_cfg["base_model_name"])

    if stage in ["prepare", "all"]:
        print("\n=== [STAGE 1/6] PREPARE DATA ===")
        from prepare_data import prepare_data_only

        prepare_data_only(data_config)

    if stage in ["baseline", "all"]:
        print("\n=== [STAGE 2/6] GENERATE BASELINE PREDICTIONS ===")
        from generate_baseline import generate_baseline

        generate_baseline(data_config)

    if stage in ["zero_shot", "all"]:
        print(f"\n=== [STAGE 3/6] INFERENCE: {model_slug} ZERO-SHOT (vLLM) ===")
        from inference import run_vllm_inference

        run_vllm_inference(
            model_identifier=model_cfg["base_model_name"],
            output_csv_name=f"{model_slug}_zero_shot.csv",
            data_cfg=data_cfg,
            model_cfg=model_cfg,
        )

    if stage in ["train", "all"]:
        print(f"\n=== [STAGE 4/6] TRAIN TRANSFORMERS ({model_slug} LoRA) & MERGE ===")
        from train_transformers import merge_weights, train_lora

        train_lora(data_cfg, model_cfg)
        merge_weights(model_cfg)

    if stage in ["lora", "all"]:
        print(f"\n=== [STAGE 5/6] INFERENCE: TRAINED {model_slug} LORA (vLLM) ===")
        from inference import run_vllm_inference

        run_vllm_inference(
            model_identifier=model_cfg["merged_dir"],
            output_csv_name=f"{model_slug}_lora.csv",
            data_cfg=data_cfg,
            model_cfg=model_cfg,
        )

    if stage in ["evaluate", "all"]:
        print("\n=== [STAGE 6/6] EVALUATION & LLM-JUDGE ===")
        from evaluate_llm_judge import run_llm_judge

        run_llm_judge(data_cfg, model_cfg)


if __name__ == "__main__":
    main()