import click
import yaml
from dataset import load_raw_data


def prepare_data_only(data_config_path: str):
    """Проверка целостности и валидация исходных данных."""
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)

    train, val, test = load_raw_data(data_config)

    print("--- Проверка структуры данных ---")
    for name, df in [("train", train), ("validation", val), ("test", test)]:
        print(
            f"{name}: размер={df.shape}, пропуски={df.isna().sum().sum()}, дубликаты={df.duplicated().sum()}"
        )


@click.command()
@click.option(
    "--data-config",
    default="configs/data_config.yaml",
    help="Путь к конфигу данных",
)
def main(data_config: str):
    prepare_data_only(data_config)


if __name__ == "__main__":
    main()