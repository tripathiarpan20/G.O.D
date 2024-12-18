import argparse
import os
import random
import secrets
import string
from typing import Any
from typing import Dict

from core.models.config_models import MinerConfig
from core.models.config_models import ValidatorConfig
from core.validators import InputValidators
from core.validators import validate_input


def generate_secure_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    password = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
    ]
    password += [secrets.choice(alphabet) for _ in range(length - 3)]
    password = list(password)  # Convert to list for shuffling
    random.shuffle(password)  # Use random.shuffle instead of secrets.shuffle
    return "".join(password)


def parse_bool_input(prompt: str, default: bool = False) -> bool:
    result = validate_input(
        f"{prompt} (y/n): (default: {'y' if default else 'n'}) ",
        InputValidators.yes_no,
        default="y" if default else "n",
    )
    return result.lower().startswith("y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate configuration file")
    parser.add_argument("--dev", action="store_true",
                        help="Use development configuration")
    parser.add_argument("--miner", action="store_true",
                        help="Generate miner configuration")
    return parser.parse_args()


def generate_miner_config(dev: bool = False) -> Dict[str, Any]:
    print("\n🤖 Let's configure your Miner! 🛠️\n")

    subtensor_network = input(
        "🌐 Enter subtensor network (default: finney): ") or "finney"
    subtensor_address = (
        validate_input(
            "🔌 Enter subtensor address (default: None): ",
            InputValidators.websocket_url,
        )
        or None
    )

    config = MinerConfig(
        wallet_name=input(
            "\n💼 Enter wallet name (default: default): ") or "default",
        hotkey_name=input(
            "🔑 Enter hotkey name (default: default): ") or "default",
        wandb_token=input(
            "📊 Enter wandb token (default: default): ") or "default",
        huggingface_token=input(
            "🤗 Enter huggingface token (default: default): ") or "default",
        repo_id=input(
            "🏗️ What is the name of the hugginface repo that you'd like to default to saving models to: "),
        subtensor_network=subtensor_network,
        subtensor_address=subtensor_address,
        refresh_nodes=True,
        netuid=241 if subtensor_network == "test" else 56,
        env="dev" if dev else "prod",
        min_stake_threshold=input(
            f"Enter MIN_STAKE_THRESHOLD (default: {'0' if subtensor_network == 'test' else '1000'}): ")
        or ("0" if subtensor_network == "test" else "1000"),
    )

    return vars(config)


def generate_validator_config(dev: bool = False) -> Dict[str, Any]:
    print("\n🎯 Let's set up your Validator! 🚀\n")

    # Check if POSTGRES_PASSWORD already exists in the environment
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    frontend_api_key = os.getenv("FRONTEND_API_KEY")

    subtensor_network = input(
        "🌐 Enter subtensor network (default: finney): ") or "finney"
    subtensor_address = (
        validate_input(
            "🔌 Enter subtensor address (default: None): ",
            InputValidators.websocket_url,
        )
        or None
    )

    wallet_name = input(
        "💼 Enter wallet name (default: default): ") or "default"
    hotkey_name = input(
        "🔑 Enter hotkey name (default: default): ") or "default"
    netuid = 241 if subtensor_network.strip() == "test" else 56
    postgres_user = "user"
    postgres_password = generate_secure_password(
    ) if not postgres_password else postgres_password
    postgres_db = "god-db"
    postgres_host = "postgresql"
    postgres_port = "5432"

    validator_port = input(
        "👀 Enter an exposed port to run the validator on (default: 9001): ") or "9001"

    s3_compatible_endpoint = input("🎯 Enter s3 compatible endpoint: ")
    s3_compatible_access_key = input("🎯 Enter s3 compatible access key: ")
    s3_compatible_secret_key = input("🎯 Enter s3 compatible secret key: ")
    s3_bucket_name = input("🎯 Enter your s3 bucket name: ")

    frontend_api_key = generate_secure_password(
    ) if not frontend_api_key else frontend_api_key

    config = ValidatorConfig(
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        subtensor_network=subtensor_network,
        subtensor_address=subtensor_address,
        netuid=netuid,
        env=env,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
        postgres_db=postgres_db,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        s3_compatible_endpoint=s3_compatible_endpoint,
        s3_compatible_access_key=s3_compatible_access_key,
        s3_compatible_secret_key=s3_compatible_secret_key,
        s3_bucket_name=s3_bucket_name,
        frontend_api_key=frontend_api_key,
        validator_port=validator_port,
        gpu_server=None,
        set_metagraph_weights=parse_bool_input(
            "Set metagraph weights when updated gets really high to not dereg?",
            default=False,
        ),
        refresh_nodes=(parse_bool_input(
            "Refresh nodes?", default=True) if dev else True),
        localhost=parse_bool_input(
            "Use localhost?", default=True) if dev else False,
    )
    return vars(config)


def generate_config(dev: bool = False, miner: bool = False) -> dict[str, Any]:
    if miner:
        return generate_miner_config(dev)
    else:
        return generate_validator_config(dev)


def write_config_to_file(config: dict[str, Any], env: str) -> None:
    filename = f".{env}.env"
    with open(filename, "w") as f:
        for key, value in config.items():
            if value is not None:
                f.write(f"{key.upper()}={value}\n")


if __name__ == "__main__":
    args = parse_args()
    print("\n✨ Welcome to the Config Environment Generator! ✨\n")

    if args.miner:
        config = generate_config(miner=True)
        name = "1"

    else:
        env = "dev" if args.dev else "prod"
        name = "vali"
        config = generate_config(dev=args.dev)

    write_config_to_file(config, name)
    print(f"Configuration has been written to .{name}.env")
