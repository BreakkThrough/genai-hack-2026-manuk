from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class AzureDIConfig:
    endpoint: str = os.getenv("AZURE_DI_ENDPOINT", "")
    key: str = os.getenv("AZURE_DI_KEY", "")


class AzureOpenAIConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    key: str = os.getenv("AZURE_OPENAI_KEY", "")
    deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
