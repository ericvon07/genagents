# D1 (System One Lab, 003-IP DS-C): the paper ships this file gitignored, with the key
# typed into it. Here every value comes from the environment, so the file holds no secret
# and is committed; the lab's runner loads <lab>/.env before importing the fork. With
# nothing set, the values are the paper's example-settings.py and PREDICTOR is `cot`, the
# paper's chain-of-thought predictor.
import os
from pathlib import Path

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
KEY_OWNER = os.environ.get("KEY_OWNER", "NAME")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")

PREDICTOR = os.environ.get("PREDICTOR", "cot")


DEBUG = os.environ.get("DEBUG", "") not in ("", "0", "false", "False")

MAX_CHUNK_SIZE = int(os.environ.get("MAX_CHUNK_SIZE", "4"))

LLM_VERS = os.environ.get("LLM_VERS", "gpt-4o-mini")

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"

## To do: Are the following needed in the new structure? Ideally Populations_Dir is for the user to define.
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations" 
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
