import os
from dotenv import load_dotenv

load_dotenv()

# Must be set before any ROCm/HIP import
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE = os.getenv("DEVICE", "cuda")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"

MAX_NEURONS = int(os.getenv("MAX_NEURONS", "20"))
PRUNE_THRESHOLD = float(os.getenv("PRUNE_THRESHOLD", "0.2"))
NEUROGENESIS_THRESHOLD = float(os.getenv("NEUROGENESIS_THRESHOLD", "0.7"))
HEBBIAN_LR = float(os.getenv("HEBBIAN_LEARNING_RATE", "0.1"))
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "./adapters")

# External judge + self-improvement
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MEMORY_CAPACITY = int(os.getenv("MEMORY_CAPACITY", "16"))
MEMORY_SCORE_THRESHOLD = float(os.getenv("MEMORY_SCORE_THRESHOLD", "0.65"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
TORCH_COMPILE = os.getenv("TORCH_COMPILE", "false").lower() == "true"
CORTEX_STATE_PATH = os.getenv("CORTEX_STATE_PATH", "./cortex_state.json")
BENCHMARK_HISTORY_PATH = os.getenv("BENCHMARK_HISTORY_PATH", "./benchmark_history.json")
ROLE_MEMORY_PATH = os.getenv("ROLE_MEMORY_PATH", "./role_memory.json")
BENCHMARK_INTERVAL = int(os.getenv("BENCHMARK_INTERVAL", "25"))
