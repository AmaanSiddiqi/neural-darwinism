import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from typing import Optional
import colony.config as cfg
from colony.training.roles import ROLES


class ModelManager:
    """
    Loads the base model once and pre-loads all role LoRA adapters.
    Switching roles uses set_adapter() — instant, no disk I/O per generation.
    """

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[PeftModel] = None  # PeftModel with all adapters loaded
        self._has_adapters: bool = False
        self._load()

    def _load(self):
        print(f"Loading {cfg.MODEL_ID} on {cfg.DEVICE}...")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        quant_config = None
        if cfg.LOAD_IN_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        base = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if not cfg.LOAD_IN_4BIT else None,
            device_map=cfg.DEVICE,
        )
        base.eval()

        # Pre-load all trained role adapters — set_adapter() is then instantaneous
        adapter_dir = Path(cfg.ADAPTER_DIR)
        trained = [r for r in ROLES if (adapter_dir / r).exists()]

        if trained:
            print(f"Loading {len(trained)} role adapters: {trained}")
            first, rest = trained[0], trained[1:]
            self.model = PeftModel.from_pretrained(base, str(adapter_dir / first), adapter_name=first)
            for role in rest:
                self.model.load_adapter(str(adapter_dir / role), adapter_name=role)
            self._has_adapters = True
            print("All adapters loaded.")
        else:
            # No adapters trained yet — wrap base model so interface is uniform
            self.model = base
            self._has_adapters = False

        if cfg.TORCH_COMPILE:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("torch.compile applied (reduce-overhead).")
            except Exception as e:
                print(f"torch.compile skipped: {e}")

    def _set_adapter(self, role: Optional[str]):
        """Switch to role adapter or disable adapters for base model behaviour."""
        if not self._has_adapters:
            return
        if role and role in ROLES and role in self.model.peft_config:
            self.model.set_adapter(role)
        else:
            # disable_adapter() is a context manager in newer PEFT; use the layer-level call instead
            self.model.disable_adapter_layers()

    def generate(
        self,
        prompt: str,
        role: Optional[str] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        self._set_adapter(role)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(cfg.DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def judge(self, task: str, response: str) -> tuple[bool, float]:
        """Score via Claude Haiku — external, non-circular quality signal."""
        from colony.judge import judge_response
        return judge_response(task, response)

    def reload_adapter(self, role: str):
        """Hot-swap a role's adapter from disk after fine-tuning."""
        adapter_path = Path(cfg.ADAPTER_DIR) / role
        if not adapter_path.exists():
            return
        if self._has_adapters:
            if role in self.model.peft_config:
                self.model.delete_adapter(role)
            self.model.load_adapter(str(adapter_path), adapter_name=role)
        else:
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), adapter_name=role)
            self._has_adapters = True
        print(f"[ModelManager] {role} adapter hot-swapped.")

    def fine_tune_role(self, role: str, entries: list):
        """
        Fine-tune a role adapter from memory entries.
        Loads a temporary base model for training so the inference model is untouched.
        """
        import gc
        from transformers import AutoModelForCausalLM
        from colony.training.lora_trainer import train_role_from_entries

        print(f"[OnlineTrain] Fine-tuning {role} on {len(entries)} examples...")
        temp_base = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            torch_dtype=torch.float16,
            device_map=cfg.DEVICE,
        )
        temp_base.enable_input_require_grads()
        try:
            train_role_from_entries(role, entries, temp_base, self.tokenizer, cfg.ADAPTER_DIR, epochs=2)
        finally:
            del temp_base
            gc.collect()
            torch.cuda.empty_cache()

        self.reload_adapter(role)
        print(f"[OnlineTrain] {role} adapter updated and hot-swapped.")

