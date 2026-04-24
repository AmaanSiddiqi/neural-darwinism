"""
Fine-tunes a LoRA adapter per role using the role-specific datasets.
Runs in float16 — no 4-bit needed for 1.5B models on 16GB VRAM.

Usage:
    python -m colony.training.lora_trainer [--roles analyst critic synthesizer explorer]
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

import colony.config as cfg
from colony.training.roles import ROLES, ROLE_PROMPTS, ROLE_EXAMPLES


def build_dataset(role: str, tokenizer, max_length: int = 512) -> Dataset:
    examples = ROLE_EXAMPLES[role]
    system = ROLE_PROMPTS[role]

    records = []
    for question, answer in examples:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)

        # Labels: mask the prompt tokens, only compute loss on the answer
        input_ids = tokenized["input_ids"]
        # Find where the answer starts by encoding up to the assistant turn
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:max_length]

        records.append({
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        })

    return Dataset.from_list(records)


def train_role(role: str, base_model, tokenizer, output_dir: str, epochs: int = 3, dataset=None):
    print(f"\n{'='*50}")
    print(f"Training adapter: {role}")
    print(f"{'='*50}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    if dataset is None:
        dataset = build_dataset(role, tokenizer)
    print(f"Dataset: {len(dataset)} examples")

    adapter_path = str(Path(output_dir) / role)

    args = TrainingArguments(
        output_dir=adapter_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save only the LoRA adapter weights, not the full model
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    # Detach adapter so base model is clean for the next role
    model.unload()
    del model
    torch.cuda.empty_cache()


def train_role_from_entries(role: str, entries, base_model, tokenizer, output_dir: str, epochs: int = 2):
    """Fine-tune a role adapter from memory bank entries (online learning)."""
    system = ROLE_PROMPTS[role]
    records = []
    for entry in entries:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": entry.task},
            {"role": "assistant", "content": entry.response},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text, max_length=512, truncation=True, padding=False)
        input_ids = tokenized["input_ids"]

        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        records.append({
            "input_ids":      input_ids[:512],
            "attention_mask": tokenized["attention_mask"][:512],
            "labels":         labels[:512],
        })

    dataset = Dataset.from_list(records)
    train_role(role, base_model, tokenizer, output_dir, epochs=epochs, dataset=dataset)


def train_all(roles: list[str], output_dir: str = "./adapters", epochs: int = 3):
    print(f"Loading base model: {cfg.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_ID,
        torch_dtype=torch.float16,
        device_map=cfg.DEVICE,
    )
    base_model.enable_input_require_grads()

    for role in roles:
        train_role(role, base_model, tokenizer, output_dir, epochs=epochs)

    print(f"\nAll adapters trained. Saved under {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", default=ROLES)
    parser.add_argument("--output-dir", default="./adapters")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train_all(roles=args.roles, output_dir=args.output_dir, epochs=args.epochs)
