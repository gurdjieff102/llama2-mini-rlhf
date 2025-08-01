import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from trl import DPOTrainer, SFTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        load_in_4bit=cfg.get("load_in_4bit", False),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=False)

    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if "sft" in args.config:
        dataset = load_dataset("json", data_files=cfg["train_dataset"])["train"]
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            learning_rate=cfg.get("learning_rate", 2e-4),
            num_train_epochs=cfg.get("num_train_epochs", 1),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        )
    else:
        dataset = load_dataset("json", data_files=cfg["train_dataset"])["train"]
        trainer = DPOTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            learning_rate=cfg.get("learning_rate", 1e-5),
            num_train_epochs=cfg.get("num_train_epochs", 1),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        )

    trainer.train()

if __name__ == "__main__":
    main()
