#!/usr/bin/env python3
"""
GPT-OSS 20B Fine-tuning Script with Unsloth + QLoRA
Optimized for RTX 5090 (32GB VRAM)
"""

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import os
from datetime import datetime

def load_model_and_tokenizer(
    model_name="openai/gpt-oss-20b",
    max_seq_length=2048,
    load_in_4bit=True,
):
    """Load GPT-OSS 20B with Unsloth + QLoRA"""

    print("=" * 60)
    print("Loading GPT-OSS 20B Model")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )

    # Apply QLoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("✅ Model loaded with QLoRA configuration")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   VRAM allocated: {memory_allocated:.2f} GB")

    return model, tokenizer


def load_harmony_dataset(dataset_path="dataset/takaichi_sanae_qa_harmony.jsonl"):
    """Load Harmony format dataset"""

    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")

    # Load JSONL dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    print(f"✅ Dataset loaded: {len(dataset)} conversations")

    return dataset


def formatting_prompts_func(examples, tokenizer):
    """
    Format examples for training with Harmony format
    Unsloth will automatically apply chat template
    Returns list of strings as required by Unsloth
    """
    # Handle both single example (dict) and batch (dict with lists)
    if isinstance(examples["messages"], list) and len(examples["messages"]) > 0:
        # Check if it's a batch or single example
        if isinstance(examples["messages"][0], list):
            # Batch of examples
            messages_list = examples["messages"]
        else:
            # Single example
            messages_list = [examples["messages"]]
    else:
        messages_list = [examples["messages"]]

    texts = []
    for messages in messages_list:
        # Apply chat template (Harmony format automatically applied for GPT-OSS)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return texts  # Return list directly, not dict


def train(
    model_name="openai/gpt-oss-20b",
    dataset_path="dataset/takaichi_sanae_qa_harmony.jsonl",
    output_dir="outputs",
    max_seq_length=2048,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=1,
    save_steps=50,
):
    """Execute fine-tuning"""

    print("=" * 60)
    print("GPT-OSS 20B Fine-tuning with Unsloth")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_train_epochs}")
    print("=" * 60 + "\n")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=max_seq_length,
    )

    # Load dataset
    dataset = load_harmony_dataset(dataset_path)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_timestamped = f"{output_dir}/gpt-oss-20b-takaichi-{timestamp}"
    os.makedirs(output_dir_timestamped, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir_timestamped}\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir_timestamped,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard for now
    )

    print("🔧 Training Arguments:")
    print(f"   Per device batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_train_epochs}")
    print(f"   Warmup ratio: {warmup_ratio}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Optimizer: adamw_8bit")
    print(f"   LR scheduler: cosine")
    print(f"   FP16/BF16: {'BF16' if is_bfloat16_supported() else 'FP16'}")
    print(f"   Logging steps: {logging_steps}")
    print(f"   Save steps: {save_steps}")
    print()

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=data_collator,
        formatting_func=lambda examples: formatting_prompts_func(examples, tokenizer),
        args=training_args,
    )

    print("=" * 60)
    print("🚀 Starting Fine-tuning")
    print("=" * 60 + "\n")

    # Enable native 2x faster training
    FastLanguageModel.for_training(model)

    # Show GPU stats before training
    if torch.cuda.is_available():
        print("📊 GPU Stats (Before Training):")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print()

    # Train!
    trainer_stats = trainer.train()

    print("\n" + "=" * 60)
    print("✅ Fine-tuning Complete!")
    print("=" * 60)

    # Show GPU stats after training
    if torch.cuda.is_available():
        print("\n📊 GPU Stats (After Training):")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"   Peak allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

    # Print training stats
    print("\n📈 Training Statistics:")
    print(f"   Train loss: {trainer_stats.training_loss:.4f}")
    print(f"   Train runtime: {trainer_stats.metrics['train_runtime']:.2f}s")
    print(f"   Train samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    print(f"   Train steps/sec: {trainer_stats.metrics['train_steps_per_second']:.2f}")

    # Save final model
    print(f"\n💾 Saving final model to {output_dir_timestamped}/final")
    model.save_pretrained(f"{output_dir_timestamped}/final")
    tokenizer.save_pretrained(f"{output_dir_timestamped}/final")

    print("\n" + "=" * 60)
    print("🎉 Fine-tuning Session Complete!")
    print("=" * 60)
    print(f"\n📁 Model saved to: {output_dir_timestamped}/final")
    print(f"📊 Training logs: {output_dir_timestamped}")
    print()

    return model, tokenizer, output_dir_timestamped


def test_finetuned_model(model, tokenizer, prompts=None):
    """Test the fine-tuned model with sample prompts"""

    if prompts is None:
        prompts = [
            "高市早苗さんについて教えてください。",
            "サナエノミクスとは何ですか？",
            "高市早苗さんの政治哲学について説明してください。",
        ]

    print("\n" + "=" * 60)
    print("🧪 Testing Fine-tuned Model")
    print("=" * 60 + "\n")

    # Switch to inference mode
    FastLanguageModel.for_inference(model)

    for i, prompt in enumerate(prompts, 1):
        print(f"Test {i}/{len(prompts)}")
        print("-" * 60)
        print(f"Prompt: {prompt}\n")

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "<|start|>assistant" in response:
            response = response.split("<|start|>assistant")[-1]

        print(f"Response:\n{response}\n")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS 20B with Unsloth")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--dataset", default="dataset/takaichi_sanae_qa_harmony.jsonl", help="Dataset path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--test", action="store_true", help="Test model after training")

    args = parser.parse_args()

    # Train
    model, tokenizer, output_dir = train(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
    )

    # Test if requested
    if args.test:
        test_finetuned_model(model, tokenizer)
