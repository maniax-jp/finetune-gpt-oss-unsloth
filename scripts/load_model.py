#!/usr/bin/env python3
"""
GPT-OSS 20B model loading script with Unsloth + QLoRA
RTX 5090 (32GB VRAM) optimized configuration
"""

import torch
from unsloth import FastLanguageModel
import os

def load_gpt_oss_20b(
    model_name="openai/gpt-oss-20b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    use_qlora=True,
):
    """
    Load GPT-OSS 20B model with Unsloth optimizations

    Args:
        model_name: HuggingFace model ID or local path
        max_seq_length: Maximum sequence length (default: 2048)
        dtype: Data type (None = auto)
        load_in_4bit: Use 4-bit quantization (QLoRA)
        use_qlora: Apply QLoRA adapters

    Returns:
        model, tokenizer
    """

    print("=" * 60)
    print("GPT-OSS 20B Model Loading with Unsloth")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"4-bit quantization: {load_in_4bit}")
    print(f"QLoRA: {use_qlora}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)

    # Load model with Unsloth
    print("\n📥 Loading model with Unsloth...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,  # None = auto detection
        load_in_4bit=load_in_4bit,
        # For RTX 5090 (Blackwell architecture)
        # Unsloth will automatically detect and use optimal settings
    )

    print("✅ Model loaded successfully!")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   VRAM allocated: {memory_allocated:.2f} GB")
        print(f"   VRAM reserved: {memory_reserved:.2f} GB")

    # Apply QLoRA if requested
    if use_qlora:
        print("\n🔧 Applying QLoRA configuration...")

        model = FastLanguageModel.get_peft_model(
            model,
            r=32,  # LoRA rank (Unsloth tested with 32)
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # MLP
            ],
            lora_alpha=32,  # LoRA alpha (usually same as rank)
            lora_dropout=0,  # Unsloth optimized (0 = no dropout)
            bias="none",     # No bias training
            use_gradient_checkpointing="unsloth",  # Unsloth gradient checkpointing
            random_state=3407,
            use_rslora=False,  # Rank stabilized LoRA
            loftq_config=None,
        )

        print("✅ QLoRA configuration applied!")
        print(f"   LoRA rank: 32")
        print(f"   Target modules: q, k, v, o, gate, up, down")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   VRAM allocated (with LoRA): {memory_allocated:.2f} GB")
            print(f"   VRAM reserved (with LoRA): {memory_reserved:.2f} GB")

    # Print model info
    print("\n📊 Model Information:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Tokenizer: {type(tokenizer).__name__}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %: {trainable_pct:.4f}%")

    print("\n" + "=" * 60)
    print("✅ Model ready for fine-tuning!")
    print("=" * 60 + "\n")

    return model, tokenizer


def test_inference(model, tokenizer, prompt="高市早苗さんについて教えてください。"):
    """Test model inference with a sample prompt"""

    print("=" * 60)
    print("Testing Inference")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    # Format as Harmony chat
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"Formatted input:\n{input_text}\n")
    print("-" * 60)

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    print(f"Input token count: {inputs.input_ids.shape[1]}")

    # Generate (fast inference mode)
    FastLanguageModel.for_inference(model)

    print("\n🤖 Generating response...\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Response:")
    print(response)
    print("\n" + "=" * 60 + "\n")

    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load GPT-OSS 20B with Unsloth")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no-qlora", action="store_true", help="Disable QLoRA")
    parser.add_argument("--test-inference", action="store_true", help="Run inference test")
    parser.add_argument("--prompt", default="高市早苗さんについて教えてください。", help="Test prompt")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_gpt_oss_20b(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        use_qlora=not args.no_qlora,
    )

    # Test inference if requested
    if args.test_inference:
        test_inference(model, tokenizer, prompt=args.prompt)
