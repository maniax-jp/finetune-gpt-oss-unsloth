#!/usr/bin/env python3
"""
Export fine-tuned model to various formats
"""

import argparse
from unsloth import FastLanguageModel


def export_merged_model(model_path: str, output_path: str):
    """Export model with LoRA weights merged (BF16)"""
    print("="*60)
    print("Exporting Merged Model (BF16)")
    print("="*60)
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Merge and save in BF16
    print("\nMerging LoRA weights and saving in BF16 format...")
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method="merged_16bit"
    )

    print(f"\n✅ Merged model saved to: {output_path}")
    print("="*60)


def export_gguf_model(model_path: str, output_path: str, quantization: str = "q8_0"):
    """Export model to GGUF format for Ollama"""
    print("="*60)
    print(f"Exporting GGUF Model ({quantization.upper()})")
    print("="*60)
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Export to GGUF
    print(f"\nExporting to GGUF format with {quantization} quantization...")
    model.save_pretrained_gguf(
        output_path,
        tokenizer,
        quantization_method=quantization
    )

    print(f"\n✅ GGUF model saved to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model (e.g., outputs/gpt-oss-20b-takaichi-xxx/final)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="exported_models/gpt-oss-20b-takaichi",
        help="Output path for exported model"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["merged", "gguf", "both"],
        default="both",
        help="Export format: merged (BF16), gguf (for Ollama), or both"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16", "f32"],
        default="q8_0",
        help="GGUF quantization method (only used with --format gguf or both)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("GPT-OSS 20B Model Export")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Format: {args.format}")
    if args.format in ["gguf", "both"]:
        print(f"Quantization: {args.quantization}")
    print("="*60 + "\n")

    # Export merged model
    if args.format in ["merged", "both"]:
        export_merged_model(
            model_path=args.model_path,
            output_path=f"{args.output_path}-merged"
        )
        print()

    # Export GGUF model
    if args.format in ["gguf", "both"]:
        export_gguf_model(
            model_path=args.model_path,
            output_path=f"{args.output_path}-gguf",
            quantization=args.quantization
        )
        print()

    print("="*60)
    print("🎉 Export Complete!")
    print("="*60)

    # Print usage instructions
    if args.format in ["gguf", "both"]:
        print("\n📝 To use with Ollama:")
        print("1. Create a Modelfile:")
        print(f"   FROM ./{args.output_path}-gguf/{args.quantization.upper()}.gguf")
        print("2. Import to Ollama:")
        print("   ollama create gpt-oss-takaichi -f Modelfile")
        print("3. Run:")
        print("   ollama run gpt-oss-takaichi")


if __name__ == "__main__":
    main()
