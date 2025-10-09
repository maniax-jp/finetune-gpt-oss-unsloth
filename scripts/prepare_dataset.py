#!/usr/bin/env python3
"""
Dataset preparation script for GPT-OSS fine-tuning
Converts JSONL format to Harmony format required by GPT-OSS
"""

import json
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_harmony_format(data, reasoning_effort="medium"):
    """
    Convert standard conversation format to Harmony format

    Args:
        data: List of conversation dictionaries
        reasoning_effort: "low", "medium", or "high"

    Returns:
        List of Harmony-formatted conversations
    """
    harmony_data = []

    for item in data:
        harmony_item = {
            "messages": item["messages"],
            "reasoning_effort": reasoning_effort
        }
        harmony_data.append(harmony_item)

    return harmony_data

def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_dataset_stats(data):
    """Get dataset statistics"""
    total_conversations = len(data)
    total_messages = sum(len(item["messages"]) for item in data)
    total_chars = sum(
        sum(len(msg["content"]) for msg in item["messages"])
        for item in data
    )

    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_characters": total_chars,
        "avg_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0,
        "avg_chars_per_conversation": total_chars / total_conversations if total_conversations > 0 else 0
    }

def main():
    # Paths
    input_file = Path("dataset/takaichi_sanae_qa.jsonl")
    output_file = Path("dataset/takaichi_sanae_qa_harmony.jsonl")

    # Load data
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)

    # Get statistics
    stats = get_dataset_stats(data)
    print("\n📊 Dataset Statistics:")
    print(f"  Total conversations: {stats['total_conversations']}")
    print(f"  Total messages: {stats['total_messages']}")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Avg messages/conversation: {stats['avg_messages_per_conversation']:.1f}")
    print(f"  Avg characters/conversation: {stats['avg_chars_per_conversation']:.1f}")

    # Convert to Harmony format
    print("\n🔄 Converting to Harmony format...")
    harmony_data = convert_to_harmony_format(data, reasoning_effort="medium")

    # Save converted data
    print(f"💾 Saving to {output_file}...")
    save_jsonl(harmony_data, output_file)

    print("\n✅ Dataset preparation complete!")
    print(f"   Input: {input_file} ({len(data)} conversations)")
    print(f"   Output: {output_file} ({len(harmony_data)} conversations)")
    print(f"   Reasoning effort: medium")

if __name__ == "__main__":
    main()
