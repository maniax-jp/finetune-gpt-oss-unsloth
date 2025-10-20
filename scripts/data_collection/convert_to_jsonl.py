#!/usr/bin/env python3
"""
Convert Harmony dataset to JSONL format for training
"""

import json

# データ読み込み
with open("data/processed/full_harmony_dataset.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# メタデータがある場合は除外
if isinstance(data, dict) and 'data' in data:
    data = data['data']

# JSONL形式で保存（1行1JSON）
output_file = "dataset/takaichi_sanae_qa_harmony_v2.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        # conversationsをmessagesに変換（role: from, content: value）
        messages = []
        for conv in item['conversations']:
            messages.append({
                "role": "user" if conv['from'] == "human" else "assistant",
                "content": conv['value']
            })

        json_line = {"messages": messages}
        f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

print(f"✅ Converted {len(data)} samples to {output_file}")
