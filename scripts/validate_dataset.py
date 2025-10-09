#!/usr/bin/env python3
"""
Dataset validation script for GPT-OSS Harmony format
Checks token lengths, format correctness, and data quality
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Unslothのトークナイザーを使用する予定だが、まずは基本的な文字数ベースの検証
# 日本語の場合、1文字≈1.5トークン程度と仮定

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def validate_harmony_format(item: Dict[str, Any], index: int) -> List[str]:
    """Validate Harmony format structure"""
    errors = []

    # Check required fields
    if "messages" not in item:
        errors.append(f"行{index}: 'messages'フィールドが見つかりません")
        return errors

    if "reasoning_effort" not in item:
        errors.append(f"行{index}: 'reasoning_effort'フィールドが見つかりません")

    # Validate reasoning_effort value
    valid_efforts = ["low", "medium", "high"]
    if item.get("reasoning_effort") not in valid_efforts:
        errors.append(f"行{index}: reasoning_effortは{valid_efforts}のいずれかである必要があります")

    # Validate messages structure
    messages = item.get("messages", [])
    if not isinstance(messages, list):
        errors.append(f"行{index}: 'messages'は配列である必要があります")
        return errors

    if len(messages) == 0:
        errors.append(f"行{index}: 'messages'が空です")
        return errors

    # Check message format
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"行{index}, メッセージ{i}: メッセージは辞書である必要があります")
            continue

        if "role" not in msg:
            errors.append(f"行{index}, メッセージ{i}: 'role'フィールドが見つかりません")
        elif msg["role"] not in ["user", "assistant", "system"]:
            errors.append(f"行{index}, メッセージ{i}: roleは'user', 'assistant', 'system'のいずれかである必要があります")

        if "content" not in msg:
            errors.append(f"行{index}, メッセージ{i}: 'content'フィールドが見つかりません")
        elif not isinstance(msg["content"], str):
            errors.append(f"行{index}, メッセージ{i}: contentは文字列である必要があります")

    return errors

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for Japanese text
    Japanese: ~1.5 tokens per character
    """
    return int(len(text) * 1.5)

def analyze_conversation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single conversation"""
    messages = item.get("messages", [])
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    total_tokens_est = estimate_tokens("".join(msg.get("content", "") for msg in messages))

    return {
        "message_count": len(messages),
        "total_chars": total_chars,
        "total_tokens_est": total_tokens_est,
        "reasoning_effort": item.get("reasoning_effort", "unknown")
    }

def validate_dataset(file_path: str) -> Dict[str, Any]:
    """Validate entire dataset"""
    print(f"📂 データセット検証: {file_path}\n")

    # Load data
    try:
        data = load_jsonl(file_path)
        print(f"✅ {len(data)}件の会話を読み込みました\n")
    except Exception as e:
        print(f"❌ エラー: ファイル読み込みに失敗しました: {e}")
        return {"valid": False}

    # Format validation
    print("🔍 フォーマット検証中...")
    all_errors = []
    for i, item in enumerate(data, 1):
        errors = validate_harmony_format(item, i)
        all_errors.extend(errors)

    if all_errors:
        print(f"❌ {len(all_errors)}件のフォーマットエラーが見つかりました:\n")
        for error in all_errors:
            print(f"  - {error}")
        print()
    else:
        print("✅ フォーマット検証: すべてクリア\n")

    # Statistical analysis
    print("📊 統計分析...")
    stats = {
        "total_conversations": len(data),
        "total_messages": 0,
        "total_chars": 0,
        "total_tokens_est": 0,
        "reasoning_distribution": {"low": 0, "medium": 0, "high": 0},
        "message_counts": [],
        "token_counts": [],
        "conversations_over_4k": 0,
        "conversations_over_8k": 0,
    }

    for item in data:
        analysis = analyze_conversation(item)
        stats["total_messages"] += analysis["message_count"]
        stats["total_chars"] += analysis["total_chars"]
        stats["total_tokens_est"] += analysis["total_tokens_est"]
        stats["message_counts"].append(analysis["message_count"])
        stats["token_counts"].append(analysis["total_tokens_est"])

        # Reasoning effort distribution
        effort = analysis["reasoning_effort"]
        if effort in stats["reasoning_distribution"]:
            stats["reasoning_distribution"][effort] += 1

        # Token length warnings
        if analysis["total_tokens_est"] > 8000:
            stats["conversations_over_8k"] += 1
        elif analysis["total_tokens_est"] > 4000:
            stats["conversations_over_4k"] += 1

    # Print statistics
    print(f"\n📈 統計サマリー:")
    print(f"  総会話数: {stats['total_conversations']}")
    print(f"  総メッセージ数: {stats['total_messages']}")
    print(f"  総文字数: {stats['total_chars']:,}")
    print(f"  推定総トークン数: {stats['total_tokens_est']:,}")
    print(f"  平均メッセージ数/会話: {stats['total_messages'] / len(data):.1f}")
    print(f"  平均トークン数/会話: {stats['total_tokens_est'] / len(data):.1f}")
    print(f"  最小トークン数: {min(stats['token_counts'])}")
    print(f"  最大トークン数: {max(stats['token_counts'])}")

    print(f"\n🧠 推論レベル分布:")
    for effort, count in stats["reasoning_distribution"].items():
        percentage = (count / len(data)) * 100
        print(f"  {effort}: {count}件 ({percentage:.1f}%)")

    # Warnings
    print(f"\n⚠️  警告:")
    if stats["conversations_over_8k"] > 0:
        print(f"  - {stats['conversations_over_8k']}件の会話が8,000トークンを超えています（メモリ使用量に注意）")
    if stats["conversations_over_4k"] > 0:
        print(f"  - {stats['conversations_over_4k']}件の会話が4,000トークンを超えています")
    if stats["conversations_over_4k"] == 0 and stats["conversations_over_8k"] == 0:
        print(f"  すべての会話が適切なトークン長です ✅")

    # Recommendations
    print(f"\n💡 推奨事項:")
    if len(data) < 50:
        print(f"  - データセットサイズが小さい（{len(data)}件）ため、概念実証用です")
        print(f"  - 本番運用には100件以上を推奨します")

    reasoning_high_pct = (stats["reasoning_distribution"]["high"] / len(data)) * 100
    if reasoning_high_pct < 30:
        print(f"  - 推論レベル'high'の会話を増やすことでモデルの推論能力が向上する可能性があります")

    # Data quality checks
    print(f"\n🔬 データ品質チェック:")
    avg_chars_per_message = stats['total_chars'] / stats['total_messages']
    if avg_chars_per_message < 20:
        print(f"  ⚠️  メッセージあたりの平均文字数が少ない（{avg_chars_per_message:.1f}文字）")
    else:
        print(f"  ✅ メッセージあたりの平均文字数: {avg_chars_per_message:.1f}文字")

    # Final verdict
    print(f"\n{'='*60}")
    if len(all_errors) == 0:
        print("✅ データセット検証: 合格")
        print("   ファインチューニングに使用できます")
        valid = True
    else:
        print("❌ データセット検証: 不合格")
        print("   エラーを修正してから再度実行してください")
        valid = False
    print(f"{'='*60}\n")

    return {
        "valid": valid,
        "errors": all_errors,
        "stats": stats
    }

if __name__ == "__main__":
    dataset_path = "dataset/takaichi_sanae_qa_harmony.jsonl"

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    result = validate_dataset(dataset_path)

    # Exit with error code if validation failed
    sys.exit(0 if result["valid"] else 1)
