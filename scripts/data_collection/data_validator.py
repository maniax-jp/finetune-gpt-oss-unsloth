#!/usr/bin/env python3
"""
Data Validator for Takaichi Sanae QA Dataset
高市早苗QAデータセット検証ツール
"""

import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QAValidator:
    """QAデータセットの検証クラス"""

    def __init__(self):
        self.valid_categories = [f"CAT-{i:02d}" for i in range(1, 11)]
        self.reliability_levels = ["A", "B", "C", "D"]
        self.question_types = [f"Q-TYPE-{i:02d}" for i in range(1, 6)]

    def validate_qa_pair(self, qa: Dict) -> Tuple[bool, List[str]]:
        """
        単一のQAペアを検証

        Args:
            qa: QAペアの辞書

        Returns:
            (検証結果, エラーリスト)のタプル
        """
        errors = []

        # 必須フィールドのチェック
        if not qa.get("id"):
            errors.append("Missing 'id' field")
        if not qa.get("question"):
            errors.append("Missing 'question' field")
        if not qa.get("answer"):
            errors.append("Missing 'answer' field")
        if not qa.get("category"):
            errors.append("Missing 'category' field")

        # カテゴリーの検証
        if qa.get("category") not in self.valid_categories:
            errors.append(f"Invalid category: {qa.get('category')}")

        # ソース検証
        source = qa.get("source", {})
        if not source:
            errors.append("Missing 'source' field")
        else:
            if not source.get("url"):
                errors.append("Missing source URL")
            if source.get("reliability") not in self.reliability_levels:
                errors.append(f"Invalid reliability level: {source.get('reliability')}")

        # 質問・回答の長さチェック
        question = qa.get("question", "")
        answer = qa.get("answer", "")

        if len(question) < 5:
            errors.append("Question is too short (<5 chars)")
        if len(question) > 200:
            errors.append("Question is too long (>200 chars)")
        if len(answer) < 3:
            errors.append("Answer is too short (<3 chars)")
        if len(answer) > 500:
            errors.append("Answer is too long (>500 chars)")

        # 質問・回答の内容チェック
        if not self._contains_takaichi_reference(question):
            errors.append("Question does not reference '高市早苗' or related terms")

        # IDフォーマットチェック
        if qa.get("id") and not qa["id"].startswith("TAKAICHI-QA-"):
            errors.append(f"Invalid ID format: {qa['id']}")

        return len(errors) == 0, errors

    def _contains_takaichi_reference(self, text: str) -> bool:
        """
        テキストに高市早苗への言及があるかチェック

        Args:
            text: チェックするテキスト

        Returns:
            言及がある場合True
        """
        keywords = ["高市早苗", "高市", "早苗", "高市氏", "高市議員", "高市大臣", "高市先生"]
        return any(keyword in text for keyword in keywords)

    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """
        データセット全体を検証

        Args:
            dataset: QAペアのリスト

        Returns:
            検証結果の辞書
        """
        results = {
            "total": len(dataset),
            "valid": 0,
            "invalid": 0,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }

        # 各QAペアを検証
        valid_qa_pairs = []
        for idx, qa in enumerate(dataset):
            is_valid, errors = self.validate_qa_pair(qa)

            if is_valid:
                results["valid"] += 1
                valid_qa_pairs.append(qa)
            else:
                results["invalid"] += 1
                results["errors"].append({
                    "index": idx,
                    "id": qa.get("id"),
                    "errors": errors
                })

        # 統計情報の計算
        if valid_qa_pairs:
            results["statistics"] = self._calculate_statistics(valid_qa_pairs)

        # 警告の生成
        results["warnings"] = self._generate_warnings(results["statistics"])

        return results

    def _calculate_statistics(self, dataset: List[Dict]) -> Dict:
        """
        データセットの統計情報を計算

        Args:
            dataset: 検証済みQAペアのリスト

        Returns:
            統計情報の辞書
        """
        # カテゴリー分布
        categories = [qa.get("category") for qa in dataset]
        category_counts = Counter(categories)

        # 信頼性レベル分布
        reliabilities = [qa.get("source", {}).get("reliability") for qa in dataset]
        reliability_counts = Counter(reliabilities)

        # 質問・回答の長さ統計
        question_lengths = [len(qa.get("question", "")) for qa in dataset]
        answer_lengths = [len(qa.get("answer", "")) for qa in dataset]

        # 検証済みの割合
        verified_count = sum(1 for qa in dataset if qa.get("verification", {}).get("verified"))
        verification_rate = (verified_count / len(dataset) * 100) if dataset else 0

        return {
            "total_samples": len(dataset),
            "category_distribution": dict(category_counts),
            "reliability_distribution": dict(reliability_counts),
            "average_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            "average_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            "min_question_length": min(question_lengths) if question_lengths else 0,
            "max_question_length": max(question_lengths) if question_lengths else 0,
            "min_answer_length": min(answer_lengths) if answer_lengths else 0,
            "max_answer_length": max(answer_lengths) if answer_lengths else 0,
            "verification_rate": verification_rate,
            "verified_count": verified_count,
            "unverified_count": len(dataset) - verified_count
        }

    def _generate_warnings(self, statistics: Dict) -> List[str]:
        """
        統計情報から警告を生成

        Args:
            statistics: 統計情報の辞書

        Returns:
            警告メッセージのリスト
        """
        warnings = []

        # カテゴリー分布の偏りチェック
        category_dist = statistics.get("category_distribution", {})
        if category_dist:
            total = sum(category_dist.values())
            for category, count in category_dist.items():
                ratio = count / total * 100
                if ratio < 5:
                    warnings.append(f"Category {category} has low representation: {ratio:.1f}%")
                elif ratio > 30:
                    warnings.append(f"Category {category} is over-represented: {ratio:.1f}%")

        # 信頼性レベルのチェック
        reliability_dist = statistics.get("reliability_distribution", {})
        total_samples = statistics.get("total_samples", 0)
        if reliability_dist and total_samples:
            a_b_count = reliability_dist.get("A", 0) + reliability_dist.get("B", 0)
            a_b_ratio = a_b_count / total_samples * 100
            if a_b_ratio < 70:
                warnings.append(f"Low A+B reliability ratio: {a_b_ratio:.1f}% (target: >85%)")

        # 検証率のチェック
        verification_rate = statistics.get("verification_rate", 0)
        if verification_rate < 100:
            warnings.append(f"Not all samples are verified: {verification_rate:.1f}%")

        # サンプル数のチェック
        if total_samples < 300:
            warnings.append(f"Sample count below target: {total_samples} (target: 300+)")

        return warnings

    def check_duplicates(self, dataset: List[Dict]) -> List[Dict]:
        """
        データセット内の重複をチェック

        Args:
            dataset: QAペアのリスト

        Returns:
            重複情報のリスト
        """
        duplicates = []
        seen_questions = {}

        for idx, qa in enumerate(dataset):
            question = qa.get("question", "").strip().lower()

            if question in seen_questions:
                duplicates.append({
                    "question": qa.get("question"),
                    "indices": [seen_questions[question], idx],
                    "ids": [dataset[seen_questions[question]].get("id"), qa.get("id")]
                })
            else:
                seen_questions[question] = idx

        return duplicates

    def generate_report(self, validation_results: Dict, output_file: str = None) -> str:
        """
        検証結果のレポートを生成

        Args:
            validation_results: validate_dataset()の結果
            output_file: レポート出力先ファイル（Noneの場合は文字列で返す）

        Returns:
            レポート文字列
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("QA Dataset Validation Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # サマリー
        report_lines.append("## Summary")
        report_lines.append(f"Total samples: {validation_results['total']}")
        report_lines.append(f"Valid samples: {validation_results['valid']}")
        report_lines.append(f"Invalid samples: {validation_results['invalid']}")
        valid_rate = (validation_results['valid'] / validation_results['total'] * 100) if validation_results['total'] else 0
        report_lines.append(f"Valid rate: {valid_rate:.1f}%")
        report_lines.append("")

        # 統計情報
        if validation_results.get("statistics"):
            stats = validation_results["statistics"]
            report_lines.append("## Statistics")
            report_lines.append(f"Average question length: {stats.get('average_question_length', 0):.1f} chars")
            report_lines.append(f"Average answer length: {stats.get('average_answer_length', 0):.1f} chars")
            report_lines.append(f"Verification rate: {stats.get('verification_rate', 0):.1f}%")
            report_lines.append("")

            # カテゴリー分布
            report_lines.append("### Category Distribution")
            cat_dist = stats.get("category_distribution", {})
            for category in sorted(cat_dist.keys()):
                count = cat_dist[category]
                ratio = count / stats['total_samples'] * 100 if stats['total_samples'] else 0
                report_lines.append(f"  {category}: {count} ({ratio:.1f}%)")
            report_lines.append("")

            # 信頼性分布
            report_lines.append("### Reliability Distribution")
            rel_dist = stats.get("reliability_distribution", {})
            for level in ["A", "B", "C", "D"]:
                count = rel_dist.get(level, 0)
                ratio = count / stats['total_samples'] * 100 if stats['total_samples'] else 0
                report_lines.append(f"  Level {level}: {count} ({ratio:.1f}%)")
            report_lines.append("")

        # エラー
        if validation_results["errors"]:
            report_lines.append("## Errors")
            for error_info in validation_results["errors"][:10]:  # 最初の10件のみ
                report_lines.append(f"  [{error_info['index']}] {error_info['id']}")
                for error in error_info["errors"]:
                    report_lines.append(f"    - {error}")
            if len(validation_results["errors"]) > 10:
                report_lines.append(f"  ... and {len(validation_results['errors']) - 10} more errors")
            report_lines.append("")

        # 警告
        if validation_results["warnings"]:
            report_lines.append("## Warnings")
            for warning in validation_results["warnings"]:
                report_lines.append(f"  - {warning}")
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")

        return report


def main():
    """メイン実行関数（テスト用）"""
    # テストデータ
    test_dataset = [
        {
            "id": "TAKAICHI-QA-0001",
            "category": "CAT-01",
            "question": "高市早苗さんは何党ですか？",
            "answer": "自由民主党（自民党）です。",
            "source": {
                "url": "https://www.sanae.gr.jp/",
                "reliability": "A"
            },
            "verification": {
                "verified": True
            }
        },
        {
            "id": "TAKAICHI-QA-0002",
            "category": "INVALID",  # 無効なカテゴリー
            "question": "短",  # 短すぎる質問
            "answer": "答",  # 短すぎる回答
            "source": {
                "url": "https://example.com/",
                "reliability": "X"  # 無効な信頼性レベル
            },
            "verification": {
                "verified": False
            }
        }
    ]

    # 検証実行
    validator = QAValidator()
    results = validator.validate_dataset(test_dataset)

    # レポート生成
    report = validator.generate_report(results)
    print(report)

    # 重複チェック
    duplicates = validator.check_duplicates(test_dataset)
    if duplicates:
        logger.info(f"\nFound {len(duplicates)} duplicate(s)")
        for dup in duplicates:
            logger.info(f"  Duplicate question: {dup['question']}")


if __name__ == "__main__":
    main()
