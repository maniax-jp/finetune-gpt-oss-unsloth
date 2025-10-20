#!/usr/bin/env python3
"""
Comprehensive Validation for Phase 3
Phase 3: 包括的な検証と品質チェック
"""

import json
import logging
import re
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple
from data_validator import QAValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveValidator:
    """包括的な検証を行うクラス"""

    def __init__(self):
        self.validator = QAValidator()
        self.issues = []
        self.warnings = []
        self.stats = {}

    def check_category_balance(self, data: List[Dict]) -> Dict:
        """カテゴリーバランスのチェック"""
        logger.info("=== Checking Category Balance ===")

        category_counts = Counter(qa['category'] for qa in data)
        total = len(data)

        balance_report = {
            'total_samples': total,
            'category_counts': dict(category_counts),
            'category_percentages': {},
            'balance_issues': []
        }

        for cat, count in category_counts.items():
            percentage = count / total * 100
            balance_report['category_percentages'][cat] = percentage
            logger.info(f"{cat}: {count} samples ({percentage:.1f}%)")

            # バランスチェック（極端な偏りがないか）
            if percentage < 5:
                warning = f"{cat} is under-represented: {percentage:.1f}%"
                balance_report['balance_issues'].append(warning)
                self.warnings.append(warning)
            elif percentage > 25:
                warning = f"{cat} is over-represented: {percentage:.1f}%"
                balance_report['balance_issues'].append(warning)
                self.warnings.append(warning)

        return balance_report

    def check_question_quality(self, data: List[Dict]) -> Dict:
        """質問の品質チェック"""
        logger.info("\n=== Checking Question Quality ===")

        quality_issues = []
        length_stats = []

        for i, qa in enumerate(data):
            question = qa['question']
            q_len = len(question)
            length_stats.append(q_len)

            # 長さチェック
            if q_len < 5:
                issue = f"[{i}] Question too short: {question}"
                quality_issues.append(issue)

            if q_len > 100:
                issue = f"[{i}] Question too long ({q_len} chars): {question[:50]}..."
                quality_issues.append(issue)

            # 疑問形チェック
            if not any(marker in question for marker in ['？', '?', 'か', 'ですか', 'ますか', 'は']):
                issue = f"[{i}] May not be a proper question: {question}"
                quality_issues.append(issue)

        avg_length = sum(length_stats) / len(length_stats) if length_stats else 0

        quality_report = {
            'total_questions': len(data),
            'quality_issues': quality_issues,
            'avg_question_length': avg_length,
            'min_question_length': min(length_stats) if length_stats else 0,
            'max_question_length': max(length_stats) if length_stats else 0
        }

        logger.info(f"Average question length: {avg_length:.1f} chars")
        logger.info(f"Question length range: {min(length_stats)}-{max(length_stats)} chars")
        logger.info(f"Quality issues found: {len(quality_issues)}")

        return quality_report

    def check_answer_quality(self, data: List[Dict]) -> Dict:
        """回答の品質チェック"""
        logger.info("\n=== Checking Answer Quality ===")

        quality_issues = []
        length_stats = []

        for i, qa in enumerate(data):
            answer = qa['answer']
            a_len = len(answer)
            length_stats.append(a_len)

            # 長さチェック
            if a_len < 3:
                issue = f"[{i}] Answer too short: {answer}"
                quality_issues.append(issue)

            if a_len > 200:
                issue = f"[{i}] Answer too long ({a_len} chars): {answer[:50]}..."
                quality_issues.append(issue)

            # 不完全な文のチェック
            if not any(end in answer for end in ['。', '.', 'です', 'ます', 'した']):
                issue = f"[{i}] Answer may be incomplete: {answer}"
                quality_issues.append(issue)

        avg_length = sum(length_stats) / len(length_stats) if length_stats else 0

        quality_report = {
            'total_answers': len(data),
            'quality_issues': quality_issues,
            'avg_answer_length': avg_length,
            'min_answer_length': min(length_stats) if length_stats else 0,
            'max_answer_length': max(length_stats) if length_stats else 0
        }

        logger.info(f"Average answer length: {avg_length:.1f} chars")
        logger.info(f"Answer length range: {min(length_stats)}-{max(length_stats)} chars")
        logger.info(f"Quality issues found: {len(quality_issues)}")

        return quality_report

    def check_consistency(self, data: List[Dict]) -> Dict:
        """一貫性のチェック"""
        logger.info("\n=== Checking Consistency ===")

        consistency_issues = []

        # 同じ質問に対する異なる回答のチェック
        question_answers = {}
        for i, qa in enumerate(data):
            q = qa['question']
            a = qa['answer']
            if q in question_answers:
                if a != question_answers[q]['answer']:
                    issue = f"Inconsistent answers for same question:\nQ: {q}\nA1: {question_answers[q]['answer']}\nA2: {a}"
                    consistency_issues.append(issue)
            else:
                question_answers[q] = {'answer': a, 'index': i}

        # 類似質問のチェック（簡易版）
        questions = [qa['question'] for qa in data]
        similar_pairs = []

        for i in range(len(questions)):
            for j in range(i+1, len(questions)):
                q1 = questions[i]
                q2 = questions[j]
                # 簡易類似度チェック（部分一致）
                if len(q1) > 10 and len(q2) > 10:
                    if q1[:10] == q2[:10] or q1[-10:] == q2[-10:]:
                        similar_pairs.append((i, j, q1, q2))

        consistency_report = {
            'duplicate_questions': len(question_answers) - len(set(questions)),
            'consistency_issues': consistency_issues,
            'similar_question_pairs': len(similar_pairs)
        }

        logger.info(f"Duplicate questions: {consistency_report['duplicate_questions']}")
        logger.info(f"Consistency issues: {len(consistency_issues)}")
        logger.info(f"Similar question pairs: {len(similar_pairs)}")

        return consistency_report

    def check_reliability_distribution(self, data: List[Dict]) -> Dict:
        """信頼性分布のチェック"""
        logger.info("\n=== Checking Reliability Distribution ===")

        reliability_counts = Counter(qa['source']['reliability'] for qa in data)
        total = len(data)

        rel_report = {
            'reliability_counts': dict(reliability_counts),
            'reliability_percentages': {}
        }

        for level in ['A', 'B', 'C', 'D']:
            count = reliability_counts.get(level, 0)
            percentage = count / total * 100 if total > 0 else 0
            rel_report['reliability_percentages'][level] = percentage
            logger.info(f"Level {level}: {count} samples ({percentage:.1f}%)")

        a_b_count = reliability_counts.get('A', 0) + reliability_counts.get('B', 0)
        a_b_ratio = a_b_count / total * 100 if total > 0 else 0
        rel_report['a_b_ratio'] = a_b_ratio

        logger.info(f"A+B ratio: {a_b_ratio:.1f}%")

        if a_b_ratio < 85:
            warning = f"Low A+B reliability ratio: {a_b_ratio:.1f}% (target: ≥85%)"
            self.warnings.append(warning)

        return rel_report

    def perform_comprehensive_validation(self, data: List[Dict]) -> Dict:
        """包括的な検証を実行"""
        logger.info("=" * 80)
        logger.info("Phase 3: Comprehensive Validation")
        logger.info("=" * 80)

        # 基本的な検証
        logger.info("\n=== Basic Validation ===")
        basic_results = self.validator.validate_dataset(data)
        logger.info(f"Valid samples: {basic_results['valid']}/{len(data)}")
        logger.info(f"Invalid samples: {basic_results['invalid']}")

        # カテゴリーバランスチェック
        balance_report = self.check_category_balance(data)

        # 質問品質チェック
        question_report = self.check_question_quality(data)

        # 回答品質チェック
        answer_report = self.check_answer_quality(data)

        # 一貫性チェック
        consistency_report = self.check_consistency(data)

        # 信頼性分布チェック
        reliability_report = self.check_reliability_distribution(data)

        # 重複チェック
        logger.info("\n=== Checking Duplicates ===")
        duplicates = self.validator.check_duplicates(data)
        logger.info(f"Duplicates found: {len(duplicates)}")

        # 統合レポート
        comprehensive_report = {
            'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(data),
            'basic_validation': basic_results,
            'category_balance': balance_report,
            'question_quality': question_report,
            'answer_quality': answer_report,
            'consistency': consistency_report,
            'reliability': reliability_report,
            'duplicates': len(duplicates),
            'issues': self.issues,
            'warnings': self.warnings
        }

        return comprehensive_report


def generate_validation_report(report: Dict) -> str:
    """検証レポートを生成"""

    report_text = f"""
================================================================================
Comprehensive Validation Report - Phase 3
================================================================================
Generated: {report['validation_date']}

## Summary
Total Samples: {report['total_samples']}
Valid Samples: {report['basic_validation']['valid']}
Invalid Samples: {report['basic_validation']['invalid']}
Validation Rate: {report['basic_validation']['valid']/report['total_samples']*100:.1f}%

## Category Balance
"""

    for cat, count in sorted(report['category_balance']['category_counts'].items()):
        percentage = report['category_balance']['category_percentages'][cat]
        report_text += f"  {cat}: {count:3d} samples ({percentage:5.1f}%)\n"

    if report['category_balance']['balance_issues']:
        report_text += "\n  Balance Issues:\n"
        for issue in report['category_balance']['balance_issues']:
            report_text += f"    - {issue}\n"

    report_text += f"""
## Question Quality
Average Length: {report['question_quality']['avg_question_length']:.1f} chars
Length Range: {report['question_quality']['min_question_length']}-{report['question_quality']['max_question_length']} chars
Quality Issues: {len(report['question_quality']['quality_issues'])}

## Answer Quality
Average Length: {report['answer_quality']['avg_answer_length']:.1f} chars
Length Range: {report['answer_quality']['min_answer_length']}-{report['answer_quality']['max_answer_length']} chars
Quality Issues: {len(report['answer_quality']['quality_issues'])}

## Consistency
Duplicate Questions: {report['consistency']['duplicate_questions']}
Consistency Issues: {len(report['consistency']['consistency_issues'])}
Similar Question Pairs: {report['consistency']['similar_question_pairs']}

## Reliability Distribution
"""

    for level in ['A', 'B', 'C', 'D']:
        count = report['reliability']['reliability_counts'].get(level, 0)
        percentage = report['reliability']['reliability_percentages'].get(level, 0)
        report_text += f"  Level {level}: {count:3d} samples ({percentage:5.1f}%)\n"

    report_text += f"\n  A+B Ratio: {report['reliability']['a_b_ratio']:.1f}% (target: ≥85%)\n"

    report_text += f"""
## Duplicates
Total Duplicates Found: {report['duplicates']}

## Issues & Warnings
Total Issues: {len(report['issues'])}
Total Warnings: {len(report['warnings'])}
"""

    if report['warnings']:
        report_text += "\n### Warnings:\n"
        for warning in report['warnings'][:10]:  # 最初の10件
            report_text += f"  - {warning}\n"
        if len(report['warnings']) > 10:
            report_text += f"  ... and {len(report['warnings']) - 10} more\n"

    report_text += """
================================================================================
"""

    return report_text


def main():
    """メイン実行関数"""
    logger.info("=== Phase 3: Comprehensive Validation ===")

    # マージ済みデータを読み込み
    with open("data/processed/merged_collection.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples for validation")

    # 包括的な検証を実行
    validator = ComprehensiveValidator()
    comprehensive_report = validator.perform_comprehensive_validation(data)

    # レポート生成
    report_text = generate_validation_report(comprehensive_report)
    print("\n" + report_text)

    # レポート保存
    report_file = "data/metadata/comprehensive_validation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # JSON形式でも保存
    json_report_file = "data/metadata/comprehensive_validation_report.json"
    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Validation report saved to:")
    logger.info(f"  - {report_file}")
    logger.info(f"  - {json_report_file}")

    # 検証結果のサマリー
    if comprehensive_report['basic_validation']['invalid'] == 0 and comprehensive_report['duplicates'] == 0:
        logger.info("\n✅ All validation checks passed!")
        return True
    else:
        logger.warning("\n⚠️ Some validation issues found. Please review the report.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Phase 3 Comprehensive Validation Complete")
    else:
        print("\n⚠️ Phase 3 Validation found some issues")
