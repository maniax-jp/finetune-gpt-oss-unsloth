#!/usr/bin/env python3
"""
Pilot Collection Script for Takaichi Sanae QA Dataset
パイロット収集スクリプト - 各カテゴリー5サンプル
"""

import json
import logging
from datetime import datetime
from web_scraper import WebScraper, QAGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_answer_text(text: str) -> str:
    """
    回答テキストをクリーニング（脚注番号除去など）

    Args:
        text: 元のテキスト

    Returns:
        クリーニング済みテキスト
    """
    import re

    # 脚注番号を除去 [1], [2]など
    text = re.sub(r'\[\d+\]', '', text)

    # 余分な空白を除去
    text = re.sub(r'\s+', ' ', text).strip()

    # 重複する句読点を除去
    text = re.sub(r'([。、])を卒業', r'\1', text)

    return text


def create_profile_qa_pairs() -> list:
    """
    CAT-01: 基本プロフィールのQAペアを作成

    Returns:
        QAペアのリスト
    """
    qa_pairs = []
    generator = QAGenerator()

    # 手動で作成（高品質を保証）
    base_info = {
        "source_url": "https://ja.wikipedia.org/wiki/高市早苗",
        "source_type": "wikipedia"
    }

    # 1. 氏名・読み方
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの正式な氏名と読み方は？",
        answer="高市早苗（たかいち さなえ）です。",
        category="CAT-01",
        source_url=base_info["source_url"],
        source_type=base_info["source_type"]
    ))

    # 2. 生年月日・年齢
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの生年月日は？",
        answer="1961年3月7日生まれです。",
        category="CAT-01",
        source_url=base_info["source_url"],
        source_type=base_info["source_type"]
    ))

    # 3. 出身地
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの出身地はどこですか？",
        answer="奈良県出身です。",
        category="CAT-01",
        source_url=base_info["source_url"],
        source_type=base_info["source_type"]
    ))

    # 4. 現在の選挙区
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの選挙区はどこですか？",
        answer="奈良県第2区（奈良市など）が選挙区です。",
        category="CAT-01",
        source_url="https://www.shugiin.go.jp/",
        source_type="official"
    ))

    # 5. 所属政党（現在）
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんは何党ですか？",
        answer="自由民主党（自民党）です。",
        category="CAT-01",
        source_url="https://www.sanae.gr.jp/",
        source_type="official_website"
    ))

    return qa_pairs


def create_political_career_qa_pairs() -> list:
    """
    CAT-03: 政治歴・役職のQAペアを作成

    Returns:
        QAペアのリスト
    """
    qa_pairs = []
    generator = QAGenerator()

    base_info = {
        "source_url": "https://www.shugiin.go.jp/",
        "source_type": "official"
    }

    # 1. 初当選
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんはいつ初当選しましたか？",
        answer="1993年の第40回衆議院議員総選挙で初当選しました。",
        category="CAT-03",
        source_url=base_info["source_url"],
        source_type=base_info["source_type"]
    ))

    # 2. 当選回数
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんは何回当選していますか？",
        answer="2024年現在、衆議院議員として9期連続当選しています。",
        category="CAT-03",
        source_url=base_info["source_url"],
        source_type=base_info["source_type"]
    ))

    # 3. 現在の党内役職
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの現在の党内役職は？",
        answer="自由民主党政務調査会長を務めています。",
        category="CAT-03",
        source_url="https://www.jimin.jp/",
        source_type="official"
    ))

    # 4. 大臣経験
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんはどの大臣を経験しましたか？",
        answer="総務大臣、内閣府特命担当大臣（経済安全保障担当）などを歴任しました。",
        category="CAT-03",
        source_url="https://ja.wikipedia.org/wiki/高市早苗",
        source_type="wikipedia"
    ))

    # 5. 総務大臣の在任期間
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんが総務大臣を務めた時期は？",
        answer="2014年9月から2017年8月まで、第2次・第3次安倍内閣で総務大臣を務めました。",
        category="CAT-03",
        source_url="https://www.soumu.go.jp/",
        source_type="official"
    ))

    return qa_pairs


def create_policy_qa_pairs() -> list:
    """
    CAT-04: 政策・主張のQAペアを作成

    Returns:
        QAペアのリスト
    """
    qa_pairs = []
    generator = QAGenerator()

    # 1. 経済政策（サナエノミクス）
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの経済政策について教えてください",
        answer="「サナエノミクス」と呼ばれる経済政策を提唱しており、積極財政と成長戦略を重視しています。",
        category="CAT-04",
        source_url="https://www.sanae.gr.jp/",
        source_type="official_website"
    ))

    # 2. 安全保障政策
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの安全保障政策の特徴は？",
        answer="防衛力の強化と日米同盟の深化を重視し、積極的な安全保障政策を主張しています。",
        category="CAT-04",
        source_url="https://www.sanae.gr.jp/",
        source_type="official_website"
    ))

    # 3. 科学技術政策
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんは科学技術政策についてどのような考えですか？",
        answer="科学技術への積極投資を主張し、特にAIやデジタル分野の発展を重視しています。",
        category="CAT-04",
        source_url="https://www.sanae.gr.jp/",
        source_type="official_website"
    ))

    # 4. エネルギー政策
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんのエネルギー政策は？",
        answer="原子力発電の活用を含むエネルギー安全保障の強化を主張しています。",
        category="CAT-04",
        source_url="https://www.sanae.gr.jp/",
        source_type="official_website"
    ))

    # 5. 経済安全保障
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの経済安全保障政策について教えてください",
        answer="内閣府特命担当大臣として経済安全保障推進法の制定に尽力し、重要技術の保護と育成を重視しています。",
        category="CAT-04",
        source_url="https://www.cao.go.jp/",
        source_type="official"
    ))

    return qa_pairs


def create_additional_categories_qa_pairs() -> list:
    """
    CAT-02, CAT-05, CAT-06 の追加QAペアを作成

    Returns:
        QAペアのリスト
    """
    qa_pairs = []
    generator = QAGenerator()

    # CAT-02: 経歴
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの出身大学は？",
        answer="神戸大学経営学部を卒業しています。",
        category="CAT-02",
        source_url="https://ja.wikipedia.org/wiki/高市早苗",
        source_type="wikipedia"
    ))

    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんは留学経験がありますか？",
        answer="はい、米国イリノイ大学に留学した経験があります。",
        category="CAT-02",
        source_url="https://ja.wikipedia.org/wiki/高市早苗",
        source_type="wikipedia"
    ))

    # CAT-05: 大臣経験
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんは総務大臣としてどのような業績がありますか？",
        answer="放送制度改革や郵政事業の見直し、マイナンバー制度の推進などに取り組みました。",
        category="CAT-05",
        source_url="https://www.soumu.go.jp/",
        source_type="official"
    ))

    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの経済安全保障担当大臣としての役割は？",
        answer="2021年から2022年まで内閣府特命担当大臣として経済安全保障推進法の制定を主導しました。",
        category="CAT-05",
        source_url="https://www.cao.go.jp/",
        source_type="official"
    ))

    # CAT-06: 選挙
    qa_pairs.append(generator._create_qa_pair(
        question="高市早苗さんの選挙区の特徴は？",
        answer="奈良県第2区は奈良市を中心とする選挙区で、古都としての歴史的背景を持つ地域です。",
        category="CAT-06",
        source_url="https://www.soumu.go.jp/senkyo/",
        source_type="official"
    ))

    return qa_pairs


def main():
    """メイン実行関数"""
    logger.info("=== Pilot Collection Start ===")
    logger.info("Target: 15+ samples across 6 categories")

    all_qa_pairs = []

    # CAT-01: 基本プロフィール（5サンプル）
    logger.info("\n--- CAT-01: Basic Profile (5 samples) ---")
    profile_qa = create_profile_qa_pairs()
    all_qa_pairs.extend(profile_qa)
    logger.info(f"Created {len(profile_qa)} QA pairs for CAT-01")

    # CAT-03: 政治歴（5サンプル）
    logger.info("\n--- CAT-03: Political Career (5 samples) ---")
    career_qa = create_political_career_qa_pairs()
    all_qa_pairs.extend(career_qa)
    logger.info(f"Created {len(career_qa)} QA pairs for CAT-03")

    # CAT-04: 政策（5サンプル）
    logger.info("\n--- CAT-04: Policy (5 samples) ---")
    policy_qa = create_policy_qa_pairs()
    all_qa_pairs.extend(policy_qa)
    logger.info(f"Created {len(policy_qa)} QA pairs for CAT-04")

    # CAT-02, 05, 06: 追加（5サンプル）
    logger.info("\n--- CAT-02, 05, 06: Additional (5 samples) ---")
    additional_qa = create_additional_categories_qa_pairs()
    all_qa_pairs.extend(additional_qa)
    logger.info(f"Created {len(additional_qa)} QA pairs for additional categories")

    # すべての回答をクリーニング
    for qa in all_qa_pairs:
        qa['answer'] = clean_answer_text(qa['answer'])

    # 統計情報
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total QA pairs created: {len(all_qa_pairs)}")

    # カテゴリー別集計
    from collections import Counter
    category_counts = Counter(qa['category'] for qa in all_qa_pairs)
    logger.info("\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  {cat}: {count} samples")

    # 信頼性レベル別集計
    reliability_counts = Counter(qa['source']['reliability'] for qa in all_qa_pairs)
    logger.info("\nReliability distribution:")
    for level, count in sorted(reliability_counts.items()):
        logger.info(f"  Level {level}: {count} samples")

    # ファイル保存
    output_file = "data/processed/pilot_collection.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Saved to {output_file}")

    # サンプル表示
    logger.info("\n=== Sample QA Pairs ===")
    for qa in all_qa_pairs[:3]:
        logger.info(f"\n[{qa['id']}] {qa['category']}")
        logger.info(f"Q: {qa['question']}")
        logger.info(f"A: {qa['answer']}")
        logger.info(f"Source: {qa['source']['url']} (Reliability: {qa['source']['reliability']})")

    logger.info("\n=== Pilot Collection Complete ===")

    return all_qa_pairs


if __name__ == "__main__":
    pilot_data = main()
