#!/usr/bin/env python3
"""
Web Scraper for Takaichi Sanae QA Dataset Collection
高市早苗QAデータセット収集用Webスクレイパー
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebScraper:
    """Webページからデータを収集するスクレイパー"""

    def __init__(self, delay: float = 1.0):
        """
        Args:
            delay: リクエスト間の遅延時間（秒）
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; TakaichiQACollector/1.0)'
        })

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        URLからページを取得してBeautifulSoupオブジェクトを返す

        Args:
            url: 取得するURL

        Returns:
            BeautifulSoupオブジェクト、エラー時はNone
        """
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay)

            return BeautifulSoup(response.content, 'html.parser')

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def scrape_wikipedia_profile(self, url: str) -> Dict:
        """
        WikipediaページからプロフィールデータをInfoboxから抽出

        Args:
            url: WikipediaページのURL

        Returns:
            抽出されたプロフィールデータ
        """
        soup = self.fetch_page(url)
        if not soup:
            return {}

        data = {
            "source_url": url,
            "scraped_date": datetime.now().isoformat(),
            "source_type": "wikipedia",
            "profile": {}
        }

        # Infoboxから情報を抽出
        infobox = soup.find('table', class_='infobox')
        if infobox:
            for row in infobox.find_all('tr'):
                header = row.find('th')
                value = row.find('td')

                if header and value:
                    key = header.get_text(strip=True)
                    val = value.get_text(strip=True)
                    data["profile"][key] = val

        # 本文の最初の段落（概要）を取得
        first_paragraph = soup.find('div', class_='mw-parser-output')
        if first_paragraph:
            paragraphs = first_paragraph.find_all('p', recursive=False)
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 50:  # 十分な長さの段落のみ
                    data["summary"] = text
                    break

        return data

    def scrape_text_content(self, url: str, selector: str = None) -> Dict:
        """
        一般的なWebページからテキストコンテンツを抽出

        Args:
            url: 対象URL
            selector: CSSセレクタ（Noneの場合は本文全体）

        Returns:
            抽出されたコンテンツ
        """
        soup = self.fetch_page(url)
        if not soup:
            return {}

        data = {
            "source_url": url,
            "scraped_date": datetime.now().isoformat(),
            "content": []
        }

        if selector:
            elements = soup.select(selector)
        else:
            # デフォルト: 主要なコンテンツ要素を探す
            elements = (
                soup.find_all('article') or
                soup.find_all('main') or
                soup.find_all('div', class_=['content', 'main', 'post'])
            )

        for elem in elements:
            text = elem.get_text(strip=True, separator='\n')
            if text:
                data["content"].append(text)

        return data


class QAGenerator:
    """スクレイプしたデータからQAペアを生成"""

    def __init__(self):
        self.qa_id_counter = 1

    def generate_qa_from_profile(
        self,
        profile_data: Dict,
        category: str = "CAT-01"
    ) -> List[Dict]:
        """
        プロフィールデータからQAペアを生成

        Args:
            profile_data: スクレイプしたプロフィールデータ
            category: カテゴリーID

        Returns:
            QAペアのリスト
        """
        qa_pairs = []
        profile = profile_data.get("profile", {})

        # 生年月日
        if "生年月日" in profile:
            qa_pairs.append(self._create_qa_pair(
                question="高市早苗さんの生年月日は？",
                answer=f"{profile['生年月日']}です。",
                category=category,
                source_url=profile_data.get("source_url"),
                source_type=profile_data.get("source_type")
            ))

        # 所属政党
        if "所属政党" in profile:
            qa_pairs.append(self._create_qa_pair(
                question="高市早苗さんは何党ですか？",
                answer=f"{profile['所属政党']}です。",
                category=category,
                source_url=profile_data.get("source_url"),
                source_type=profile_data.get("source_type")
            ))

        # 選挙区
        if "選挙区" in profile:
            qa_pairs.append(self._create_qa_pair(
                question="高市早苗さんの選挙区はどこですか？",
                answer=f"{profile['選挙区']}です。",
                category="CAT-06",
                source_url=profile_data.get("source_url"),
                source_type=profile_data.get("source_type")
            ))

        # 出身校
        if "出身校" in profile:
            qa_pairs.append(self._create_qa_pair(
                question="高市早苗さんの出身大学は？",
                answer=f"{profile['出身校']}を卒業しています。",
                category="CAT-02",
                source_url=profile_data.get("source_url"),
                source_type=profile_data.get("source_type")
            ))

        return qa_pairs

    def _create_qa_pair(
        self,
        question: str,
        answer: str,
        category: str,
        source_url: str,
        source_type: str = "web"
    ) -> Dict:
        """
        QAペアのデータ構造を作成

        Args:
            question: 質問文
            answer: 回答文
            category: カテゴリーID
            source_url: 情報源URL
            source_type: 情報源タイプ

        Returns:
            QAペアの辞書
        """
        qa_id = f"TAKAICHI-QA-{self.qa_id_counter:04d}"
        self.qa_id_counter += 1

        # 信頼性レベルの判定
        reliability = self._determine_reliability(source_url, source_type)

        return {
            "id": qa_id,
            "category": category,
            "question": question,
            "answer": answer,
            "source": {
                "type": source_type,
                "url": source_url,
                "access_date": datetime.now().isoformat()[:10],
                "reliability": reliability
            },
            "verification": {
                "verified": False,
                "verified_by": None,
                "verification_date": None,
                "cross_check_sources": []
            },
            "metadata": {
                "created_date": datetime.now().isoformat()[:10],
                "last_updated": datetime.now().isoformat()[:10],
                "version": "1.0",
                "tags": []
            }
        }

    def _determine_reliability(self, url: str, source_type: str) -> str:
        """
        URLと情報源タイプから信頼性レベルを判定

        Args:
            url: 情報源URL
            source_type: 情報源タイプ

        Returns:
            信頼性レベル (A, B, C, D)
        """
        domain = urlparse(url).netloc.lower()

        # Aレベル: 公式サイト・政府機関
        if any(d in domain for d in ['sanae.gr.jp', 'shugiin.go.jp', 'go.jp', 'jimin.jp']):
            return "A"

        # Bレベル: Wikipedia、国会図書館
        if any(d in domain for d in ['wikipedia.org', 'ndl.go.jp']):
            return "B"

        # Cレベル: 主要メディア
        if any(d in domain for d in ['nhk.or.jp', 'asahi.com', 'yomiuri.co.jp', 'mainichi.jp', 'nikkei.com']):
            return "C"

        # デフォルトはD
        return "D"


def main():
    """メイン実行関数（テスト用）"""

    # Wikipediaからテスト収集
    scraper = WebScraper(delay=2.0)
    generator = QAGenerator()

    wikipedia_url = "https://ja.wikipedia.org/wiki/高市早苗"

    logger.info("=== Wikipedia Profile Scraping Test ===")
    profile_data = scraper.scrape_wikipedia_profile(wikipedia_url)

    if profile_data:
        logger.info(f"Scraped profile data: {json.dumps(profile_data, ensure_ascii=False, indent=2)}")

        # QAペア生成
        qa_pairs = generator.generate_qa_from_profile(profile_data)

        logger.info(f"\n=== Generated {len(qa_pairs)} QA pairs ===")
        for qa in qa_pairs:
            logger.info(f"\nQ: {qa['question']}")
            logger.info(f"A: {qa['answer']}")
            logger.info(f"Category: {qa['category']}")
            logger.info(f"Reliability: {qa['source']['reliability']}")

        # JSONファイルに保存（テスト）
        output_file = "data/raw/wikipedia/test_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"\nSaved to {output_file}")
    else:
        logger.error("Failed to scrape Wikipedia profile")


if __name__ == "__main__":
    main()
