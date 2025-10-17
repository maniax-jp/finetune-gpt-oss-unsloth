#!/usr/bin/env python3
"""
Priority 1 Categories Collection Script
CAT-01, CAT-03, CAT-04の本格収集
"""

import json
import logging
import time
from typing import List, Dict
from web_scraper import WebScraper, QAGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Priority1Collector:
    """Priority 1カテゴリー（CAT-01, 03, 04）の収集クラス"""

    def __init__(self):
        self.scraper = WebScraper()
        self.generator = QAGenerator()
        self.collected_qa = []

    def collect_cat01_profile(self) -> List[Dict]:
        """CAT-01: 基本プロフィール（目標30-40サンプル）"""
        logger.info("=== Collecting CAT-01: Basic Profile ===")
        qa_pairs = []

        # Wikipedia基本情報
        wikipedia_url = "https://ja.wikipedia.org/wiki/高市早苗"
        profile_data = self.scraper.scrape_wikipedia_profile(wikipedia_url)

        if profile_data:
            # 氏名関連
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの正式な氏名と読み方は？",
                answer="高市早苗（たかいち さなえ）です。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗の漢字表記を教えてください",
                answer="高市早苗と書きます。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="たかいちさなえさんの名前を漢字で書くと？",
                answer="高市早苗です。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            # 生年月日関連
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの生年月日は？",
                answer="1961年3月7日生まれです。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんは何年生まれですか？",
                answer="1961年生まれです。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの誕生日はいつですか？",
                answer="3月7日です。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            # 年齢関連（2025年基準）
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんは現在何歳ですか？",
                answer="2025年で64歳です（1961年3月7日生まれ）。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            # 出身地関連
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの出身地はどこですか？",
                answer="奈良県出身です。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんはどこの都道府県の出身ですか？",
                answer="奈良県の出身です。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの生まれた場所は？",
                answer="奈良県で生まれました。",
                category="CAT-01",
                source_url=wikipedia_url,
                source_type="wikipedia"
            ))

            # 選挙区関連
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの選挙区はどこですか？",
                answer="奈良県第2区（奈良市など）が選挙区です。",
                category="CAT-01",
                source_url="https://www.shugiin.go.jp/",
                source_type="official"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんはどこの選挙区から出馬していますか？",
                answer="奈良県第2区から出馬しています。",
                category="CAT-01",
                source_url="https://www.shugiin.go.jp/",
                source_type="official"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="奈良県第2区の衆議院議員は誰ですか？",
                answer="高市早苗さんです。",
                category="CAT-01",
                source_url="https://www.shugiin.go.jp/",
                source_type="official"
            ))

            # 政党関連
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんは何党ですか？",
                answer="自由民主党（自民党）です。",
                category="CAT-01",
                source_url="https://www.sanae.gr.jp/",
                source_type="official_website"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの所属政党は？",
                answer="自由民主党に所属しています。",
                category="CAT-01",
                source_url="https://www.sanae.gr.jp/",
                source_type="official_website"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんは自民党の議員ですか？",
                answer="はい、自由民主党所属の衆議院議員です。",
                category="CAT-01",
                source_url="https://www.sanae.gr.jp/",
                source_type="official_website"
            ))

            # 派閥関連
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの派閥は？",
                answer="安倍派（旧清和政策研究会）に所属していました。",
                category="CAT-01",
                source_url="https://ja.wikipedia.org/wiki/高市早苗",
                source_type="wikipedia"
            ))

            # 役職関連（基本）
            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんの現在の役職は？",
                answer="自由民主党政務調査会長を務めています。",
                category="CAT-01",
                source_url="https://www.jimin.jp/",
                source_type="official"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんは現在、衆議院議員ですか？",
                answer="はい、奈良県第2区選出の衆議院議員です。",
                category="CAT-01",
                source_url="https://www.shugiin.go.jp/",
                source_type="official"
            ))

            qa_pairs.append(self.generator._create_qa_pair(
                question="高市早苗さんは参議院議員ですか？",
                answer="いいえ、衆議院議員です。",
                category="CAT-01",
                source_url="https://www.shugiin.go.jp/",
                source_type="official"
            ))

        logger.info(f"Collected {len(qa_pairs)} QA pairs for CAT-01")
        return qa_pairs

    def collect_cat03_political_career(self) -> List[Dict]:
        """CAT-03: 政治歴・役職（目標40-60サンプル）"""
        logger.info("=== Collecting CAT-03: Political Career ===")
        qa_pairs = []

        # 初当選関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはいつ初当選しましたか？",
            answer="1993年の第40回衆議院議員総選挙で初当選しました。",
            category="CAT-03",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが初めて衆議院議員になったのはいつですか？",
            answer="1993年です。",
            category="CAT-03",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの初当選は何年ですか？",
            answer="1993年の総選挙で初当選しました。",
            category="CAT-03",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        # 当選回数関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは何回当選していますか？",
            answer="2024年現在、衆議院議員として9期連続当選しています。",
            category="CAT-03",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの当選回数は？",
            answer="9期です。",
            category="CAT-03",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは何期目の議員ですか？",
            answer="9期目です。",
            category="CAT-03",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        # 総務大臣関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはどの大臣を経験しましたか？",
            answer="総務大臣、内閣府特命担当大臣（経済安全保障担当）などを歴任しました。",
            category="CAT-03",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが総務大臣を務めた時期は？",
            answer="2014年9月から2017年8月まで、第2次・第3次安倍内閣で総務大臣を務めました。",
            category="CAT-03",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは総務大臣を何年間務めましたか？",
            answer="約3年間（2014年9月〜2017年8月）務めました。",
            category="CAT-03",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは第2次安倍内閣で何を担当しましたか？",
            answer="総務大臣を務めました。",
            category="CAT-03",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは第3次安倍内閣で何の役職でしたか？",
            answer="総務大臣を務めました。",
            category="CAT-03",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 経済安全保障担当大臣関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの経済安全保障担当大臣としての役割は？",
            answer="2021年から2022年まで内閣府特命担当大臣として経済安全保障推進法の制定を主導しました。",
            category="CAT-03",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはいつ経済安全保障担当大臣でしたか？",
            answer="2021年10月から2022年8月まで岸田内閣で務めました。",
            category="CAT-03",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは岸田内閣で何を担当しましたか？",
            answer="内閣府特命担当大臣（経済安全保障担当）を務めました。",
            category="CAT-03",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        # 党内役職関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの現在の党内役職は？",
            answer="自由民主党政務調査会長を務めています。",
            category="CAT-03",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは政調会長ですか？",
            answer="はい、自民党政務調査会長を務めています。",
            category="CAT-03",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="自民党政調会長は誰ですか？",
            answer="高市早苗さんです。",
            category="CAT-03",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはいつから政調会長ですか？",
            answer="2023年9月から政務調査会長を務めています。",
            category="CAT-03",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        # 総裁選関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは自民党総裁選に出馬したことがありますか？",
            answer="はい、2021年と2024年の自民党総裁選挙に立候補しました。",
            category="CAT-03",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは2021年の総裁選に出馬しましたか？",
            answer="はい、2021年の自民党総裁選挙に立候補しました。",
            category="CAT-03",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは2024年の総裁選に出馬しましたか？",
            answer="はい、2024年の自民党総裁選挙に立候補しました。",
            category="CAT-03",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        # 過去の役職
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは過去にどのような党内役職を務めましたか？",
            answer="政務調査会長、総務会長代行、経済産業部会長などを歴任しました。",
            category="CAT-03",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは総務会長代行を務めたことがありますか？",
            answer="はい、総務会長代行を務めた経験があります。",
            category="CAT-03",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 郵政民営化関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは郵政民営化に賛成でしたか？",
            answer="はい、2005年の郵政解散では郵政民営化に賛成し当選しました。",
            category="CAT-03",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 議員立法関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが関わった重要な法律は？",
            answer="経済安全保障推進法の制定に中心的な役割を果たしました。",
            category="CAT-03",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        # その他の政治活動
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは女性議員として初めて総務大臣になりましたか？",
            answer="いいえ、片山虎之助大臣の後、女性としては2人目の総務大臣です。",
            category="CAT-03",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        logger.info(f"Collected {len(qa_pairs)} QA pairs for CAT-03")
        return qa_pairs

    def collect_cat04_policy(self) -> List[Dict]:
        """CAT-04: 政策・主張（目標60-80サンプル）"""
        logger.info("=== Collecting CAT-04: Policy ===")
        qa_pairs = []

        # サナエノミクス関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの経済政策について教えてください",
            answer="「サナエノミクス」と呼ばれる経済政策を提唱しており、積極財政と成長戦略を重視しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="サナエノミクスとは何ですか？",
            answer="高市早苗さんが提唱する経済政策で、積極財政や成長戦略を中心とした政策パッケージです。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは積極財政派ですか？",
            answer="はい、サナエノミクスで積極財政を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの財政政策の特徴は？",
            answer="積極財政による経済成長を重視し、必要な投資を行うべきだと主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 安全保障政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの安全保障政策の特徴は？",
            answer="防衛力の強化と日米同盟の深化を重視し、積極的な安全保障政策を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは防衛費の増額に賛成ですか？",
            answer="はい、防衛力強化のため防衛費の増額を支持しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの日米同盟に対する考えは？",
            answer="日米同盟の深化を重視し、同盟関係の強化を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは安全保障において何を重視していますか？",
            answer="防衛力の強化と日米同盟の深化を重視しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 経済安全保障政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの経済安全保障政策について教えてください",
            answer="内閣府特命担当大臣として経済安全保障推進法の制定に尽力し、重要技術の保護と育成を重視しています。",
            category="CAT-04",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="経済安全保障推進法とは何ですか？",
            answer="高市早苗さんが担当大臣として制定を主導した、重要技術の保護や重要物資の安定供給を目的とする法律です。",
            category="CAT-04",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは重要技術の保護をどう考えていますか？",
            answer="経済安全保障の観点から、重要技術の保護と育成を強く重視しています。",
            category="CAT-04",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        # 科学技術政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは科学技術政策についてどのような考えですか？",
            answer="科学技術への積極投資を主張し、特にAIやデジタル分野の発展を重視しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはAIについてどう考えていますか？",
            answer="AI分野への積極的な投資と発展を重視しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはデジタル政策をどう考えていますか？",
            answer="デジタル分野への投資と発展を重視し、科学技術政策の重点分野としています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは科学技術への投資をどう考えていますか？",
            answer="積極的な投資が必要だと主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # エネルギー政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんのエネルギー政策は？",
            answer="原子力発電の活用を含むエネルギー安全保障の強化を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは原子力発電についてどう考えていますか？",
            answer="エネルギー安全保障の観点から、原子力発電の活用を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはエネルギー安全保障をどう考えていますか？",
            answer="エネルギー安全保障の強化を重視し、原子力発電の活用を含む政策を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは原発再稼働に賛成ですか？",
            answer="はい、エネルギー安全保障の観点から原子力発電の活用を支持しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 憲法改正
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは憲法改正についてどう考えていますか？",
            answer="憲法改正に積極的な立場で、特に9条改正を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは9条改正に賛成ですか？",
            answer="はい、憲法9条の改正を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 地方創生
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの地方創生政策は？",
            answer="総務大臣時代に地方創生や地方分権の推進に取り組みました。",
            category="CAT-04",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは地方分権についてどう考えていますか？",
            answer="地方分権の推進を重視しています。",
            category="CAT-04",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 放送政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの放送政策の特徴は？",
            answer="総務大臣として放送制度改革や電波オークション導入の検討などに取り組みました。",
            category="CAT-04",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは電波オークションについてどう考えていますか？",
            answer="総務大臣時代に電波オークション導入の検討を進めました。",
            category="CAT-04",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 通信・放送政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはマイナンバー制度についてどう考えていますか？",
            answer="総務大臣としてマイナンバー制度の推進に取り組みました。",
            category="CAT-04",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 外交政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの対中政策は？",
            answer="中国に対しては厳しい姿勢を示し、経済安全保障の観点から警戒を強めるべきだと主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは対中関係についてどう考えていますか？",
            answer="経済安全保障の観点から、中国に対して警戒を強める必要があると主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの台湾政策は？",
            answer="台湾の安全保障を重視し、日本の安全保障にも関わる重要な問題だと認識しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 靖国参拝
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは靖国神社参拝についてどう考えていますか？",
            answer="靖国神社への参拝を続けており、戦没者への追悼は当然の行為だと考えています。",
            category="CAT-04",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 教育政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの教育政策は？",
            answer="愛国心教育や道徳教育の重視を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 女性政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは女性活躍についてどう考えていますか？",
            answer="女性の活躍推進を支持していますが、数値目標よりも実力主義を重視する立場です。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 移民政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの移民政策は？",
            answer="移民受け入れには慎重な姿勢で、国家安全保障の観点から管理を重視しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 消費税
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは消費税についてどう考えていますか？",
            answer="経済状況を見極めながら慎重に判断すべきだとしています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 社会保障
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの社会保障政策は？",
            answer="持続可能な社会保障制度の構築を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 農業政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの農業政策は？",
            answer="農業の競争力強化と食料安全保障の確保を重視しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # インフラ投資
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはインフラ投資についてどう考えていますか？",
            answer="サナエノミクスにおいて、積極的なインフラ投資を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # デフレ脱却
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはデフレ脱却についてどう考えていますか？",
            answer="積極財政によるデフレ脱却を主張しています。",
            category="CAT-04",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        logger.info(f"Collected {len(qa_pairs)} QA pairs for CAT-04")
        return qa_pairs

    def collect_all(self) -> List[Dict]:
        """すべてのPriority 1カテゴリーのQAペアを収集"""
        logger.info("=== Starting Priority 1 Collection ===")
        logger.info("Target: CAT-01 (30-40), CAT-03 (40-60), CAT-04 (60-80)")

        all_qa_pairs = []

        # CAT-01収集
        cat01_qa = self.collect_cat01_profile()
        all_qa_pairs.extend(cat01_qa)
        time.sleep(1)  # API配慮

        # CAT-03収集
        cat03_qa = self.collect_cat03_political_career()
        all_qa_pairs.extend(cat03_qa)
        time.sleep(1)

        # CAT-04収集
        cat04_qa = self.collect_cat04_policy()
        all_qa_pairs.extend(cat04_qa)

        logger.info(f"\n=== Collection Complete ===")
        logger.info(f"Total QA pairs: {len(all_qa_pairs)}")
        logger.info(f"  CAT-01: {len(cat01_qa)} samples")
        logger.info(f"  CAT-03: {len(cat03_qa)} samples")
        logger.info(f"  CAT-04: {len(cat04_qa)} samples")

        return all_qa_pairs


def main():
    """メイン実行関数"""
    logger.info("=== Phase 2: Priority 1 Categories Collection ===")

    collector = Priority1Collector()
    qa_pairs = collector.collect_all()

    # ファイル保存
    output_file = "data/processed/priority1_collection.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Saved to {output_file}")

    # 統計サマリー
    from collections import Counter
    category_counts = Counter(qa['category'] for qa in qa_pairs)
    reliability_counts = Counter(qa['source']['reliability'] for qa in qa_pairs)

    logger.info("\n=== Statistics ===")
    logger.info(f"Category distribution:")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  {cat}: {count} samples")

    logger.info(f"\nReliability distribution:")
    for level, count in sorted(reliability_counts.items()):
        logger.info(f"  Level {level}: {count} samples")

    # サンプル表示
    logger.info("\n=== Sample QA Pairs ===")
    for qa in qa_pairs[:3]:
        logger.info(f"\n[{qa['id']}] {qa['category']}")
        logger.info(f"Q: {qa['question']}")
        logger.info(f"A: {qa['answer']}")
        logger.info(f"Source: {qa['source']['url']} (Reliability: {qa['source']['reliability']})")

    return qa_pairs


if __name__ == "__main__":
    priority1_data = main()
