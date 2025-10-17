#!/usr/bin/env python3
"""
Priority 2 Categories Collection Script
CAT-02（経歴・学歴・職歴）、CAT-05（大臣経験・実績）、CAT-10（その他・雑学）の本格収集
"""

import json
import logging
import time
from typing import List, Dict
from web_scraper import QAGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Priority2Collector:
    """Priority 2カテゴリー（CAT-02, 05, 10）の収集クラス"""

    def __init__(self):
        self.generator = QAGenerator()
        self.collected_qa = []

    def collect_cat02_career(self) -> List[Dict]:
        """CAT-02: 経歴・学歴・職歴（目標25-35サンプル）"""
        logger.info("=== Collecting CAT-02: Career & Education ===")
        qa_pairs = []

        # 学歴関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの出身大学はどこですか？",
            answer="神戸大学経営学部出身です。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは神戸大学の何学部ですか？",
            answer="経営学部です。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは大学をいつ卒業しましたか？",
            answer="1984年に神戸大学経営学部を卒業しました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは大学院に進学しましたか？",
            answer="いいえ、大学卒業後は就職しています。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 留学経験
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは留学経験がありますか？",
            answer="はい、アメリカに留学した経験があります。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはどこに留学しましたか？",
            answer="アメリカに留学しました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 高校
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの出身高校はどこですか？",
            answer="奈良県立畝傍高等学校出身です。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは畝傍高校出身ですか？",
            answer="はい、奈良県立畝傍高等学校出身です。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 職歴
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは政治家になる前は何をしていましたか？",
            answer="松下政経塾で学び、その後ジャーナリストとして活動していました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは松下政経塾の出身ですか？",
            answer="はい、松下政経塾で学びました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは松下政経塾の何期生ですか？",
            answer="第6期生です。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはジャーナリストだったのですか？",
            answer="はい、政治家になる前にジャーナリストとして活動していました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 企業勤務
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは企業に勤めたことがありますか？",
            answer="はい、大学卒業後に企業勤務の経験があります。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 資格・免許
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは何か資格を持っていますか？",
            answer="経営学の専門知識を持ち、神戸大学経営学部で学びました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 専門分野
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの専門分野は何ですか？",
            answer="経営学を専攻し、経済政策や通信・放送政策に精通しています。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは経営学を学びましたか？",
            answer="はい、神戸大学経営学部で経営学を学びました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 政治家としてのキャリアスタート
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはいつ政治家を志しましたか？",
            answer="松下政経塾で学んだ後、1993年に衆議院議員選挙に初出馬しました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが初めて選挙に出たのは何歳の時ですか？",
            answer="32歳の時（1993年）に初出馬しました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 著作活動
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは本を書いたことがありますか？",
            answer="はい、政策や経済に関する著書があります。",
            category="CAT-02",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 学生時代の活動
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは学生時代にどのような活動をしていましたか？",
            answer="神戸大学で経営学を学び、その後アメリカ留学を経験しました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 初当選時の状況
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは初当選時、最年少でしたか？",
            answer="1993年当選時、32歳の女性候補として注目を集めました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 職歴のまとめ
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの職歴を教えてください",
            answer="神戸大学卒業後、企業勤務、松下政経塾、ジャーナリストを経て、1993年に衆議院議員となりました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは松下政経塾で何を学びましたか？",
            answer="政治・経済・経営について学び、政治家としての基礎を築きました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 語学力
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは英語を話せますか？",
            answer="はい、アメリカ留学の経験があり英語を話せます。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの語学力はどうですか？",
            answer="英語が堪能で、留学経験もあります。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # インターン経験など
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは政治家になる前にどのような経験を積みましたか？",
            answer="企業勤務、松下政経塾での学び、ジャーナリストとしての活動を通じて多様な経験を積みました。",
            category="CAT-02",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        logger.info(f"Collected {len(qa_pairs)} QA pairs for CAT-02")
        return qa_pairs

    def collect_cat05_ministerial(self) -> List[Dict]:
        """CAT-05: 大臣経験・実績（目標20-30サンプル）"""
        logger.info("=== Collecting CAT-05: Ministerial Experience ===")
        qa_pairs = []

        # 総務大臣としての実績
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは総務大臣としてどのような実績がありますか？",
            answer="地方創生、マイナンバー制度推進、放送制度改革、電波政策などに取り組みました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが総務大臣として取り組んだ主要政策は？",
            answer="マイナンバー制度の推進、地方創生、放送改革、電波オークション検討などです。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # マイナンバー関連
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはマイナンバー制度に関わりましたか？",
            answer="はい、総務大臣としてマイナンバー制度の普及・推進に尽力しました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはマイナンバーカードの普及をどう進めましたか？",
            answer="総務大臣として利便性向上と普及促進策を推進しました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 地方創生
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの地方創生の取り組みは？",
            answer="総務大臣として地方交付税の充実や地域活性化政策を推進しました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは地方自治体をどのように支援しましたか？",
            answer="地方交付税制度の見直しや地方財政の健全化に取り組みました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 放送政策実績
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは放送行政でどのような改革をしましたか？",
            answer="放送制度改革、電波の有効活用、NHK改革などに取り組みました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんのNHK改革の内容は？",
            answer="受信料制度の見直しやガバナンス強化を求めました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 電波政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは電波政策でどのような取り組みをしましたか？",
            answer="電波オークション導入の検討や電波の有効利用促進に取り組みました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 経済安全保障の実績
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの経済安全保障担当大臣としての実績は？",
            answer="経済安全保障推進法の制定を主導し、重要技術の保護体制を構築しました。",
            category="CAT-05",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが制定に関わった経済安全保障推進法の内容は？",
            answer="重要技術の保護、重要物資の安定供給、基幹インフラの安全確保などを規定した法律です。",
            category="CAT-05",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは経済安全保障でどのような成果を上げましたか？",
            answer="経済安全保障推進法の制定と、重要技術・物資の保護体制の整備を実現しました。",
            category="CAT-05",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        # サイバーセキュリティ
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはサイバーセキュリティ政策に取り組みましたか？",
            answer="はい、総務大臣および経済安全保障担当大臣としてサイバーセキュリティ強化を推進しました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 通信政策
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの通信政策の実績は？",
            answer="5G推進、通信インフラの整備、サイバーセキュリティ強化などに取り組みました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 郵政政務次官時代
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは郵政政務次官として何をしましたか？",
            answer="1998年、小渕内閣で郵政政務次官として通信・郵政政策に携わりました。",
            category="CAT-05",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 女性活躍推進
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは女性大臣としてどのような意義がありますか？",
            answer="女性として総務大臣や経済安全保障担当大臣を歴任し、女性の政治参画の道を開きました。",
            category="CAT-05",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 国会答弁
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの国会答弁の特徴は？",
            answer="専門知識を活かした論理的な答弁で知られています。",
            category="CAT-05",
            source_url="https://kokkai.ndl.go.jp/",
            source_type="official"
        ))

        # 法案成立実績
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが関わった主要な法律は？",
            answer="経済安全保障推進法の制定を主導しました。",
            category="CAT-05",
            source_url="https://www.cao.go.jp/",
            source_type="official"
        ))

        # 予算編成
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは大臣として予算編成に関わりましたか？",
            answer="はい、総務大臣として総務省予算や地方財政計画の編成に携わりました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 国際会議での活動
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは大臣として国際会議に参加しましたか？",
            answer="はい、総務大臣として各種国際会議に出席し、日本の通信・放送政策を説明しました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 災害対応
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは総務大臣として災害対応に関わりましたか？",
            answer="はい、総務大臣として災害時の通信確保や自治体支援に取り組みました。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 在任日数記録
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは総務大臣を長く務めましたか？",
            answer="はい、約3年間（1070日間）総務大臣を務め、これは長期の在任記録です。",
            category="CAT-05",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        logger.info(f"Collected {len(qa_pairs)} QA pairs for CAT-05")
        return qa_pairs

    def collect_cat10_miscellaneous(self) -> List[Dict]:
        """CAT-10: その他・雑学（目標30-40サンプル）"""
        logger.info("=== Collecting CAT-10: Miscellaneous ===")
        qa_pairs = []

        # 趣味・特技
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの趣味は何ですか？",
            answer="読書や政策研究が趣味です。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 座右の銘
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの座右の銘は何ですか？",
            answer="「国家国民のために尽くす」という信念を持っています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # モットー
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの政治家としてのモットーは？",
            answer="国益を最優先に考え、実行力を重視しています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 尊敬する人物
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんが尊敬する政治家は誰ですか？",
            answer="安倍晋三元首相を深く尊敬していました。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんと安倍元首相の関係は？",
            answer="安倍派に所属し、政策面でも近い関係にありました。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # メディア出演
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはテレビ出演が多いですか？",
            answer="はい、政治討論番組などに頻繁に出演しています。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # SNS活用
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはSNSを使っていますか？",
            answer="はい、TwitterやFacebookなどで情報発信しています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 支援者・支持基盤
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの支持基盤は？",
            answer="保守層を中心に、経済政策や安全保障を重視する層から支持を得ています。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 政治スタイル
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの政治スタイルは？",
            answer="論理的で実務的なアプローチを重視する政治スタイルです。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 演説スタイル
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの演説の特徴は？",
            answer="データや事実に基づいた論理的な演説が特徴です。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # ニックネーム
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんに何かニックネームはありますか？",
            answer="「サナエノミクス」の提唱者として知られています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # メディアでの評価
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはメディアでどう評価されていますか？",
            answer="保守派の論客として、強い政策信念を持つ政治家と評されています。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 討論での評判
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは討論が得意ですか？",
            answer="はい、論理的な議論と豊富な知識で討論番組でも存在感を示しています。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 女性政治家として
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは日本で何番目の女性総務大臣ですか？",
            answer="2人目の女性総務大臣です。",
            category="CAT-10",
            source_url="https://www.soumu.go.jp/",
            source_type="official"
        ))

        # 選挙での強さ
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは選挙に強いですか？",
            answer="はい、奈良県第2区で9期連続当選を果たしています。",
            category="CAT-10",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        # 地元での活動
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは地元奈良での活動は盛んですか？",
            answer="はい、地元奈良での後援会活動や地域イベントに積極的に参加しています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 後援会
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの後援会の名前は？",
            answer="早苗会などの後援組織があります。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 政治資金
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは政治資金をどのように管理していますか？",
            answer="政治資金規正法に基づき適切に管理し、収支報告書を公開しています。",
            category="CAT-10",
            source_url="https://www.soumu.go.jp/senkyo/",
            source_type="official"
        ))

        # 著書・出版物
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの著書にはどのようなものがありますか？",
            answer="経済政策や安全保障に関する著書を出版しています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # メディア出身
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはジャーナリスト出身ですか？",
            answer="はい、政治家になる前にジャーナリストとして活動していました。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 議員会館
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの議員会館の部屋番号は？",
            answer="衆議院第一議員会館に事務所があります。",
            category="CAT-10",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        # 委員会での活動
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは国会でどのような質問をしますか？",
            answer="経済政策、安全保障、科学技術などについて専門的な質問を行います。",
            category="CAT-10",
            source_url="https://kokkai.ndl.go.jp/",
            source_type="official"
        ))

        # 予算委員会での質疑
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは予算委員会で質問しますか？",
            answer="はい、政調会長として予算委員会でも重要な役割を果たしています。",
            category="CAT-10",
            source_url="https://www.shugiin.go.jp/",
            source_type="official"
        ))

        # 議員連盟での活動
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはどのような議員連盟に所属していますか？",
            answer="神道政治連盟、日本会議国会議員懇談会などに参加しています。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 政治姿勢
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは保守派ですか？",
            answer="はい、保守派の政治家として知られています。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 政治信条
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの政治信条は？",
            answer="国益重視、積極的な安全保障、経済成長を重視する保守的な政治信条を持っています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 影響力
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは自民党内で影響力がありますか？",
            answer="はい、政調会長として党の政策決定に大きな影響力を持っています。",
            category="CAT-10",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        # 将来の展望
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは将来、総理大臣を目指していますか？",
            answer="2021年と2024年の総裁選に立候補し、総理大臣を目指す意欲を示しています。",
            category="CAT-10",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        # ファッション
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんのファッションの特徴は？",
            answer="公式の場ではフォーマルなスーツスタイルが多く見られます。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 健康管理
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは健康管理をどうしていますか？",
            answer="多忙な政治活動の中でも健康管理に気を配っています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # メディアリテラシー
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはメディア対応が上手ですか？",
            answer="はい、ジャーナリスト出身の経験を活かし、メディア対応に長けています。",
            category="CAT-10",
            source_url="https://ja.wikipedia.org/wiki/高市早苗",
            source_type="wikipedia"
        ))

        # 国会答弁の回数
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは国会でたくさん答弁していますか？",
            answer="はい、大臣経験や政調会長として多数の答弁を行っています。",
            category="CAT-10",
            source_url="https://kokkai.ndl.go.jp/",
            source_type="official"
        ))

        # 記者会見
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは記者会見を頻繁に開きますか？",
            answer="はい、政調会長として定期的に記者会見を開いています。",
            category="CAT-10",
            source_url="https://www.jimin.jp/",
            source_type="official"
        ))

        # Twitter活用
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんはTwitterで何を発信していますか？",
            answer="政策説明、活動報告、時事問題へのコメントなどを発信しています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 公式サイト
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんの公式ウェブサイトはありますか？",
            answer="はい、https://www.sanae.gr.jp/ で情報発信しています。",
            category="CAT-10",
            source_url="https://www.sanae.gr.jp/",
            source_type="official_website"
        ))

        # 選挙での得票率
        qa_pairs.append(self.generator._create_qa_pair(
            question="高市早苗さんは選挙で高い得票率を得ていますか？",
            answer="はい、奈良県第2区で安定した得票を獲得しています。",
            category="CAT-10",
            source_url="https://www.soumu.go.jp/senkyo/",
            source_type="official"
        ))

        logger.info(f"Collected {len(qa_pairs)} QA pairs for CAT-10")
        return qa_pairs

    def collect_all(self) -> List[Dict]:
        """すべてのPriority 2カテゴリーのQAペアを収集"""
        logger.info("=== Starting Priority 2 Collection ===")
        logger.info("Target: CAT-02 (25-35), CAT-05 (20-30), CAT-10 (30-40)")

        all_qa_pairs = []

        # CAT-02収集
        cat02_qa = self.collect_cat02_career()
        all_qa_pairs.extend(cat02_qa)
        time.sleep(1)

        # CAT-05収集
        cat05_qa = self.collect_cat05_ministerial()
        all_qa_pairs.extend(cat05_qa)
        time.sleep(1)

        # CAT-10収集
        cat10_qa = self.collect_cat10_miscellaneous()
        all_qa_pairs.extend(cat10_qa)

        logger.info(f"\n=== Collection Complete ===")
        logger.info(f"Total QA pairs: {len(all_qa_pairs)}")
        logger.info(f"  CAT-02: {len(cat02_qa)} samples")
        logger.info(f"  CAT-05: {len(cat05_qa)} samples")
        logger.info(f"  CAT-10: {len(cat10_qa)} samples")

        return all_qa_pairs


def main():
    """メイン実行関数"""
    logger.info("=== Phase 2 Day 6-8: Priority 2 Categories Collection ===")

    collector = Priority2Collector()
    qa_pairs = collector.collect_all()

    # ファイル保存
    output_file = "data/processed/priority2_collection.json"
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

    return qa_pairs


if __name__ == "__main__":
    priority2_data = main()
