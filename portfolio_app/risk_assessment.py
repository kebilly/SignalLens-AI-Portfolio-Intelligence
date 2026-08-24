from __future__ import annotations

from typing import Any

DIMENSION_WEIGHTS = {
    "financial_status": 0.25,
    "investment_experience": 0.20,
    "investment_goal": 0.20,
    "risk_tolerance": 0.35,
}

RISK_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "income",
        "dimension": "financial_status",
        "question": "您的主要收入來源是？",
        "options": {"固定薪資": 85, "自由業／彈性收入": 55, "投資收益": 65, "無固定收入": 10},
    },
    {
        "id": "emergency_fund",
        "dimension": "financial_status",
        "question": "目前的應急資金可維持多久的生活開支？",
        "options": {"6 個月以上": 100, "3–6 個月": 70, "1–3 個月": 35, "不到 1 個月": 0},
    },
    {
        "id": "debt_ratio",
        "dimension": "financial_status",
        "question": "目前負債占收入的比例約為？",
        "options": {"無負債": 100, "低於 30%": 75, "30%–50%": 40, "50% 以上": 0},
    },
    {
        "id": "responsibilities",
        "dimension": "financial_status",
        "question": "目前有哪些持續性的財務責任？（可複選）",
        "kind": "multiselect",
        "options": ["無重大財務責任", "房貸或長期租金", "子女教育", "扶養家人", "其他長期支出"],
    },
    {
        "id": "current_allocation",
        "dimension": "financial_status",
        "question": "目前個人資產配置較接近哪一項？",
        "options": {"主要為現金／存款": 40, "現金與投資平均分配": 75, "主要為投資資產": 65},
    },
    {
        "id": "experience_years",
        "dimension": "investment_experience",
        "question": "您有多少年投資經驗？",
        "options": {"5 年以上": 100, "3–5 年": 75, "1–3 年": 45, "1 年以下或無經驗": 10},
    },
    {
        "id": "known_instruments",
        "dimension": "investment_experience",
        "question": "您實際了解哪些投資工具？（可複選）",
        "kind": "multiselect",
        "options": ["目前都不熟悉", "股票", "ETF", "債券", "基金", "衍生性商品"],
    },
    {
        "id": "review_frequency",
        "dimension": "investment_experience",
        "question": "您通常多久檢視一次投資組合？",
        "options": {"每日或每週": 90, "每月": 70, "每季": 45, "一年一次或更少": 20},
    },
    {
        "id": "invested_assets",
        "dimension": "investment_experience",
        "question": "目前投資資產約占總資產多少？",
        "options": {"10% 以下": 25, "10%–30%": 50, "30%–50%": 75, "50% 以上": 90},
    },
    {
        "id": "horizon",
        "dimension": "investment_goal",
        "question": "預計的投資時間範圍是？",
        "options": {"10 年以上": 100, "5–10 年": 75, "1–5 年": 40, "1 年以下": 5},
    },
    {
        "id": "primary_goal",
        "dimension": "investment_goal",
        "question": "目前最重要的投資目標是？",
        "options": {"保本為主": 10, "穩定收入": 35, "資本增值": 70, "追求高報酬": 100},
    },
    {
        "id": "liquidity_need",
        "dimension": "investment_goal",
        "question": "未來五年可能需要動用多少比例的投資資金？",
        "options": {"0%": 100, "25% 以下": 70, "25%–50%": 35, "50% 以上": 0},
    },
    {
        "id": "expected_return",
        "dimension": "investment_goal",
        "question": "期望的長期年化報酬率範圍是？",
        "options": {"3% 以下": 15, "3%–8%": 45, "8%–15%": 75, "15% 以上": 100},
    },
    {
        "id": "loss_reaction",
        "dimension": "risk_tolerance",
        "question": "若投資短期虧損 20%，您較可能怎麼做？",
        "options": {"立即全部賣出": 0, "賣出部分持倉": 30, "持有並重新檢視": 65, "在評估後分批加碼": 100},
    },
    {
        "id": "max_loss",
        "dimension": "risk_tolerance",
        "question": "您能接受的最大帳面損失比例是？",
        "options": {"5% 以下": 5, "5%–15%": 35, "15%–30%": 70, "30% 以上": 100},
    },
    {
        "id": "probability_choice",
        "dimension": "risk_tolerance",
        "question": "A 有 80% 機會獲利 10%；B 有 40% 機會獲利 25%，您傾向？",
        "options": {"選擇 A": 30, "兩者各配置一部分": 60, "選擇 B": 90},
    },
    {
        "id": "volatility_acceptance",
        "dimension": "risk_tolerance",
        "question": "您對投資價值波動的接受程度是？",
        "options": {"希望盡量穩定": 5, "接受小幅波動": 35, "能接受適度波動": 70, "可承受大幅波動": 100},
    },
    {
        "id": "investment_belief",
        "dimension": "risk_tolerance",
        "question": "哪一項最符合您的投資理念？",
        "options": {"安全優先，接受較低報酬": 10, "在安全與報酬間取得平衡": 55, "願承擔較高風險追求報酬": 95},
    },
    {
        "id": "correction_reaction",
        "dimension": "risk_tolerance",
        "question": "市場修正使投資下跌 12% 時，您較可能怎麼做？",
        "options": {"降低持倉並轉向低波動資產": 20, "維持配置並重新平衡": 60, "依原策略分批投入": 90},
    },
    {
        "id": "decision_basis",
        "dimension": "risk_tolerance",
        "question": "您的投資決策通常主要根據什麼？",
        "options": {"情緒與直覺": 15, "親友或市場意見": 30, "基本面與技術資料": 70, "事先制定的系統化策略": 95},
    },
]

DIMENSION_META = {
    "financial_status": ("財務狀況", "評估承受市場損失的客觀財務能力"),
    "investment_experience": ("投資經驗", "評估對投資工具與組合管理的熟悉度"),
    "investment_goal": ("投資目標", "評估投資期限、流動性需求與報酬期待"),
    "risk_tolerance": ("風險心理承受度", "評估面對虧損與價格波動時的反應"),
}


def questions_for(dimension: str) -> list[dict[str, Any]]:
    return [question for question in RISK_QUESTIONS if question["dimension"] == dimension]


def score_answer(question: dict[str, Any], answer: Any) -> float:
    if question.get("kind") != "multiselect":
        return float(question["options"][answer])
    selected = list(answer or [])
    if question["id"] == "responsibilities":
        if "無重大財務責任" in selected:
            return 100.0 if len(selected) == 1 else 70.0
        return float(max(10, 85 - 15 * len(selected)))
    if "目前都不熟悉" in selected:
        if len(selected) > 1:
            raise ValueError("「目前都不熟悉」不能與其他投資工具同時選擇。")
        return 0.0
    return float(min(100, 20 * len(selected)))


def risk_type(score: float) -> str:
    if score < 20:
        return "保守型"
    if score < 40:
        return "穩健型"
    if score < 60:
        return "平衡型"
    if score < 80:
        return "成長型"
    return "積極型"


def calculate_assessment(answers: dict[str, Any]) -> dict[str, Any]:
    missing = [question["id"] for question in RISK_QUESTIONS if not answers.get(question["id"])]
    if missing:
        raise ValueError("問卷尚有未完成題目：" + ", ".join(missing))
    dimension_scores: dict[str, float] = {}
    for dimension in DIMENSION_WEIGHTS:
        questions = questions_for(dimension)
        values = [score_answer(question, answers[question["id"]]) for question in questions]
        dimension_scores[dimension] = sum(values) / len(values)
    total = sum(dimension_scores[key] * weight for key, weight in DIMENSION_WEIGHTS.items())
    spread = max(dimension_scores.values()) - min(dimension_scores.values())
    return {
        "score": round(total, 1),
        "risk_type": risk_type(total),
        "dimension_scores": {key: round(value, 1) for key, value in dimension_scores.items()},
        "alignment_warning": spread >= 35,
        "answers": dict(answers),
    }
