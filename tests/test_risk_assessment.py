import unittest

from portfolio_app.risk_assessment import (
    DIMENSION_WEIGHTS,
    RISK_QUESTIONS,
    calculate_assessment,
    questions_for,
    risk_type,
)


def answer_set(high: bool) -> dict:
    answers = {}
    for question in RISK_QUESTIONS:
        if question.get("kind") == "multiselect":
            if question["id"] == "responsibilities":
                answers[question["id"]] = (
                    ["無重大財務責任"] if high else question["options"][1:]
                )
            else:
                answers[question["id"]] = (
                    list(question["options"])[1:] if high else [question["options"][0]]
                )
            continue
        ranked = sorted(question["options"], key=question["options"].get)
        answers[question["id"]] = ranked[-1] if high else ranked[0]
    return answers


class RiskAssessmentTests(unittest.TestCase):
    def test_questionnaire_has_twenty_questions_in_four_dimensions(self):
        self.assertEqual(len(RISK_QUESTIONS), 20)
        self.assertEqual(set(DIMENSION_WEIGHTS), {q["dimension"] for q in RISK_QUESTIONS})
        self.assertEqual([len(questions_for(key)) for key in DIMENSION_WEIGHTS], [5, 4, 4, 7])

    def test_high_answers_produce_higher_explainable_score(self):
        low = calculate_assessment(answer_set(False))
        high = calculate_assessment(answer_set(True))

        self.assertLess(low["score"], high["score"])
        self.assertEqual(low["risk_type"], "保守型")
        self.assertEqual(high["risk_type"], "積極型")
        self.assertEqual(set(high["dimension_scores"]), set(DIMENSION_WEIGHTS))

    def test_incomplete_answers_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_assessment({})

    def test_unknown_instruments_is_zero_and_mutually_exclusive(self):
        low = calculate_assessment(answer_set(False))
        self.assertEqual(low["dimension_scores"]["investment_experience"], 13.8)

        contradictory = answer_set(False)
        contradictory["known_instruments"] = ["目前都不熟悉", "股票"]
        with self.assertRaises(ValueError):
            calculate_assessment(contradictory)

    def test_risk_type_boundaries(self):
        self.assertEqual(risk_type(19.9), "保守型")
        self.assertEqual(risk_type(20), "穩健型")
        self.assertEqual(risk_type(40), "平衡型")
        self.assertEqual(risk_type(60), "成長型")
        self.assertEqual(risk_type(80), "積極型")
