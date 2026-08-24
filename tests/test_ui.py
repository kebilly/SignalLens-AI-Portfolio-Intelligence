import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def find_by_label(elements, label):
    return next(element for element in elements if element.label == label)


def fill_questionnaire_step(app: AppTest) -> None:
    for radio in app.radio:
        if radio.label != "功能導覽":
            radio.set_value(radio.options[0])
    for multiselect in app.multiselect:
        multiselect.set_value([multiselect.options[0]])


class StreamlitUITests(unittest.TestCase):
    def test_home_page_smoke_test(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        navigation = find_by_label(app.radio, "功能導覽")
        self.assertEqual(
            navigation.options,
            [
                "總覽",
                "風險評估",
                "整合分析",
                "投資組合風險",
                "新聞情緒研究",
                "ETF 曝險比較",
                "產品警示",
                "研究報告",
            ],
        )

    def test_complete_questionnaire_and_apply_result(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        find_by_label(app.radio, "功能導覽").set_value("風險評估").run(timeout=30)

        for step in range(4):
            self.assertEqual(len(app.exception), 0)
            fill_questionnaire_step(app)
            button_label = "計算評估結果" if step == 3 else "儲存並繼續"
            find_by_label(app.button, button_label).click().run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("綜合風險承受分數" in str(markdown.value) for markdown in app.markdown)
        )
        find_by_label(app.button, "套用至投資分析").click().run(timeout=30)

        risk_type_select = find_by_label(app.selectbox, "風險類型")
        self.assertTrue(risk_type_select.disabled)
        self.assertIn(
            risk_type_select.value,
            ["保守型", "穩健型", "平衡型", "成長型", "積極型"],
        )
        self.assertTrue(
            any("目前投資分析正在使用" in success.value for success in app.success)
        )
