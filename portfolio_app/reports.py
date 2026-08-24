from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _styles():
    font_path = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSansTC-VF.ttf"
    if "NotoSansTC" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("NotoSansTC", str(font_path)))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ChineseTitle",
            parent=base["Title"],
            fontName="NotoSansTC",
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#174A5B"),
        ),
        "heading": ParagraphStyle(
            "ChineseHeading",
            parent=base["Heading2"],
            fontName="NotoSansTC",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#176B87"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ChineseBody",
            parent=base["BodyText"],
            fontName="NotoSansTC",
            fontSize=9.2,
            leading=14,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "ChineseSmall",
            parent=base["BodyText"],
            fontName="NotoSansTC",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#52616B"),
        ),
    }


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("NotoSansTC", 8)
    canvas.setFillColor(colors.HexColor("#667780"))
    canvas.drawString(18 * mm, 12 * mm, "SignalLens - Portfolio Risk Research")
    canvas.drawRightString(192 * mm, 12 * mm, f"第 {doc.page} 頁")
    canvas.restoreState()


def portfolio_report_pdf(
    profile: dict[str, Any],
    assets: list[dict[str, Any]],
    portfolio_risk: float,
    ai_report: str,
    sentiment_summary: list[dict[str, Any]] | None = None,
) -> bytes:
    output = BytesIO()
    styles = _styles()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="投資組合風險分析報告",
        author="SignalLens Portfolio Intelligence",
    )
    story = [
        Paragraph("投資組合風險分析報告", styles["title"]),
        Spacer(1, 5 * mm),
        Paragraph(f"報告日期：{datetime.now():%Y-%m-%d %H:%M}", styles["small"]),
        Paragraph("一、風險評估摘要", styles["heading"]),
    ]
    summary = [
        ["風險屬性", profile["risk_profile"], "風險承受分數", f"{profile['risk_score']:.1f}/100"],
        ["可投資資金", f"USD {profile['capital']:,.0f}", "組合風險分數", f"{portfolio_risk:.1f}/100"],
    ]
    story.append(_table(summary, [34 * mm, 46 * mm, 38 * mm, 42 * mm]))
    story.extend([Spacer(1, 3 * mm), Paragraph("二、資產明細", styles["heading"])])
    data = [["代碼", "配置", "年化波動", "年回報", "最大回撤", "Beta", "產業"]]
    for asset in assets:
        data.append(
            [
                str(asset.get("symbol", "")),
                f"{float(asset.get('allocation', 0)):.0f}%",
                f"{float(asset.get('annual_volatility', 0)):.1f}%",
                f"{float(asset.get('annual_return', 0)):.1f}%",
                f"{float(asset.get('max_drawdown', 0)):.1f}%",
                str(asset.get("beta") if asset.get("beta") is not None else "N/A"),
                str(asset.get("sector", "Unknown")),
            ]
        )
    story.append(_table(data, [19 * mm, 18 * mm, 25 * mm, 22 * mm, 23 * mm, 17 * mm, 36 * mm]))
    if sentiment_summary:
        story.extend([Spacer(1, 3 * mm), Paragraph("三、持倉新聞情緒摘要", styles["heading"])])
        sentiment_data = [["代碼", "新聞數", "股票情緒", "文章情緒", "相關性", "主要標籤"]]
        for row in sentiment_summary:
            sentiment_data.append(
                [
                    str(row.get("symbol", "")),
                    str(row.get("count", 0)),
                    f"{float(row.get('ticker_score', 0)):+.3f}",
                    f"{float(row.get('overall_score', 0)):+.3f}",
                    f"{float(row.get('relevance', 0)):.3f}",
                    str(row.get("dominant_label", "No data")),
                ]
            )
        story.append(_table(sentiment_data, [22 * mm, 20 * mm, 28 * mm, 28 * mm, 24 * mm, 38 * mm]))
    analysis_heading = "四、AI 整合研究分析" if sentiment_summary else "三、AI 風險教育分析"
    story.extend([Spacer(1, 4 * mm), Paragraph(analysis_heading, styles["heading"])])
    for line in _clean_markdown(ai_report).splitlines():
        if not line.strip():
            story.append(Spacer(1, 2 * mm))
        elif line.startswith("##"):
            story.append(Paragraph(line.lstrip("# "), styles["heading"]))
        else:
            story.append(Paragraph(line, styles["body"]))
    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph("免責聲明", styles["heading"]),
            Paragraph(
                "本報告僅供學術研究與教育用途，不構成投資或財務建議。歷史表現不代表未來結果；使用者應自行判斷並承擔投資風險。",
                styles["small"],
            ),
        ]
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()


def _table(data, widths):
    wrapped = [
        [Paragraph(str(cell), _styles()["small"]) for cell in row]
        for row in data
    ]
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCECEF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#174A5B")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AABBC1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _clean_markdown(value: str) -> str:
    value = re.sub(r"```.*?```", "", value, flags=re.DOTALL)
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", value)
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
