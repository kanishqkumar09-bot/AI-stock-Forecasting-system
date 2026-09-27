from io import BytesIO
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
)


def generate_investment_report(
    selected_company,
    latest_date,
    current_price,
    prediction,
    expected_growth,
    rsi,
    volatility,
    sma20,
    sma50,
    market_trend,
    risk_level,
    score,
    recommendation,
    confidence,
    df=None,
):
    """
    Creates a professional PDF investment report.

    df is optional. If supplied, the report includes a recent closing-price
    chart. The existing dashboard works even if df is not supplied.
    """

    buffer = BytesIO()

    # ------------------------------------------------------------
    # DOCUMENT
    # ------------------------------------------------------------
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=48,
        bottomMargin=50,
        title=f"{selected_company} Investment Report",
        author="AI Financial Intelligence",
        subject="AI-based stock investment analysis",
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        leading=29,
        spaceAfter=8,
        textColor=colors.HexColor("#102A43"),
    )

    company_title = ParagraphStyle(
        "CompanyTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=27,
        leading=32,
        spaceAfter=8,
        textColor=colors.HexColor("#0B3D91"),
    )

    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#52606D"),
    )

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=17,
        leading=22,
        spaceBefore=5,
        spaceAfter=12,
        textColor=colors.HexColor("#102A43"),
    )

    subheading = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading3"],
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=7,
        textColor=colors.HexColor("#243B53"),
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=16,
        spaceAfter=9,
        textColor=colors.HexColor("#243B53"),
    )

    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#52606D"),
    )

    centered_small = ParagraphStyle(
        "CenteredSmall",
        parent=small,
        alignment=TA_CENTER,
    )

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def fmt_date(value):
        try:
            return pd.to_datetime(value).strftime("%d %b %Y")
        except Exception:
            return str(value)

    def money(value):
        return f"₹{float(value):,.2f}"

    def make_table(data, widths=(3.0 * inch, 2.5 * inch)):
        table = Table(data, colWidths=list(widths), repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#243B53")),
                    ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#BCCCDC")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F5F7FA")],
                    ),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return table

    def score_bar(score_value):
        # The project's score can be negative or positive.
        # Display it on a simple -5 to +5 scale.
        normalized = max(0, min(100, ((float(score_value) + 5) / 10) * 100))
        filled = int(round(normalized / 10))
        return "■" * filled + "□" * (10 - filled)

    def create_price_chart():
        if df is None or not isinstance(df, pd.DataFrame):
            return None

        if "Date" not in df.columns or "Close" not in df.columns:
            return None

        chart_df = df[["Date", "Close"]].copy()
        chart_df["Date"] = pd.to_datetime(chart_df["Date"], errors="coerce")
        chart_df["Close"] = pd.to_numeric(chart_df["Close"], errors="coerce")
        chart_df = chart_df.dropna().tail(60)

        if len(chart_df) < 2:
            return None

        fig, ax = plt.subplots(figsize=(8, 3.3))
        ax.plot(chart_df["Date"], chart_df["Close"], linewidth=2)
        ax.axhline(float(prediction), linestyle="--", linewidth=1.5)
        ax.set_title(f"{selected_company} — Recent Closing Price & Model Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()

        chart_buffer = BytesIO()
        fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        chart_buffer.seek(0)
        return chart_buffer

    def footer(canvas, doc_obj):
        canvas.saveState()
        width, _ = A4

        canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
        canvas.line(42, 32, width - 42, 32)

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#7B8794"))
        canvas.drawString(
            42,
            20,
            "AI Financial Intelligence • Educational Project"
        )
        canvas.drawRightString(
            width - 42,
            20,
            f"Page {doc_obj.page}"
        )
        canvas.restoreState()

    # ------------------------------------------------------------
    # PROFESSIONAL AI-STYLE NARRATIVE
    # ------------------------------------------------------------
    if expected_growth > 5:
        growth_text = (
            f"The model projects a relatively strong positive short-term movement, "
            f"with expected growth of {expected_growth:.2f}% from the current price."
        )
    elif expected_growth > 0:
        growth_text = (
            f"The model projects a modest positive short-term movement, "
            f"with expected growth of {expected_growth:.2f}%."
        )
    elif expected_growth == 0:
        growth_text = (
            "The model projects approximately no short-term price movement "
            "relative to the current price."
        )
    else:
        growth_text = (
            f"The model projects a negative short-term movement of "
            f"{abs(expected_growth):.2f}%."
        )

    if market_trend == "BULLISH":
        trend_text = (
            "The current price is above the 20-day moving average and the "
            "20-day moving average is above the 50-day moving average, "
            "which the dashboard classifies as bullish."
        )
    elif market_trend == "BEARISH":
        trend_text = (
            "The current price is below the 20-day moving average and the "
            "20-day moving average is below the 50-day moving average, "
            "which the dashboard classifies as bearish."
        )
    else:
        trend_text = (
            "The moving-average relationship does not satisfy the dashboard's "
            "bullish or bearish conditions, so the trend is classified as neutral."
        )

    if risk_level == "LOW":
        risk_text = (
            "The 20-day volatility falls in the dashboard's low-risk range."
        )
    elif risk_level == "MODERATE":
        risk_text = (
            "The 20-day volatility falls in the dashboard's moderate-risk range."
        )
    else:
        risk_text = (
            "The 20-day volatility falls in the dashboard's high-risk range."
        )

    if rsi >= 70:
        rsi_text = (
            f"RSI is {rsi:.2f}, which the dashboard treats as a potentially "
            "overbought condition for scoring purposes."
        )
    elif rsi <= 30:
        rsi_text = (
            f"RSI is {rsi:.2f}, which the dashboard treats as a potentially "
            "oversold condition for scoring purposes."
        )
    elif 40 <= rsi <= 65:
        rsi_text = (
            f"RSI is {rsi:.2f}, which falls inside the dashboard's positive "
            "RSI scoring range."
        )
    else:
        rsi_text = (
            f"RSI is {rsi:.2f}, outside the dashboard's preferred positive "
            "scoring range but not in its extreme RSI penalty zones."
        )

    ai_summary = (
        f"<b>AI Financial Intelligence assessment:</b> {selected_company} receives "
        f"a <b>{recommendation}</b> recommendation with <b>{confidence}</b> confidence. "
        f"{growth_text} {trend_text} {risk_text} {rsi_text} "
        f"The final score is {score}, calculated from expected growth, market trend, "
        f"volatility-based risk and RSI according to the project's recommendation logic. "
        f"This narrative is generated from the dashboard's calculated indicators and "
        f"does not claim certainty about future market performance."
    )

    # ------------------------------------------------------------
    # STORY
    # ------------------------------------------------------------
    story = []

    # ========================= COVER ============================
    story += [
        Spacer(1, 0.45 * inch),
        Paragraph("AI FINANCIAL INTELLIGENCE", title),
        Paragraph("Professional Investment Analysis Report", subtitle),
        Spacer(1, 0.28 * inch),
        Paragraph(str(selected_company), company_title),
        Paragraph(
            f"Market analysis date: {fmt_date(latest_date)}",
            subtitle
        ),
        Spacer(1, 0.28 * inch),
    ]

    # Recommendation badge
    badge_bg = {
        "BUY": "#D9F7E8",
        "HOLD": "#FFF3CD",
        "AVOID": "#FDE2E1",
    }.get(str(recommendation), "#E9ECEF")

    badge_text = {
        "BUY": "#147D4A",
        "HOLD": "#8A5A00",
        "AVOID": "#B42318",
    }.get(str(recommendation), "#243B53")

    badge = Table(
        [[Paragraph(
            f"<b>{recommendation}</b><br/><font size=9>Confidence: {confidence}</font>",
            ParagraphStyle(
                "Badge",
                parent=body,
                alignment=TA_CENTER,
                fontSize=20,
                leading=25,
                textColor=colors.HexColor(badge_text),
            ),
        )]],
        colWidths=[3.4 * inch],
        rowHeights=[0.8 * inch],
    )
    badge.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(badge_bg)),
                ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(badge_text)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )

    badge_wrapper = Table([[badge]], colWidths=[6.0 * inch])
    badge_wrapper.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(badge_wrapper)

    story += [
        Spacer(1, 0.3 * inch),
        make_table(
            [
                ["Report Snapshot", "Value"],
                ["Current Price", money(current_price)],
                ["Predicted Price", money(prediction)],
                ["Expected Growth", f"{expected_growth:+.2f}%"],
                ["Risk Level", str(risk_level)],
                ["Market Trend", str(market_trend)],
                ["Model Score", str(score)],
            ]
        ),
        Spacer(1, 0.35 * inch),
        Paragraph(
            "Prepared automatically from the AI Financial Intelligence dashboard.",
            centered_small,
        ),
        Spacer(1, 0.15 * inch),
        Paragraph(
            f"Report generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
            centered_small,
        ),
        PageBreak(),
    ]

    # ================= SECTION 1 ===============================
    story += [
        Paragraph("1. Company & Market Overview", heading),
        Paragraph(
            "This section presents the latest company-level market snapshot used "
            "by the dashboard's investment recommendation engine.",
            body,
        ),
        make_table(
            [
                ["Metric", "Value"],
                ["Company", str(selected_company)],
                ["Latest Analysis Date", fmt_date(latest_date)],
                ["Current Price", money(current_price)],
                ["20-Day SMA", money(sma20)],
                ["50-Day SMA", money(sma50)],
                ["Market Trend", str(market_trend)],
            ]
        ),
        Spacer(1, 16),
        Paragraph("Technical Market Interpretation", subheading),
        Paragraph(trend_text, body),
        Paragraph(
            f"The current price is {money(current_price)}, compared with a "
            f"20-day SMA of {money(sma20)} and a 50-day SMA of {money(sma50)}.",
            body,
        ),
    ]

    chart = create_price_chart()
    if chart is not None:
        story += [
            Paragraph("Recent Price Movement", subheading),
            Image(chart, width=6.35 * inch, height=2.62 * inch),
            Paragraph(
                "The chart uses the most recent available closing prices from the "
                "selected company's processed dataset. The dashed reference line "
                "represents the model's predicted price.",
                small,
            ),
        ]

    story.append(PageBreak())

    # ================= SECTION 2 ===============================
    story += [
        Paragraph("2. Stock Price Prediction", heading),
        Paragraph(
            "The dashboard uses a trained linear-regression model to generate the "
            "next-day price estimate from the latest feature row.",
            body,
        ),
        make_table(
            [
                ["Prediction Metric", "Value"],
                ["Current Price", money(current_price)],
                ["Predicted Price", money(prediction)],
                ["Expected Growth", f"{expected_growth:+.2f}%"],
                ["20-Day SMA", money(sma20)],
                ["50-Day SMA", money(sma50)],
            ]
        ),
        Spacer(1, 16),
        Paragraph("Model Interpretation", subheading),
        Paragraph(growth_text, body),
        Paragraph(
            f"The predicted price differs from the current price by "
            f"{money(abs(prediction - current_price))}. "
            f"A positive difference indicates projected upside, while a negative "
            f"difference indicates projected downside.",
            body,
        ),
        Paragraph(
            "The prediction is a machine-learning output based on historical "
            "patterns and engineered technical features. It should not be treated "
            "as a guaranteed future price.",
            body,
        ),
        PageBreak(),
    ]

    # ================= SECTION 3 ===============================
    story += [
        Paragraph("3. Risk Analysis", heading),
        Paragraph(
            "Risk is evaluated using the dashboard's 20-day volatility thresholds "
            "and supported by RSI and market-trend information.",
            body,
        ),
        make_table(
            [
                ["Risk Indicator", "Value"],
                ["RSI (14)", f"{rsi:.2f}"],
                ["20-Day Volatility", f"{volatility:.4f}"],
                ["Risk Level", str(risk_level)],
                ["Market Trend", str(market_trend)],
            ]
        ),
        Spacer(1, 16),
        Paragraph("Risk Interpretation", subheading),
        Paragraph(risk_text, body),
        Paragraph(rsi_text, body),
        Paragraph(
            "Risk classification in this project is rule-based and depends on the "
            "volatility thresholds implemented in the dashboard. It is not a "
            "complete measure of all investment risks such as valuation, liquidity, "
            "company fundamentals or macroeconomic conditions.",
            body,
        ),
        PageBreak(),
    ]

    # ================= SECTION 4 ===============================
    story += [
        Paragraph("4. Investment Recommendation", heading),
        Paragraph(
            "The final recommendation combines the project's model forecast and "
            "technical scoring framework.",
            body,
        ),
        make_table(
            [
                ["Decision Factor", "Result"],
                ["Final Recommendation", str(recommendation)],
                ["Confidence", str(confidence)],
                ["Model Score", str(score)],
                ["Expected Growth", f"{expected_growth:+.2f}%"],
                ["Risk Level", str(risk_level)],
                ["Market Trend", str(market_trend)],
                ["RSI (14)", f"{rsi:.2f}"],
            ]
        ),
        Spacer(1, 12),
        Paragraph("Recommendation Strength", subheading),
        Paragraph(
            f"<b>Score: {score}</b> &nbsp;&nbsp; {score_bar(score)}",
            body,
        ),
        Paragraph("AI-Generated Professional Company Report", subheading),
        Paragraph(ai_summary, body),
        Paragraph(
            "<b>Decision rationale:</b> The recommendation is not based on a single "
            "indicator. The score reflects the interaction of projected growth, "
            "moving-average trend, volatility-based risk and RSI conditions in the "
            "dashboard's rule set.",
            body,
        ),
        Spacer(1, 10),
    ]

    # Final decision box
    decision_color = {
        "BUY": "#147D4A",
        "HOLD": "#8A5A00",
        "AVOID": "#B42318",
    }.get(str(recommendation), "#243B53")

    decision_box = Table(
        [[Paragraph(
            f"<b>FINAL SIGNAL: {recommendation}</b><br/>"
            f"Confidence: {confidence}<br/>"
            f"Score: {score}",
            ParagraphStyle(
                "DecisionBox",
                parent=body,
                alignment=TA_CENTER,
                fontSize=13,
                leading=19,
                textColor=colors.HexColor(decision_color),
            ),
        )]],
        colWidths=[6.0 * inch],
    )
    decision_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(decision_color)),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(decision_box)

    story += [
        Spacer(1, 18),
        Paragraph(
            "<b>Important Disclaimer:</b> This report is an educational output of "
            "the AI Financial Intelligence project. It uses historical market data, "
            "technical indicators and machine-learning predictions. Predictions are "
            "uncertain and may be wrong. This report is not financial, investment, "
            "tax or legal advice and should not be used as the sole basis for any "
            "investment decision. Always perform independent research and consider "
            "your own risk tolerance and financial circumstances.",
            small,
        ),
    ]

    # ------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    buffer.seek(0)
    return buffer
