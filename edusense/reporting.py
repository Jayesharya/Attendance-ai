"""PDF report rendering from aggregate session metrics."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape


def make_pdf_report(
    started_at: str,
    source_label: str,
    summary: dict[str, object],
    metrics: pd.DataFrame,
) -> bytes:
    """Create an in-memory PDF containing only classroom-level aggregates."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="EduSense AI session summary",
        author="EduSense AI",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("EduSense AI · Session summary", styles["Title"]),
        Paragraph(f"Started: {started_at}", styles["BodyText"]),
        Paragraph(f"Source: {escape(source_label)}", styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph(
            "Classroom signals are approximate and should be interpreted with context.",
            styles["BodyText"],
        ),
        Spacer(1, 0.15 * inch),
    ]
    kpis = [
        ["Activity signal", "Present profiles", "Roster coverage", "Audio average"],
        [
            f"{float(summary['mean_activity_signal']):.1f}/100",
            f"{summary['present_profiles']}/{summary['tracked_profiles']}",
            f"{summary['roster_coverage']:.1f}%"
            if summary["roster_coverage"] is not None
            else "Not set",
            f"{summary['mean_audio_dbfs']:.1f} dBFS"
            if summary["mean_audio_dbfs"] is not None
            else "Not provided",
        ],
    ]
    story.append(
        Table(
            kpis,
            colWidths=[1.6 * inch, 1.55 * inch, 1.65 * inch, 1.55 * inch],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B4FE4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F1F3FC")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCD2E8")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]),
        )
    )

    columns = [
        column for column in (
            "elapsed_seconds",
            "detected_faces",
            "present_profiles",
            "motion_signal",
            "audio_activity",
            "activity_signal",
        ) if column in metrics.columns
    ]
    story.extend([Spacer(1, 0.2 * inch), Paragraph("Activity timeline", styles["Heading2"])])
    rows = [columns]
    samples = metrics[columns]
    if len(samples) > 20:
        samples = samples.iloc[::max(1, len(samples) // 20)].head(20)
    rows.extend([[str(value)[:28] for value in row] for row in samples.itertuples(index=False, name=None)])
    if len(rows) == 1:
        rows.append(["No samples"] + [""] * (len(columns) - 1))
    story.append(
        Table(rows, repeatRows=1, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3B4FE4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCD2E8")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F6FB")]),
        ])),
    )
    document.build(story)
    return output.getvalue()
