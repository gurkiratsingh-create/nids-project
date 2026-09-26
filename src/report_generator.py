"""Automated PDF Executive Report Generation for SOC Investigations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ReportLab imports with fallback
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("ReportLab library not installed. PDF reporting will generate formatted text fallback.")


class ReportGenerator:
    """Generate formal executive security audit PDF reports from database incidents."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the report generator with output paths."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.reports_dir = Path(output_dir) if output_dir is not None else self.project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(
        self,
        incidents: List[Dict[str, Any]],
        output_filename: str = "soc_incident_report.pdf",
    ) -> Path:
        """Build and save a multi-page executive security audit PDF report."""
        pdf_path = self.reports_dir / output_filename
        logger.info("Generating PDF security report at %s", pdf_path)

        if not HAS_REPORTLAB:
            # Fallback text file report if ReportLab is absent
            txt_path = pdf_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=== SOC SECURITY INCIDENT AUDIT REPORT ===\n\n")
                f.write(f"Total Incidents Logged: {len(incidents)}\n\n")
                for inc in incidents[:20]:
                    f.write(f"ID: {inc.get('id')} | Time: {inc.get('timestamp')} | Src: {inc.get('source_ip')} -> Dst: {inc.get('destination_ip')} | Attack: {inc.get('attack_type')} | Risk: {inc.get('risk_score')}\n")
            logger.info("Generated fallback text report at %s", txt_path)
            return txt_path

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=12,
        )
        subtitle_style = ParagraphStyle(
            "SubTitleStyle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            "Heading2Style",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=14,
            spaceAfter=8,
        )

        elements = []

        # Title Block
        elements.append(Paragraph("Enterprise NIDS - SOC Security Audit Report", title_style))
        elements.append(Paragraph("Automated Threat Intelligence, Explainable AI & Incident Investigation Summary", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=15))

        # KPI Summary Table
        total_inc = len(incidents)
        critical_inc = sum(1 for i in incidents if i.get("risk_score", 0) >= 0.8)
        high_inc = sum(1 for i in incidents if 0.5 <= i.get("risk_score", 0) < 0.8)

        kpi_data = [
            ["Total Flagged Incidents", "Critical Risk (>= 0.80)", "High Risk (0.50 - 0.79)", "Compliance Status"],
            [str(total_inc), str(critical_inc), str(high_inc), "ACTIVE MONITORING"],
        ]
        kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 130])
        kpi_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor("#dc2626")),
            ])
        )
        elements.append(kpi_table)
        elements.append(Spacer(1, 15))

        # Detailed Incident Table
        elements.append(Paragraph("Logged Threat Incidents", heading_style))
        table_headers = ["ID", "Timestamp", "Source IP", "Dest IP", "Attack Type", "Risk", "Recommendation"]
        table_rows = [table_headers]

        for inc in incidents[:25]:  # Limit top 25 rows for concise PDF length
            table_rows.append([
                str(inc.get("id", "")),
                str(inc.get("timestamp", ""))[:16],
                str(inc.get("source_ip", "")),
                str(inc.get("destination_ip", "")),
                str(inc.get("attack_type", "")),
                f"{inc.get('risk_score', 0.0):.2f}",
                str(inc.get("recommendation", ""))[:30] + "...",
            ])

        inc_table = Table(table_rows, colWidths=[30, 95, 75, 75, 80, 45, 140])
        inc_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ])
        )
        elements.append(inc_table)
        
        doc.build(elements)
        logger.info("PDF generation complete: %s", pdf_path)
        return pdf_path
