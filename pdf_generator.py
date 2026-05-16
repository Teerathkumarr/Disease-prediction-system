"""
MediSense — PDF Report Generator
Generates professional clinical reports using ReportLab.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import os


# Color palette
DARK_BG    = colors.HexColor("#050A0E")
CYAN       = colors.HexColor("#00E5FF")
RED_RISK   = colors.HexColor("#FF3366")
AMBER_RISK = colors.HexColor("#FFB300")
GREEN_RISK = colors.HexColor("#00FF88")
GREY_TEXT  = colors.HexColor("#5A7A8A")
WHITE      = colors.white
LIGHT_GREY = colors.HexColor("#F5F7FA")


class MediSenseReport:
    """
    Generates a professional PDF clinical report for MediSense predictions.
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        self.styles.add(ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=DARK_BG,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#0C1419"),
            spaceBefore=12,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name="BodyText2",
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#333333"),
            spaceAfter=4,
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            name="MonoSmall",
            fontName="Courier",
            fontSize=8,
            textColor=GREY_TEXT
        ))

    def generate(self, patient_data: dict, prediction: dict, output_path: str) -> str:
        """
        Generate full PDF report.
        
        Args:
            patient_data: dict with patient demographics and input features
            prediction: dict with risk scores, SHAP values, recommendation
            output_path: where to save the PDF
        
        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2.5*cm, bottomMargin=2*cm
        )
        story = []
        module = prediction.get("module", "cardiovascular").upper()

        # ── HEADER ────────────────────────────────────────────
        story.append(Paragraph("MediSense", ParagraphStyle(
            "Logo", fontName="Helvetica-Bold", fontSize=28,
            textColor=DARK_BG, spaceAfter=2
        )))
        story.append(Paragraph(
            "AI-Powered Early Disease Detection — Clinical Report",
            ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=10,
                           textColor=GREY_TEXT, spaceAfter=2)
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=CYAN, spaceAfter=12))

        # Report metadata
        meta_data = [
            ["Report ID:", f"MDS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
             "Generated:", datetime.now().strftime("%B %d, %Y at %H:%M")],
            ["Module:", module, "Version:", "MediSense v1.0"],
        ]
        meta_table = Table(meta_data, colWidths=[3*cm, 7*cm, 3*cm, 4*cm])
        meta_table.setStyle(TableStyle([
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",  (0,0), (-1,-1), 8),
            ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#333333")),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.5*cm))

        # ── PATIENT INFORMATION ────────────────────────────────
        story.append(Paragraph("Patient Information", self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREY))
        story.append(Spacer(1, 0.2*cm))

        pat = patient_data
        patient_rows = [
            ["Name:", pat.get("name", "Anonymous"),
             "Age:", str(pat.get("age", "—"))],
            ["Gender:", pat.get("gender", "—"),
             "Patient ID:", pat.get("patient_id", "N/A")],
        ]
        pt = Table(patient_rows, colWidths=[3*cm, 7*cm, 3*cm, 4*cm])
        pt.setStyle(TableStyle([
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
            ("BACKGROUND",(0,0), (-1,-1), LIGHT_GREY),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_GREY, colors.white]),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.5*cm))

        # ── RISK SUMMARY ───────────────────────────────────────
        story.append(Paragraph("Risk Assessment Summary", self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREY))
        story.append(Spacer(1, 0.2*cm))

        risk_score   = prediction.get("risk_percent", 0)
        risk_level   = prediction.get("risk_level", "UNKNOWN")
        risk_color   = (RED_RISK if risk_level == "HIGH"
                        else AMBER_RISK if risk_level == "MODERATE" else GREEN_RISK)
        confidence   = prediction.get("confidence", 0)
        uncertainty  = prediction.get("uncertainty", 0)

        risk_data = [
            ["RISK SCORE", "RISK LEVEL", "CONFIDENCE", "UNCERTAINTY"],
            [f"{risk_score:.1f}%", risk_level, f"{confidence*100:.0f}%", f"±{uncertainty*100:.0f}%"]
        ]
        rt = Table(risk_data, colWidths=[4.25*cm]*4)
        rt.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1, 0), DARK_BG),
            ("TEXTCOLOR",    (0,0), (-1, 0), WHITE),
            ("FONTNAME",     (0,0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1, 0), 8),
            ("FONTNAME",     (0,1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE",     (0,1), (-1, 1), 16),
            ("TEXTCOLOR",    (0,1), (0, 1), risk_color),
            ("TEXTCOLOR",    (1,1), (1, 1), risk_color),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1), (-1,1), [colors.HexColor("#F9FAFB")]),
            ("BOX",          (0,0), (-1,-1), 1, colors.HexColor("#E0E0E0")),
            ("INNERGRID",    (0,0), (-1,-1), 0.5, colors.HexColor("#E0E0E0")),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.5*cm))

        # ── FEATURE IMPORTANCE (SHAP) ──────────────────────────
        story.append(Paragraph("Key Contributing Factors (SHAP Analysis)",
                                self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREY))
        story.append(Spacer(1, 0.2*cm))

        shap_vals = prediction.get("shap_values", [])
        shap_rows = [["Feature", "SHAP Value", "Impact Direction"]]
        for s in shap_vals:
            direction = "↑ Increases Risk" if s["value"] > 0 else "↓ Reduces Risk"
            shap_rows.append([
                s["feature"],
                f"{s['value']:+.3f}",
                direction
            ])
        if len(shap_rows) > 1:
            shap_table = Table(shap_rows, colWidths=[6*cm, 4*cm, 7*cm])
            shap_table.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1, 0), colors.HexColor("#0C1419")),
                ("TEXTCOLOR",    (0,0), (-1, 0), WHITE),
                ("FONTNAME",     (0,0), (-1,-1), "Helvetica"),
                ("FONTNAME",     (0,0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT_GREY, colors.white]),
                ("TOPPADDING",   (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                ("BOX",          (0,0), (-1,-1), 0.5, colors.HexColor("#E0E0E0")),
            ]))
            story.append(shap_table)
        story.append(Spacer(1, 0.5*cm))

        # ── CLINICAL RECOMMENDATION ────────────────────────────
        story.append(Paragraph("Clinical Recommendation", self.styles["SectionHeader"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREY))
        story.append(Spacer(1, 0.2*cm))

        rec_text = prediction.get("recommendation", "Please consult a physician.")
        rec_data = [[Paragraph(rec_text, self.styles["BodyText2"])]]
        rec_table = Table(rec_data, colWidths=[17*cm])
        rec_color = (colors.HexColor("#FFF0F3") if risk_level == "HIGH"
                     else colors.HexColor("#FFFBF0") if risk_level == "MODERATE"
                     else colors.HexColor("#F0FFF8"))
        rec_table.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), rec_color),
            ("BOX",          (0,0), (-1,-1), 1, risk_color),
            ("TOPPADDING",   (0,0), (-1,-1), 10),
            ("BOTTOMPADDING",(0,0), (-1,-1), 10),
            ("LEFTPADDING",  (0,0), (-1,-1), 12),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 0.5*cm))

        # ── DISCLAIMER ─────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_GREY))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "⚠️  DISCLAIMER: This report is generated by an AI system for informational purposes only. "
            "It does not constitute a medical diagnosis and should not replace professional medical advice. "
            "All predictions should be reviewed and validated by a licensed healthcare professional. "
            "MediSense is intended as a clinical decision support tool only.",
            ParagraphStyle("Disclaimer", fontName="Helvetica", fontSize=7.5,
                           textColor=GREY_TEXT, leading=11)
        ))

        doc.build(story)
        return output_path


def generate_report(patient_data: dict, prediction: dict) -> bytes:
    """Helper: generate report and return as bytes for API endpoint."""
    buf = io.BytesIO()
    reporter = MediSenseReport()
    
    tmp_path = f"/tmp/medisense_report_{datetime.now().strftime('%H%M%S')}.pdf"
    reporter.generate(patient_data, prediction, tmp_path)
    
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
    os.unlink(tmp_path)
    return pdf_bytes
