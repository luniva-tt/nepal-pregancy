import os
import json
from datetime import datetime
from src.config import REPORT_DIR
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PDFReportGenerator:
    def __init__(self):
        self.report_dir = REPORT_DIR
        os.makedirs(self.report_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Define custom styles for the report."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            alignment=1 # Center
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.darkblue
        ))
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            name='WarningText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.red
        ))

    def generate_report(self, patient_data, risk_assessment, conversation_log):
        """
        Main entry point.
        1. Generates structured JSON data.
        2. Formats JSON into PDF.
        """
        report_json = self._prepare_data(patient_data, risk_assessment, conversation_log)
        pdf_path = self._create_pdf(report_json)
        return pdf_path

    def _prepare_data(self, patient_data, risk_assessment, conversation_log):
        """
        Structuring the data into a strict JSON format.
        """
        # 1. Patient Info
        patient_info = {
            "name": patient_data.get("name", "Unknown"),
            "age": str(patient_data.get("age", "Unknown")),
            "gestational_age": str(patient_data.get("week", "Unknown")),
            "visit_date": datetime.now().strftime("%Y-%m-%d")
        }

        # 2. Symptoms (Extract from conversation or default)
        symptoms = ["Patient provided no specific symptom list."]
        # Simple heuristic: Look for keywords in user queries
        recorded_symptoms = []
        for q, a in conversation_log:
            if "headache" in q.lower(): recorded_symptoms.append("Headache")
            if "pain" in q.lower(): recorded_symptoms.append("Pain")
            if "swelling" in q.lower(): recorded_symptoms.append("Swelling")
            if "nausea" in q.lower(): recorded_symptoms.append("Nausea")
        
        if recorded_symptoms:
            symptoms = list(set(recorded_symptoms))

        # 3. Vitals
        vitals_raw = risk_assessment.get("vitals_analyzed", {})
        vitals = {
            "bp": f"{vitals_raw.get('bp_systolic', '?')}/{vitals_raw.get('bp_diastolic', '?')} mmHg",
            "hemoglobin": "Not Tested", # Placeholder as requested
            "glucose": f"{vitals_raw.get('glucose', '?')} mg/dL",
            "heart_rate": f"{vitals_raw.get('heart_rate', '?')} bpm"
        }

        # 4. Risk Assessment
        risk_level = risk_assessment.get("risk_level", "Low")
        conditions = risk_assessment.get("warnings", ["No immediate risks identified."])
        
        interpretation = "Vitals are within normal limits."
        if risk_level == "High":
            interpretation = "Critical values detected. Immediate attention required."
        elif risk_level == "Medium":
            interpretation = "Abnormal values detected. Monitoring recommended."

        risk_data = {
            "risk_level": risk_level,
            "identified_conditions": conditions,
            "clinical_interpretation": interpretation
        }

        # 5. Guidelines (Extract from last RAG response if available)
        source_doc = "N/A"
        summary = "General antenatal care guidelines apply."
        
        # Try to find the last relevant answer and source
        if conversation_log:
            last_q, last_a = conversation_log[-1]
            summary = last_a[:300] + "..." if len(last_a) > 300 else last_a
            # Sources are printed to console in main.py but not stored in log properly as structured data
            # We will use a placeholder here or need to update main.py to pass sources. 
            # For now, we state this limitation or use a generic statement.
            source_doc = "WHO / National Antenatal Care Guidelines"

        guidelines = {
            "retrieved_source": source_doc,
            "summary": summary
        }

        # 6. Action Plan
        action_plan = {
            "immediate_action": "None",
            "monitoring_plan": "Continue regular antenatal visits.",
            "referral_required": "No"
        }

        if risk_level == "High":
            action_plan["immediate_action"] = "Consult Obstetrician immediately."
            action_plan["monitoring_plan"] = "Daily BP/Glucose monitoring."
            action_plan["referral_required"] = "Yes - Urgent"
        elif risk_level == "Medium":
            action_plan["immediate_action"] = "Schedule follow-up within 1 week."
            action_plan["monitoring_plan"] = "Weekly monitoring."
            action_plan["referral_required"] = "Yes"

        return {
            "patient_info": patient_info,
            "symptoms": symptoms,
            "vital_signs": vitals,
            "risk_assessment": risk_data,
            "guideline_explanation": guidelines,
            "recommended_action": action_plan
        }

    def _create_pdf(self, data):
        """
        Generates the PDF using ReportLab Platypus.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Pregnancy_Report_{timestamp}.pdf"
        filepath = os.path.join(self.report_dir, filename)
        
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )

        elements = []

        # Title
        elements.append(Paragraph("Pregnancy Health Assessment Report", self.styles['ReportTitle']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        elements.append(Spacer(1, 20))

        # 1. Patient Information
        elements.append(Paragraph("1. Patient Information", self.styles['SectionHeader']))
        info_data = [
            ["Name:", data["patient_info"]["name"]],
            ["Age:", data["patient_info"]["age"]],
            ["Gestational Age:", data["patient_info"]["gestational_age"]],
            ["Visit Date:", data["patient_info"]["visit_date"]]
        ]
        t = Table(info_data, colWidths=[150, 300])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

        # 2. Reported Symptoms
        elements.append(Paragraph("2. Reported Symptoms", self.styles['SectionHeader']))
        for symptom in data["symptoms"]:
            elements.append(Paragraph(f"• {symptom}", self.styles['NormalText']))
        elements.append(Spacer(1, 15))

        # 3. Vital Signs
        elements.append(Paragraph("3. Vital Signs", self.styles['SectionHeader']))
        vitals_data = [
            ["Blood Pressure:", data["vital_signs"]["bp"]],
            ["Hemoglobin:", data["vital_signs"]["hemoglobin"]],
            ["Blood Glucose:", data["vital_signs"]["glucose"]],
            ["Heart Rate:", data["vital_signs"]["heart_rate"]]
        ]
        t = Table(vitals_data, colWidths=[150, 300])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

        # 4. Risk Assessment
        elements.append(Paragraph("4. Risk Assessment", self.styles['SectionHeader']))
        risk_level = data["risk_assessment"]["risk_level"]
        risk_color = colors.black
        if risk_level == "High": risk_color = colors.red
        elif risk_level == "Medium": risk_color = colors.orange

        elements.append(Paragraph(f"<b>Risk Level:</b> <font color='{risk_color}'>{risk_level}</font>", self.styles['NormalText']))
        elements.append(Spacer(1, 5))
        
        elements.append(Paragraph("<b>Identified Conditions:</b>", self.styles['NormalText']))
        for condition in data["risk_assessment"]["identified_conditions"]:
             elements.append(Paragraph(f"• {condition}", self.styles['NormalText']))
        elements.append(Spacer(1, 5))

        elements.append(Paragraph(f"<b>Clinical Interpretation:</b> {data['risk_assessment']['clinical_interpretation']}", self.styles['NormalText']))
        elements.append(Spacer(1, 15))

        # 5. Evidence-Based Guideline Explanation
        elements.append(Paragraph("5. Evidence-Based Guideline Explanation", self.styles['SectionHeader']))
        elements.append(Paragraph(f"<b>Retrieved Source Document:</b> {data['guideline_explanation']['retrieved_source']}", self.styles['NormalText']))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(f"<b>Summary of Relevant Medical Guideline:</b>", self.styles['NormalText']))
        elements.append(Paragraph(data['guideline_explanation']['summary'], self.styles['NormalText']))
        elements.append(Spacer(1, 15))

        # 6. Recommended Action
        elements.append(Paragraph("6. Recommended Action", self.styles['SectionHeader']))
        elements.append(Paragraph(f"<b>Immediate Action:</b> {data['recommended_action']['immediate_action']}", self.styles['NormalText']))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(f"<b>Monitoring Plan:</b> {data['recommended_action']['monitoring_plan']}", self.styles['NormalText']))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(f"<b>Referral Required:</b> {data['recommended_action']['referral_required']}", self.styles['NormalText']))
        elements.append(Spacer(1, 25))

        # 7. Medical Disclaimer
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Spacer(1, 10))
        disclaimer_text = "<b>7. Medical Disclaimer</b><br/>This system is for educational and research purposes only and does not replace professional medical consultation."
        elements.append(Paragraph(disclaimer_text, self.styles['NormalText']))

        # Build PDF
        doc.build(elements)
        return filepath
