from src.config import RISK_THRESHOLDS

class RiskEvaluator:
    def __init__(self):
        self.thresholds = RISK_THRESHOLDS

    def assess_risk(self, vitals):
        """
        Evaluates patient risk based on vitals.
        vitals: dict {
            "bp_systolic": int,
            "bp_diastolic": int,
            "glucose": int, 
            "heart_rate": int
        }
        Returns: dict { "risk_level": "High/Medium/Low", "warnings": [str] }
        """
        warnings = []
        risk_score = 0

        # Blood Pressure Check
        if (vitals.get("bp_systolic", 0) >= self.thresholds["bp_systolic_high"] or 
            vitals.get("bp_diastolic", 0) >= self.thresholds["bp_diastolic_high"]):
            warnings.append(f"High Blood Pressure detected ({vitals['bp_systolic']}/{vitals['bp_diastolic']}). Possible Pre-eclampsia risk.")
            risk_score += 2

        # Glucose Check
        if vitals.get("glucose", 0) >= self.thresholds["glucose_high"]:
            warnings.append(f"High Glucose level ({vitals['glucose']} mg/dL). Gestational Diabetes risk.")
            risk_score += 2

        # Heart Rate Check
        hr = vitals.get("heart_rate", 0)
        if hr >= self.thresholds["heart_rate_high"]:
            warnings.append(f"Tachycardia detected ({hr} bpm).")
            risk_score += 1
        elif hr > 0 and hr <= self.thresholds["heart_rate_low"]:
            warnings.append(f"Bradycardia detected ({hr} bpm).")
            risk_score += 1

        # Determine Risk Level
        if risk_score >= 2:
            risk_level = "High"
        elif risk_score == 1:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "risk_level": risk_level,
            "warnings": warnings,
            "vitals_analyzed": vitals
        }
