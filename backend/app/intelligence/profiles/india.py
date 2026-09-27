"""India Country & Road-Standard Profile for RoadLens AI (Staged Specification).

Architectural implementation stub for Indian driving environments under:
- Indian Roads Congress: IRC:67-2022 (Code of Practice for Road Signs)
    - Mandatory / Regulatory Signs (Circular with red ring, Stop octagonal)
    - Cautionary / Warning Signs (Equilateral triangle pointing upwards with red ring)
    - Informatory Signs (Rectangular green/blue boards)
- Ministry of Road Transport and Highways (MoRTH):
    - Central Motor Vehicles Rules (CMVR) & High Security Registration Plates (HSRP)
    - State-coded plates: [State 2 letters] [District 2 digits] [Series 1-3 letters] [Number 4 digits]
    - Bharat (BH) Series: [Year 2 digits] BH [Number 4 digits] [Letters 1-2]

STATUS: STAGED_SPECIFICATION
Active recognition benchmark for Mini Challenge 2 is USA. This profile provides the
concrete architectural structure for future Indian road perception expansion without
claiming premature accuracy.
"""
import re
from typing import Dict, Any, List, Optional
from backend.app.intelligence.profiles.base import BaseCountryProfile
from backend.app.schemas import DetectionItem, OCRResultItem, RoadIntelligenceDetails


INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL",
    "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}


class IndiaProfile(BaseCountryProfile):
    """India Road Standard Profile conforming to IRC:67 and MoRTH HSRP specifications."""

    @property
    def code(self) -> str:
        return "india"

    @property
    def name(self) -> str:
        return "India (IRC / MoRTH)"

    @property
    def speed_unit(self) -> str:
        return "km/h"

    @property
    def status(self) -> str:
        return "staged_specification"

    @property
    def standards(self) -> List[str]:
        return [
            "IRC:67-2022 (Road Signs Code of Practice)",
            "MoRTH Central Motor Vehicles Rules (CMVR)",
            "MoRTH High Security Registration Plate (HSRP) Standard"
        ]

    def validate_license_plate(
        self, plate_text: str, bounding_box: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Validates Indian license plate against MoRTH state and BH series patterns."""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", plate_text).upper()

        detected_state_code: Optional[str] = None
        is_bh_series = False
        is_valid = False

        # 1. Check BH Series: e.g. 22BH1234AA
        bh_match = re.match(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$", cleaned)
        if bh_match:
            is_bh_series = True
            is_valid = True
        else:
            # 2. Check Standard State Plate: e.g. MH12DE1432
            std_match = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$", cleaned)
            if std_match:
                state_candidate = std_match.group(1)
                if state_candidate in INDIAN_STATE_CODES:
                    detected_state_code = state_candidate
                    is_valid = True

        return {
            "valid": is_valid,
            "plate_code": cleaned if cleaned else None,
            "detected_state_code": detected_state_code,
            "is_bh_series": is_bh_series,
            "standard": "MoRTH CMVR / HSRP Specification",
            "staged_note": "Validation rule verified against MoRTH syntax; full Indian optical benchmark staged for future release."
        }

    def interpret_speed_sign(
        self, detection: DetectionItem, ocr_results: List[OCRResultItem]
    ) -> Dict[str, Any]:
        """Interprets IRC:67 Mandatory Speed Limit Signs (circular with red border, black numerals)."""
        target_tokens = []
        if detection.bounding_box and len(detection.bounding_box) == 4 and any(detection.bounding_box):
            x1, y1, x2, y2 = detection.bounding_box
            for r in ocr_results:
                if r.bounding_box and len(r.bounding_box) == 4 and any(r.bounding_box):
                    tx1, ty1, tx2, ty2 = r.bounding_box
                    tcx = (tx1 + tx2) / 2.0
                    tcy = (ty1 + ty2) / 2.0
                    if x1 <= tcx <= x2 and y1 <= tcy <= y2:
                        target_tokens.append(r)
        if not target_tokens:
            tokens_with_boxes = [r for r in ocr_results if r.bounding_box and any(r.bounding_box)]
            if not tokens_with_boxes:
                target_tokens = ocr_results

        combined_text = " ".join(r.text.upper() for r in target_tokens)
        speed_match = re.search(r"\b(20|30|40|50|60|70|80|90|100|120)\b", combined_text)
        detected_speed = int(speed_match.group(1)) if speed_match else None

        meaning = (
            f"Regulatory maximum speed limit {detected_speed} km/h (IRC:67)"
            if detected_speed
            else "Regulatory speed limit sign (IRC:67 Mandatory)"
        )

        return {
            "object_type": "speed_limit_sign",
            "regulatory_standard": "IRC:67-2022 Mandatory",
            "meaning": meaning,
            "value": detected_speed,
            "unit": self.speed_unit if detected_speed is not None else None,
        }

    def interpret_warning_sign(
        self, detection: DetectionItem, ocr_results: List[OCRResultItem]
    ) -> Dict[str, Any]:
        """Interprets IRC:67 Cautionary Signs (equilateral triangle pointing upward)."""
        return {
            "object_type": "warning_sign",
            "regulatory_standard": "IRC:67-2022 Cautionary",
            "meaning": "Cautionary road hazard (IRC:67 triangular standard)",
            "hazard_description": "Cautionary alert",
        }

    def interpret_scene(
        self,
        detections: List[DetectionItem],
        ocr_results: List[OCRResultItem],
        image_quality: Dict[str, Any],
    ) -> RoadIntelligenceDetails:
        """Synthesizes scene under Indian IRC:67 / MoRTH rules with explicit staging banner."""
        all_states: List[Dict[str, Any]] = []

        primary_object: Optional[str] = None
        meaning: Optional[str] = None
        value: Optional[Any] = None
        unit: Optional[str] = None
        det_conf: Optional[float] = None
        ocr_conf: Optional[float] = None
        primary_std: Optional[str] = None
        plate_val_info: Optional[Dict[str, Any]] = None

        combined_text = " ".join(r.text.upper() for r in ocr_results)

        for det in detections:
            cname = det.class_name
            state_item: Dict[str, Any] = {
                "class_name": cname,
                "confidence": det.confidence,
                "bounding_box": det.bounding_box,
            }

            if cname == "STOP_SIGN":
                state_item.update({
                    "regulatory_standard": "IRC:67 Mandatory Stop",
                    "meaning": "Mandatory full stop (IRC:67)",
                    "action": "STOP",
                })
                if not primary_object:
                    primary_object = "stop_sign"
                    meaning = "Full stop mandatory (IRC:67)"
                    primary_std = "IRC:67 Mandatory"
                    det_conf = det.confidence
                    ocr_conf = max([r.confidence for r in ocr_results], default=None)

            elif cname in ["SPEED_LIMIT_SIGN", "ADVISORY_SPEED"]:
                sign_intel = self.interpret_speed_sign(det, ocr_results)
                state_item.update(sign_intel)
                sign_tokens = [
                    r for r in ocr_results
                    if r.bounding_box and det.bounding_box and
                    det.bounding_box[0] <= (r.bounding_box[0] + r.bounding_box[2]) / 2.0 <= det.bounding_box[2] and
                    det.bounding_box[1] <= (r.bounding_box[1] + r.bounding_box[3]) / 2.0 <= det.bounding_box[3]
                ]
                sign_ocr_conf = max([r.confidence for r in sign_tokens], default=None)
                if not primary_object or (value is None and sign_intel["value"] is not None):
                    primary_object = sign_intel["object_type"]
                    meaning = sign_intel["meaning"]
                    value = sign_intel["value"]
                    unit = sign_intel["unit"]
                    primary_std = sign_intel["regulatory_standard"]
                    det_conf = det.confidence
                    ocr_conf = sign_ocr_conf

            elif cname in ["WARNING_SIGN", "WORK_ZONE_SIGN"]:
                warn_intel = self.interpret_warning_sign(det, ocr_results)
                state_item.update(warn_intel)
                if not primary_object:
                    primary_object = warn_intel["object_type"]
                    meaning = warn_intel["meaning"]
                    primary_std = warn_intel["regulatory_standard"]
                    det_conf = det.confidence
                    ocr_conf = max([r.confidence for r in ocr_results], default=None)

            elif cname == "LICENSE_PLATE":
                plate_tokens = [r for r in ocr_results if r.bounding_box and det.bounding_box and det.bounding_box[0] <= (r.bounding_box[0] + r.bounding_box[2]) / 2.0 <= det.bounding_box[2] and det.bounding_box[1] <= (r.bounding_box[1] + r.bounding_box[3]) / 2.0 <= det.bounding_box[3]]
                plate_text = " ".join(r.text.upper() for r in plate_tokens)
                val = self.validate_license_plate(plate_text, det.bounding_box)
                plate_val_info = val
                state_item.update({
                    "regulatory_standard": val["standard"],
                    "meaning": "Indian Vehicle Registration Plate (MoRTH)",
                    "plate_validation": val,
                })
                if not primary_object or (value is None and val.get("plate_code")):
                    primary_object = "license_plate"
                    meaning = f"Indian License Plate ({val['detected_state_code'] or 'State Unspecified'})"
                    primary_std = val["standard"]
                    value = val["plate_code"]
                    det_conf = det.confidence
                    plate_tokens = [
                        r for r in ocr_results
                        if r.bounding_box and det.bounding_box and
                        det.bounding_box[0] <= (r.bounding_box[0] + r.bounding_box[2]) / 2.0 <= det.bounding_box[2] and
                        det.bounding_box[1] <= (r.bounding_box[1] + r.bounding_box[3]) / 2.0 <= det.bounding_box[3]
                    ]
                    ocr_conf = max([r.confidence for r in plate_tokens], default=None)

            all_states.append(state_item)

        staging_note = "India profile is in staged specification mode. Active benchmark profile is USA for Mini Challenge 2."

        if det_conf is not None and ocr_conf is not None:
            interp_conf = round((det_conf + ocr_conf) / 2.0, 2)
        elif det_conf is not None:
            interp_conf = round(det_conf, 2)
        else:
            interp_conf = None

        return RoadIntelligenceDetails(
            country_profile=self.code,
            regulatory_standard=primary_std or "IRC:67-2022",
            object_type=primary_object or "unclassified_road_scene",
            meaning=meaning or "Road scene evaluated (India IRC Standard)",
            value=value,
            unit=unit,
            detection_confidence=det_conf,
            ocr_confidence=ocr_conf,
            interpretation_confidence=interp_conf,
            advisory_notes=staging_note,
            plate_validation=plate_val_info,
            all_states=all_states,
        )
