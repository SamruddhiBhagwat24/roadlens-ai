"""USA Country & Road-Standard Profile for RoadLens AI.

Implements the official profile required by Mini Challenge 2 (AMD AI Academy):
- FHWA MUTCD (Manual on Uniform Traffic Control Devices)
    - Regulatory Stop Signs (MUTCD R1-1)
    - Regulatory Speed Limit Signs (MUTCD R2-1)
    - Advisory Speed Plaques (MUTCD W13-1P, number-only yellow plaques)
    - Warning & Work-Zone Signs (MUTCD W20 Series, multi-line construction alerts)
- US Standard License Plates (50 States + DC)
    - Aspect ratio verification (~2.0:1)
    - 50-State header identification
    - Alphanumeric plate code syntax validation
"""
import re
from typing import Dict, Any, List, Optional
from backend.app.intelligence.profiles.base import BaseCountryProfile
from backend.app.schemas import DetectionItem, OCRResultItem, RoadIntelligenceDetails


US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
    "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "NEW HAMPSHIRE",
    "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
    "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
    "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA", "D.C."
}

US_WORK_ZONE_KEYWORDS = [
    "ROAD WORK", "WORK ZONE", "LANE CLOSED", "DETOUR", "FLAGGER", "MEN WORKING",
    "CONSTRUCTION", "RIGHT LANE", "LEFT LANE", "MERGE", "SHOULDER WORK", "SPEED REDUCED"
]


class USAProfile(BaseCountryProfile):
    """USA Road Standard Profile conforming to FHWA MUTCD and US Plate Standards."""

    @property
    def code(self) -> str:
        return "usa"

    @property
    def name(self) -> str:
        return "United States (MUTCD)"

    @property
    def speed_unit(self) -> str:
        return "mph"

    @property
    def status(self) -> str:
        return "production_active"

    @property
    def standards(self) -> List[str]:
        return [
            "FHWA MUTCD R-Series (Regulatory)",
            "FHWA MUTCD W-Series (Warning / Work-Zone)",
            "AAMVA Standard 12x6 Inch License Plates"
        ]

    def validate_license_plate(
        self,
        plate_text: str,
        bounding_box: Optional[List[int]] = None,
        tokens: Optional[List[OCRResultItem]] = None,
    ) -> Dict[str, Any]:
        """Validates US license plate syntax and extracts state header if visible."""
        detected_state: Optional[str] = None
        plate_code: Optional[str] = None

        if tokens:
            remaining: List[OCRResultItem] = []
            for t in tokens:
                clean_t = re.sub(r"[^A-Za-z0-9\s]", "", t.text).strip().upper()
                matched_state = False
                for state in US_STATES:
                    if state in clean_t:
                        detected_state = state
                        matched_state = True
                        break
                if not matched_state and clean_t:
                    remaining.append(t)

            if remaining and bounding_box and len(bounding_box) == 4:
                x1, y1, x2, y2 = bounding_box
                ph = max(1, y2 - y1)
                core: List[OCRResultItem] = []
                for t in remaining:
                    if t.bounding_box and len(t.bounding_box) == 4:
                        tx1, ty1, tx2, ty2 = t.bounding_box
                        th = max(1, ty2 - ty1)
                        mid_y = (ty1 + ty2) / 2.0
                        rel_y = (mid_y - y1) / float(ph)
                        rel_h = th / float(ph)
                        is_header_noise = (rel_y < 0.32 and rel_h < 0.38) or (len(t.text.strip()) <= 2 and t.confidence < 0.35)
                        is_footer_noise = (rel_y > 0.82 and rel_h < 0.30)
                        if not is_header_noise and not is_footer_noise:
                            core.append(t)
                    else:
                        core.append(t)
                if core:
                    core.sort(key=lambda item: item.bounding_box[0] if item.bounding_box else 0)
                    plate_code = "".join(re.sub(r"[^A-Za-z0-9]", "", item.text).upper() for item in core)
                else:
                    best = max(remaining, key=lambda item: (len(item.text), item.confidence))
                    plate_code = re.sub(r"[^A-Za-z0-9]", "", best.text).upper()
            elif remaining:
                best = max(remaining, key=lambda item: (len(item.text), item.confidence))
                plate_code = re.sub(r"[^A-Za-z0-9]", "", best.text).upper()

        if not plate_code:
            cleaned = re.sub(r"[^A-Za-z0-9\s]", "", plate_text).strip().upper()
            for state in US_STATES:
                if state in cleaned:
                    detected_state = state
                    cleaned = cleaned.replace(state, "").strip()
                    break
            plate_code = re.sub(r"\s+", "", cleaned)

        # Typical US plates have 4 to 8 alphanumeric characters
        is_valid_syntax = 4 <= len(plate_code) <= 8

        # Aspect ratio check if bounding box is provided
        aspect_ok = True
        if bounding_box and len(bounding_box) == 4:
            x1, y1, x2, y2 = bounding_box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            aspect = w / float(h)
            # US 12" x 6" ratio is 2.0; allow range 1.5 to 2.8 for perspective skew
            aspect_ok = 1.5 <= aspect <= 2.8

        return {
            "valid": is_valid_syntax and aspect_ok,
            "plate_code": plate_code if plate_code else None,
            "detected_state": detected_state,
            "aspect_ratio_valid": aspect_ok,
            "standard": "AAMVA 12x6 Inch Standard",
        }

    def interpret_speed_sign(
        self, detection: DetectionItem, ocr_results: List[OCRResultItem]
    ) -> Dict[str, Any]:
        """Interprets MUTCD R2-1 Speed Limit signs."""
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
        speed_match = re.search(r"\b(15|20|25|30|35|40|45|50|55|60|65|70|75|80|85)\b", combined_text)
        detected_speed = int(speed_match.group(1)) if speed_match else None

        meaning = (
            f"Regulatory maximum speed limit {detected_speed} MPH"
            if detected_speed
            else "Regulatory speed limit sign (MUTCD R2-1)"
        )

        return {
            "object_type": "speed_limit_sign",
            "regulatory_standard": "FHWA MUTCD R2-1",
            "meaning": meaning,
            "value": detected_speed,
            "unit": self.speed_unit if detected_speed is not None else None,
        }

    def interpret_advisory_speed(
        self, detection: DetectionItem, ocr_results: List[OCRResultItem]
    ) -> Dict[str, Any]:
        """Interprets MUTCD W13-1P Advisory Speed Plaques (number-only plaques)."""
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
        speed_match = re.search(r"\b(15|20|25|30|35|40|45|50|55)\b", combined_text)
        detected_speed = int(speed_match.group(1)) if speed_match else None

        meaning = (
            f"Cautionary advisory speed {detected_speed} MPH for upcoming road segment"
            if detected_speed
            else "Advisory speed plaque (MUTCD W13-1P)"
        )

        return {
            "object_type": "advisory_speed_plaque",
            "regulatory_standard": "FHWA MUTCD W13-1P",
            "meaning": meaning,
            "value": detected_speed,
            "unit": self.speed_unit if detected_speed is not None else None,
            "advisory_notes": "Advisory speed plaque carries a cautionary recommendation rather than an absolute legal maximum.",
        }

    def interpret_warning_sign(
        self, detection: DetectionItem, ocr_results: List[OCRResultItem]
    ) -> Dict[str, Any]:
        """Interprets MUTCD W20 Series Construction and Warning signs."""
        combined_text = " ".join(r.text.upper() for r in ocr_results)
        cname = detection.class_name

        matched_phrases = [kw for kw in US_WORK_ZONE_KEYWORDS if kw in combined_text]
        hazard_desc = ", ".join(matched_phrases) if matched_phrases else "Road hazard / work zone"

        std = "FHWA MUTCD W20 Series (Work Zone)" if cname == "WORK_ZONE_SIGN" else "FHWA MUTCD W-Series (Warning)"

        return {
            "object_type": cname.lower(),
            "regulatory_standard": std,
            "meaning": f"{hazard_desc} detected",
            "hazard_description": hazard_desc,
        }

    def interpret_scene(
        self,
        detections: List[DetectionItem],
        ocr_results: List[OCRResultItem],
        image_quality: Dict[str, Any],
    ) -> RoadIntelligenceDetails:
        """Synthesizes all localized entities under US MUTCD and Plate rules."""
        all_states: List[Dict[str, Any]] = []

        primary_object: Optional[str] = None
        meaning: Optional[str] = None
        value: Optional[Any] = None
        unit: Optional[str] = None
        det_conf: Optional[float] = None
        ocr_conf: Optional[float] = None
        advisory: Optional[str] = None
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
                    "regulatory_standard": "FHWA MUTCD R1-1",
                    "meaning": "Mandatory full stop required at intersection",
                    "action": "STOP",
                    "confidence": det.confidence,
                })
                if not primary_object:
                    primary_object = "stop_sign"
                    meaning = "Full stop mandatory (MUTCD R1-1)"
                    primary_std = "FHWA MUTCD R1-1"
                    det_conf = det.confidence
                    ocr_conf = max([r.confidence for r in ocr_results], default=None)

            elif cname == "SPEED_LIMIT_SIGN":
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

            elif cname == "ADVISORY_SPEED":
                adv_intel = self.interpret_advisory_speed(det, ocr_results)
                state_item.update(adv_intel)
                adv_tokens = [
                    r for r in ocr_results
                    if r.bounding_box and det.bounding_box and
                    det.bounding_box[0] <= (r.bounding_box[0] + r.bounding_box[2]) / 2.0 <= det.bounding_box[2] and
                    det.bounding_box[1] <= (r.bounding_box[1] + r.bounding_box[3]) / 2.0 <= det.bounding_box[3]
                ]
                adv_ocr_conf = max([r.confidence for r in adv_tokens], default=None)
                if not primary_object or (value is None and adv_intel["value"] is not None):
                    primary_object = adv_intel["object_type"]
                    meaning = adv_intel["meaning"]
                    value = adv_intel["value"]
                    unit = adv_intel["unit"]
                    primary_std = adv_intel["regulatory_standard"]
                    det_conf = det.confidence
                    ocr_conf = adv_ocr_conf
                    advisory = adv_intel.get("advisory_notes")

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
                plate_tokens = []
                if det.bounding_box:
                    px1, py1, px2, py2 = det.bounding_box
                    for r in ocr_results:
                        if r.bounding_box:
                            tx1, ty1, tx2, ty2 = r.bounding_box
                            tcx = (tx1 + tx2) / 2.0
                            tcy = (ty1 + ty2) / 2.0
                            if px1 <= tcx <= px2 and py1 <= tcy <= py2:
                                plate_tokens.append(r)
                if not plate_tokens:
                    plate_tokens = ocr_results

                val = self.validate_license_plate(combined_text, det.bounding_box, tokens=plate_tokens)
                plate_val_info = val
                state_item.update({
                    "regulatory_standard": val["standard"],
                    "meaning": "US Vehicle Registration Plate",
                    "plate_validation": val,
                })
                if not primary_object or (value is None and val.get("plate_code")):
                    primary_object = "license_plate"
                    meaning = f"US License Plate ({val['detected_state'] or 'State Unspecified'})"
                    primary_std = val["standard"]
                    value = val["plate_code"]
                    det_conf = det.confidence
                    ocr_conf = max([r.confidence for r in plate_tokens], default=None)

            all_states.append(state_item)

        # Quality-based advisory note
        adverse = image_quality.get("adverse_conditions", [])
        if adverse:
            adverse_note = f"Adverse visual conditions active: {', '.join(adverse)}. Sensor reliability degraded."
            advisory = f"{advisory} • {adverse_note}" if advisory else adverse_note

        # Compute semantic confidence dynamically from actual measured upstream stages
        if det_conf is not None and ocr_conf is not None:
            interp_conf = round((det_conf + ocr_conf) / 2.0, 2)
        elif det_conf is not None:
            interp_conf = round(det_conf, 2)
        else:
            interp_conf = None

        return RoadIntelligenceDetails(
            country_profile=self.code,
            regulatory_standard=primary_std,
            object_type=primary_object or "unclassified_road_scene",
            meaning=meaning or "Road scene evaluated",
            value=value,
            unit=unit,
            detection_confidence=det_conf,
            ocr_confidence=ocr_conf,
            interpretation_confidence=interp_conf,
            advisory_notes=advisory,
            plate_validation=plate_val_info,
            all_states=all_states,
        )
