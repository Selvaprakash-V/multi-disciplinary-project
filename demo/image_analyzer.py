"""
Solar Panel Image Crack Analyzer & Diagnostic Engine.

Provides automated optical / electroluminescence (EL) solar cell crack detection,
defect segmentation, crack path tracing, power loss estimation, and preset generation.
"""

import base64
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def generate_preset_cell(preset_name: str) -> np.ndarray:
    """Generate high-fidelity solar cell images representing realistic inspection states."""
    size = 512
    # Base silicon wafer texture
    img = np.full((size, size), 212, dtype=np.uint8)

    # Solar cell busbars (3 vertical silver conductors)
    busbars = [int(size * 0.25), int(size * 0.50), int(size * 0.75)]
    for bx in busbars:
        cv2.line(img, (bx, 0), (bx, size), 70, 5)

    # Grid fingers (fine horizontal collection lines every 14 pixels)
    for y in range(14, size - 14, 14):
        cv2.line(img, (8, y), (size - 8, y), 170, 1)

    # Wafer border
    cv2.rectangle(img, (6, 6), (size - 6, size - 6), 55, 2)

    # Chamfered corners (monocrystalline pseudo-square)
    ch = 36
    cv2.fillPoly(img, [np.array([(0, 0), (ch, 0), (0, ch)])], 245)
    cv2.fillPoly(img, [np.array([(size, 0), (size - ch, 0), (size, ch)])], 245)
    cv2.fillPoly(img, [np.array([(0, size), (ch, size), (0, size - ch)])], 245)
    cv2.fillPoly(img, [np.array([(size, size), (size - ch, size), (size, size - ch)])], 245)

    # Realistic silicon wafer grain noise
    seed_map = {"intact": 101, "microcrack": 202, "cross_crack": 303, "busbar_fracture": 404}
    rng = np.random.default_rng(seed=seed_map.get(preset_name, 505))
    grain = rng.normal(0, 2.5, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    if preset_name == "microcrack":
        # Jagged diagonal micro-crack across silicon grain
        pts = [
            (95, 110), (135, 145), (175, 160), (210, 185),
            (255, 230), (290, 260), (330, 310), (370, 340), (410, 395)
        ]
        # Inactive dark zone on one side of the crack (typical in EL imaging)
        mask_poly = pts + [(pts[-1][0], size - 30), (pts[0][0], size - 30)]
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(mask_poly)], 150)
        img = cv2.addWeighted(img, 0.75, overlay, 0.25, 0)

        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            cv2.line(img, p1, p2, 20, 2)
            cv2.line(img, p1, p2, 60, 4)

    elif preset_name == "cross_crack":
        # Dendritic intersecting crack caused by mechanical impact / hail
        branch1 = [(180, 95), (210, 170), (240, 240), (265, 320), (290, 415)]
        branch2 = [(110, 220), (170, 235), (240, 240), (320, 255), (415, 280)]
        branch3 = [(240, 240), (290, 180), (350, 150)]

        for branch in (branch1, branch2, branch3):
            for i in range(len(branch) - 1):
                p1, p2 = branch[i], branch[i + 1]
                cv2.line(img, p1, p2, 15, 2)
                cv2.line(img, p1, p2, 55, 4)

    elif preset_name == "busbar_fracture":
        # Transverse crack severing across two busbars with localized finger interruption
        pts = [
            (85, 210), (128, 215), (150, 235), (195, 225),
            (250, 240), (275, 265), (320, 255), (395, 275)
        ]
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            cv2.line(img, p1, p2, 20, 3)
            cv2.line(img, p1, p2, 65, 5)

        # Sever busbar electrical continuity at the crossing
        cv2.circle(img, (busbars[0], 215), 6, 210, -1)
        cv2.circle(img, (busbars[1], 240), 6, 210, -1)

    return img


def analyze_solar_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Process an uploaded solar panel or cell image.
    Extracts crack contours, calculates physical/geometric metrics,
    and returns annotated visualization with diagnostic classification.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image file. Please provide a valid JPG or PNG.")

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Edge detection with bilateral/Gaussian filtering
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 95)

    # 2. Mask border margin to avoid frame/wafer edge detection
    margin_x = max(10, int(w * 0.05))
    margin_y = max(10, int(h * 0.05))
    edges[:margin_y, :] = 0
    edges[-margin_y:, :] = 0
    edges[:, :margin_x] = 0
    edges[:, -margin_x:] = 0

    # 3. Detect and remove straight regular grid lines (busbars and fingers)
    min_line_len = max(20, int(min(h, w) * 0.06))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=min_line_len, maxLineGap=6)
    straight_mask = np.zeros_like(edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            # Lines parallel to horizontal fingers (0/180 deg) or vertical busbars (90 deg)
            if angle < 14 or angle > 166 or (76 < angle < 104):
                cv2.line(straight_mask, (x1, y1), (x2, y2), 255, 7)

    # 4. Extract irregular crack edges (subtract straight grid lines)
    irregular_edges = cv2.subtract(edges, straight_mask)

    # 5. Connected components & contour analysis
    contours, _ = cv2.findContours(irregular_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cracks: List[np.ndarray] = []
    total_crack_length_px = 0.0
    bounding_boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        arc_len = cv2.arcLength(cnt, False)
        if arc_len >= 38.0:
            xb, yb, wb, hb = cv2.boundingRect(cnt)
            # Genuine micro-cracks span across multiple dimensions, unlike residual grid line fragments
            if wb > 8 and hb > 8:
                detected_cracks.append(cnt)
                total_crack_length_px += (arc_len / 2.0)
                bounding_boxes.append((xb, yb, wb, hb))

    # Standard 156 mm wafer scale reference
    pixel_to_mm = 156.0 / float(min(h, w))
    total_crack_length_mm = round(total_crack_length_px * pixel_to_mm, 1)

    is_crack_detected = len(detected_cracks) > 0 and total_crack_length_mm >= 10.0
    num_crack_segments = len(detected_cracks)

    # Diagnostic Classification
    if not is_crack_detected:
        verdict = "NO CRACK DETECTED"
        classification = "Intact Solar Cell (Grade A)"
        severity = "None"
        confidence_pct = 99.2
        power_loss_est_pct = 0.0
        recommendation = "Cell passes quality inspection. Normal operational efficiency."
        risk_level = "Low"
    else:
        verdict = "CRACK DETECTED"
        risk_level = "High" if total_crack_length_mm > 50 else "Moderate"

        if num_crack_segments >= 4:
            classification = "Dendritic Cross-Crack / Impact Fracture"
            severity = "Severe"
            power_loss_est_pct = min(28.0, round(5.0 + total_crack_length_mm * 0.10, 1))
            recommendation = "High risk of localized hot-spots and bypass diode overload. Immediate cell replacement advised."
        elif any(wb > w * 0.40 for _, _, wb, _ in bounding_boxes):
            classification = "Transverse Wafer Micro-Crack"
            severity = "High"
            power_loss_est_pct = min(18.0, round(3.5 + total_crack_length_mm * 0.07, 1))
            recommendation = "Active crack traversing finger lines. Schedule replacement during next scheduled maintenance."
        else:
            classification = "Incipient Micro-Fracture"
            severity = "Moderate"
            power_loss_est_pct = min(10.0, round(2.0 + total_crack_length_mm * 0.05, 1))
            recommendation = "Monitor degradation rate; perform secondary thermographic scan to check for hot-spot formation."

        confidence_pct = min(99.6, round(89.0 + min(num_crack_segments * 2.5, 10.0), 1))

    # 6. Create clean engineering-style annotation image
    annotated = bgr.copy()

    if is_crack_detected:
        # Bold high-contrast red highlighted crack paths
        cv2.drawContours(annotated, detected_cracks, -1, (0, 0, 230), 3, lineType=cv2.LINE_AA)

        # High-contrast bounding boxes
        for (xb, yb, wb, hb) in bounding_boxes:
            pad = 8
            x1 = max(0, xb - pad)
            y1 = max(0, yb - pad)
            x2 = min(w, xb + wb + pad)
            y2 = min(h, yb + hb + pad)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 200), 2)

    # Encode images to base64 PNG
    _, buffer = cv2.imencode(".png", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    _, orig_buffer = cv2.imencode(".png", bgr)
    original_b64 = base64.b64encode(orig_buffer).decode("utf-8")

    return {
        "verdict": verdict,
        "is_crack_detected": is_crack_detected,
        "classification": classification,
        "severity": severity,
        "risk_level": risk_level,
        "confidence_pct": confidence_pct,
        "crack_length_mm": total_crack_length_mm,
        "crack_segments_count": num_crack_segments,
        "power_loss_est_pct": power_loss_est_pct,
        "recommendation": recommendation,
        "image_width_px": w,
        "image_height_px": h,
        "annotated_image_b64": f"data:image/png;base64,{annotated_b64}",
        "original_image_b64": f"data:image/png;base64,{original_b64}",
    }
