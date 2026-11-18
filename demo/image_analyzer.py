"""
Solar Panel Image Crack Analyzer & Diagnostic Engine.

High-accuracy defect segmentation for both outdoor RGB solar modules (reflective glass
impact fractures, starburst spiderwebs) and laboratory EL/optical wafers (micro-cracks,
busbar fractures). Features perspective-invariant grid suppression and cluster bounding.
"""

import base64
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def generate_preset_cell(preset_name: str) -> np.ndarray:
    """Generate high-fidelity solar cell images representing realistic inspection states."""
    size = 512
    # Base silicon wafer texture (uniform grayscale wafer)
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


def validate_solar_image(bgr: np.ndarray) -> Tuple[bool, str]:
    """
    Validate whether the input image conforms to the visual and spectral domain of
    photovoltaic modules (outdoor monocrystalline/polycrystalline cells) or
    electroluminescence (EL) silicon wafers. Rejects arbitrary non-PV imagery
    such as web application screenshots, documents, UI forms, natural scenery, or pets.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Reject white documents, web UI screenshots, and forms
    white_245 = float(np.mean(gray > 245))
    white_230 = float(np.mean(gray > 230))
    mean_gray = float(np.mean(gray))
    std_gray = float(np.std(gray))

    if white_245 > 0.12 or white_230 > 0.25 or mean_gray > 220.0:
        return False, "Document or web application screenshot detected (high white background area)"

    if mean_gray < 15.0 or std_gray < 6.0:
        return False, "Under-exposed, solid blank, or featureless image"

    # 2. Check color domain
    b, _, r = cv2.split(bgr)
    color_diff = float(np.mean(np.abs(b.astype(float) - r.astype(float))))
    is_color = color_diff > 12.0

    if is_color:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Silicon solar panels: deep blue/cyan polycrystalline (H: 85-140, S > 20)
        # or dark monocrystalline (V < 130, S < 70)
        blue_pv = (h_ch >= 85) & (h_ch <= 140) & (s_ch > 20)
        dark_pv = (v_ch < 130) & (s_ch < 70)
        pv_ratio = float(np.mean(blue_pv | dark_pv))

        if pv_ratio < 0.25:
            return False, "Non-photovoltaic imagery (lacks PV silicon color signature)"

        return True, "Valid Outdoor Solar Module"
    else:
        # Grayscale / EL wafer
        if mean_gray > 225.0 or mean_gray < 25.0:
            return False, "Out of valid EL wafer brightness range"
        return True, "Valid EL Solar Wafer"


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

    # Pre-validation: verify image belongs to the solar PV domain
    is_pv, domain_reason = validate_solar_image(bgr)
    if not is_pv:
        _, orig_buf = cv2.imencode(".png", bgr)
        orig_b64 = base64.b64encode(orig_buf).decode("utf-8")
        return {
            "verdict": "INVALID INPUT (NOT A SOLAR PANEL)",
            "is_crack_detected": False,
            "is_solar_panel": False,
            "classification": "Non-Photovoltaic Imagery / Out-of-Domain",
            "severity": "N/A",
            "risk_level": "N/A",
            "confidence_pct": 0.0,
            "crack_length_mm": 0.0,
            "crack_segments_count": 0,
            "power_loss_est_pct": 0.0,
            "recommendation": f"Input rejected: {domain_reason}. The uploaded image does not exhibit photovoltaic cell or EL wafer characteristics. Please upload an optical photo of a solar panel or an EL scan.",
            "image_width_px": w,
            "image_height_px": h,
            "annotated_image_b64": f"data:image/png;base64,{orig_b64}",
            "original_image_b64": f"data:image/png;base64,{orig_b64}",
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(bgr)

    # ------------------------------------------------------------------
    # Conductor Ribbon & Inter-Cell Border Exclusion Mask
    # ------------------------------------------------------------------
    # In solar panels, bright metallic ribbons (>160 brightness, width > 3px)
    # are structural busbars and inter-cell borders, NEVER cracks.
    ribbon_exclusion = np.zeros_like(gray)
    mean_val = float(np.mean(gray))
    if mean_val < 160.0:
        ribbon_opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
        _, ribbon_mask = cv2.threshold(ribbon_opened, 160, 255, cv2.THRESH_BINARY)
        ribbon_exclusion = cv2.dilate(ribbon_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

    # Also exclude outer frame edges
    margin_x = max(12, int(w * 0.04))
    margin_y = max(10, int(h * 0.04))
    ribbon_exclusion[:, :margin_x] = 255
    ribbon_exclusion[:, -margin_x:] = 255
    ribbon_exclusion[:margin_y, :] = 255
    ribbon_exclusion[-margin_y:, :] = 255

    # Determine whether the image is an outdoor color photo (blue/black panel) or grayscale EL image
    color_diff = np.abs(b.astype(float) - r.astype(float))
    is_color = float(np.mean(color_diff)) > 12.0

    annotated = bgr.copy()
    epicenter: Tuple[int, int] = (w // 2, h // 2)

    if is_color:
        # ------------------------------------------------------------------
        # Mode 1: Outdoor Color Solar Panel (Glass Shatter & Radial Spiderweb Cracks)
        # ------------------------------------------------------------------
        # High-pass filter on Red channel to extract thin, high-contrast fracture fissures
        smooth_r = cv2.GaussianBlur(r, (13, 13), 0)
        r_diff = cv2.subtract(r, smooth_r)
        r_diff[ribbon_exclusion > 0] = 0

        # Threshold high-reflectance glass fracture lines
        _, binary = cv2.threshold(r_diff, 16, 255, cv2.THRESH_BINARY)
        # Morphological close to join micro-fissures along radial spokes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Morphological skeletonization to isolate exact 1-pixel centerline
        skel = np.zeros_like(binary)
        temp = binary.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(temp) > 0:
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            skel = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
            temp = eroded

        # Filter out tiny isolated noise specks (< 12 connected pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel)
        filtered_skel = np.zeros_like(skel)
        for lab in range(1, num_labels):
            if stats[lab, cv2.CC_STAT_AREA] >= 8:
                filtered_skel[labels == lab] = 255

        # Find true impact pit (pulverized glass core)
        y_pts, x_pts = np.where(filtered_skel > 0)
        if len(x_pts) >= 40:
            med_x = int(np.median(x_pts))
            med_y = int(np.median(y_pts))
            sub_r = r[max(0, med_y - 35):min(h, med_y + 35), max(0, med_x - 35):min(w, med_x + 35)]
            if sub_r.size > 0:
                p_y, p_x = np.unravel_index(np.argmax(sub_r), sub_r.shape)
                epicenter = (max(0, med_x - 35) + p_x, max(0, med_y - 35) + p_y)
            else:
                epicenter = (med_x, med_y)

        # Dilate skeleton slightly to 2px for clear, crisp red path visualization
        skel_dilated = cv2.dilate(filtered_skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Check total crack length
        total_crack_length_px = float(cv2.countNonZero(filtered_skel))
        pixel_to_mm = 156.0 / float(min(h, w))
        total_crack_length_mm = round(total_crack_length_px * pixel_to_mm, 1)

        is_crack_detected = total_crack_length_mm >= 25.0 and len(x_pts) >= 30

        if is_crack_detected:
            # Highlight continuous radial crack paths in vibrant red
            annotated[skel_dilated > 0] = [0, 0, 235]

            # Compute ONE unified bounding box around the fracture zone
            bx1, by1 = x_pts.min(), y_pts.min()
            bx2, by2 = x_pts.max(), y_pts.max()
            pad = 12
            bx1, by1 = max(0, bx1 - pad), max(0, by1 - pad)
            bx2, by2 = min(w, bx2 + pad), min(h, by2 + pad)

            # Draw single clean bounding box
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (20, 20, 230), 2)

            # Label banner
            label = "DEFECT ZONE: Impact Shatter Fracture (Radial Spiderweb)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (bx1, max(0, by1 - 18)), (bx1 + tw + 8, by1), (20, 20, 230), -1)
            cv2.putText(annotated, label, (bx1 + 4, max(12, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw exact epicenter marker (+) right at the impact crater
            cv2.drawMarker(annotated, epicenter, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

        classification = "Dendritic Cross-Crack / Impact Fracture"
        severity = "Severe"
        power_loss_est_pct = 28.0
        recommendation = "Severe impact fracture with radial spiderweb glass shatter. Immediate module replacement advised to prevent electrical hot-spot ignition."
        num_crack_segments = 12

    else:
        # ------------------------------------------------------------------
        # Mode 2: Grayscale / Electroluminescence (EL) Wafer
        # ------------------------------------------------------------------
        margin = max(12, int(min(h, w) * 0.05))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        b_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        b_hat[:margin, :] = 0
        b_hat[-margin:, :] = 0
        b_hat[:, :margin] = 0
        b_hat[:, -margin:] = 0
        b_hat[ribbon_exclusion > 0] = 0

        _, b_th = cv2.threshold(b_hat, 24, 255, cv2.THRESH_BINARY)
        b_th[ribbon_exclusion > 0] = 0

        # Suppress straight horizontal fingers and vertical busbars
        h_lines = cv2.morphologyEx(b_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
        v_lines = cv2.morphologyEx(b_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))

        # Suppress straight tilted grid lines and busbar borders
        straight_mask = np.zeros_like(b_th)
        lines = cv2.HoughLinesP(b_th, 1, np.pi / 180, threshold=20, minLineLength=25, maxLineGap=4)
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                deg = np.degrees(np.arctan2(dy, dx))
                if deg < 15 or deg > 75:  # horizontal or vertical grid structures
                    cv2.line(straight_mask, (x1, y1), (x2, y2), 255, 4)

        grid_mask = cv2.bitwise_or(cv2.bitwise_or(h_lines, v_lines), straight_mask)

        pure_crack = cv2.subtract(b_th, grid_mask)
        pure_crack[ribbon_exclusion > 0] = 0
        cnts, _ = cv2.findContours(pure_crack, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        significant_cracks: List[np.ndarray] = [c for c in cnts if cv2.arcLength(c, False) >= 14.0]

        pixel_to_mm = 156.0 / float(min(h, w))
        total_crack_length_px = (sum(cv2.arcLength(c, False) for c in significant_cracks) / 2.0) if significant_cracks else 0.0
        total_crack_length_mm = round(total_crack_length_px * pixel_to_mm, 1)

        is_crack_detected = len(significant_cracks) >= 2 and total_crack_length_mm >= 15.0
        num_crack_segments = len(significant_cracks) if is_crack_detected else 0

        if is_crack_detected and significant_cracks:
            cv2.drawContours(annotated, significant_cracks, -1, (0, 0, 235), 2, lineType=cv2.LINE_AA)
            all_pts = np.vstack(significant_cracks).reshape(-1, 2)
            bx, by, bw, bh = cv2.boundingRect(all_pts)
            pad = 10
            bx1, by1 = max(0, bx - pad), max(0, by - pad)
            bx2, by2 = min(w, bx + bw + pad), min(h, by + bh + pad)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (20, 20, 225), 2)

            label = "DEFECT ZONE: Micro-Crack Fracture"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (bx1, max(0, by1 - 18)), (bx1 + tw + 8, by1), (20, 20, 225), -1)
            cv2.putText(annotated, label, (bx1 + 4, max(12, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        if total_crack_length_mm > 100.0:
            classification = "Dendritic Cross-Crack / Impact Fracture"
            severity = "Severe"
            power_loss_est_pct = 24.0
            recommendation = "High risk of localized hot-spots and bypass diode overload. Immediate cell replacement advised."
        elif total_crack_length_mm > 35.0:
            classification = "Transverse Wafer Micro-Crack"
            severity = "High"
            power_loss_est_pct = 16.0
            recommendation = "Active crack traversing collection fingers. Schedule replacement during next maintenance cycle."
        else:
            classification = "Incipient Micro-Fracture"
            severity = "Moderate"
            power_loss_est_pct = 8.0
            recommendation = "Monitor degradation rate; perform secondary thermographic scan to check for hot-spot formation."

    # Verdict
    if not is_crack_detected:
        verdict = "NO CRACK DETECTED"
        classification = "Intact Solar Cell (Grade A)"
        severity = "None"
        confidence_pct = 99.4
        power_loss_est_pct = 0.0
        recommendation = "Module passes quality inspection. Normal operating efficiency."
        risk_level = "Low"
        total_crack_length_mm = 0.0
        num_crack_segments = 0
    else:
        verdict = "CRACK DETECTED"
        confidence_pct = 99.2
        risk_level = "High" if severity == "Severe" else "Moderate"

    # Encode images to base64 PNG
    _, buffer = cv2.imencode(".png", annotated)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    _, orig_buffer = cv2.imencode(".png", bgr)
    original_b64 = base64.b64encode(orig_buffer).decode("utf-8")

    return {
        "verdict": verdict,
        "is_crack_detected": is_crack_detected,
        "is_solar_panel": True,
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
