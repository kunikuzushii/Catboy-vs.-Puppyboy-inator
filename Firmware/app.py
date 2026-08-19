"""
Catboy vs Puppyboy Detector — application prototype

Runs on a laptop webcam (no hardware needed) so the full ML pipeline can be
tested and demoed before the ESP32/CAD/firmware side exists. Press SPACE to
scan, 'q' or ESC to quit.

Pipeline:
    camera frame -> MediaPipe landmarks -> heuristic score (Judge 1)
                                         -> classifier score (Judge 2, optional)
                                         -> fusion -> on-screen display

Install:
    pip install mediapipe opencv-python numpy
    # optional, only needed once you've trained/exported a classifier:
    pip install tensorflow  (or tflite-runtime on constrained machines)
"""

import time
import os
import numpy as np
import cv2
import mediapipe as mp

# --------------------------------------------------------------------------
# 1. Landmark extraction
# --------------------------------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_landmarks(frame_bgr):
    """Returns a list of (x, y) pixel-coordinate landmarks, or None if no face."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    h, w, _ = frame_bgr.shape
    lm = results.multi_face_landmarks[0].landmark
    return [(pt.x * w, pt.y * h) for pt in lm]


# --------------------------------------------------------------------------
# 2. Judge 1 — landmark heuristic ("vibes" model)
# --------------------------------------------------------------------------
# Landmark indices below are standard MediaPipe Face Mesh points (468-point
# map). These are reasonable starting points, not fixed truth — recalibrate
# per section 4.5 of the tutorial doc once you've tested on real faces.

IDX = {
    "eye_top": 159, "eye_bottom": 145, "eye_outer": 33, "eye_inner": 133,
    "face_top": 10, "face_bottom": 152, "face_left": 234, "face_right": 454,
    "nose_bridge": 168, "nose_base": 2, "nose_left": 98, "nose_right": 327,
    "mouth_left": 61, "mouth_right": 291, "mouth_top": 13, "mouth_bottom": 14,
    "cheek_left": 234, "cheek_right": 454, "jaw_left": 172, "jaw_right": 397,
    "brow_left": 105, "eye_ref": 159,
}

WEIGHTS = {
    "eye_shape": 0.25,
    "face_shape": 0.20,
    "nose": 0.15,
    "mouth": 0.20,
    "cheekbones": 0.10,
    "eyebrows": 0.10,
}


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _normalize(value, low, high):
    """Maps a raw ratio to roughly [-1, 1] given an expected human range."""
    return float(np.clip((value - low) / (high - low) * 2 - 1, -1, 1))


def heuristic_score(landmarks):
    """Returns (cat_pct, dog_pct, features_dict) from -1 (cat) to +1 (dog)."""
    L = landmarks

    eye_ratio = _dist(L[IDX["eye_top"]], L[IDX["eye_bottom"]]) / _dist(
        L[IDX["eye_outer"]], L[IDX["eye_inner"]]
    )
    face_ratio = _dist(L[IDX["face_left"]], L[IDX["face_right"]]) / _dist(
        L[IDX["face_top"]], L[IDX["face_bottom"]]
    )
    nose_ratio = _dist(L[IDX["nose_bridge"]], L[IDX["nose_base"]]) / _dist(
        L[IDX["nose_left"]], L[IDX["nose_right"]]
    )
    mouth_ratio = _dist(L[IDX["mouth_left"]], L[IDX["mouth_right"]]) / _dist(
        L[IDX["face_left"]], L[IDX["face_right"]]
    )
    cheek_ratio = _dist(L[IDX["cheek_left"]], L[IDX["cheek_right"]]) / _dist(
        L[IDX["jaw_left"]], L[IDX["jaw_right"]]
    )
    brow_height = _dist(L[IDX["brow_left"]], L[IDX["eye_ref"]])

    features = {
        "eye_shape": _normalize(eye_ratio, 0.2, 0.5),
        "face_shape": _normalize(face_ratio, 0.6, 1.0),
        "nose": _normalize(nose_ratio, 0.3, 0.8),
        "mouth": _normalize(mouth_ratio, 0.3, 0.6),
        "cheekbones": _normalize(cheek_ratio, 0.9, 1.1),
        "eyebrows": _normalize(brow_height, 10, 30),
    }

    dog_score = sum(WEIGHTS[k] * features[k] for k in WEIGHTS)  # -1..+1
    cat_pct = round((1 - dog_score) / 2 * 100)
    dog_pct = 100 - cat_pct
    return cat_pct, dog_pct, features


# --------------------------------------------------------------------------
# 3. Judge 2 — trained classifier (pluggable, optional)
# --------------------------------------------------------------------------
# Works with no model present (falls back to a neutral 50/50 vote) so the
# app runs standalone before a classifier has been trained. Once you export
# a Teachable Machine / TFLite model, drop it in as MODEL_PATH and this
# picks it up automatically.

MODEL_PATH = "classifier.tflite"
LABELS = ["catboy", "puppyboy"]  # must match training class order

_classifier_interpreter = None


def _load_classifier():
    global _classifier_interpreter
    if _classifier_interpreter is not None or not os.path.exists(MODEL_PATH):
        return
    try:
        import tensorflow as tf
        _classifier_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _classifier_interpreter.allocate_tensors()
    except Exception as e:
        print(f"[classifier] could not load model, falling back to neutral: {e}")


def classifier_score(frame_bgr):
    """Returns (cat_pct, dog_pct). Falls back to neutral 50/50 if no model."""
    _load_classifier()
    if _classifier_interpreter is None:
        return 50, 50

    input_details = _classifier_interpreter.get_input_details()
    output_details = _classifier_interpreter.get_output_details()
    _, in_h, in_w, _ = input_details[0]["shape"]

    img = cv2.resize(frame_bgr, (in_w, in_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    _classifier_interpreter.set_tensor(input_details[0]["index"], img)
    _classifier_interpreter.invoke()
    output = _classifier_interpreter.get_tensor(output_details[0]["index"])[0]

    cat_pct = round(float(output[LABELS.index("catboy")]) * 100)
    dog_pct = 100 - cat_pct
    return cat_pct, dog_pct


# --------------------------------------------------------------------------
# 4. Fusion
# --------------------------------------------------------------------------

CONFLICT_THRESHOLD = 30  # percentage points of disagreement to flag as "conflicted"


def fuse(heuristic_cat_pct, classifier_cat_pct, w_heuristic=0.5, w_classifier=0.5):
    fused_cat_pct = round(w_heuristic * heuristic_cat_pct + w_classifier * classifier_cat_pct)
    fused_dog_pct = 100 - fused_cat_pct
    disagreement = abs(heuristic_cat_pct - classifier_cat_pct)
    conflicted = disagreement > CONFLICT_THRESHOLD
    return {
        "cat_pct": fused_cat_pct,
        "dog_pct": fused_dog_pct,
        "conflicted": conflicted,
        "disagreement": disagreement,
    }


# --------------------------------------------------------------------------
# 5. Display (desktop stand-in for the device's LED/TFT screen)
# --------------------------------------------------------------------------

CAT_COLOR = (180, 105, 255)   # BGR — pink/magenta
DOG_COLOR = (255, 170, 60)    # BGR — blue/orange
NEUTRAL = (230, 230, 230)


def draw_bar(frame, x, y, w, h, pct, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), NEUTRAL, 1)
    fill_w = int(w * pct / 100)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)


def draw_overlay(frame, heuristic_result, classifier_result, fused, status):
    h, w, _ = frame.shape
    panel_h = 150
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (30, 28, 24)

    h_cat, h_dog, _ = heuristic_result
    c_cat, c_dog = classifier_result

    cv2.putText(panel, f"heuristic: cat {h_cat}% / dog {h_dog}%", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEUTRAL, 1, cv2.LINE_AA)
    draw_bar(panel, 12, 30, 250, 10, h_cat, CAT_COLOR)

    cv2.putText(panel, f"classifier: cat {c_cat}% / dog {c_dog}%", (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEUTRAL, 1, cv2.LINE_AA)
    draw_bar(panel, 12, 70, 250, 10, c_cat, CAT_COLOR)

    verdict = "CATBOY" if fused["cat_pct"] >= 50 else "PUPPYBOY"
    verdict_color = CAT_COLOR if fused["cat_pct"] >= 50 else DOG_COLOR
    cv2.putText(panel, f"VERDICT: {verdict} ({max(fused['cat_pct'], fused['dog_pct'])}%)",
                (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, verdict_color, 2, cv2.LINE_AA)

    if fused["conflicted"]:
        cv2.putText(panel, "the machine is CONFLICTED", (12, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1, cv2.LINE_AA)

    cv2.putText(panel, status, (w - 220, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (140, 140, 140), 1, cv2.LINE_AA)

    return np.vstack([frame, panel])


# --------------------------------------------------------------------------
# 6. Main loop
# --------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0). Check camera permissions/index.")

    last_heuristic = (50, 50, {})
    last_classifier = (50, 50)
    last_fused = fuse(50, 50)
    status = "press SPACE to scan"

    print("Catboy vs Puppyboy Detector — SPACE to scan, 'q'/ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        display_frame = draw_overlay(frame, last_heuristic, last_classifier, last_fused, status)
        cv2.imshow("Catboy vs Puppyboy Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord(" "):  # SPACE = scan, mirrors the physical button press
            landmarks = get_landmarks(frame)
            if landmarks is None:
                status = "no face detected"
                continue
            last_heuristic = heuristic_score(landmarks)
            last_classifier = classifier_score(frame)
            last_fused = fuse(last_heuristic[0], last_classifier[0])
            status = f"scanned at {time.strftime('%H:%M:%S')}"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
