import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SMOOTHING_ALPHA = 0.3
FINGER_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# ---------------------------------------------------------
# UTIL : Dessiner les landmarks (version simplifiée)
# ---------------------------------------------------------
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = rgb_image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            landmark_list = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ]
            )

            mp_drawing.draw_landmarks(
                annotated_image,
                landmark_list,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
    return annotated_image


def compute_palm_normal(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z], dtype=np.float32)
    index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z], dtype=np.float32)
    pinky_mcp = np.array([hand_landmarks[17].x, hand_landmarks[17].y, hand_landmarks[17].z], dtype=np.float32)

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm == 0:
        return None

    return normal / norm


def smooth_vector(previous, current, alpha=SMOOTHING_ALPHA):
    if current is None:
        return None

    if previous is None:
        previous = current

    smoothed = previous * (1.0 - alpha) + current * alpha
    norm = np.linalg.norm(smoothed)
    if norm == 0:
        return current

    return smoothed / norm


def smooth_scalar(previous, current, alpha=SMOOTHING_ALPHA):
    if current is None:
        return previous

    if previous is None:
        return current

    return previous * (1.0 - alpha) + current * alpha


def compute_hand_axes(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z], dtype=np.float32)
    middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y, hand_landmarks[9].z], dtype=np.float32)

    forward = compute_palm_normal(hand_landmarks)
    if forward is None:
        return None

    up = middle_mcp - wrist
    up_norm = np.linalg.norm(up)
    if up_norm == 0:
        return None
    up = up / up_norm

    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm == 0:
        return None
    right = right / right_norm

    # Re-orthogonalize up using the computed right and forward to avoid drift.
    up = np.cross(forward, right)
    up_norm = np.linalg.norm(up)
    if up_norm == 0:
        return None
    up = up / up_norm

    return {
        "forward": forward,
        "up": up,
        "right": right,
    }


def draw_palm_normal(image, hand_landmarks, normal, label):
    h, w, _ = image.shape
    wrist = hand_landmarks[0]

    start = (int(wrist.x * w), int(wrist.y * h))
    scale = 120
    end = (
        int(start[0] + normal[0] * scale),
        int(start[1] + normal[1] * scale),
    )

    cv2.arrowedLine(image, start, end, (0, 255, 255), 2, tipLength=0.25)
    text = f"{label} n=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f})"
    text_origin = (start[0] + 10, start[1] - 10)
    cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_hand_axes(image, hand_landmarks, axes):
    colors = {
        "forward": (0, 0, 255),
        "up": (0, 255, 0),
        "right": (255, 0, 0),
    }

    h, w, _ = image.shape
    wrist = hand_landmarks[0]
    origin = (int(wrist.x * w), int(wrist.y * h))
    scale = 100

    for name, vec in axes.items():
        end = (
            int(origin[0] + vec[0] * scale),
            int(origin[1] + vec[1] * scale),
        )
        cv2.arrowedLine(image, origin, end, colors[name], 2, tipLength=0.25)
        cv2.putText(
            image,
            name[0].upper(),
            (end[0] + 5, end[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[name],
            1,
            cv2.LINE_AA,
        )


def landmark_to_np(landmark):
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def angle_between(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return None

    dot = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
    return np.arccos(dot)


def compute_finger_curls(hand_landmarks):
    configs = {
        "Thumb": (1, 2, 4),
        "Index": (5, 6, 8),
        "Middle": (9, 10, 12),
        "Ring": (13, 14, 16),
        "Pinky": (17, 18, 20),
    }

    curls = {}
    for finger, (a_idx, b_idx, c_idx) in configs.items():
        a = landmark_to_np(hand_landmarks[a_idx])
        b = landmark_to_np(hand_landmarks[b_idx])
        c = landmark_to_np(hand_landmarks[c_idx])

        vec_ab = a - b
        vec_cb = c - b
        angle = angle_between(vec_ab, vec_cb)
        if angle is None:
            continue

        curl = 1.0 - min(angle / np.pi, 1.0)
        curls[finger] = curl

    return curls


def draw_finger_curl_ui(image, curls):
    bar_width = 200
    bar_height = 18
    margin_left = 30
    margin_top = 30
    spacing = 12

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (margin_left - 20, margin_top - 20),
        (margin_left + bar_width + 40, margin_top + (bar_height + spacing) * len(FINGER_ORDER)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    for idx, finger in enumerate(FINGER_ORDER):
        value = np.clip(curls.get(finger, 0.0), 0.0, 1.0)
        top = margin_top + idx * (bar_height + spacing)
        start = (margin_left, top)
        end = (margin_left + bar_width, top + bar_height)
        fill_end = (margin_left + int(bar_width * value), top + bar_height)

        cv2.rectangle(image, start, end, (80, 80, 80), 1)
        cv2.rectangle(image, start, fill_end, (0, 165, 255), -1)
        label = f"{finger}: {value:.2f}"
        cv2.putText(
            image,
            label,
            (end[0] + 15, end[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
# ---------------------------------------------------------
# MAIN : Webcam + MediaPipe Tasks HandLandmarker
# ---------------------------------------------------------
def main():
    # STEP 1: Create HandLandmarker
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    detector = vision.HandLandmarker.create_from_options(options)

    smoothed_normals = {}
    smoothed_axes = {}
    smoothed_curls = {}

    # STEP 2: OpenCV webcam loop
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Impossible d’ouvrir la webcam.")
        return

    print("🎉 HandOrientationDetector démarré. Appuie sur Q pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame non lue.")
            break

        # Convertir frame en mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # STEP 3: Détection
        detection_result = detector.detect(mp_image)

        # STEP 4: Dessin
        annotated = draw_landmarks_on_image(rgb_frame, detection_result)

        right_index = None
        handedness_list = detection_result.handedness or []
        for idx, handedness in enumerate(handedness_list):
            if handedness and handedness[0].category_name.lower() == "right":
                right_index = idx
                break

        if (
            detection_result.hand_landmarks
            and right_index is not None
            and right_index < len(detection_result.hand_landmarks)
        ):
            hand_landmarks = detection_result.hand_landmarks[right_index]
            label = "Right"
            if handedness_list and handedness_list[right_index]:
                label = handedness_list[right_index][0].category_name

            normal = compute_palm_normal(hand_landmarks)
            if normal is not None:
                smoothed_normal = smooth_vector(smoothed_normals.get(label), normal)
                smoothed_normals[label] = smoothed_normal
                draw_palm_normal(annotated, hand_landmarks, smoothed_normal, label)

            axes = compute_hand_axes(hand_landmarks)
            if axes:
                prev_axes = smoothed_axes.get(label, {})
                blended_axes = {}
                smoothed_axes[label] = {}

                for axis_name, axis_vec in axes.items():
                    blended = smooth_vector(prev_axes.get(axis_name), axis_vec)
                    if blended is None:
                        continue
                    smoothed_axes[label][axis_name] = blended
                    blended_axes[axis_name] = blended

                if blended_axes:
                    draw_hand_axes(annotated, hand_landmarks, blended_axes)

            curls = compute_finger_curls(hand_landmarks)
            if curls:
                for finger, value in curls.items():
                    smoothed_curls[finger] = smooth_scalar(smoothed_curls.get(finger), value)

                draw_finger_curl_ui(annotated, smoothed_curls)

        # STEP 5: Affichage
        cv2.imshow("Hand Orientation Detector", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
