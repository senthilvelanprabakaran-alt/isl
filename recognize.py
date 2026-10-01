import cv2
import time
import os
import string
import numpy as np
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

MEDIAPIPE_MODEL = "hand_landmarker.task"
CLASSIFIER_MODEL = "model.h5"

WINDOW_NAME = "Signify - ISL Recognition | 0-9 + A-Z"

CONFIDENCE_THRESHOLD = 0.40


# ============================================================
# LABELS
# ============================================================

# Signify mapping:
# 0-9 + A-Z = 36 classes

LABELS = (
    [str(i) for i in range(10)]
    + list(string.ascii_uppercase)
)


# ============================================================
# HAND SKELETON
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (5, 9),
    (9, 13),
    (13, 17),
]


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_landmarks(hand_landmarks):

    """
    21 landmarks
    X + Y
    Wrist becomes origin
    Normalize
    Output = 42 features
    """

    points = []

    for landmark in hand_landmarks:

        points.append([
            landmark.x,
            landmark.y
        ])

    points = np.array(
        points,
        dtype=np.float32
    )

    if points.shape != (21, 2):

        return np.zeros(
            42,
            dtype=np.float32
        )

    # Wrist = origin
    points[:, 0] -= points[0][0]
    points[:, 1] -= points[0][1]

    # Flatten
    features = points.flatten()

    # Normalize
    max_value = np.max(
        np.abs(features)
    )

    if max_value > 0:

        features /= max_value

    return features.astype(
        np.float32
    )


# ============================================================
# DRAW SKELETON
# ============================================================

def draw_skeleton(
    frame,
    hand_landmarks
):

    height, width, _ = frame.shape

    points = []

    for landmark in hand_landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        x = max(
            0,
            min(
                x,
                width - 1
            )
        )

        y = max(
            0,
            min(
                y,
                height - 1
            )
        )

        points.append(
            (x, y)
        )

        # Landmark
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    # Skeleton
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(
    MEDIAPIPE_MODEL
):

    print(
        f"ERROR: {MEDIAPIPE_MODEL} not found."
    )

    raise SystemExit


if not os.path.exists(
    CLASSIFIER_MODEL
):

    print(
        f"ERROR: {CLASSIFIER_MODEL} not found."
    )

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading TensorFlow model...")

model = tf.keras.models.load_model(
    CLASSIFIER_MODEL,
    compile=False
)

print("Model loaded.")

print(
    "Input:",
    model.input_shape
)

print(
    "Output:",
    model.output_shape
)


# ============================================================
# CHECK CLASSES
# ============================================================

num_classes = int(
    model.output_shape[-1]
)

print(
    "Classes:",
    num_classes
)

print(
    "Labels:",
    ", ".join(LABELS)
)

EXPECTED_CLASSES = 36

if num_classes != EXPECTED_CLASSES:

    print(
        f"WARNING: Expected {EXPECTED_CLASSES} model classes, "
        f"but model has {num_classes}."
    )

if num_classes != len(LABELS):

    print(
        "WARNING: Model output classes and label list differ."
    )


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MEDIAPIPE_MODEL
)

options = vision.HandLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)

landmarker = (
    vision.HandLandmarker.create_from_options(
        options
    )
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Camera not found."
    )

    landmarker.close()

    raise SystemExit


# ============================================================
# FULLSCREEN
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# TIMESTAMP
# ============================================================

start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    # RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Timestamp
    timestamp_ms = int(
        (
            time.time()
            -
            start_time
        ) * 1000
    )

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # STORE ALL PREDICTIONS
    # ========================================================

    predictions = []


    # ========================================================
    # PROCESS DETECTED HANDS
    # ========================================================

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            # -----------------------------------------------
            # DRAW SKELETON
            # -----------------------------------------------

            draw_skeleton(
                frame,
                hand_landmarks
            )


            # -----------------------------------------------
            # 42 FEATURES
            # -----------------------------------------------

            features = preprocess_landmarks(
                hand_landmarks
            )


            # -----------------------------------------------
            # MODEL INPUT
            # -----------------------------------------------

            model_input = features.reshape(
                1,
                42
            )


            # -----------------------------------------------
            # PREDICTION
            # -----------------------------------------------

            try:

                output = model(
                    model_input,
                    training=False
                ).numpy()[0]


                # Convert logits to probability if needed
                if not np.isclose(
                    np.sum(output),
                    1.0,
                    atol=0.05
                ):

                    output = tf.nn.softmax(
                        output
                    ).numpy()


                class_index = int(
                    np.argmax(output)
                )


                confidence = float(
                    output[class_index]
                )


                if (
                    class_index
                    <
                    len(LABELS)
                ):

                    prediction = LABELS[
                        class_index
                    ]

                else:

                    prediction = "?"


                # Store prediction
                predictions.append({
                    "label": prediction,
                    "confidence": confidence
                })


            except Exception as e:

                print(
                    "Prediction error:",
                    e
                )


    # ========================================================
    # FINAL SINGLE PREDICTION
    # ========================================================

    final_label = "-"
    final_confidence = 0.0


    if predictions:

        # Choose the highest-confidence hand
        best_prediction = max(
            predictions,
            key=lambda x: x["confidence"]
        )

        final_label = best_prediction[
            "label"
        ]

        final_confidence = best_prediction[
            "confidence"
        ]


    # ========================================================
    # SHOW ONLY ONE ANSWER
    # ========================================================

    if final_confidence >= CONFIDENCE_THRESHOLD:

        # Large prediction only
        cv2.putText(
            frame,
            final_label,
            (
                50,
                100
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 0),
            6
        )


        cv2.putText(
            frame,
            f"{final_confidence * 100:.1f}%",
            (
                55,
                150
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Stopped.")