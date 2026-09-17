import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import pytesseract


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    print("WARNING: Tesseract not found.")
    print("OCR will be disabled.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera started successfully.")


# ============================================================
# SETTINGS
# ============================================================

WINDOW_NAME = "AIR WRITING AI STUDIO"

pen_colors = [
    (0, 255, 0),       # Green
    (255, 0, 0),       # Blue
    (0, 0, 255),       # Red
    (255, 255, 0),     # Cyan
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Yellow
    (255, 255, 255)    # White
]

color_names = [
    "GREEN",
    "BLUE",
    "RED",
    "CYAN",
    "MAGENTA",
    "YELLOW",
    "WHITE"
]

color_index = 0
pen_color = pen_colors[color_index]

pen_size = 8
eraser_size = 35

drawing = False
eraser_mode = False

current_stroke = []
strokes = []

mode = "WRITE"

last_point = None

# Cooldowns
last_clear_time = 0
last_undo_time = 0
last_color_time = 0
last_size_time = 0
last_save_time = 0
last_symbol_time = 0

COOLDOWN = 0.7


# ============================================================
# SYMBOLS
# ============================================================

symbols = [
    "+",
    "-",
    "x",
    "/",
    "=",
    "->",
    "<-",
    "@",
    "#",
    "%",
    "*",
    "sqrt"
]

symbol_index = 0
selected_symbol = symbols[symbol_index]


# ============================================================
# SHAPE MODE
# ============================================================

shape_points = []


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def finger_states(lm):
    """
    Returns:
    index, middle, ring, pinky
    """

    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    return index, middle, ring, pinky


def is_pinch(lm):
    return distance(lm[4], lm[8]) < 0.055


def is_open_palm(lm):

    index, middle, ring, pinky = finger_states(lm)

    return (
        index and
        middle and
        ring and
        pinky
    )


def is_index_only(lm):

    index, middle, ring, pinky = finger_states(lm)

    return (
        index and
        not middle and
        not ring and
        not pinky
    )


def is_two_fingers(lm):

    index, middle, ring, pinky = finger_states(lm)

    return (
        index and
        middle and
        not ring and
        not pinky
    )


def is_fist(lm):

    index, middle, ring, pinky = finger_states(lm)

    return (
        not index and
        not middle and
        not ring and
        not pinky
    )


def is_thumb_up(lm):

    index, middle, ring, pinky = finger_states(lm)

    other_fingers_down = (
        not index and
        not middle and
        not ring and
        not pinky
    )

    thumb_up = (
        lm[4].y < lm[3].y and
        lm[4].y < lm[2].y
    )

    return other_fingers_down and thumb_up


def is_thumb_down(lm):

    index, middle, ring, pinky = finger_states(lm)

    other_fingers_down = (
        not index and
        not middle and
        not ring and
        not pinky
    )

    thumb_down = (
        lm[4].y > lm[3].y and
        lm[4].y > lm[2].y
    )

    return other_fingers_down and thumb_down


# ============================================================
# ACTION COOLDOWN
# ============================================================

def ready(last_time):
    return time.time() - last_time > COOLDOWN


# ============================================================
# UNDO
# ============================================================

def undo():

    global strokes

    if len(strokes) > 0:
        strokes.pop()


# ============================================================
# CLEAR
# ============================================================

def clear_all():

    global strokes
    global current_stroke
    global shape_points
    global last_point

    strokes.clear()
    current_stroke.clear()
    shape_points.clear()
    last_point = None


# ============================================================
# COLOR
# ============================================================

def next_color():

    global color_index
    global pen_color

    color_index += 1

    if color_index >= len(pen_colors):
        color_index = 0

    pen_color = pen_colors[color_index]


# ============================================================
# PEN SIZE
# ============================================================

def increase_size():

    global pen_size

    pen_size += 2

    if pen_size > 30:
        pen_size = 30


def decrease_size():

    global pen_size

    pen_size -= 2

    if pen_size < 2:
        pen_size = 2


# ============================================================
# DRAW SAVED STROKES
# ============================================================

def draw_strokes(canvas):

    for stroke in strokes:

        if len(stroke["points"]) < 2:
            continue

        points = stroke["points"]
        color = stroke["color"]
        size = stroke["size"]
        eraser = stroke["eraser"]

        for i in range(1, len(points)):

            p1 = points[i - 1]
            p2 = points[i]

            if eraser:

                cv2.line(
                    canvas,
                    p1,
                    p2,
                    (0, 0, 0),
                    eraser_size
                )

            else:

                cv2.line(
                    canvas,
                    p1,
                    p2,
                    color,
                    size,
                    cv2.LINE_AA
                )


# ============================================================
# SHAPE DETECTION
# ============================================================

def recognize_shape(points):

    if len(points) < 10:
        return None

    pts = np.array(points)

    x, y, w, h = cv2.boundingRect(pts)

    if w < 30 or h < 30:
        return None

    # Close the shape
    start = np.array(points[0])
    end = np.array(points[-1])

    closing_distance = np.linalg.norm(start - end)

    # Approximation
    epsilon = 0.04 * cv2.arcLength(
        pts.reshape((-1, 1, 2)),
        False
    )

    approx = cv2.approxPolyDP(
        pts.reshape((-1, 1, 2)),
        epsilon,
        False
    )

    number_of_corners = len(approx)

    aspect = w / float(h)

    if number_of_corners == 3:
        return "TRIANGLE"

    if number_of_corners == 4:

        if 0.8 <= aspect <= 1.2:
            return "SQUARE"

        return "RECTANGLE"

    if number_of_corners >= 5:

        if closing_distance < max(w, h) * 0.25:
            return "CIRCLE"

    return None


# ============================================================
# OCR
# ============================================================

def recognize_text(frame):

    if not os.path.exists(TESSERACT_PATH):
        return "TESSERACT NOT FOUND"

    try:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        text = pytesseract.image_to_string(
            threshold,
            config="--psm 6"
        )

        text = text.strip()

        if text == "":
            return "NO TEXT FOUND"

        return text

    except Exception as e:

        return "OCR ERROR"


# ============================================================
# SAVE
# ============================================================

def save_image(frame):

    filename = (
        "air_writing_"
        + time.strftime("%Y%m%d_%H%M%S")
        + ".png"
    )

    cv2.imwrite(filename, frame)

    print("Saved:", filename)

    return filename


# ============================================================
# DRAW SYMBOL
# ============================================================

def draw_symbol(canvas, symbol, position):

    x, y = position

    # OpenCV cannot display many Unicode symbols,
    # therefore ASCII symbols are used.

    cv2.putText(
        canvas,
        symbol,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        pen_color,
        3,
        cv2.LINE_AA
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Frame not received.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    # --------------------------------------------------------
    # Identify hands
    # --------------------------------------------------------

    right_hand = None
    left_hand = None

    if results.multi_hand_landmarks:

        for i, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            label = (
                results
                .multi_handedness[i]
                .classification[0]
                .label
            )

            if label == "Right":
                right_hand = hand_landmarks

            elif label == "Left":
                left_hand = hand_landmarks

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # ========================================================
    # LEFT HAND CONTROL
    # ========================================================

    if left_hand is not None:

        lm = left_hand.landmark

        # LEFT PINCH = CHANGE COLOR
        if is_pinch(lm):

            if ready(last_color_time):

                next_color()
                last_color_time = time.time()

        # LEFT TWO FINGERS = ERASER
        elif is_two_fingers(lm):

            eraser_mode = True
            mode = "ERASER"

        # LEFT INDEX = SYMBOL SELECT
        elif is_index_only(lm):

            if ready(last_symbol_time):

                symbol_index += 1

                if symbol_index >= len(symbols):
                    symbol_index = 0

                selected_symbol = symbols[symbol_index]

                last_symbol_time = time.time()

        # LEFT THUMB UP = BIGGER
        elif is_thumb_up(lm):

            if ready(last_size_time):

                increase_size()
                last_size_time = time.time()

        # LEFT THUMB DOWN = SMALLER
        elif is_thumb_down(lm):

            if ready(last_size_time):

                decrease_size()
                last_size_time = time.time()

        # LEFT OPEN PALM = NORMAL WRITE
        elif is_open_palm(lm):

            eraser_mode = False

            if mode == "ERASER":
                mode = "WRITE"

    # ========================================================
    # RIGHT HAND CONTROL
    # ========================================================

    if right_hand is not None:

        lm = right_hand.landmark

        # --------------------------------------------
        # RIGHT FIST = CLEAR
        # --------------------------------------------

        if is_fist(lm):

            drawing = False
            last_point = None

            if ready(last_clear_time):

                clear_all()
                last_clear_time = time.time()

            mode = "CLEAR"

        # --------------------------------------------
        # RIGHT PINCH = PAUSE
        # --------------------------------------------

        elif is_pinch(lm):

            drawing = False
            last_point = None

            mode = "PAUSE"

        # --------------------------------------------
        # RIGHT TWO FINGERS = SHAPE MODE
        # --------------------------------------------

        elif is_two_fingers(lm):

            drawing = False
            last_point = None

            mode = "SHAPE"

        # --------------------------------------------
        # RIGHT INDEX = WRITE
        # --------------------------------------------

        elif is_index_only(lm):

            mode = "ERASER" if eraser_mode else "WRITE"

            # Index fingertip
            x = int(lm[8].x * width)
            y = int(lm[8].y * height)

            # ----------------------------------------
            # NORMAL WRITING
            # ----------------------------------------

            if not eraser_mode:

                if last_point is None:

                    current_stroke = [(x, y)]

                    strokes.append({
                        "points": current_stroke,
                        "color": pen_color,
                        "size": pen_size,
                        "eraser": False
                    })

                else:

                    strokes[-1]["points"].append(
                        (x, y)
                    )

                last_point = (x, y)

            # ----------------------------------------
            # ERASER
            # ----------------------------------------

            else:

                cv2.circle(
                    frame,
                    (x, y),
                    eraser_size,
                    (255, 255, 255),
                    2
                )

                # Remove points near eraser
                new_strokes = []

                for stroke in strokes:

                    remaining = []

                    for p in stroke["points"]:

                        d = math.sqrt(
                            (p[0] - x) ** 2 +
                            (p[1] - y) ** 2
                        )

                        if d > eraser_size:
                            remaining.append(p)

                    if len(remaining) > 1:

                        stroke["points"] = remaining
                        new_strokes.append(stroke)

                strokes = new_strokes

        # --------------------------------------------
        # RIGHT THUMB = UNDO
        # --------------------------------------------

        elif is_thumb_up(lm):

            drawing = False
            last_point = None

            if ready(last_undo_time):

                undo()
                last_undo_time = time.time()

            mode = "UNDO"

        else:

            drawing = False
            last_point = None

    else:

        drawing = False
        last_point = None


    # ========================================================
    # DRAW ALL STROKES
    # ========================================================

    drawing_layer = np.zeros_like(frame)

    draw_strokes(drawing_layer)

    # Glow effect
    glow = cv2.GaussianBlur(
        drawing_layer,
        (0, 0),
        12
    )

    # Combine
    frame = cv2.addWeighted(
        frame,
        1.0,
        glow,
        0.35,
        0
    )

    frame = cv2.addWeighted(
        frame,
        1.0,
        drawing_layer,
        1.0,
        0
    )


    # ========================================================
    # TOP INFORMATION BAR
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 115),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "AIR WRITING AI STUDIO",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "MODE: " + mode,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        pen_color,
        2
    )

    cv2.putText(
        frame,
        "COLOR: " + color_names[color_index],
        (250, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        pen_color,
        2
    )

    cv2.putText(
        frame,
        "SIZE: " + str(pen_size),
        (500, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SYMBOL: " + selected_symbol,
        (650, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # BOTTOM CONTROLS
    # ========================================================

    controls = [
        "INDEX: WRITE",
        "PINCH: PAUSE",
        "FIST: CLEAR",
        "THUMB: UNDO",
        "L-PINCH: COLOR",
        "L-2F: ERASER",
        "L-INDEX: SYMBOL",
        "L-THUMB: SIZE"
    ]

    y = height - 65

    cv2.putText(
        frame,
        " | ".join(controls[:4]),
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "L-PINCH COLOR | L-2F ERASER | L-INDEX SYMBOL | "
        "S SAVE | R OCR | C CLEAR | U UNDO | Q EXIT",
        (15, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SYMBOL PREVIEW
    # ========================================================

    cv2.rectangle(
        frame,
        (width - 150, 125),
        (width - 15, 210),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "SYMBOL",
        (width - 135, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        selected_symbol,
        (width - 105, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        pen_color,
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # KEYBOARD BACKUP CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):
        break

    # C = clear
    elif key == ord("c"):

        clear_all()
        mode = "CLEAR"

    # U = undo
    elif key == ord("u"):

        undo()
        mode = "UNDO"

    # S = save
    elif key == ord("s"):

        save_image(frame)

    # R = OCR
    elif key == ord("r"):

        print("\nRecognizing text...")

        result = recognize_text(
            drawing_layer
        )

        print("OCR RESULT:")
        print(result)

        cv2.putText(
            frame,
            "OCR: " + result[:50],
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        cv2.waitKey(1500)

    # COLOR
    elif key == ord("1"):

        color_index = 0
        pen_color = pen_colors[color_index]

    elif key == ord("2"):

        color_index = 1
        pen_color = pen_colors[color_index]

    elif key == ord("3"):

        color_index = 2
        pen_color = pen_colors[color_index]

    elif key == ord("4"):

        color_index = 3
        pen_color = pen_colors[color_index]

    elif key == ord("5"):

        color_index = 4
        pen_color = pen_colors[color_index]

    elif key == ord("6"):

        color_index = 5
        pen_color = pen_colors[color_index]

    elif key == ord("7"):

        color_index = 6
        pen_color = pen_colors[color_index]

    # PEN SIZE
    elif key == ord("+") or key == ord("="):

        increase_size()

    elif key == ord("-"):

        decrease_size()


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Air Writing Studio closed.")