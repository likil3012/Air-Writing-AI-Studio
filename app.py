import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import threading

st.set_page_config(
    page_title="Air Writing AI Studio",
    page_icon="🖐️",
    layout="wide"
)

st.title("🖐️ Air Writing AI Studio")
st.write("Use your RIGHT index finger to write in the air.")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

canvas = None
previous_point = None
lock = threading.Lock()


class AirWriting(VideoTransformerBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def transform(self, frame):

        global canvas, previous_point

        img = frame.to_ndarray(format="bgr24")

        if canvas is None:
            canvas = np.zeros_like(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = img.shape

                index_tip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]

                index_pip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_PIP
                ]

                # Index finger pointing upward
                if index_tip.y < index_pip.y:

                    x = int(index_tip.x * w)
                    y = int(index_tip.y * h)

                    if previous_point is not None:

                        cv2.line(
                            canvas,
                            previous_point,
                            (x, y),
                            (0, 255, 0),
                            8
                        )

                    previous_point = (x, y)

                    cv2.circle(
                        img,
                        (x, y),
                        10,
                        (0, 255, 0),
                        -1
                    )

                else:
                    previous_point = None

        else:
            previous_point = None

        with lock:
            output = cv2.addWeighted(
                img,
                1,
                canvas,
                1,
                0
            )

        return output


if st.button("🗑️ Clear Canvas"):

    with lock:
        canvas = None

    previous_point = None

st.info("☝️ Raise your RIGHT index finger and move it to write.")

webrtc_streamer(
    key="air-writing",
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    video_transformer_factory=AirWriting,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
