import av
import cv2
import math
import threading
import mediapipe as mp
import streamlit as st

from streamlit_webrtc import webrtc_streamer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Virtual Keyboard",
    page_icon="⌨️",
    layout="wide"
)

st.title("⌨️ Virtual Keyboard")
st.write("Use your index finger to select a key and pinch to press it.")


# =========================================================
# KEYBOARD
# =========================================================

keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["SPACE", "BACKSPACE", "ENTER"]
]

key_width = 55
key_height = 55
key_gap = 8
start_y = 200


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class VideoProcessor:

    def __init__(self):

        self.lock = threading.Lock()

        self.typed_text = ""

        self.pinch_was_active = False

        self.landmarker = None

        self.create_landmarker()


    # -----------------------------------------------------
    # CREATE MEDIAPIPE HAND LANDMARKER
    # -----------------------------------------------------

    def create_landmarker(self):

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="hand_landmarker.task"
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.landmarker = HandLandmarker.create_from_options(
            options
        )


    # =====================================================
    # DRAW KEYBOARD
    # =====================================================

    def draw_keyboard(self, frame, selected_key=None):

        key_positions = []

        frame_width = frame.shape[1]

        for row_index, row in enumerate(keys):

            widths = []

            for key in row:

                if key == "SPACE":
                    widths.append(150)

                elif key == "BACKSPACE":
                    widths.append(120)

                elif key == "ENTER":
                    widths.append(90)

                else:
                    widths.append(key_width)

            row_width = (
                sum(widths)
                + (len(row) - 1) * key_gap
            )

            row_x = (frame_width - row_width) // 2

            current_x = row_x

            for i, key in enumerate(row):

                current_width = widths[i]

                x1 = current_x
                y1 = start_y + row_index * (
                    key_height + key_gap
                )

                x2 = x1 + current_width
                y2 = y1 + key_height


                # Highlight selected key
                if key == selected_key:

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        -1
                    )

                    text_color = (0, 0, 0)

                else:

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

                    text_color = (255, 255, 255)


                # Key text
                font_scale = 0.55

                text_size = cv2.getTextSize(
                    key,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    2
                )[0]

                text_x = (
                    x1
                    + (current_width - text_size[0]) // 2
                )

                text_y = (
                    y1
                    + (key_height + text_size[1]) // 2
                )

                cv2.putText(
                    frame,
                    key,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    2
                )


                key_positions.append(
                    (
                        key,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                current_x += (
                    current_width + key_gap
                )

        return key_positions


    # =====================================================
    # FIND KEY
    # =====================================================

    def get_key_at_position(
        self,
        finger_x,
        finger_y,
        key_positions
    ):

        for key, x1, y1, x2, y2 in key_positions:

            if (
                x1 <= finger_x <= x2
                and
                y1 <= finger_y <= y2
            ):

                return key

        return None


    # =====================================================
    # PINCH DETECTION
    # =====================================================

    def is_pinching(self, hand):

        thumb_tip = hand[4]

        index_tip = hand[8]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2
            +
            (thumb_tip.y - index_tip.y) ** 2
            +
            (thumb_tip.z - index_tip.z) ** 2
        )

        return distance < 0.06


    # =====================================================
    # PROCESS KEY
    # =====================================================

    def process_key(self, key):

        if key == "SPACE":

            self.typed_text += " "


        elif key == "BACKSPACE":

            self.typed_text = (
                self.typed_text[:-1]
            )


        elif key == "ENTER":

            self.typed_text += "\n"


        else:

            self.typed_text += key


    # =====================================================
    # PROCESS VIDEO FRAME
    # =====================================================

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera
        img = cv2.flip(img, 1)


        # -------------------------------------------------
        # MEDIAPIPE INPUT
        # -------------------------------------------------

        rgb_img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_img
        )


        # -------------------------------------------------
        # HAND DETECTION
        # -------------------------------------------------

        result = self.landmarker.detect(
            mp_image
        )


        selected_key = None

        pinching = False


        # -------------------------------------------------
        # HAND FOUND
        # -------------------------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # Index fingertip
            index_tip = hand[8]


            finger_x = int(
                index_tip.x * img.shape[1]
            )

            finger_y = int(
                index_tip.y * img.shape[0]
            )


            # -------------------------------------------------
            # TEMPORARY KEY POSITIONS
            # -------------------------------------------------

            key_positions = self.draw_keyboard(
                img
            )


            selected_key = (
                self.get_key_at_position(
                    finger_x,
                    finger_y,
                    key_positions
                )
            )


            # -------------------------------------------------
            # PINCH
            # -------------------------------------------------

            pinching = self.is_pinching(
                hand
            )


            # -------------------------------------------------
            # DRAW FINGERTIP
            # -------------------------------------------------

            cv2.circle(
                img,
                (finger_x, finger_y),
                12,
                (0, 255, 0),
                -1
            )


            # -------------------------------------------------
            # PRESS KEY
            # -------------------------------------------------

            if pinching:

                cv2.putText(
                    img,
                    "PINCH",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                # Only press once per pinch
                if (
                    not self.pinch_was_active
                    and selected_key is not None
                ):

                    with self.lock:

                        self.process_key(
                            selected_key
                        )

                    self.pinch_was_active = True


            else:

                self.pinch_was_active = False


        else:

            cv2.putText(
                img,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            self.pinch_was_active = False


        # =================================================
        # DRAW FINAL KEYBOARD
        # =================================================

        self.draw_keyboard(
            img,
            selected_key
        )


        # =================================================
        # DISPLAY TEXT
        # =================================================

        with self.lock:

            text = self.typed_text


        cv2.rectangle(
            img,
            (20, 70),
            (img.shape[1] - 20, 135),
            (0, 0, 0),
            -1
        )


        # Show last characters
        display_text = text[-50:]


        cv2.putText(
            img,
            "Text: " + display_text,
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # =================================================
        # RETURN FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================================================
# STREAMLIT WEBRTC
# =========================================================

ctx = webrtc_streamer(
    key="virtual-keyboard",

    video_processor_factory=VideoProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)


# =========================================================
# INSTRUCTIONS
# =========================================================

st.markdown("---")

st.subheader("How to use")

st.write(
    """
    1. Click **START**.
    2. Allow camera permission.
    3. Show your hand to the camera.
    4. Move your index finger over a key.
    5. Bring your thumb and index finger together.
    6. Release the pinch.
    """
)

st.info(
    "The green fingertip shows the position of your index finger."
)