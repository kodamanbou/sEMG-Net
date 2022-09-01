import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def capture_frame():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print('Error empty frame.')
                continue

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.imwrite('frame.png', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


def capture_semg():
    print()
