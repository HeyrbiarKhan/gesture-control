import cv2
import numpy as np
import time
import platform
import math

import mediapipe as mp
import pyautogui

IS_WINDOWS = platform.system().lower() == "windows"
if IS_WINDOWS:
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

CAM_INDEX = 0
SMOOTHING = 0.25
PINCH_CLICK_THRESH = 30
CLICK_COOLDOWN = 0.2
SHOW_OVERLAY = True

VOL_MIN_DIST = 20
VOL_MAX_DIST = 200

screen_w, screen_h = pyautogui.size()
prev_mouse_x, prev_mouse_y = pyautogui.position()

if IS_WINDOWS:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_min, vol_max, _ = volume.GetVolumeRange()

def set_volume_percent(pct: float):
    pct = max(0, min(100, pct))
    if IS_WINDOWS:
        target = vol_min + (pct / 100.0) * (vol_max - vol_min)
        volume.SetMasterVolumeLevel(target, None)

def fingers_up(landmarks, handedness):
    TIP = [4, 8, 12, 16, 20]
    PIP = [3, 6, 10, 14, 18]

    up = [False]*5
    if handedness == "Right":
        up[0] = landmarks[TIP[0]].x < landmarks[PIP[0]].x
    else:
        up[0] = landmarks[TIP[0]].x > landmarks[PIP[0]].x

    for i in range(1, 5):
        up[i] = landmarks[TIP[i]].y < landmarks[PIP[i]].y

    return up

def lerp(a, b, t):
    return a + (b - a) * t

def stabilize_hand_classification(hand_landmarks_list, handedness_list):
    if len(hand_landmarks_list) != 2:
        return handedness_list
    
    wrist_positions = []
    for landmarks in hand_landmarks_list:
        wrist_x = landmarks.landmark[0].x
        wrist_positions.append(wrist_x)
    
    if wrist_positions[0] < wrist_positions[1]:
        hand_0_label = "Right"
        hand_1_label = "Left"
    else:
        hand_0_label = "Left"
        hand_1_label = "Right"
    
    corrected_handedness = []
    corrected_handedness.append(type('obj', (object,), {'classification': [type('obj', (object,), {'label': hand_0_label})]})())
    corrected_handedness.append(type('obj', (object,), {'classification': [type('obj', (object,), {'label': hand_1_label})]})())
    
    return corrected_handedness

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

last_click_time = 0
mode_text = "Hand tracking mode (Right hand: pointer control, Left hand: volume control)"
current_volume_pct = None

hand_history = {"Left": [], "Right": []}
HISTORY_SIZE = 5
HAND_SWITCH_THRESHOLD = 0.3

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hand_results = hands.process(rgb)

        mode_text = "Hand tracking mode (Right hand: pointer, Left hand: volume)"

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            
            hand_landmarks_list = hand_results.multi_hand_landmarks
            handedness_list = hand_results.multi_handedness
            
            for hand_landmarks, handedness_info in zip(hand_landmarks_list, handedness_list):
                handedness_classification = handedness_info.classification[0]
                handedness = handedness_classification.label
                confidence = handedness_classification.score
                
                if SHOW_OVERLAY:
                    lm = hand_landmarks.landmark
                    debug_pos = (int(lm[0].x * w), int(lm[0].y * h) + 20)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", debug_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                lm = hand_landmarks.landmark
                
                up = fingers_up(lm, handedness)
                thumb_up, index_up = up[0], up[1]
                
                def to_xy(idx):
                    return int(lm[idx].x * w), int(lm[idx].y * h)
                x4, y4 = to_xy(4)
                x8, y8 = to_xy(8)
                
                thumb_index_dist = math.hypot(x8 - x4, y8 - y4)
                
                if handedness == "Right":
                    other_fingers_down = (not thumb_up) and (not up[2]) and (not up[3]) and (not up[4])
                    
                    if index_up and other_fingers_down:
                        mode_text = "Right hand: Cursor mode"
                        target_x = np.interp(x8, [0, w], [0, screen_w])
                        target_y = np.interp(y8, [0, h], [0, screen_h])

                        mouse_x = lerp(prev_mouse_x, target_x, 1.0 - SMOOTHING)
                        mouse_y = lerp(prev_mouse_y, target_y, 1.0 - SMOOTHING)
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0)
                        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                    if thumb_up and index_up:
                        now = time.time()
                        if thumb_index_dist < PINCH_CLICK_THRESH and (now - last_click_time) > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = now
                            if SHOW_OVERLAY:
                                cv2.putText(frame, "CLICK!", (x8+10, y8-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if SHOW_OVERLAY:
                            cv2.line(frame, (x4, y4), (x8, y8), (255, 255, 0), 2)
                            mid_x, mid_y = (x4 + x8) // 2, (y4 + y8) // 2
                            cv2.putText(frame, f"Dist: {int(thumb_index_dist)}", (mid_x, mid_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                elif handedness == "Left":
                    mode_text = "Left hand: Volume mode"
                    
                    d = np.clip(thumb_index_dist, VOL_MIN_DIST, VOL_MAX_DIST)
                    pct = np.interp(d, [VOL_MIN_DIST, VOL_MAX_DIST], [0, 100]).astype(float)
                    current_volume_pct = pct
                    set_volume_percent(pct)

                    if SHOW_OVERLAY:
                        cv2.putText(frame, f"VOL {int(pct)}%", (30, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        bar_h = int(np.interp(pct, [0, 100], [0, 200]))
                        cv2.rectangle(frame, (30, 140), (60, 340), (0, 255, 0), 2)
                        cv2.rectangle(frame, (30, 340-bar_h), (60, 340), (0, 255, 0), -1)
                        
                        cv2.line(frame, (x4, y4), (x8, y8), (0, 255, 255), 2)
                        mid_x, mid_y = (x4 + x8) // 2, (y4 + y8) // 2
                        cv2.putText(frame, f"Dist: {int(thumb_index_dist)}", (mid_x, mid_y-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if SHOW_OVERLAY:
                    connection_color = (255, 0, 0) if handedness == "Right" else (0, 255, 0)
                    landmark_color = (0, 0, 255) if handedness == "Right" else (0, 255, 0)
                    
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=connection_color, thickness=2))
                    
                    hand_label = f"{handedness} Hand"
                    label_pos = (int(lm[0].x * w), int(lm[0].y * h) - 20)
                    cv2.putText(frame, hand_label, label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, connection_color, 2)

        if SHOW_OVERLAY:
            cv2.putText(frame, mode_text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "Right hand: Point with index only, or thumb+index to pinch-click", 
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Left hand: Thumb+index distance for volume (always active)", 
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            if SHOW_OVERLAY:
                cv2.putText(frame, "Show your hands to the camera",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Dual Hand Gesture Control (Esc to quit)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
            
        if cv2.getWindowProperty("Dual Hand Gesture Control (Esc to quit)", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
