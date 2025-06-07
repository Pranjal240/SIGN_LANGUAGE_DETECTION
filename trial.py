# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import deque
import threading

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

import pyttsx3

# Allow numbers 1–36 (0 unused)
MAX_CLASSES = 37

# Debounce state with pointer gesture counters
data_state = {
    'hand_last': None,
    'hand_count': 0,
    'spoken_hand': None,
    'pointer_last': None,
    'pointer_last_detected': None,
    'pointer_count': 0
}

# Initialize single TTS engine and lock
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)
tts_lock = threading.Lock()

def speak_text(text):
    # Non-blocking with thread and lock to avoid run loop conflicts
    def _speak():
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960, help='cap width')
    parser.add_argument("--height", type=int, default=540, help='cap height')
    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def select_mode(key, mode):
    number = -1
    # case-insensitive check for n, k, h
    try:
        ch = chr(key).lower()
    except ValueError:
        ch = ''
    if ch == 'n':
        mode = 0
    elif ch == 'k':
        mode = 1
    elif ch == 'h':
        mode = 2
    else:
        if 48 <= key <= 57:         # '0'–'9'
            number = key - 48 + 1
        elif 65 <= key <= 90 or 97 <= key <= 122:  # 'A'–'Z' or 'a'–'z'
            base = 65 if key <= 90 else 97
            number = 11 + (key - base)
    return number, mode

def calc_bounding_rect(image, landmarks):
    ih, iw = image.shape[:2]
    pts = np.array([[min(int(lm.x * iw), iw - 1), min(int(lm.y * ih), ih - 1)]
                    for lm in landmarks.landmark])
    x, y, w, h = cv.boundingRect(pts)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    ih, iw = image.shape[:2]
    return [[min(int(lm.x * iw), iw - 1), min(int(lm.y * ih), ih - 1)]
            for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] -= base_x; p[1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_v = max(map(abs, flat)) or 1
    return [v / max_v for v in flat]

def pre_process_point_history(image, point_history):
    ih, iw = image.shape[:2]
    temp = copy.deepcopy(point_history)
    if not temp:
        return []
    bx, by = temp[0]
    for p in temp:
        p[0] = (p[0] - bx) / iw; p[1] = (p[1] - by) / ih
    return list(itertools.chain.from_iterable(temp))

def draw_landmarks(image, lp, color):
    for start, end in mp.solutions.hands.HAND_CONNECTIONS:
        cv.line(image, tuple(lp[start]), tuple(lp[end]), color, 2)
    for p in lp:
        cv.circle(image, tuple(p), 5, color, -1)
    return image

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 28), (255, 255, 255), -1)
    info = handedness.classification[0].label
    if hand_text:
        info += ':' + hand_text
    cv.putText(image, info, (brect[0] + 5, brect[1] - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return image

def draw_point_history(image, ph):
    for i, p in enumerate(ph):
        if p[0] and p[1]:
            cv.circle(image, tuple(p), 1 + i // 2, (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number, fg_text=None):
    cv.putText(image, f"FPS:{fps}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if fg_text:
        cv.putText(image, fg_text, (10, 68),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    modes = ['KeyPoint', 'PointHist']
    if 1 <= mode <= len(modes):
        cv.putText(image, f"MODE:{modes[mode-1]}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if 1 <= number < MAX_CLASSES:
        cv.putText(image, f"NUM:{number}", (10, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image

def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Prepare keypoint CSV file
    kp_file = open('Keypoints.csv', 'a', newline='')
    kp_writer = csv.writer(kp_file)

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        gesture_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)

    mode = 0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = image.copy()

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        fg_text = ''
        hand_text = ''

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                lm_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_lm = pre_process_landmark(lm_list)

                # Write keypoints when KeyPoint mode is active
                if mode == 1:
                    kp_writer.writerow(pre_lm)

                pre_ph = pre_process_point_history(debug_image, point_history)

                # Hand sign detection
                hand_id = keypoint_classifier(pre_lm)
                hand_text = keypoint_labels[hand_id]
                if hand_text == data_state['hand_last']:
                    data_state['hand_count'] += 1
                else:
                    data_state.update({'hand_last': hand_text, 'hand_count': 1})
                if data_state['hand_count'] >= 10 and hand_text != data_state['spoken_hand']:
                    data_state['spoken_hand'] = hand_text
                    msg = 'thank you' if hand_text.lower() in ['thanks', 'thankyou'] else hand_text
                    print(f"Detected hand sign: {msg}", flush=True)
                    # Speak only the gesture itself
                    speak_text(msg)

                # Point gesture history
                if hand_id == 2:
                    point_history.append(lm_list[8])
                else:
                    point_history.append([0, 0])

                # Pointer gesture detection
                if len(pre_ph) == history_length * 2:
                    ph_id = point_history_classifier(pre_ph)
                    fg_text = gesture_labels[ph_id]
                    if fg_text in ['Clockwise', 'Counter Clockwise']:
                        if fg_text == data_state['pointer_last_detected']:
                            data_state['pointer_count'] += 1
                        else:
                            data_state['pointer_last_detected'] = fg_text
                            data_state['pointer_count'] = 1
                        if data_state['pointer_count'] >= 3 and fg_text != data_state['pointer_last']:
                            data_state['pointer_last'] = fg_text
                            print(f"Pointer gesture: {fg_text}", flush=True)
                            speak_text(fg_text)

                # Draw on image
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_landmarks(debug_image, lm_list, (0, 0, 255)
                                             if handedness.classification[0].label == 'Right' else (255, 0, 0))
                debug_image = draw_info_text(debug_image, brect, handedness, hand_text)
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number, fg_text)

        cv.imshow('Hand Gesture Recognition', debug_image)

    kp_file.close()
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
