import cv2
import mediapipe as mp
# Added imports for voice feedback
import pyttsx3
import threading

def main():
    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize video capture.
    cap = cv2.VideoCapture(0)

    # Initialize TTS engine (offline).
    engine = pyttsx3.init()

    # Tracking the last spoken values to avoid repetition
    last_sign = None
    last_gesture = None

    # Define a function to speak text (runs in background thread).
    def speak_text(text):
        engine.say(text)
        engine.runAndWait()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert color for MediaPipe.
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Perform hand detection.
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        sign = None
        gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame.
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # *** Placeholder for sign and gesture detection logic ***
                # Here you would compute `sign` and `gesture` from landmarks.
                # For example:
                # sign = detect_sign(hand_landmarks)
                # gesture = detect_finger_gesture(hand_landmarks)
                
                # For demonstration, we use dummy values:
                sign = "Hello"         # Replace with actual sign result
                gesture = "Thumbs Up"  # Replace with actual gesture result
                break  # Only process first hand for this example

        else:
            # No hand detected: you can reset tracking or skip speaking.
            last_sign = None
            last_gesture = None
            # Show the frame without overlay and continue.
            cv2.imshow('Sign Language Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Check if sign or gesture has changed.
        if (sign != last_sign) or (gesture != last_gesture):
            # Only speak if both sign and gesture are not None.
            if sign is not None and gesture is not None:
                message = f"Detected hand sign: {sign}, Finger gesture: {gesture}"
                # Start a background thread for speaking.
                threading.Thread(target=speak_text, args=(message,), daemon=True).start()
                last_sign = sign
                last_gesture = gesture

        # Display the resulting frame.
        cv2.putText(frame, f"Sign: {sign}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Sign Language Detection', frame)

        # Exit on 'q' key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
