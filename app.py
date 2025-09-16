from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings


# Suppress warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")


# Flask + SocketIO setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)
    model = None


# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


def generate_frames():
    cap = cv2.VideoCapture(0)
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv2.flip(frame, 1)  # Flip horizontally
        H, W, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = hands.process(rgb_frame)


        if not results.multi_hand_landmarks:
            # No hand detected
            socketio.emit('prediction', {'text': 'None', 'confidence': 0})
        else:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )


                # Normalize landmarks like training
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_list))
                    data_aux.append(lm.y - min(y_list))


                # Predict
                try:
                    pred = model.predict([np.array(data_aux)])
                    pred_proba = model.predict_proba([np.array(data_aux)])
                    confidence = max(pred_proba[0])
                    predicted_char = labels_dict[int(pred[0])]


                    # Emit to frontend
                    socketio.emit('prediction', {'text': predicted_char, 'confidence': confidence})


                    # Draw bounding box
                    x1, y1 = int(min(x_list)*W)-10, int(min(y_list)*H)-10
                    x2, y2 = int(max(x_list)*W)+10, int(max(y_list)*H)+10
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 3)
                    cv2.putText(frame, f"{predicted_char} ({confidence*100:.1f}%)", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
                except Exception as e:
                    print("Prediction error:", e)


        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=True)
