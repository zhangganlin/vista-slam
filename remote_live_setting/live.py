import cv2
from flask import Flask, Response

app = Flask(__name__)
cap = cv2.VideoCapture(1)  # your Windows webcam

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

@app.route('/video')
def video():
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000)
