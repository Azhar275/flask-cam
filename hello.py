from flask import Flask, Response, render_template
from ultralytics import YOLO
import supervision as sv
import numpy as np

import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

ZONE_POLYGON = np.array([
    [0, 344],[1358, 344],[1358, 488],[0, 488],[0, 344]
])

def gen_frames():
    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    # zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple([1280, 720]))

    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)

            # To determine what object to detect
            detections = detections[detections.class_id == 0]
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections
            ]
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels
            )
            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return 'Index Page'

@app.route('/world')
def world():
    return 'Hello, World'

@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')