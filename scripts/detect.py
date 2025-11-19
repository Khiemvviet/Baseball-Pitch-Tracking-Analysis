import os
import cv2
import pandas as pd
import numpy as np
import argparse
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to input video")
parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
parser.add_argument("--out", default="output/detect_video_result.mp4", help="Output annotated video path")
parser.add_argument("--pixels_per_meter", type=float, default=120, help="Pixels per meter scale (adjust for correct speed)")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)


model = YOLO(args.model)
cap = cv2.VideoCapture(args.video)

fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))


trajectory = []
velocity = 0
zone_lock = None
data = []  


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    ball_box = None
    detected_class = None

    for box in results.boxes:
        cls_index = int(box.cls)
        cls_name = results.names[cls_index]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if "ball" in cls_name.lower():
            ball_box = (x1, y1, x2, y2)
            detected_class = cls_name

        if "zone" in cls_name.lower():
            zone_lock = (x1, y1, x2, y2)
            zone_class = cls_name

    if zone_lock:
        zx1, zy1, zx2, zy2 = zone_lock
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 255, 255), 3)
        if zone_class:
            cv2.putText(frame, zone_class, (zx1, zy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    if ball_box and zone_lock:
        cx = (ball_box[0] + ball_box[2]) // 2
        cy = (ball_box[1] + ball_box[3]) // 2

        zx1, zy1, zx2, zy2 = zone_lock

       
        inside = zx1 < cx < zx2 and zy1 < cy < zy2
        label = "Inside" if inside else "Outside"
        label_color = (255, 255, 0) if inside else (0, 255, 0)

        trajectory.append((cx, cy))

        speed_mph = velocity
        if len(trajectory) > 1:
            px, py = trajectory[-2]
            pixel_dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            meter = pixel_dist / args.pixels_per_meter
            m_per_s = meter * fps
            speed = m_per_s * 2.23694
            velocity = 0.7 * velocity + 0.3 * speed
            speed_mph = velocity

      
       
        cv2.circle(frame, (cx, cy), 10, label_color, 2)

        cv2.putText(frame, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{speed_mph:.1f} mph", (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], label_color, 6)

        data.append ({
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "cx": cx,
            "cy": cy,
            "Class": detected_class,
            "Location": label,
            "Speed_mph": speed_mph
        })

    out.write(frame)
    # cv2.imshow("Pitch Analysis", frame)
    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

os.makedirs("output", exist_ok=True)
if data:
    df = pd.DataFrame([data])
    df.to_csv("output/Pitch.csv", index=False)
    print("Pitch data saved to output/Pitch.csv")
else:
    print("No ball detected in the video.")
