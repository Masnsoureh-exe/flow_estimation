from pathlib import Path
import cv2
import pandas as pd

videos= Path("../data/dataset")

rows= []

for video_path in videos.glob("*"):

    if video_path.suffix.lower() not in [".avi",".mp4"]:
        continue

    cap= cv2.VideoCapture(str(video_path))

    fps= cap.get(cv2.CAP_PROP_FPS)
    frame_count= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration= frame_count / fps if fps else 0

    rows.append({
        "file": video_path.name,
        "fps": fps,
        "frames": frame_count,
        "duration": duration,
        "width": width,
        "height": height
    })
    cap.release()

df= pd.DataFrame(rows)
print(df)
df.to_csv("video_statistics.csv", index=False)
