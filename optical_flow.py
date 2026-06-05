import cv2
import  numpy as np
from pathlib import Path
from preprocessing import preprocess_frame
import pandas as pd


def compute_flow(video_path, video_name):

    cap = cv2.VideoCapture(str(video_path))

    ret,prev = cap.read()
    if not ret:
        return None
    
    # resize and grayscale previous frame
    prev = preprocess_frame(prev, video_name)
    
    magnitudes = []

    frame_idx = 0

    while True:

        ret,frame = cap.read()
        if not ret:
            break

        # resize and grayscale current frame
        frame = preprocess_frame(frame, video_name)

        # calculate optical flow between two frames
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            frame, 
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # movement = sqrt(dx^2 + dy^2)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

        # each frame one movement number 
        magnitudes.append(np.mean(mag))

        prev = frame

        #track progress
        frame_idx += 1
        if frame_idx % 20 == 0:
            print("Processed frames: ", frame_idx)


    cap.release()

    magnitudes = np.array(magnitudes)

    return {
        "video": video_name,
        "mean_flow": np.mean(magnitudes),
        "std_flow": np.std(magnitudes),
        "median_flow": np.median(magnitudes),
        "max_flow": np.max(magnitudes),
        "p90_flow": np.percentile(magnitudes, 90),
        "p95_flow": np.percentile(magnitudes, 95),
    }

def all_videos_features(video_dir):

    rows = []

    for video_path in Path(video_dir).glob("*"):

        if video_path.suffix.lower() not in [".mp4", ".avi"]:
            continue
        print(" processing: ", video_path.name)
        
        features =  compute_flow(video_path, video_path.name)

        if features:
            rows.append(features)

    # save flow features
    df =  pd.DataFrame(rows)
    df.to_csv("optical_flow_features.csv", index=False)

    print("saved to optical_flow_features.csv")

if __name__ == "__main__":
    all_videos_features("../data/dataset")


        
 