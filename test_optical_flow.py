import cv2
from preprocessing import preprocess_frame

video_path = "..\data\dataset\Arrow_GoPro_Seeded_Stabilised.avi"

cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()

if not ret:
    raise Exception("Could not read first frame")

prev_frame = preprocess_frame(
    prev_frame,
    "Arrow_GoPro_Seeded_Stabilised.avi"
)

ret, current_frame = cap.read()

if not ret:
    raise Exception("Could not read second frame")

current_frame = preprocess_frame(
    current_frame,
    "Arrow_GoPro_Seeded_Stabilised.avi"
)

print(prev_frame.shape)
print(current_frame.shape)

cap.release()