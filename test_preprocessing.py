
import cv2
from preprocessing import preprocess_frame

# Test the preprocessing 

cap = cv2.VideoCapture(
    "../data/dataset/BrentaFLIR.avi"
)

#print("Opened:", cap.isOpened())

ret, frame = cap.read()

#print("ret =", ret)
#print("frame is None =", frame is None)

processed = preprocess_frame(frame, "BrentaFLIR.avi")

print("Original:", frame.shape)
print("Processed:", processed.shape)

cap.release()