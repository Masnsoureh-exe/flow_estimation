import cv2

def preprocess_frame(frame, video_name, target_width=320):

    # Cut the 
    if "BrentaFLIR" in video_name:
        frame = frame[:, :-80]

    # Convert to grayscale if video has color
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame
    h, w = frame.shape[:2]

    scale = target_width / w
    target_height = int(h * scale)

    frame = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    return frame



