import cv2
from create_mask import ROIMask
from dust_detect import detect_blur_fft


def run_video(roi_comp):
    video_file = roi_comp.video_file
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()

    roi_mask = ROIMask(frame.shape)
    frame_shape = frame.shape

    if not ret:
        print("Error: Unable to read the first frame.")
        return

    back = cv2.createBackgroundSubtractorKNN()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (mean, blurry) = detect_blur_fft(gray, size=60)

        # Apply masks using ROIMask class methods
        masked_frame = roi_mask.apply_masks(frame, roi_comp, back)

        # Dusty no Dusty
        text = "Dusty ({:.4f})" if blurry else "Not Dusty ({:.4f})"
        text = text.format(mean)
        color = (0, 0, 255) if blurry else (0, 255, 0)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("frame", masked_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame_shape
