import numpy as np
import cv2


# Function to capture video
def capture_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap


def detect_edges(roi_gray, min_threshold=50, max_threshold=150):
    edges = cv2.Canny(roi_gray, min_threshold, max_threshold)
    return edges


def extract_resize_roi(frame, roi_coordinates, target_size=(100, 100)):
    # Extract the ROI using the provided coordinates
    roi_pts = np.array([[roi_coordinates['x1'], roi_coordinates['y1']],
                        [roi_coordinates['x2'], roi_coordinates['y2']],
                        [roi_coordinates['x3'], roi_coordinates['y3']],
                        [roi_coordinates['x4'], roi_coordinates['y4']]], dtype=np.int32)
    mask = np.zeros_like(frame[:, :, 0])
    cv2.fillPoly(mask, [roi_pts], (255, 255, 255))
    roi = cv2.bitwise_and(frame, frame, mask=mask)

    # Resize ROI
    roi_resized = cv2.resize(roi, target_size)

    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    return roi_gray, mask


def apply_fft_to_roi(frame, roi_coordinates):
    roi_gray, mask = extract_resize_roi(frame, roi_coordinates, target_size=(100, 100))

    roi_fft = np.fft.rfft2(roi_gray)
    roi_filtered = np.fft.irfft2(roi_fft)

    # Resize the filtered ROI back to the size of the original frame
    roi_filtered_resized = cv2.resize(roi_filtered, (frame.shape[1], frame.shape[0]))

    return roi_filtered_resized, mask


# Load the video
def process_video(video_path):
    cap = capture_video(video_path)

    # Define the ROI coordinates
    roi_coordinates = {
        "x1": 250,
        "y1": 580,
        "x2": 300,
        "y2": 300,
        "x3": 980,
        "y3": 300,
        "x4": 800,
        "y4": 580
    }

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply FFT to ROI
        roi_filtered, mask = apply_fft_to_roi(frame, roi_coordinates)

        # Replace the original ROI in the frame with the filtered ROI
        frame_roi = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)



        # Display the frame with ROI and edges
        cv2.imshow('Filtered Frame with Edges', frame_roi)

        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to process the video
process_video("data/crusher_bin_bridge2.mkv")
