from collections import deque

import numpy as np
import cv2

from dust_detect import detect_blur_fft


def capture_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap


def extract_resize_roi(frame, roi_coordinates, target_size=(100, 100)):
    # Extract the ROI using the provided coordinates
    roi_pts = np.array([[roi_coordinates['x1'], roi_coordinates['y1']],
                        [roi_coordinates['x2'], roi_coordinates['y2']],
                        [roi_coordinates['x3'], roi_coordinates['y3']],
                        [roi_coordinates['x4'], roi_coordinates['y4']]], dtype=np.int32)

    mask = np.zeros_like(frame[:, :, 0])
    cv2.fillPoly(mask, [roi_pts], (255, 255, 255))
    roi = cv2.bitwise_and(frame, frame, mask=mask)

    roi_resized = cv2.resize(roi, target_size)
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    return roi_gray, mask


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def extract_features_and_target(frame, roi_coords):
    roi_gray, _ = extract_resize_roi(frame, roi_coords, target_size=(100, 100))
    roi_fft = np.fft.fft2(roi_gray)
    roi_filtered = np.fft.ifft2(roi_fft).real

    normalized_roi = normalize_image(roi_filtered)

    features = normalized_roi.reshape(-1, 1)

    return features


def apply_fft_to_roi(frame, roi_coords):
    roi_gray, mask = extract_resize_roi(frame, roi_coords, target_size=(100, 100))

    roi_fft = np.fft.fft2(roi_gray)
    roi_filtered = np.fft.ifft2(roi_fft).real  # Real part of inverse FFT

    resized = cv2.resize(roi_filtered, (frame.shape[1], frame.shape[0]))

    return resized, mask


def process_video(video_path, output_path):
    cap = capture_video(video_path)

    cap.set(cv2.CAP_PROP_POS_MSEC, 700 * 1.0e3)

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

    roi_coordinates2 = {
        "x1": 700,
        "y1": 580,
        "x2": 700,
        "y2": 300,
        "x3": 1200,
        "y3": 300,
        "x4": 1200,
        "y4": 580
    }

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    roi_filtered_deque = deque(maxlen=200)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi_filtered, mask = apply_fft_to_roi(frame, roi_coordinates2)
        roi_filtered_deque.append(roi_filtered)

        features = extract_features_and_target(frame, roi_coordinates2)

        # Replace the original ROI in the frame with the filtered ROI
        frame_roi = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Draw text on frame_roi
        mean, blurry = detect_blur_fft(frame, size=60, thresh=10)
        text = "Dusty ({:.4f})" if blurry else "Not Dusty ({:.4f})"
        text = text.format(mean)
        color = (0, 0, 255) if blurry else (0, 255, 0)
        cv2.putText(frame_roi, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        sum_features = np.sum(features.flatten())
        rounded_sum_features = round(sum_features, 4)

        if sum_features < 800:
            color = (0, 0, 255)
            cv2.putText(frame_roi, f"Bridge ({rounded_sum_features})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 1)
        else:
            color = (0, 255, 0)
            cv2.putText(frame_roi, f"No Bridge ({rounded_sum_features})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color,
                        1)

        cv2.imshow('Filtered Frame ', frame_roi)

        out.write(frame_roi)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video capture, writer, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Call the function to process the video and save the output
process_video("data/crusher_bin_bridge.mkv", "output_video2.mp4")

