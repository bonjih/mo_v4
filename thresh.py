import numpy as np
import cv2

import global_params_variables

# setup
params = global_params_variables.ParamsDict()
ws = params.get_value('change_window')['size']


def calculate_rate_of_change(frame1, frame2, roi_key, window_size):
    """
    Calculate the rate of change of pixel intensities between two frames within specified windows.

    Parameters:
        frame1 (numpy.ndarray): The first frame.
        frame2 (numpy.ndarray): The second frame.
        window_size (tuple): A tuple (window_height, window_width) specifying the size of the window.

    Returns:
        numpy.ndarray: An array containing the calculated rate of change for each window.

    Note:
        The mean pixel intensity difference is calculated for each window
    """
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(gray_frame1, gray_frame2)

    window_height, window_width = window_size

    rate_of_change = []

    for y in range(0, gray_frame1.shape[0], window_height):
        for x in range(0, gray_frame1.shape[1], window_width):
            window = frame_diff[y:y + window_height, x:x + window_width]
            rate_of_change.append(np.mean(window))

    max_value = np.max(rate_of_change)

    return roi_key, max_value


def extract_resize_roi(frame, roi_pts, target_size=(100, 100)):
    """
    Extract a region of interest (ROI) from the frame, resize it, and convert it to grayscale.

    Parameters:
        frame (numpy.ndarray): The original frame.
        roi_pts (list): List of points defining the region of interest (ROI).
        target_size (tuple): Target size of the ROI after resizing. Default is (100, 100).

    Returns:
        numpy.ndarray: The ROI resized and converted to grayscale.
        numpy.ndarray: The mask used to extract the ROI.
    """
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


def apply_fft_to_roi(frame, prev_frame, roi_coords):
    """
        Apply Fast Fourier Transform (FFT) to a region of interest (ROI) within a frame.

        Parameters:
            frame (numpy.ndarray): Current frame.
            prev_frame (numpy.ndarray): Previous frame.
            roi_coords (list): List of coordinates defining the region of interest (ROI).

        Returns:
            numpy.ndarray: Resized ROI after FFT processing.
            numpy.ndarray: Mask used for extracting the ROI.
            numpy.ndarray: Features extracted from the FFT-processed ROI.
        """
    key_roi, max_window_val = calculate_rate_of_change(frame[1], prev_frame, frame[0], (ws, ws))

    roi_gray, mask = extract_resize_roi(frame[1], roi_coords, target_size=(100, 100))
    roi_fft = np.fft.fft2(roi_gray)
    roi_filtered = np.fft.ifft2(roi_fft).real  # Real part of inverse FFT

    normalized_roi = normalize_image(roi_filtered)
    features = normalized_roi.reshape(-1, 1)

    resized = cv2.resize(roi_filtered, (frame[1].shape[1], frame[1].shape[0]))
    return resized, mask, features, (key_roi, max_window_val)
