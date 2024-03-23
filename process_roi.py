import os
import pickle

import numpy as np
import cv2
import imutils


def pre_process_frames(frame, back):
    img_array = []
    orig_array = [frame]

    frame = back.apply(frame)

    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    img_array.append(blur)
    x = np.array(img_array)
    o = np.array(orig_array)

    q = compute_fft(x)
    angle = np.angle(q)
    phase_spectrum_array = np.exp(1j * angle)
    reconstructed_array = compute_ifft(phase_spectrum_array)

    for i in range(o.shape[0]):
        abs_reconstructed_array = abs(reconstructed_array[i])
        binary_frame, processed_frame = process_frame(abs_reconstructed_array, o[i])

        return binary_frame, processed_frame


def compute_fft(x):
    if os.access('stream_fft.pickle', os.R_OK):
        with open('stream_fft.pickle', 'rb') as f:
            q = pickle.load(f)
    else:
        q = np.fft.fftn(x)
        with open('stream_fft.pickle', 'wb') as f:
            pickle.dump(q, f)
    return q


def compute_ifft(phase_spectrum_array):
    if os.access('stream_ifft.pickle', os.R_OK):
        with open('stream_ifft.pickle', 'rb') as f:
            reconstructed_array = pickle.load(f)
    else:
        reconstructed_array = np.fft.ifftn(phase_spectrum_array)
        with open('stream_ifft.pickle', 'wb') as f:
            pickle.dump(reconstructed_array, f)
    return reconstructed_array


def process_frame(reconstructed_array, orig_frame):
    frame = abs(reconstructed_array)
    filteredFrame = cv2.GaussianBlur(frame, (13, 13), 0)
    mean_value = np.mean(filteredFrame)
    ret, binary_frame = cv2.threshold(filteredFrame, 1.6 * mean_value, 255, cv2.THRESH_BINARY)
    npbinary = np.uint8(binary_frame)  # Convert to uint8 data type
    npbinary = cv2.cvtColor(npbinary, cv2.COLOR_BGR2GRAY)  # Ensure it's single-channel

    # Ensure binary image is properly binarized
    binary_frame = cv2.threshold(npbinary, 127, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(binary_frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 180:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return binary_frame, orig_frame


def apply_processed_mask(frame, mask, roi):
    """
    applies the processed mask to the ROI
    :param frame:
    :param mask:
    :param roi:
    :return:
    """
    masked_frame = frame.copy()
    roi_points = roi.get_polygon_points()
    cv2.polylines(masked_frame, [roi_points], True, (255, 0, 0), 1)
    masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=mask)
    return masked_frame


def apply_mask_to_rois(frame, mask, rois, back):
    """
    Applies mask to each ROI and combines them.

    :param back: Background subtraction
    :param frame: Original frame
    :param mask: Mask to apply
    :param rois: List of ROIs
    :return: Combined masked frame
    """

    # Pre-process the frame
    binary_frame, processed_frame = pre_process_frames(frame, back)

    # Initialise blank masked frame
    masked_frame = np.zeros_like(frame)

    # apply mask to each ROI and combine them
    for roi in rois:
        masked_frame_roi = apply_processed_mask(processed_frame, mask, roi)
        masked_frame = cv2.bitwise_or(masked_frame, masked_frame_roi)

    return masked_frame
