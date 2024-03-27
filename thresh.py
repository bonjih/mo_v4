import numpy as np
import cv2


def apply_masks(frame, roi_comp):
    """
    Apply the mask to each ROI created in process_roi()
    :param frame: Original frame
    :param roi_comp: Object containing ROIs
    :return: Frame with only ROI regions outlined
    """

    frame_with_masks = frame.copy()

    for roi in roi_comp.rois:
        roi_points = roi.get_polygon_points()
        cv2.polylines(frame_with_masks, [roi_points], isClosed=True, color=(0, 255, 255), thickness=2)

    return frame_with_masks


def extract_resize_roi(frame, roi_pts, target_size=(100, 100)):
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


def apply_fft_to_roi(frame, roi_coords):
    roi_gray, mask = extract_resize_roi(frame, roi_coords, target_size=(100, 100))
    roi_fft = np.fft.fft2(roi_gray)
    roi_filtered = np.fft.ifft2(roi_fft).real  # Real part of inverse FFT

    normalized_roi = normalize_image(roi_filtered)
    features = normalized_roi.reshape(-1, 1)

    resized = cv2.resize(roi_filtered, (frame.shape[1], frame.shape[0]))
    return resized, mask, features
