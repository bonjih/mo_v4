import numpy as np
import cv2


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


def apply_mask_to_rois(frame, mask, rois):
    """
    Applies mask to each ROI and combines them.

    :param back: Background subtraction
    :param frame: Original frame
    :param mask: Mask to apply
    :param rois: List of ROIs
    :return: Combined masked frame
    """

    # Initialise blank masked frame
    masked_frame = np.zeros_like(frame)

    # apply mask to each ROI and combine them
    for roi in rois:
        masked_frame_roi = apply_processed_mask(masked_frame, mask, roi)
        masked_frame = cv2.bitwise_or(masked_frame, masked_frame_roi)

    return masked_frame
