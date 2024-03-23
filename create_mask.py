import cv2
import numpy as np

from process_roi import apply_mask_to_rois


class ROIMask:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape

    def apply_black_mask(self, roi):
        """
        creates a black mask so to exclude processing outside the ROI
        :param roi:
        :return: black mask
        """
        # mask covering the entire frame
        black_mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)

        # ROI polygon on the mask
        roi_points = roi.get_polygon_points()
        cv2.fillPoly(black_mask, [roi_points], (255, 255, 255))  # fill ROI area with white color

        # invert the mask to cover the area outside the ROI
        black_mask = cv2.bitwise_not(black_mask)

        return black_mask

    def apply_masks(self, frame, roi_comp, back):
        """
        applies the mask to each ROI created in process_roi()
        :param back: background subtraction
        :param frame:
        :param roi_comp:
        :return:
        """

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Draw ROI polygons on the mask
        for roi in roi_comp.rois:
            roi_points = roi.get_polygon_points()
            cv2.fillPoly(mask, [roi_points], 255)  # Fill ROI area with white color

        # apply mask to each ROI and combine them
        masked_frame = apply_mask_to_rois(frame, mask, roi_comp.rois, back)

        # creates a white frame to only process the ROI's
        full_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # apply black mask to exclude processing outside the ROI's
        for roi in roi_comp.rois:
            black_mask_roi = self.apply_black_mask(roi)
            full_mask = cv2.bitwise_and(full_mask, full_mask, mask=black_mask_roi)

        # combine original frame with the masked ROI
        final_frame = cv2.bitwise_or(cv2.bitwise_and(frame, frame, mask=full_mask), masked_frame)

        return final_frame
