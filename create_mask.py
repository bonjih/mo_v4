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

    @staticmethod
    def apply_masks(frame, roi_comp):
        """
        applies the mask to each ROI created in process_roi()
        :param frame: Original frame
        :param roi_comp: Object containing ROIs
        :return: Frame with only ROIs visible
        """

        # Create a blank frame to show only the ROIs
        roi_frame = np.zeros_like(frame)

        # Draw ROI polygons directly on the frame
        for roi in roi_comp.rois:
            roi_points = roi.get_polygon_points()
            cv2.fillPoly(roi_frame, [roi_points], (255, 255, 255))  # Fill ROI area with white color

        # Combine original frame with the ROI frame
        final_frame = cv2.bitwise_or(frame, roi_frame)

        return final_frame
