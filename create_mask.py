import cv2
import numpy as np

from thresh import apply_fft_to_roi


class ROIMask:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape

    @staticmethod
    def apply_masks(frame, roi_comp):
        """
        Apply the mask to each ROI created in process_roi()
        :param frame: Original frame
        :param roi_comp: Object containing ROIs
        :return: Frame with only ROI regions outlined
        """

        for roi in roi_comp.rois:
            roi_points = roi.get_polygon_points()
            cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=2)  # Draw ROI outline

            # Apply FFT to ROI
            roi_filtered, mask = apply_fft_to_roi(frame, roi_points)

            #Replace the original ROI in the frame with the filtered ROI
            frame_roi = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return frame
