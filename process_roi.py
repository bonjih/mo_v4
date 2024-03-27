import cv2
import numpy as np

from dust_detect import detect_blur_fft
from thresh import apply_fft_to_roi


class FrameProcessor:
    def __init__(self, roi_comp, dust_size=60, dust_thresh=2):
        self.roi_comp = roi_comp
        self.dust_size = dust_size
        self.dust_thresh = dust_thresh

    def process_frame(self, frame):
        frame_roi = frame.copy()
        text_y = 25

        mean, blurry = detect_blur_fft(frame, size=self.dust_size, thresh=self.dust_thresh)
        dusty_text = f"Dusty ({mean:.4f})" if blurry else f"Not Dusty ({mean:.4f})"
        dusty_color = (0, 0, 255) if blurry else (0, 255, 0)

        cv2.putText(frame_roi, dusty_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dusty_color, 1)

        if not blurry:
            text_y += 30

            for roi_key in self.roi_comp.rois:
                roi = self.roi_comp.rois[roi_key]
                roi_points = roi.get_polygon_points()
                roi_filtered, mask, features = apply_fft_to_roi(frame, roi_points)
                frame_roi = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask))
                frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                # Determine threshold based on ROI key
                sum_features = np.sum(features.flatten())

                if sum_features < 250:
                    bridge_text = f"{roi_key}: No Bridge ({sum_features:.4f})"
                elif 250 <= sum_features < 400:
                    bridge_text = f"{roi_key}: No Bridge ({sum_features:.4f})"
                elif 400 <= sum_features < 500:
                    bridge_text = f"{roi_key}: Bridge ({sum_features:.4f})"
                else:
                    bridge_text = f"{roi_key}: No Bridge ({sum_features:.4f})"

                # Determine text color based on condition
                bridge_color = (0, 0, 255) if "Bridge" in bridge_text else (0, 255, 0)

                cv2.putText(frame_roi, bridge_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bridge_color, 1)

                text_y += 30

            return frame_roi

        else:

            return frame_roi
