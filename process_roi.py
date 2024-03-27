import cv2
import numpy as np

from dust_detect import detect_blur_fft
from thresh import apply_fft_to_roi


class FrameProcessor:
    def __init__(self, roi_comp):
        self.roi_comp = roi_comp

    def process_frame(self, frame):
        frame_roi = frame.copy()
        text_y = 25  # Starting y-coordinate for text

        for idx, roi_key in enumerate(self.roi_comp.rois, start=1):
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()
            roi_filtered, mask, features = apply_fft_to_roi(frame, roi_points)
            mean, blurry = detect_blur_fft(frame_roi, size=60, thresh=10)

            frame_roi = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask))
            frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Add ROI key to Dusty/Not Dusty text
            dusty_text = f"{roi_key}: Dusty ({mean:.4f})" if blurry else f"{roi_key}: Not Dusty ({mean:.4f})"
            color = (0, 0, 255) if blurry else (0, 255, 0)
            cv2.putText(frame_roi, dusty_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

            # Add Bridge/No Bridge text based on sum features
            sum_features = np.sum(features.flatten())
            rounded_sum_features = round(sum_features, 4)
            bridge_text = f"{roi_key}: Bridge" if (idx == 1 and sum_features < 800) else f"{roi_key}: No Bridge"
            bridge_text += f" ({rounded_sum_features})"
            cv2.putText(frame_roi, bridge_text, (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 1)

            text_y += 60  # Increment y-coordinate for next ROI text

        return frame_roi
