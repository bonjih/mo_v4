import cv2
import numpy as np
import pandas as pd
from collections import deque

from dust_detect import detect_blur_fft
from logistic_reg import perform_regression
from thresh import apply_fft_to_roi
from utils import make_csv, make_histo


class FrameProcessor:
    def __init__(self, roi_comp, dust_size=60, dust_thresh=2, audit_path="output/audit_data.csv"):
        self.roi_comp = roi_comp
        self.dust_size = dust_size
        self.dust_thresh = dust_thresh
        self.audit_path = audit_path

    def process_frame(self, frame, ts):
        audit_data = []
        ts = ts / 1000
        frame_roi = frame.copy()
        text_y = 25

        mean, dusty = detect_blur_fft(frame, size=self.dust_size, thresh=self.dust_thresh)
        dusty_text = f"Dusty ({mean:.4f})" if dusty else f"Not Dusty ({mean:.4f})"
        dusty_color = (0, 0, 255) if dusty else (0, 255, 0)

        cv2.putText(frame_roi, dusty_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dusty_color, 1)

        deque_roi = deque(maxlen=200)

        text_y += 30

        for idx, roi_key in enumerate(self.roi_comp.rois):
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()

            roi_filtered, mask, features = apply_fft_to_roi(frame, roi_points)
            deque_roi.append(roi_filtered)

            frame_roi = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask))
            # perform_regression(deque_roi)
            # make_histo(deque_roi)
            flattened_data = np.concatenate(deque_roi)
            normalized_data = (flattened_data - np.min(flattened_data)) / (
                    np.max(flattened_data) - np.min(flattened_data))

            for roi_filtered in deque_roi:
                frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Determine threshold based on ROI key
            sum_features = np.sum(normalized_data.flatten())

            if idx == 0:  # ts from roi_1
                cv2.putText(frame_roi, f"TS(s): {ts}", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1)

            if sum_features < 29999:
                bridge_text = f"{roi_key}: No Bridge ({sum_features:.4f})"
                is_bridge = True
            elif 30000 <= sum_features < 50000:
                bridge_text = f"{roi_key}: Bridge ({sum_features:.4f})"
                is_bridge = True
            else:
                bridge_text = f"{roi_key}: No Bridge ({sum_features:.4f})"
                is_bridge = False

            if not dusty:
                # Determine text color based on condition
                bridge_color = (0, 0, 255) if "Bridge" in bridge_text else (0, 255, 0)
                cv2.putText(frame_roi, bridge_text, (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bridge_color, 1)

            # display which roi
            cv2.putText(frame_roi, roi_key, (roi_points[0:1][0][-1], roi_points[0:1][0][-1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            text_y += 30

            audit_data.append([ts, sum_features, is_bridge])

        make_csv(audit_data, self.audit_path)

        return frame_roi

        # else:
        #     return frame_roi
