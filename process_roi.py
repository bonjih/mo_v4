import cv2
import numpy as np
from collections import deque

import global_params_variables
from dust_detect import detect_blur_fft
from thresh import apply_fft_to_roi
from utils import make_csv

# setup
params = global_params_variables.ParamsDict()

dust_size = params.get_value('dust')['size']
dust_thresh = params.get_value('dust')['thresh']
audit_path = params.get_value('output_audit_path')
len_deque_features = params.get_value('offset')

deque_features = deque(maxlen=len_deque_features)
audit_deque = deque(maxlen=len_deque_features)


class FrameProcessor:
    def __init__(self, roi_comp, prev_frame):
        self.roi_comp = roi_comp
        self.prev_frame = prev_frame

    def process_frame(self, frame, prev_frame, ts):
        audit_data = []
        ts = ts / 1000
        frame_roi = frame.copy()
        text_y = 25
        is_bridge = None
        bridge_text = None

        deque_roi = deque(maxlen=200)

        mean, dusty = detect_blur_fft(frame)
        dusty_text = f"Dusty ({mean:.4f})" if dusty else f"Not Dusty ({mean:.4f})"
        dusty_color = (0, 0, 255) if dusty else (0, 255, 0)

        cv2.putText(frame_roi, dusty_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dusty_color, 1)

        text_y += 30

        for roi_key in self.roi_comp.rois:
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()
            frame_roi = [roi_key, frame_roi]

            roi_filtered, mask, features, max_window_val = apply_fft_to_roi(frame_roi, prev_frame, roi_points)
            deque_roi.append(roi_filtered)

            frame_roi = cv2.bitwise_and(frame_roi[1], frame_roi[1], mask=cv2.bitwise_not(mask))

            flattened_data = np.concatenate(roi_filtered)
            normalized_data = (flattened_data - np.min(flattened_data)) / (
                    np.max(flattened_data) - np.min(flattened_data))

            for roi_filtered in deque_roi:
                frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            sum_features = np.std(normalized_data.flatten()) * 100

            if roi_key == 'roi_1':  # ts from roi_1
                cv2.putText(frame_roi, f"TS(s): {ts}", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            if max_window_val > 40:
                bridge_text = f"{roi_key}: No Bridge ({sum_features:.4f}) {max_window_val}"
                is_bridge = False
            else:
                bridge_text = f"{roi_key}: Bridge ({sum_features:.4f}) {max_window_val}"
                is_bridge = True

            # if 13.5000 <= sum_features < 25.0000:
            #     bridge_text = f"{roi_key}: Bridge ({sum_features:.4f}) {max_window_val}"
            #     is_bridge = True
            # elif sum_features < 13.4999 and max_window_val > 80:
            #     bridge_text = f"{roi_key}: No Bridge ({sum_features:.4f}) {max_window_val}"
            #     is_bridge = False

            if not dusty:
                bridge_color = (0, 0, 255) if "Bridge" in bridge_text else (0, 255, 0)
                cv2.putText(frame_roi, bridge_text, (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bridge_color, 1)

            cv2.putText(frame_roi, roi_key, (roi_points[1:2, 0:1][-1][-1], roi_points[1:2, 1:][-1][-1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

            text_y += 30

            audit_data.append([ts, sum_features, is_bridge, max_window_val])
            deque_features.append([sum_features, is_bridge])
            audit_deque.append([sum_features, is_bridge])
            # perform_regression(roi_key, list(deque_features))

        make_csv(audit_data, audit_path)

        return frame_roi
