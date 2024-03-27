import numpy as np
import cv2
from collections import deque
from dust_detect import detect_blur_fft
from thresh import apply_fft_to_roi


class VideoProcessor:
    def __init__(self, video_path, output_path, roi_comp):
        self.video_path = video_path
        self.output_path = output_path
        self.roi_comp = roi_comp

    def capture_video(self):
        cap = cv2.VideoCapture(self.video_path)
        return cap

    def process_video(self):
        cap = self.capture_video()

        cap.set(cv2.CAP_PROP_POS_MSEC, 700 * 1.0e3)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_roi = frame.copy()

            text_y = 25  # Starting y-coordinate for text

            for idx, roi_key in enumerate(self.roi_comp.rois, start=1):
                roi = self.roi_comp.rois[roi_key]

                roi_points = roi.get_polygon_points()

                roi_filtered, mask, features = apply_fft_to_roi(frame, roi_points)

                mean, blurry = detect_blur_fft(frame_roi, size=60, thresh=10)

                frame_roi = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask))
                frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                # Add ROI information to the text
                text = f"{roi_key}: Dusty ({mean:.4f}), Sum Features: {np.sum(features.flatten()):.4f}" if blurry else f"{roi_key}: Not Dusty ({mean:.4f}), Sum Features: {np.sum(features.flatten()):.4f}"
                color = (0, 0, 255) if blurry else (0, 255, 0)
                cv2.putText(frame_roi, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

                text_y += 30  # Increment y-coordinate for next ROI text

            sum_features = np.sum(features.flatten())
            rounded_sum_features = round(sum_features, 4)

            # Add ROI information to the "Bridge" or "Not Bridge" text
            bridge_text = ""
            for roi_key in self.roi_comp.rois:
                if sum_features < 800:
                    bridge_text += f"{roi_key}, "
            bridge_text = bridge_text[:-2]

            if sum_features < 430:
                color = (0, 0, 255)
                cv2.putText(frame_roi, f"Bridge ({bridge_text}): {rounded_sum_features}", (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 1)
            else:
                color = (0, 255, 0)
                cv2.putText(frame_roi, f"No Bridge ({bridge_text}): {rounded_sum_features}", (10, text_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

            cv2.imshow('Filtered Frame ', frame_roi)
            out.write(frame_roi)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
