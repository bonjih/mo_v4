import cv2

from process_roi import FrameProcessor


class VideoProcessor:
    def __init__(self, video_path, output_path, roi_comp):
        self.video_path = video_path
        self.output_path = output_path
        self.roi_comp = roi_comp
        self.frame_processor = FrameProcessor(roi_comp)

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

            processed_frame = self.frame_processor.process_frame(frame)
            cv2.imshow('Filtered Frame ', processed_frame)
            out.write(processed_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()






