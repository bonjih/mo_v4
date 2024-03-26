
from queue import Queue
from threading import Thread
import cv2
import time

from dust_detect import detect_blur_fft, dusty_labels
from create_mask import ROIMask


class VideoStream:
    def __init__(self, cam_name, path, roi_points, transform=None, queue_size=128):
        self.roi_comp = None
        self.cam_name = cam_name
        self.path = path
        self.stream = cv2.VideoCapture()
        self.stopped = False
        self.transform = transform
        self.queue_size = queue_size
        self.Q = Queue(maxsize=self.queue_size)
        self.frame_queue = Queue(maxsize=1)
        self.prev_frame = None
        self.roi_points = roi_points
        self.roi_mask = None
        self.save_video = False
        self.video_writer = None
        self.fps = None
        self.thread_read = Thread(target=self.read_frames, args=())
        self.thread_display = Thread(target=self.display_frames, args=())
        self.thread_check_stream = Thread(target=self.check_stream, args=())
        self.thread_write_video = None
        self.thread_read.daemon = True
        self.thread_display.daemon = True
        self.thread_check_stream.daemon = True

    def start(self, roi_comp, start_time_seconds=816):
        self.stream.open(self.path)
        self.stream.set(cv2.CAP_PROP_POS_MSEC, start_time_seconds * 1.0e3)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        _, self.prev_frame = self.stream.read()
        self.roi_mask = ROIMask(self.prev_frame.shape)
        self.roi_comp = roi_comp
        self.thread_read.start()
        self.thread_display.start()
        self.thread_check_stream.start()
        if self.save_video:
            self.thread_write_video = Thread(target=self.write_video, args=())
            self.thread_write_video.start()
        return self

    def read_frames(self):
        while not self.stopped:
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    print(f"Video stream {self.cam_name} has ended. Restarting...")
                    self.restart_stream()
                    continue

                if self.transform:
                    frame = self.transform(frame, self.prev_frame, self.roi_points)

                self.Q.put(frame)
                self.prev_frame = frame

            else:
                time.sleep(0.01)

    def display_frames(self):
        while not self.stopped:
            if not self.Q.empty():
                frame = self.Q.get()
                mean, blurry = detect_blur_fft(frame, size=60)
                dusty_labels(frame, mean, blurry)
                masked_frame = self.roi_mask.apply_masks(frame, self.roi_comp)
                self.frame_queue.put(masked_frame)
            else:
                time.sleep(0.01)

    def show_frame(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            cv2.imshow(self.cam_name, frame)
            cv2.waitKey(1)

    def write_video(self):
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter('output.avi', fourcc, self.fps,
                                                (self.prev_frame.shape[1], self.prev_frame.shape[0]))
            while not self.stopped:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.video_writer.write(frame)
                else:
                    time.sleep(0.01)

    def check_stream(self):
        while not self.stopped:
            if not self.stream.isOpened():
                print(f"Stream for {self.cam_name} is down or cannot be opened. Trying to restart...")
                self.restart_stream()
            time.sleep(1)

    def restart_stream(self):
        self.stream.release()
        self.stream.open(self.path)

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        return not self.Q.empty()

    def stop(self):
        self.stopped = True
        self.thread_read.join()
        self.thread_display.join()
        self.thread_check_stream.join()
        if self.thread_write_video:
            self.thread_write_video.join()
            self.stop_video_saving()

    def stop_video_saving(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def save_video_with_flag(self, save_flag):
        if save_flag:
            self.save_video = True
        else:
            self.save_video = False
            self.stop_video_saving()
