from threading import Thread
import cv2
import time
from queue import Queue

from dust_detect import dusty_labels


class VideoStream:
    def __init__(self, cam_name, path, roi_points, transform=None, queue_size=128):
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
        self.thread_read = Thread(target=self.read_frames, args=())
        self.thread_display = Thread(target=self.display_frames, args=())
        self.thread_check_stream = Thread(target=self.check_stream, args=())
        self.thread_read.daemon = True
        self.thread_display.daemon = True
        self.thread_check_stream.daemon = True

    def start(self):
        self.stream.open(self.path)
        _, self.prev_frame = self.stream.read()
        self.thread_read.start()
        self.thread_display.start()
        self.thread_check_stream.start()
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
                time.sleep(0.1)

    def display_frames(self):
        while not self.stopped:
            if not self.Q.empty():
                frame = self.Q.get()
                self.frame_queue.put(frame)
            else:
                time.sleep(0.1)

    def show_frame(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            dusty_labels(frame)
            cv2.imshow(self.cam_name, frame)
            cv2.waitKey(1)

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

