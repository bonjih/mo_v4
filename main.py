import json
import cv2

from RoiMultiClass import ComposeROI
from video_streamer import VideoStream


def main():
    with open('params.json', 'r') as f:
        data = json.load(f)

    comp_roi = ComposeROI(data)

    video_stream = VideoStream("YourCamName", comp_roi.video_file, comp_roi.roi_points)
    video_stream.start(comp_roi)

    while video_stream.running():
        video_stream.show_frame()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
