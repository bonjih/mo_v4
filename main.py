import json
import os

from RoiMultiClass import ComposeROI
from video_stream2 import VideoProcessor


def main():
    with open('params.json', 'r') as f:
        data = json.load(f)

    roi_config = ComposeROI(data)
    video_path = roi_config.video_file

    if not os.path.exists(video_path) or not os.path.isfile(video_path):
        print("Input video path or file does not exist.")
        return

    output_path = data.get("output_video_path", None)

    if output_path is None:
        output_dir = data.get("output_video_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output_video.mkv")

    is_watching = data.get("is_watching", False)
    is_save_video = data.get("is_save_video", False)
    offset = data.get("offset", 0)

    video_processor = VideoProcessor(video_path, output_path, roi_config,
                                     is_watching=is_watching,
                                     is_save_video=is_save_video,
                                     offset=offset)
    video_processor.process_video()


if __name__ == '__main__':
    main()
