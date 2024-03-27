import json

from RoiMultiClass import ComposeROI
from video_stream2 import VideoProcessor


def main():
    with open('params.json', 'r') as f:
        data = json.load(f)

    roi_config = ComposeROI(data)

    video_path = roi_config.video_file
    output_path = data.get("output_video_path", "crusher_bin_bridge_2.mkv")

    video_processor = VideoProcessor(video_path, output_path, roi_config)
    video_processor.process_video()


if __name__ == '__main__':
    main()
