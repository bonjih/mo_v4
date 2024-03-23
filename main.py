import json
from RoiMultiClass import ComposeROI
from video_streamer import run_video

with open('params.json', 'r') as f:
    data = json.load(f)

input_video_path = data["input_video_path"]

comp_roi = ComposeROI(data)
run_video(comp_roi)
