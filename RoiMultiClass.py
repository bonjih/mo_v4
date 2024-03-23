import numpy as np


class ROI:
    def __init__(self, **kwargs):
        self.points = kwargs

    def get_polygon_points(self):
        points = [(self.points[f'x{i}'], self.points[f'y{i}']) for i in range(1, 5)]
        return np.array(points, np.int32)


class ComposeROI:
    def __init__(self, data):
        self.roi_points = self.extract_roi_points(data)
        self.rois = []
        self.thresholds = []
        self.video_file = None

        for key, value in data.items():
            if key.startswith("roi"):
                roi = ROI(**value)
                self.rois.append(roi)
            elif key == "thresholds":
                for thresh_key, thresh_value in value.items():
                    thresh = ROI(**thresh_value)
                    self.thresholds.append(thresh)
            elif key == "input_video_path":
                self.video_file = value

    @staticmethod
    def extract_roi_points(data):
        roi_points = []
        for key, value in data.items():
            if key.startswith("roi"):
                roi_points.append(value)
        return roi_points

    def add_roi(self, roi):
        self.rois.append(roi)

    def add_threshold(self, thresh):
        self.thresholds.append(thresh)

    def __iter__(self):
        return iter(self.rois + self.thresholds)
