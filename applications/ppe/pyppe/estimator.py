

from ctypes import Array


class estimator:
    def __init__(self, filepath) -> None:
        self.videofile = filepath
        print("processing video file : ", self.videofile)

        self.video = cv2.VideoCapture(self.videofile)

    def estimate():
        pass

    def estimate_pos_2d() -> Array:
        pass

    def estimate_pos_3d() -> Array:
        pass