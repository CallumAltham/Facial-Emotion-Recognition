import cv2

class video_handler:
    """
        video_handler.py
        ------------------

        Used to handle input from video source and 
        provide said input to facial_rec class.
    """
    def __init__(self, source):
        self.source = source
        self.videoStream = cv2.VideoCapture(self.source)

    def get_current_frame(self):
        ret, frame = self.videoStream.read()
        return ret, frame