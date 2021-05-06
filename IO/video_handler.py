import cv2

class video_handler:
    """
        video_handler.py
        ------------------

        Used to handle input from video source and 
        provide said input to facial_rec class.
    """

    """Initialises class wide variables"""
    def __init__(self, source):
        try:
            self.source = source
            self.videoStream = cv2.VideoCapture(self.source)
        except:
            print("[ERROR] Unable to access video file, please ensure it is available and not corrupted")
            
    """Retrieve current video frame from video stream"""
    def get_current_frame(self):
        ret, frame = self.videoStream.read()
        return ret, frame