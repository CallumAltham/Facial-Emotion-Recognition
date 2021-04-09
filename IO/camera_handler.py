import cv2
class camera_handler:
    """
        camera_handler.py
        ------------------

        Used to handle input from camera source and provide said input to 
        facial_rec class.

    """

    def __init__(self, source=0):
        try:
            self.source = source
            self.camStream = cv2.VideoCapture(self.source)
        except:
            print("[ERROR] Cannot access camera. Please check it is plugged in and not being used by another application")

    def get_current_frame(self):
        ret, frame = self.camStream.read()
        return ret, frame