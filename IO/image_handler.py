import cv2

from PIL import Image, ImageTk

class image_handler:
    """
        image_handler.py
        ------------------

        Used to handle input from image source (e.g. photo) and 
        provide said input to GUI class.
    """

    """Used to initialise class wide variables"""
    def __init__(self, source):
        self.source = source

    """Used to load the user specified file into the system"""
    def load_image(self):
        try:
            img = cv2.cvtColor(cv2.imread(self.source, 1), cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            return img, height, width
        except:
            print("[ERROR] Unable to access image file, please ensure it is available and not corrupted")