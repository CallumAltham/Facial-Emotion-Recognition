"""Module Imports"""
import operator
import time

class sequence_analyser:

    """
        sequence_analyser.py
        ------------------

        Used to handle the process of sequential analysis for video and camera data.

    """

    """Method to initialise class wide variables"""
    def __init__(self):
        self.prediction_counter = 0
        self.recheck_counter = 0
        self.current_prediction = ""
        self.confidence_values = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surpise": 0}

    """ Used to reset confidence values to new model parameters when model changed"""
    def set_confidence_values(self, dict):
        self.confidence_values = {}
        for value in dict.values():
            self.confidence_values[value] = 0


    """Used by prediction system to check if a prediction should be made"""
    def check_for_prediction(self):
        if self.prediction_counter < 50:
            return True
        elif self.prediction_counter == 50:
            self.assign_current_prediction()
            return self.current_prediction 
        elif self.prediction_counter > 50:
            return self.check_recheck_counter()
            
    """Used to add a new prediction to the system by updating the confidence value"""
    def add_new_prediction(self, prediction):
        self.confidence_values[prediction] += 1
        self.update_prediction_counter()

    """Used to check if new prediction is different to currently assigned common prediction and assign new prediction if so, keep old prediction if not"""
    def check_new_prediction(self, prediction):
        if prediction != self.current_prediction:
            self.reset_counters()
            return prediction
        elif prediction == self.current_prediction:
            self.reset_recheck_counter()
            return self.current_prediction
        
    """Used to keep track of frames analysed after most recent common prediction assignment and to determine if prediction check should be made"""
    def check_recheck_counter(self):
        if self.recheck_counter >= 150:
            return "similarity"
        elif self.recheck_counter < 150:
            self.update_recheck_counter()
            return self.get_current_prediction()

    """Used to reset all counters and confidence values within the system"""
    def reset_counters(self):
        self.prediction_counter = 0
        self.confidence_values = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surpise": 0}
        self.current_prediction = ""
        self.recheck_counter = 0

    """Used to reset recheck counter to 0"""
    def reset_recheck_counter(self):
        self.recheck_counter = 0

    """Used to assign current prediction as the highest confidence value"""
    def assign_current_prediction(self):
        self.current_prediction = max(self.confidence_values.items(), key=operator.itemgetter(1))[0]

    """Update the number of predictions counted by one"""
    def update_prediction_counter(self):
        self.prediction_counter += 1

    """Used to update recheck counter by one"""
    def update_recheck_counter(self):
        self.recheck_counter += 1

    """Used to retrieve current common prediction"""
    def get_current_prediction(self):
        return self.current_prediction