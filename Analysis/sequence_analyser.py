import operator
import time

class sequence_analyser:

    def __init__(self):
        self.prediction_counter = 0
        self.recheck_counter = 0
        self.current_prediction = ""
        self.confidence_values = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surpise": 0}

    def check_for_prediction(self):
        if self.prediction_counter < 50:
            return True
        elif self.prediction_counter == 50:
            self.assign_current_prediction()
            return self.current_prediction 
        elif self.prediction_counter > 50:
            return self.check_recheck_counter()
            
    def add_new_prediction(self, prediction):
        self.confidence_values[prediction] += 1
        self.update_prediction_counter()

    def check_new_prediction(self, prediction):
        if prediction != self.current_prediction:
            self.reset_counters()
            return prediction
        elif prediction == self.current_prediction:
            self.reset_recheck_counter()
            return self.current_prediction
        
    def check_recheck_counter(self):
        if self.recheck_counter >= 150:
            return "similarity"
        elif self.recheck_counter < 150:
            self.update_recheck_counter()
            return self.get_current_prediction()

    def reset_counters(self):
        self.prediction_counter = 0
        self.confidence_values = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surpise": 0}
        self.current_prediction = ""
        self.recheck_counter = 0

    def reset_recheck_counter(self):
        self.recheck_counter = 0

    def assign_current_prediction(self):
        self.current_prediction = max(self.confidence_values.items(), key=operator.itemgetter(1))[0]

    def update_prediction_counter(self):
        self.prediction_counter += 1

    def update_recheck_counter(self):
        self.recheck_counter += 1

    def get_current_prediction(self):
        return self.current_prediction