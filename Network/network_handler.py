import Network.housekeeping
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Analysis.sequence_analyser import sequence_analyser
import dlib
from tkinter import * 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

"""
        network_handler.py
        ------------------

        Used to handle the relationship between prediction, annotation, metrics and sequence analyser systems and the main facial rec class

    """
class networkHandler:

    """Initialise class wide variables"""
    def __init__(self):
        self.model_name = ""
        self.model_height = 0
        self.model_width = 0
        self.model_num_classes = 0
        self.model_classes = {}
        self.model_channels = 0
        self.sequenceAnalyserPres = False
        try:
            self.model = load_model("Models/resnet50_2021-05-03_02-01-02.model")
            self.check_model("resnet50_2021-05-03_02-01-02.model")
        except:
            print("Default prediction model not found. Please add file provided to Models directory or run train_model.py file with default parameters")
            quit()

        self.sequenceAnalyser = sequence_analyser()
        self.sequenceAnalyserPres = True
        self.detector = dlib.get_frontal_face_detector()
        
        try:
            self.shape_predicter = dlib.shape_predictor("Network/shape_predictor_68_face_landmarks.dat")
        except:
            print("DLIB face landmarks file cannot be found. Please add shape_predictor_68_face_landmarks.dat to Network directory")
            quit()

    """Load specified model into the system for prediction"""
    def load_model(self, model):
        self.model = load_model("Models/" + model)
        self.check_model(model)


    """Check if model parameters available within configuration panel and load them into the system if found"""
    def check_model(self, model_name):
        try:
            file = open("Utilities/model_config.txt", "r")
            for line in file:
                items = line.replace("\n", "").split(" ")
                model_nme = items[0]
                if model_nme == model_name:
                    height = int(items[1])
                    width = int(items[2])
                    num_classes = int(items[3])
                    classes = items[4:-1]
                    num_channels = int(items[-1])

                    self.model_name = model_nme
                    self.model_height = height
                    self.model_width = width
                    self.model_num_classes = num_classes
                    self.model_channels = num_channels
                    
                    keys = range(self.model_num_classes)
                    values = classes
                    classes_dict = {}

                    for i in keys:
                        classes_dict[i] = values[i]

                    self.model_classes = classes_dict
                    if self.sequenceAnalyserPres:
                        self.sequenceAnalyser.reset_counters()
                        self.sequenceAnalyser.set_confidence_values(classes_dict)
            file.close()
        except:
            print("[ERROR] Cannot access model config txt to load in model. Please ensure model file is available containing default model paramaters from README")
            quit()

    """Instruct sequence analyser to reset all counters and confidence values"""
    def clear_sequence_analyser(self):
        self.sequenceAnalyser.reset_counters()

    """Make prediction upon a provided image"""
    def make_prediction(self, image, category):
        prediction = ""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if category == "video":
            make_pred = self.sequenceAnalyser.check_for_prediction()
            if make_pred == True:
                for rect in rects:
                    prediction = self.get_prediction(image, rect, "sequence")
                    shape = self.shape_predicter(gray, rect)
                    for i in range(shape.num_parts):
                        p = shape.part(i)
                        cv2.circle(image, (p.x, p.y), 1, (0, 0, 255), -1)
                    self.sequenceAnalyser.add_new_prediction(prediction)
                    image = self.annotate_image(image, rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top(), prediction)
            elif make_pred == "similarity":
                for rect in rects:
                    prediction = self.get_prediction(image, rect, "sequence")
                    prediction = self.sequenceAnalyser.check_new_prediction(prediction)
                    shape = self.shape_predicter(gray, rect)
                    for i in range(shape.num_parts):
                        p = shape.part(i)
                        cv2.circle(image, (p.x, p.y), 1, (0, 0, 255), -1)
                    image = self.annotate_image(image, rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top(), prediction)
            elif make_pred != True and make_pred != "similarity":
                for rect in rects:
                    prediction = make_pred
                    self.sequenceAnalyser.update_prediction_counter()
                    shape = self.shape_predicter(gray, rect)
                    for i in range(shape.num_parts):
                        p = shape.part(i)
                        cv2.circle(image, (p.x, p.y), 1, (0, 0, 255), -1)
                    image = self.annotate_image(image, rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top(), prediction)
        else:
            for rect in rects:
                if category == "metric":
                    prediction = self.get_prediction(image, rect, "sequence")
                    return prediction

                image = self.get_prediction(image, rect, None)
                shape = self.shape_predicter(gray, rect)

                for i in range(shape.num_parts):
                    p = shape.part(i)
                    cv2.circle(image, (p.x, p.y), 1, (0, 0, 255), -1)

        return image

    """Used to retrieve actual prediction and annotated image from prediction system"""
    def get_prediction(self, image, rect, category):
        if self.model_channels == 1:
            image_to_crop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_to_crop = image

        crop = image_to_crop[rect.top():rect.bottom(), rect.left():rect.right()]
        face = cv2.resize(crop, (self.model_height, self.model_width))

        img_scaled = face / 255.0
        reshape = np.reshape(img_scaled, (1, self.model_height, self.model_width, self.model_channels))
        img = np.vstack([reshape])
        result = self.model.predict(img)
        class_predict = np.argmax(result, axis=1)

        prediction = self.model_classes[class_predict[0]]
        if category == "sequence":
            return prediction
        else:
            image = self.annotate_image(image, rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top(), prediction)
            return image

    """Used to annotate given prediction onto an image"""
    def annotate_image(self, img, x, y, w, h, prediction):
        image = img
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
        cv2.rectangle(image,(x,y-50),(x+w,y),(255,0,0),-1)
        cv2.putText(image,prediction,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        return image

    """Used to instruct generation of metrics using a set of 10 images in Network/metrics_images"""
    def generate_metrics(self, metric):
        actual = ["Sad", "Disgust", "Angry", "Fear", "Happy", "Surprise", "Angry", "Surprise", "Sad", "Surprise"]
        if metric == "Confusion Matrix":
            return self.gen_conf_matrix(actual)
        elif metric == "Normalized Confusion Matrix":
            return self.gen_conf_matrix_norm(actual)
        elif metric == "F-Score, Precision and Recall":
            return self.gen_fscore(actual)
        elif metric == "MAE and MSE":
            return self.gen_mae_mse(actual)

    """Used to generate a confusion matrix regarding images found in Network/metrics_images"""
    def gen_conf_matrix(self, actual):
        fig = Figure(figsize = (5, 5), dpi = 100)

        predictions = []

        for image in os.listdir("Network/metrics_images"):
            image = "Network/metrics_images/" + image
            prediction = self.make_prediction(cv2.imread(image), "metric")
            predictions.append(prediction)

        y_actu = pd.Series(actual, name='Actual')
        y_pred = pd.Series(predictions, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        
        axes = fig.add_subplot(111)
        
        # using the matshow() function 
        caxes = axes.matshow(df_confusion)
        fig.colorbar(caxes)

        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")

        return fig

    """Used to generate a normalised confusion matrix regarding images found in Network/metrics_images"""
    def gen_conf_matrix_norm(self, actual):
        fig = Figure(figsize = (5, 5), dpi = 100)

        predictions = []

        for image in os.listdir("Network/metrics_images"):
            image = "Network/metrics_images/" + image
            prediction = self.make_prediction(cv2.imread(image), "metric")
            predictions.append(prediction)

        y_actu = pd.Series(actual, name='Actual')
        y_pred = pd.Series(predictions, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        df_conf_norm = df_confusion / df_confusion.sum(axis=1)
        
        axes = fig.add_subplot(111)
        
        # using the matshow() function 
        caxes = axes.matshow(df_conf_norm)
        fig.colorbar(caxes)

        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")

        return fig

    """Used to generate f-score, precision and recall regarding images found in Network/metrics_images"""
    def gen_fscore(self, actual):
        fig = Figure(figsize = (5, 5), dpi = 100)

        predictions = []

        for image in os.listdir("Network/metrics_images"):
            image = "Network/metrics_images/" + image
            prediction = self.make_prediction(cv2.imread(image), "metric")
            predictions.append(prediction)

        y_true = np.array(actual)
        y_pred = np.array(predictions)
        results = precision_recall_fscore_support(y_true, y_pred, average="macro")
        
        
        axes = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        axes.axis([0, 10, 0, 10])
        axes.text(2, 6, "F-Score: " + str(round(results[2], 2)), fontsize=15)
        axes.text(2, 5, "Precision: " + str(round(results[0], 2)), fontsize=15)
        axes.text(2, 4, "Recall: " + str(round(results[1], 2)), fontsize=15)

        return fig


    """Used to generate mean absolute error and mean squared error regarding images found in Network/metrics_images"""
    def gen_mae_mse(self, actual):
        fig = Figure(figsize = (5, 5), dpi = 100)

        predictions = []

        for image in os.listdir("Network/metrics_images"):
            image = "Network/metrics_images/" + image
            prediction = self.make_prediction(cv2.imread(image), "metric")
            predictions.append(prediction)

        le = preprocessing.LabelEncoder()
        le.fit(actual)
        
        actual = le.transform(actual)
        predictions = le.transform(predictions)

        y_true = np.array(actual)
        y_pred = np.array(predictions)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        
        axes = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        axes.axis([0, 10, 0, 10])
        axes.text(2, 6, "MSE: " + str(round(mse, 2)), fontsize=15)
        axes.text(2, 5, "MAE: " + str(round(mae, 2)), fontsize=15)

        return fig
    