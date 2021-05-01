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

class networkHandler:

    def __init__(self):
        self.model_name = ""
        self.model_height = 0
        self.model_width = 0
        self.model_num_classes = 0
        self.model_classes = {}
        self.model_channels = 0
        try:
            self.model = load_model("Models/face_rec_2021-04-22_17-24-32.model")
            self.check_model("face_rec_2021-04-22_17-24-32.model")
        except:
            print("Default prediction model not found. Please add file provided to Models directory or run train_model.py file with default parameters")
            quit()

        self.sequenceAnalyser = sequence_analyser()
        self.detector = dlib.get_frontal_face_detector()
        
        try:
            self.shape_predicter = dlib.shape_predictor("Network/shape_predictor_68_face_landmarks.dat")
        except:
            print("DLIB face landmarks file cannot be found. Please add shape_predictor_68_face_landmarks.dat to Network directory")
            quit()

    def load_model(self, model):
        #try:
        self.model = load_model("Models/" + model)
        self.check_model(model)
        #except:
        #    print("[ERROR] Issue with accessing model file, please check it is available and is in the correct format for use (h5 format with a .model extension)")


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
            file.close()
        except:
            print("[ERROR] Cannot access model config txt to load in model. Please ensure model file is available containing default model paramaters from README")
            quit()

    def clear_sequence_analyser(self):
        self.sequenceAnalyser.reset_counters()

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
                image = self.get_prediction(image, rect, None)
                shape = self.shape_predicter(gray, rect)

                for i in range(shape.num_parts):
                    p = shape.part(i)
                    cv2.circle(image, (p.x, p.y), 1, (0, 0, 255), -1)

        return image

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

    def annotate_image(self, img, x, y, w, h, prediction):
        image = img
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
        cv2.rectangle(image,(x,y-50),(x+w,y),(255,0,0),-1)
        cv2.putText(image,prediction,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        return image

    def generate_metrics(self, metric):
        if metric == "Confusion Matrix":
            return self.gen_conf_matrix()
        elif metric == "Normalized Confusion Matrix":
            return self.gen_conf_matrix_norm()
        elif metric == "F-Score, Precision and Recall":
            pass
        elif metric == "MAE and MSE":
            pass

    def gen_conf_matrix(self):
        fig = Figure(figsize = (5, 5), dpi = 100)

        y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
        y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        
        axes = fig.add_subplot(111)
        
        # using the matshow() function 
        caxes = axes.matshow(df_confusion)
        fig.colorbar(caxes)

        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")

        return fig

    def gen_conf_matrix_norm(self):
        fig = Figure(figsize = (5, 5), dpi = 100)

        y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
        y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        df_conf_norm = df_confusion / df_confusion.sum(axis=1)
        
        axes = fig.add_subplot(111)
        
        # using the matshow() function 
        caxes = axes.matshow(df_conf_norm)
        fig.colorbar(caxes)

        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")

        return fig

    def gen_fscore(self):
        pass

    def gen_mae_mse(self):
        pass
    