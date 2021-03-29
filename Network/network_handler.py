import Network.housekeeping
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Analysis.sequence_analyser import sequence_analyser
import dlib
from tkinter import * 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import pandas as pd
import numpy as np

class networkHandler:

    def __init__(self):
        try:
            self.model = load_model("Models/InceptionV3 - 87 acc.model")
        except:
            print("Default prediction model not found. Please add file provided to Models directory or run train_model.py file")
            quit()

        self.sequenceAnalyser = sequence_analyser()
        self.detector = dlib.get_frontal_face_detector()
        
        try:
            self.shape_predicter = dlib.shape_predictor("Network/shape_predictor_68_face_landmarks.dat")
        except:
            print("DLIB face landmarks file cannot be found. Please add shape_predictor_68_face_landmarks.dat to Network directory")
            quit()

    def load_model(self, model):
        self.model = load_model("Models/" + model)

    def check_model(self):
        print("Selected model: " + str(self.model))

    def clear_sequence_analyser(self):
        self.sequenceAnalyser.reset_counters()

    def make_image_prediction(self, image):
        label = {0:"female", 1:"male"}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = self.detector(gray, 0)

        for rect in rects:
            crop = image[rect.top():rect.bottom(), rect.left():rect.right()]
            face = cv2.resize(crop, (229, 229))

            img_scaled = face / 255.0
            reshape = np.reshape(img_scaled, (1, 229, 229, 3))
            img = np.vstack([reshape])
            result = self.model.predict(img)
        
            max_val = np.max(result[0])
            max_index = np.where(result[0] == max_val)
            idx = max_index[0]
            if idx == 0:
                cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
                cv2.putText(image,label[0],(rect.left(),rect.top()-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            elif idx == 1:
                cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)
                cv2.putText(image,label[0],(rect.left(),rect.top()-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            shape = self.shape_predicter(gray, rect)

            for i in range(shape.num_parts):
                p = shape.part(i)
                cv2.circle(image, (p.x, p.y), 1, (0, 0, 255), -1)
        
        return image

    def make_video_prediction(self, image):
        prediction = ""
        label = {0:"female", 1:"male"}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        make_pred = self.sequenceAnalyser.check_for_prediction()

        if make_pred == True:
            #REPLACE WITH REAL PREDICTION CODE WHEN MODEL MADE
            for rect in rects:

                crop = image[rect.top():rect.bottom(), rect.left():rect.right()]
                face = cv2.resize(crop, (229, 229))

                img_scaled = face / 255.0
                reshape = np.reshape(img_scaled, (1, 229, 229, 3))
                img = np.vstack([reshape])
                result = self.model.predict(img)
            
                max_val = np.max(result[0])
                max_index = np.where(result[0] == max_val)
                idx = max_index[0]
                prediction = label[int(idx)]

                self.sequenceAnalyser.add_new_prediction(prediction)
                image = self.annotate_image(image, rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top(), prediction)
        elif make_pred == "similarity":
            for rect in rects:

                crop = image[rect.top():rect.bottom(), rect.left():rect.right()]
                face = cv2.resize(crop, (229, 229))

                img_scaled = face / 255.0
                reshape = np.reshape(img_scaled, (1, 229, 229, 3))
                img = np.vstack([reshape])
                result = self.model.predict(img)
            
                max_val = np.max(result[0])
                max_index = np.where(result[0] == max_val)
                idx = max_index[0]
                prediction = label[int(idx)]

                self.sequenceAnalyser.check_new_prediction(prediction)

                image = self.annotate_image(image, rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top(), prediction)

        elif make_pred != True:
            prediction = make_pred
            #REPLACE WITH REAL PREDICTION CODE WHEN MODEL MADE
            for rect in rects:
                self.sequenceAnalyser.update_prediction_counter()
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
    