import matplotlib
matplotlib.use("Agg")
import Network.housekeeping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Activation, SeparableConv2D, ZeroPadding2D
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
import datetime

# Construct argument parser to parse console arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-lr", "--learning", required=False, type=float, default=0.01, help="Learning rate for model training. Default value of 0.001")
argparser.add_argument("-e", "--epochs", required=False, default=50, help="Number of epochs for training. Default value of 25 epochs")
argparser.add_argument("-bs", "--batch", required=False, default=32, help="Batch size used in model training. Default value of 32")
argparser.add_argument("-n", "--classes", required=False, default=7, help="Number of classes within training and validation directories")
argparser.add_argument("-t", "--train", required=False, default="datasets/fer2013/data", help="Directory for training dataset")
argparser.add_argument("-v", "--val", required=False, default="datasets/fer2013/validation", help="Directory for validation dataset")
argparser.add_argument("-he", "--height", required=False, default=48, help="Height of images used within training")
argparser.add_argument("-w", "--width", required=False, default=48, help="Width of images used within training")
arguments = vars(argparser.parse_args())

INIT_LR = (arguments["learning"])
EPOCHS = int(arguments["epochs"])
BS = int(arguments["batch"])
NUM_CLASSES = int(arguments["classes"])
TRAIN_DIR = arguments["train"]
VAL_DIR = arguments["val"]
HEIGHT = int(arguments["height"])
WIDTH = int(arguments["width"])
IMAGE_SIZE = (int(arguments["height"]), int(arguments["width"]))

print("\n[INFO] Creating Datasets...")
print("\nTraining:")
train_datagen = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                validation_split=0.2
)

train_ds = train_datagen.flow_from_directory(TRAIN_DIR,
                                    class_mode="categorical",
                                    color_mode="grayscale",
                                    batch_size=BS,
                                    target_size=IMAGE_SIZE,
                                    shuffle=True,
                                    subset="training")

print("\nValidation:")
val_datagen = ImageDataGenerator(validation_split=0.2)

validation_ds = val_datagen.flow_from_directory(TRAIN_DIR,
                                            class_mode="categorical",
                                            color_mode="grayscale",
                                            batch_size=BS,
                                            target_size=IMAGE_SIZE,
                                            subset="validation")


print("\n[INFO] Creating Model...")
input_shape = (HEIGHT, WIDTH, 1)
model = Sequential()

model.add(Conv2D(64, (5, 5), padding="same", activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (5, 5), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))


model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(NUM_CLASSES, activation='softmax'))

#model.summary()

#model = Model(inputs=baseModel.input, outputs=headModel)
print("\n[INFO] Model Ready For Compiling")

print("\n[INFO] Compiling Model...")
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])

lr_reducer = ReduceLROnPlateau(monitor='val_loss' , factor=0.1, patience=2, min_lr=0.00001, model='auto')

print("\n[INFO] Model Training In Progress...")
start = time.perf_counter()
H = model.fit(train_ds, validation_data = validation_ds, callbacks=[lr_reducer], epochs=EPOCHS, verbose=1)
end = time.perf_counter()
print(f"\n[INFO] Model Training Completed Successfully In {end-start}s")

print("\n[INFO] Saving Model to Models/face_rec_"+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +".model")
filename = "face_rec_"+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +".model"
model.save("Models/"+filename, save_format="h5")

print(f"\n[INFO] Updating Configuration File. Please do not delete any data not added to the file by you")
file = open("Utilities/model_config.txt", "a")
sentence = "\n" + filename + " " + str(HEIGHT) + " " + str(WIDTH) + " " + str(NUM_CLASSES) + " " + " ".join(list(train_ds.class_indices.keys()))
file.write(sentence)
file.close()
