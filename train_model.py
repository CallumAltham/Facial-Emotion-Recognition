import matplotlib
matplotlib.use("Agg")
import housekeeping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
from datetime import datetime

# Construct argument parser to parse console arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-lr", "--learning", required=False, default=0.001, help="Learning rate for model training. Default value of 0.001")
argparser.add_argument("-e", "--epochs", required=False, default=25, help="Number of epochs for training. Default value of 25 epochs")
argparser.add_argument("-bs", "--batch", required=False, default=32, help="Batch size used in model training. Default value of 32")
argparser.add_argument("-n", "--classes", required=False, default=7, help="Number of classes within training and validation directories")
argparser.add_argument("-t", "--train", required=False, default="data/train", help="Directory for training dataset")
argparser.add_argument("-v", "--val", required=False, default="data/val", help="Directory for validation dataset")
argparser.add_argument("-he", "--height", required=False, default=299, help="Height of images used within training")
argparser.add_argument("-w", "--width", required=False, default=299, help="Width of images used within training")
arguments = vars(argparser.parse_args())

INIT_LR = int(arguments["learning"])
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
train_datagen = ImageDataGenerator(dtype='float32',
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                preprocessing_function=preprocess_input)

train_ds = train_datagen.flow_from_directory(TRAIN_DIR,
                                    class_mode="binary",
                                    batch_size=BS,
                                    target_size=IMAGE_SIZE)

print("\nValidation:")
val_datagen = ImageDataGenerator(dtype='float32', 
                                preprocessing_function=preprocess_input)
validation_ds = val_datagen.flow_from_directory(VAL_DIR,
                                            class_mode="binary",
                                            batch_size=BS,
                                            target_size=IMAGE_SIZE)

print("\n[INFO] Creating Model...")
input_shape = (HEIGHT, WIDTH, 3)
baseModel = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dense(NUM_CLASSES, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
print("\n[INFO] Model Ready For Compiling")

print("\n[INFO] Compiling Model...")
model.compile(loss="binary_crossentropy", optimizer=Adam(INIT_LR), metrics=["accuracy"])

print("\n[INFO] Model Training In Progres...")
start = time.perf_counter()
H = model.fit(train_ds, validation_data = validation_ds, epochs=EPOCHS, verbose=1)
end = time.perf_counter()
print(f"\n[INFO] Model Training Completed Successfully {end-start}s")

print("\n[INFO] Saving Model to Models/face rec.model")
model.save("Models/face rec.model", save_format="h5")