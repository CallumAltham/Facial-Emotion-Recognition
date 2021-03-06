import Network.housekeeping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD,
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
import datetime

# Construct argument parser to parse console arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-lr", "--learning", required=False, type=float, default=1e-3, help="Learning rate for model training. Default value of 1e-3")
argparser.add_argument("-e", "--epochs", required=False, default=100, help="Number of epochs for training. Default value of 100 epochs")
argparser.add_argument("-bs", "--batch", required=False, default=32, help="Batch size used in model training. Default value of 32")
argparser.add_argument("-n", "--classes", required=False, default=7, help="Number of classes within training and validation directories")
argparser.add_argument("-t", "--train", required=False, default="datasets/fer2013/data", help="Directory for dataset")
argparser.add_argument("-he", "--height", required=False, default=229, help="Height of images used within training")
argparser.add_argument("-w", "--width", required=False, default=229, help="Width of images used within training")
arguments = vars(argparser.parse_args())

# Specification of model arguments
INIT_LR = (arguments["learning"])
EPOCHS = int(arguments["epochs"])
BS = int(arguments["batch"])
NUM_CLASSES = int(arguments["classes"])
TRAIN_DIR = arguments["train"]
HEIGHT = int(arguments["height"])
WIDTH = int(arguments["width"])
IMAGE_SIZE = (int(arguments["height"]), int(arguments["width"]), 3)

# Creation of datasets
print("\n[INFO] Creating Datasets...")
print("\nTraining:")
train_datagen = ImageDataGenerator(dtype='float32',
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2,
                                   preprocessing_function=preprocess_input
)

train_ds = train_datagen.flow_from_directory(TRAIN_DIR,
                                    class_mode="categorical",
                                    batch_size=BS,
                                    target_size=IMAGE_SIZE,
                                    subset="training")

print("\nValidation:")
val_datagen = ImageDataGenerator(dtype='float32',
                                validation_split=0.2,
                                preprocessing_function=preprocess_input)

validation_ds = val_datagen.flow_from_directory(TRAIN_DIR,
                                            class_mode="categorical",
                                            batch_size=BS,
                                            target_size=IMAGE_SIZE,
                                            subset="validation")

# Creation of models
print("\n[INFO] Creating Model...")
baseModel = InceptionV3(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE)
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
predictions = Dense(7, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=predictions)
model.summary()

#Compiling and training initial layers
print("\n[INFO] Model Ready For Compiling")

for layer in baseModel.layers:
    layer.trainable = False

print("\n[INFO] Compiling Upper Layers")
opt = Adam(lr = INIT_LR, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("\n[INFO] Upper Layers Training In Progress...")
H = model.fit(train_ds, validation_data = validation_ds, epochs=10, verbose=1)

# Compiling and training complete model
for layer in model.layers:
    layer.trainable = True

print("\n[INFO] Compiling Model...")
opt = SGD(lr = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True)


reduce_lr = ReduceLROnPlateau(
monitor 	= 'val_loss',
	factor		= 0.4,
	patience	= 5,
	mode 		= 'auto',
	min_lr		= 1e-6)

early_stop = EarlyStopping(
	monitor 	= 'val_loss',
	patience 	= 20,
	mode 		= 'auto')

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("\n[INFO] Complete Model Training In Progress...")
start = time.perf_counter()
H = model.fit(train_ds, validation_data = validation_ds, callbacks=[reduce_lr, early_stop], epochs=EPOCHS, verbose=1)
end = time.perf_counter()
print(f"\n[INFO] Model Training Completed Successfully In {end-start}s")

# Save model to disk
print("\n[INFO] Saving Model to Models/inceptionv3_"+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +".model")
filename = "inceptionv3_"+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +".model"
model.save("Models/"+filename, save_format="h5")

# Update configuration parameters
print(f"\n[INFO] Updating Configuration File. Please do not delete any data not added to the file by you")
file = open("Utilities/model_config.txt", "a")
sentence = "\n" + filename + " " + str(HEIGHT) + " " + str(WIDTH) + " " + str(NUM_CLASSES) + " " + " ".join(list(train_ds.class_indices.keys())) + " " + 3
file.write(sentence)
file.close()

