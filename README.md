# FACIAL RECOGNITION SYSTEM - README

## DEPENDENCIES AND INSTALLATION
======================================
```bash
dlib: 19.21.1
pandas: 1.1.5
tensorflow: 2.3.1
matplotlib: 3.3.3
numpy: 1.18.5
opencv: 4.4.0.46
Pillow: 8.1.2
```

Use the package manager to install required dependencies with requirements.txt file found in root folder.

```bash
pip install -r requirements.txt
```

![Alt Text](https://raw.githubusercontent.com/CallumAltham/Facial-Emotion-Recognition/main/Documentation/gifs/requirements-install.gif)

Note: Tensorflow may not install GPU support correctly without presence of CUDA and CuDNN. Please see link below for more info:
https://www.tensorflow.org/install/gpu

## USAGE - MODEL TRAINING
===========================

If no models are provided, run train_model.py as follows

```bash
python model_train.py -lr NUM -e NUM -bs NUM -n NUM -t DIR -v DIR -he NUM -w NUM
```

```bash
AVAILABLE PARAMETERS
========================

-h, --help 
        Show the help message and exit

-lr --learning
            Learning rate for model training. Default value of 0.001

-e --epochs
            Number of epochs for training. Default value of 25 epochs

-bs --batch
            Batch size used in model training. Default value of 32

-n --classes   
            Number of classes within training and validation directories

-t --train
            Directory for training dataset

-v --val
            Directory for validation dataset

-he --height
            Height of images used within training

-w --width
            Width of images used within training

```

## USAGE - RUNNING MAIN APPLICATION
========================================

Running the main application is a simple task, simply open a terminal or CMD window in the root application directory and enter the following command.

NOTE: Please ignore any warnings entered into console. These are due to an inbuilt script used to limit memory usage when using NVIDIA GPUs.

```bash
python facial_rec.py
```

![Alt Text](https://raw.githubusercontent.com/CallumAltham/Facial-Emotion-Recognition/main/Documentation/gifs/application_open.gif)
