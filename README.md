# FACIAL EMOTION RECOGNITION SYSTEM - README

## DEPENDENCIES AND INSTALLATION
## ======================================
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

<br />

## USAGE - MODEL TRAINING
## ===========================

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
<br />

## USAGE - RUNNING MAIN APPLICATION
## ========================================

Running the main application is a simple task, simply open a terminal or CMD window in the root application directory and enter the following command.

NOTE: Please ignore any warnings entered into console. These are due to an inbuilt script used to limit memory usage when using NVIDIA GPUs.

```bash
python facial_rec.py
```

![Alt Text](https://raw.githubusercontent.com/CallumAltham/Facial-Emotion-Recognition/main/Documentation/gifs/application_open.gif)

<br />

## USAGE - CONTROLLING INPUT PLAYBACK
## ==========================================

Three buttons are available to control playback, with each button being available for differing input types:
- Play: Can only control playback of video input. Will be unable to be pressed when no video input selected, or when video is still playing
- Pause: Can only control playback of video input. Will be unable to be pressed when no vidoe input selected, or when video is currently paused
- End: Can end use of all three input types, and reset's canvas back to blank when pressed.

<br />

## USAGE - SELECTING AN INPUT FOR PREDICTION
## ==========================================

Three buttons are available to select the type of input used. Namely:
- Image Input: Image based input only (e.g. jpg/png)
- Video Input: Video based input only (e.g. mp4/avi)
- Camera Input: Camera stream input only 
    - Note: Camera will open default camera in system. At present there is no way to select camera

<br />

<br />

## USAGE - SELECTING AN MODEL FOR PREDICTION
## ==========================================

A dropdown menu is available for selecting the model to be used in prediction.

All models found within "Models" directory in application file structure will be usable. Model will not be usable unless in this folder.

NOTE: Models must be saved in a .h5 format by Keras with a .model extension. No other files will be accepted as models

<br />

## USAGE - VIEW HELP
## ==========================================

A small subset of help information is available within the application.

To access this information, click the button labeled "Help" on the bottom left side of the application.

To close this panel, simply click the help button once more and the panel will close.

<br />

## USAGE - GENERATE METRICS
## ==========================================

ADD LATER

<br />

## USAGE - CHANGE SETTINGS
## ==========================================

ADD LATER

<br />

## TROUBLESHOOTING
## ====================================

The following table may show potential solutions to problems that may arise. If not, please feel free to attempt to find a solution yourself within the code or email me at callumaltham1@gmail.com for aid with the subject line "Facial Emotion Recognition System Issue".

NOTE: Please inform me of any issues you find, even if you manage to fix them yourselves so it can be added to the main code repository.

| Issue      |Potential Solution|
| ------------- |:-------------:|
| col 3 is      | right-aligned |
| col 2 is      | centered      |
| zebra stripes | are neat      |

<br />

## THANK YOU FOR READING
## ====================================

Any questions please don't hestitate to email me at callumaltham1@gmail.com. Please start the subject line of the email as "Facial Emotion Recognition System Query" 

<br />

![Gif of Cat waving goodbye](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)