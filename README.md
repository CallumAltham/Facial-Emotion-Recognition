# FACIAL EMOTION RECOGNITION SYSTEM - README

## DEPENDENCIES AND INSTALLATION
```bash
tensorflow==2.3.1
pandas==1.1.5
dlib==19.21.1
matplotlib==3.3.3
numpy==1.18.5
opencv_python==4.4.0.46
Pillow==8.2.0
scikit_learn==0.24.2

```

Use the package manager to install required dependencies with requirements.txt file found in root folder.

```bash
pip install -r requirements.txt
```

Note: Tensorflow may not install GPU support correctly without presence of CUDA and CuDNN. Please see link below for more info:
https://www.tensorflow.org/install/gpu

<br />

## USAGE - MODEL TRAINING

If no models are provided, run train_model.py as follows where 'NUM' is replaced with your chosen parameter value.

```bash
python train_model.py -lr NUM -e NUM -bs NUM -n NUM -t DIR -v DIR -he NUM -w NUM
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
            Directory for dataset

-he --height
            Height of images used within training

-w --width
            Width of images used within training

```
<br />

## USAGE - RUNNING MAIN APPLICATION

Running the main application is a simple task, simply open a terminal or CMD window in the root application directory and enter the following command.

NOTE: Please ignore any warnings entered into console. These are due to an inbuilt script used to limit memory usage when using NVIDIA GPUs.

```bash
python facial_rec.py
```
<br />

## USAGE - CONTROLLING INPUT PLAYBACK

Three buttons are available to control playback, with each button being available for differing input types:
- Play: Can only control playback of video input. Will be unable to be pressed when no video input selected, or when video is still playing
- Pause: Can only control playback of video input. Will be unable to be pressed when no vidoe input selected, or when video is currently paused
- End: Can end use of all three input types, and reset's canvas back to blank when pressed.

<br />

## USAGE - SELECTING AN INPUT FOR PREDICTION

Three buttons are available to select the type of input used. Namely:
- Image Input: Image based input only (e.g. jpg/png)
- Video Input: Video based input only (e.g. mp4/avi)
- Camera Input: Camera stream input only 
    - Note: Camera will open default camera in system. At present there is no way to select camera

<br />

<br />

## USAGE - SELECTING AN MODEL FOR PREDICTION

A dropdown menu is available for selecting the model to be used in prediction.

All models found within "Models" directory in application file structure will be usable. Model will not be usable unless in this folder.

NOTE: Models must be saved in a .h5 format by Keras with a .model extension. No other files will be accepted as models

<br />

## USAGE - VIEW HELP

A small subset of help information is available within the application.

To access this information, click the button labeled "Help" on the bottom left side of the application.

To close this panel, simply click the help button once more and the panel will close.

<br />

## USAGE - GENERATE METRICS

Within the application, a separate panel is available to allow for the generation of metrics.

To access this panel, click the button labelled "Metrics" at the bottom middle of the application.

Within this panel is a dropdown menu and a button. To generate a metric, please select an option from the dropdown menu and the metric will appear on screen.

To save this metric to file, press the save to disk button next to the dropdown menu and select a save location, filename and extension. The currently selected metric will then be saved in the selected location.

<br />

## USAGE - CHANGE SETTINGS

Within the application, a separate panel is available to allow for the generation of metrics.

To access this panel, click the button labelled "Settings" at the bottom right of the application.

Within this panel are a series of dropdown menus and a button. To change an application setting, please select an item from a dropdown.

Once your chosen parameters have been chosen from the dropdown, press the save settings button below to change and update application parameters

<br />

## THANK YOU FOR READING

Any questions please don't hestitate to email me at callumaltham1@gmail.com. Please start the subject line of the email as "Facial Emotion Recognition System Query" 

<br />