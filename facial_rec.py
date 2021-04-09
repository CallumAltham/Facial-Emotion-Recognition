from IO.video_handler import video_handler
from Network.network_handler import networkHandler
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import cv2
from IO.camera_handler import camera_handler
from IO.image_handler import image_handler
from PIL import Image, ImageTk
import os
from os import listdir
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class facial_rec:
    """

        facial_rec.py
        -----------
        Used to handle display of GUI and major application functions
    """

    def __init__(self):

        """WINDOW PROPERTIES AND MAIN WINDOW FRAME"""
        self.root = Tk()
        self.root.title("Facial Recognition System")
        self.root.config(bg="lightgrey")

        """CAPTURE STREAMS AND JOBS FOR RUNNING INPUT"""
        self.cam_cap = None
        self.cam_job = None

        self.vid_cap = None
        self.vid_job = None
        
        self.img_cap = None

        self.network_handler = networkHandler()

        self.saveFlag = False
        self.saveConfirmationFlag = False

        self.videoWriter = None
        self.image_to_save = None

        self.vid_height = None
        self.vid_width = None

        """OVERRIDING INTERVAL USED TO CREATE NATURAL FRAME COUNT FOR VIDEO AND CAMERA INPUT"""
        self.interval = 50

        """FLAGS USED TO DETERMINE CURRENT STATE OF APPLICATION FEATURES
            E.G. INPUT TYPE, 
            VIDEO PLAYBACK STATE, 
            PANEL VISIBILITY STATE
        """
        self.input_flag = None
        self.pause_flag = False

        self.helpFlag = False
        self.settingsFlag = False
        self.metricsFlag = False

        """ROOT WINDOW LAYOUT SPECIFICATION"""     

        self.mainPanel = Frame(self.root, bg="lightgrey")
        self.mainPanel.pack(side=LEFT)

        self.secondaryPanel = Frame(self.root, bg="lightgrey")
        self.secondaryPanel.pack(side=RIGHT)

        self.main_title = Label(self.mainPanel, text="Facial Recognition System", bg="lightgrey")
        self.main_title.config(font=(14))
        self.main_title.pack(pady=10)
 
        self.canvas = Canvas(self.mainPanel, height=450, width=450)
        self.canvas.pack(pady=10, padx=10)

        videoControlFrame = Frame(self.mainPanel, bg="lightgrey")
        videoControlFrame.pack()

        self.playButton = Button(videoControlFrame, text="Play", width=22, command=lambda: self.button_handler("play"))
        self.playButton.pack(side=LEFT, padx=5, pady=5)

        self.pauseButton = Button(videoControlFrame, text="Pause", width=22, command=lambda: self.button_handler("pause"))
        self.pauseButton.pack(side=LEFT, padx=5, pady=5)

        self.endButton = Button(videoControlFrame, text="End", width=22, command=lambda: self.button_handler("end"))
        self.endButton.pack(side=LEFT, padx=5, pady=5)

        inputSourceFrame = Frame(self.mainPanel, bg="lightgrey")
        inputSourceFrame.pack(fill='x')

        self.imageButton = Button(inputSourceFrame, text="Image Input", width=22, command=lambda: self.button_handler("image"))
        self.imageButton.pack(side=LEFT, padx=5, pady=5)

        self.videoButton = Button(inputSourceFrame, text="Video Input", width=22, command=lambda: self.button_handler("video"))
        self.videoButton.pack(side=LEFT, padx=5, pady=5)

        self.cameraButton = Button(inputSourceFrame, text="Camera Input", width=22, command=lambda: self.button_handler("camera"))
        self.cameraButton.pack(side=LEFT, padx=5, pady=5)

        self.modelSelect = StringVar(self.mainPanel)
        self.modelSelect.set("Prediction Model Select")

        modelSelectMenu = OptionMenu(self.mainPanel, self.modelSelect, *(f for f in listdir("Models") if f.endswith('.model')))
        modelSelectMenu.pack(fill='x', padx=5, pady=5,)

        self.modelSelect.trace("w", self.model_dropdown_select)

        menuPanelFrame = Frame(self.mainPanel, bg="lightgrey")
        menuPanelFrame.pack(fill='x')

        helpButton = Button(menuPanelFrame, text="Help", width=22, command=lambda: self.button_handler("help"))
        helpButton.pack(side=LEFT, padx=5, pady=5)

        metricsButton = Button(menuPanelFrame, text="Metrics", width=22, command=lambda: self.button_handler("metrics"))
        metricsButton.pack(side=LEFT, padx=5, pady=5)

        settingsButton = Button(menuPanelFrame, text="Settings", width=22, command=lambda: self.button_handler("settings"))
        settingsButton.pack(side=LEFT, padx=5, pady=5)

        self.helpPanelFrame = Frame(self.secondaryPanel, bg="lightgrey")

        help_title = Label(self.helpPanelFrame, text="Help", bg="lightgrey")
        help_title.config(font=(14))
        help_title.pack(pady=10)

        helpText = Text(self.helpPanelFrame, bg="lightgrey", bd=0, relief=FLAT)

        helpScroll = Scrollbar(self.helpPanelFrame)
        helpScroll.pack(side=RIGHT, fill=Y)
        helpText.pack(side=LEFT, fill='y', padx=5, pady=5)
        helpScroll.config(command=helpText.yview)
        helpText.config(yscrollcommand=helpScroll.set)

        with open('Utilities/help.txt', 'r') as file:
            data_text = file.read()

        helpText.insert(END, data_text)
        helpText['state'] = DISABLED

        self.settingsPanelFrame = Frame(self.secondaryPanel, bg="lightgrey")
        
        settings_title = Label(self.settingsPanelFrame, text="Settings", bg="lightgrey")
        settings_title.config(font=(14))
        settings_title.pack(pady=10)

        self.metricsPanelFrame = Frame(self.secondaryPanel, bg="lightgrey")

        metrics_title = Label(self.metricsPanelFrame, text="Metrics", bg="lightgrey")
        metrics_title.config(font=(14))
        metrics_title.pack(pady=10)
        
        self.metricsCanvas = None
        self.current_fig = None

        metricsButtonsFrame = Frame(self.metricsPanelFrame, bg="lightgrey")
        metricsButtonsFrame.pack()
        
        self.metricSelect = StringVar(metricsButtonsFrame)
        self.metricSelect.set("Metric Select") # default value

        metrics = ["Confusion Matrix", "Normalized Confusion Matrix", "F-Score, Precision and Recall", "MAE and MSE"]

        metricsSelectMenu = OptionMenu(metricsButtonsFrame, self.metricSelect, *metrics)
        metricsSelectMenu.pack(side=LEFT, padx=5, pady=5,)
        self.metricSelect.trace("w", self.metric_dropdown_select)

        saveButton = Button(metricsButtonsFrame, text="Save to Disk", width=22, command=lambda: self.button_handler("save"))
        saveButton.pack(side=RIGHT, padx=5, pady=5)

    """METHOD USED TO HANDLE SELECTION OF ITEMS WITHIN MODEL SELECTOR DROPDOWN"""
    def model_dropdown_select(self, *args):
        model_name = self.modelSelect.get()

        model_pres = False

        file = open("Utilities/model_config.txt", "r")
        for line in file:
            items = line.replace("\n", "").split(" ")
            model_nme = items[0]
            if model_nme == model_name:
                model_pres = True
        file.close()

        if model_pres:
            self.network_handler.load_model(model_name)
        else:
            width = simpledialog.askstring("Parameter Input", "What is the width used in the model?", parent=self.root)
            height = simpledialog.askstring("Parameter Input", "What is the height used in the model?", parent=self.root)
            num_classes = simpledialog.askstring("Parameter Input", "How many classes are used in the model?", parent=self.root)
            classes = simpledialog.askstring("Parameter Input", "What are the classes used in the model? (Classes must be in the same order as specified during training and separated by a space)")

            params = [width, height, num_classes, classes]
            res = None in params

            if res:
                messagebox.showwarning("Model Configuration Error", "Cannot use selected model as not all parameters provided")
                self.modelSelect.set("Prediction Model Select")
            else:
                file = open("Utilities/model_config.txt", "a")
                file.write(model_name + " " + height + " " + width + " " + num_classes + " " + classes + "\n")
                file.close()
                self.network_handler.load_model(model_name)

    
    def metric_dropdown_select(self, *args):
        metric = self.network_handler.generate_metrics(self.metricSelect.get())
        self.current_fig = metric
        try: 
            self.metricsCanvas.get_tk_widget().pack_forget()
        except AttributeError: 
            pass  
        self.metricsCanvas = FigureCanvasTkAgg(metric, self.metricsPanelFrame)
        self.metricsCanvas.draw()
        self.metricsCanvas.get_tk_widget().pack(pady=10, padx=10)

    """METHOD USED TO START GUI LOOP AND SET MEDIA PLAYBACK FUNCTIONS AS DISABLED"""
    def start_gui(self):
        self.playButton['state'] = DISABLED
        self.pauseButton['state'] = DISABLED
        self.endButton['state'] = DISABLED
        self.root.mainloop()
        
    """METHOD USED TO HANDLE FUNCTION OF INDIVIDUAL BUTTONS WITHIN GUI"""
    def button_handler(self, button):
        if button == "camera":
            self.input_flag = "camera"    

            self.imageButton['state'] = DISABLED
            self.videoButton['state'] = DISABLED
            self.cameraButton['state'] = DISABLED

            self.endButton['state'] = NORMAL

            self.cam_cap = camera_handler(0)
            self.update_cam_image()

        if button == "video":
            filename =  filedialog.askopenfilename(initialdir = "/", title = "Select Video file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))

            if filename:
                self.input_flag = "video"

                self.playButton['state'] = DISABLED
                self.pauseButton['state'] = NORMAL
                self.endButton['state'] = NORMAL

                self.imageButton['state'] = DISABLED
                self.videoButton['state'] = DISABLED
                self.cameraButton['state'] = DISABLED

                self.vid_cap = video_handler(filename)
                self.update_vid_image()

        if button == "image":
            filename =  filedialog.askopenfilename(initialdir = "/",title = "Select Image file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

            if filename:
                self.imageButton['state'] = DISABLED
                self.videoButton['state'] = DISABLED
                self.cameraButton['state'] = DISABLED
                self.endButton['state'] = NORMAL

                self.img_cap = image_handler(filename)

                i, height, width = self.img_cap.load_image()
                i = self.network_handler.make_image_prediction(i)

                self.img = Image.fromarray(i)
                self.img = ImageTk.PhotoImage(self.img)

                self.canvas.config(width=width, height=height)
                self.canvas.create_image(0, 0, anchor=NW, image=self.img)

                MsgBox = messagebox.askquestion ('Save Image To File','Do you want to save the annotated image to disk', icon = 'question')
                if MsgBox == 'yes':
                    filename = filedialog.asksaveasfilename(initialdir="/", title="Select Location To Save Annotated Image", 
                    defaultextension=".*", filetypes=(("jpeg files","*.jpg"), ("png files", "*.png"),("all files","*.*")))
                    if filename:
                        cv2.imwrite(filename, cv2.cvtColor(i, cv2.COLOR_BGR2RGB))


        if button == "pause":
            self.pause_flag = True
            self.playButton['state'] = NORMAL
            self.pauseButton['state'] = DISABLED

        if button =="play":
            self.pause_flag = False
            self.playButton['state'] = DISABLED
            self.pauseButton['state'] = NORMAL

        if button == "end":
            self.playButton['state'] = DISABLED
            self.pauseButton['state'] = DISABLED
            self.endButton['state'] = DISABLED

            self.imageButton['state'] = NORMAL
            self.videoButton['state'] = NORMAL
            self.cameraButton['state'] = NORMAL

            if self.input_flag == "video":
                self.root.after_cancel(self.vid_job)
                self.vid_cap = None
            elif self.input_flag == "camera":
                self.root.after_cancel(self.cam_job)
                self.cam_cap = None

            self.input_flag = None

            self.reset_canvas()
            self.network_handler.clear_sequence_analyser()

            if self.saveFlag:
                self.releaseWriter()

        if button == "help":
            if self.helpFlag != True:
                self.helpPanelFrame.pack()
                self.secondaryPanel.pack(side=RIGHT, fill='y')
                self.helpFlag = True

                self.settingsPanelFrame.forget()
                self.settingsFlag = False

                self.metricsPanelFrame.forget()
                self.metricsFlag = False

            else:
                self.helpPanelFrame.forget()
                self.secondaryPanel.forget()
                self.helpFlag = False

        if button == "settings":
            if self.settingsFlag != True:
                self.settingsPanelFrame.pack()
                self.secondaryPanel.pack(side=RIGHT, fill='y')
                self.settingsFlag = True

                self.helpPanelFrame.forget()
                self.helpFlag = False

                self.metricsPanelFrame.forget()
                self.metricsFlag = False

            else:
                self.settingsPanelFrame.forget()
                self.secondaryPanel.forget()
                self.settingsFlag = False

        if button == "metrics":
            if self.metricsFlag != True:
                self.metricsPanelFrame.pack()
                self.secondaryPanel.pack(side=RIGHT, fill='y')
                self.metricsFlag = True

                self.settingsPanelFrame.forget()
                self.settingsFlag = False

                self.helpPanelFrame.forget()
                self.helpFlag = False
            else:
                self.metricsPanelFrame.forget()
                self.secondaryPanel.forget()
                self.metricsFlag = False

        if button == "save":
            if self.current_fig:
                filename =  filedialog.asksaveasfilename(initialdir = "/", title = "Select Location To Save Metric Image File", 
                defaultextension=".*", filetypes=(("jpeg files","*.jpg"), ("png files", "*.png"),("all files","*.*")))
                if filename:
                    self.current_fig.savefig(filename)
            else:
                MsgBox = messagebox.showwarning('No Metric Selected','Please select a metric to save to disk. A metric can be saved when an image is visible', icon = 'warning')

    """FUNCTION USED TO CONTINUALLY UPDATE CANVAS CONTENT TO NEW CAMERA FRAME AT 20FPS RATE"""
    def update_cam_image(self):
        frame = self.cam_cap.get_current_frame()[1]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas.config(width=image.shape[:2][1], height=image.shape[:2][0])

        if not self.saveConfirmationFlag:
            MsgBox = messagebox.askquestion('Save Video To File','Do you want to save the processed video file to disk', icon = 'question')
            if MsgBox == 'yes':
                filename = filename = filedialog.asksaveasfilename(initialdir="/", title="Select Location To Save Annotated Video", 
                    defaultextension=".*", filetypes=(("avi files", "*.avi"),("all files","*.*")))
                if filename:
                    self.setVideoWriter(image.shape[:2][0], image.shape[:2][1], filename)
            self.saveConfirmationFlag = True

        image = self.network_handler.make_video_prediction(image)

        if self.saveFlag:
            self.writeFrame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        image = Image.fromarray(image)
        self.img = ImageTk.PhotoImage(image)

        self.canvas.create_image(0,0,anchor=NW,image=self.img)
        self.cam_job = self.root.after(self.interval, self.update_cam_image)

    """FUNCTION USED TO CONTINUALLY UPDATE CANVAS CONTENT TO NEW VIDEO FRAME AT 20FPS RATE"""
    def update_vid_image(self):
        if self.pause_flag != True:
            ret, frame = self.vid_cap.get_current_frame()

            if ret:
                vidimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not self.saveConfirmationFlag:
                    MsgBox = messagebox.askquestion('Save Video To File','Do you want to save the processed video file to disk', icon = 'question')
                    if MsgBox == 'yes':
                        filename = filename = filedialog.asksaveasfilename(initialdir="/users/callu/documents", title="Select Location To Save Annotated Video", 
                            defaultextension=".*", filetypes=(("avi files", "*.avi"),("all files","*.*")))
                        if filename:
                            self.setVideoWriter(vidimage.shape[:2][0], vidimage.shape[:2][1], filename)
                    self.saveConfirmationFlag = True

                self.canvas.config(width=vidimage.shape[:2][1], height=vidimage.shape[:2][0])

                vidimage = self.network_handler.make_video_prediction(vidimage)

                if self.saveFlag:
                    self.writeFrame(cv2.cvtColor(vidimage, cv2.COLOR_RGB2BGR))

                vidimage = Image.fromarray(vidimage)
                self.img = ImageTk.PhotoImage(vidimage)
                self.canvas.create_image(0,0,anchor=NW,image=self.img)

                self.vid_job = self.root.after(16, self.update_vid_image)

            else:
                self.root.after_cancel(self.vid_job)
                self.input_flag = None
                self.reset_canvas()
                self.network_handler.clear_sequence_analyser()
                if self.saveFlag:
                    self.releaseWriter()

    def setVideoWriter(self, height, width, filename):
        self.videoWriter = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))
        self.saveFlag = True

    def writeFrame(self, frame):
        self.videoWriter.write(frame)

    def releaseWriter(self):
        self.videoWriter.release()

    def reset_canvas(self):
        self.canvas.delete("all")
        self.canvas.config(width=450, height=450)
        self.playButton['state'] = DISABLED
        self.pauseButton['state'] = DISABLED
        self.endButton['state'] = DISABLED
        self.imageButton['state'] = NORMAL
        self.videoButton['state'] = NORMAL
        self.cameraButton['state'] = NORMAL

if __name__ == '__main__':
    app = facial_rec()
    app.start_gui()
