from IO.video_handler import video_handler
from IO.camera_handler import camera_handler
from IO.image_handler import image_handler
from Network.network_handler import networkHandler

from tkinter import *
import tkinter.font as tkFont
from tkinter import filedialog, messagebox, simpledialog

import cv2
from PIL import Image, ImageTk
from os import listdir
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time

class facial_rec:
    """

        facial_rec.py
        -----------
        Used to handle display of GUI and major application functions
    """

    """USED TO INITIALISE GLOBAL CLASS VARIABLES"""
    def __init__(self):

        """WINDOW PROPERTIES AND MAIN WINDOW FRAME"""
        self.root = Tk()
        self.root.title("Facial Emotion Recognition System")
        self.root.config(bg="lightgrey")
        self.root.resizable(False, False) 

        self.fontStyle = tkFont.Font(family="Lucia Grande", size=10)
        self.fontStyleSize = 0
        self.fontStyleFamily = ""
        self.fontStyleColour = ""

        """CAPTURE STREAMS AND JOBS FOR RUNNING INPUT"""
        self.cam_cap = None
        self.cam_job = None

        self.vid_cap = None
        self.vid_job = None
        
        self.img_cap = None

        """NETWORK HANDLER USED TO MANAGE PREDICTION SYSTEM"""
        self.network_handler = networkHandler()

        """STATE FLAGS USED TO MANAGE SYSTEM STATES"""
        self.saveFlag = False
        self.saveConfirmationFlag = False

        """PARAMETERS FOR VIDEO WRITING TO DISK"""
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

        self.main_title = Label(self.mainPanel, text="Facial Emotion Recognition System", font=self.fontStyle, bg="lightgrey")
        self.main_title.pack(pady=10)
 
        self.canvas = Canvas(self.mainPanel, height=450, width=450)
        self.canvas.pack(pady=10, padx=10)

        self.videoControlFrame = Frame(self.mainPanel, bg="lightgrey")
        self.videoControlFrame.pack()

        self.playButton = Button(self.videoControlFrame, text="Play", font=self.fontStyle, width=22, command=lambda: self.button_handler("play"))
        self.playButton.pack(side=LEFT, padx=5, pady=5)

        self.pauseButton = Button(self.videoControlFrame, text="Pause", font=self.fontStyle, width=22, command=lambda: self.button_handler("pause"))
        self.pauseButton.pack(side=LEFT, padx=5, pady=5)

        self.endButton = Button(self.videoControlFrame, text="End", font=self.fontStyle, width=22, command=lambda: self.button_handler("end"))
        self.endButton.pack(side=LEFT, padx=5, pady=5)

        self.playButton['state'] = DISABLED
        self.pauseButton['state'] = DISABLED
        self.endButton['state'] = DISABLED

        self.inputSourceFrame = Frame(self.mainPanel, bg="lightgrey")
        self.inputSourceFrame.pack(fill='x')

        self.imageButton = Button(self.inputSourceFrame, text="Image Input", font=self.fontStyle, width=22, command=lambda: self.button_handler("image"))
        self.imageButton.pack(side=LEFT, padx=5, pady=5)

        self.videoButton = Button(self.inputSourceFrame, text="Video Input", font=self.fontStyle, width=22, command=lambda: self.button_handler("video"))
        self.videoButton.pack(side=LEFT, padx=5, pady=5)

        self.cameraButton = Button(self.inputSourceFrame, text="Camera Input", font=self.fontStyle, width=22, command=lambda: self.button_handler("camera"))
        self.cameraButton.pack(side=LEFT, padx=5, pady=5)

        self.modelSelect = StringVar(self.mainPanel)
        self.modelSelect.set("Prediction Model Select")

        models = []
        files = listdir("models")
        for file in files:
            if file.endswith(".model"):
                models.append(file)
        models.append("Refresh List")


        self.modelSelectMenu = OptionMenu(self.mainPanel, self.modelSelect, *models)
        self.modelSelectMenu.config(font=self.fontStyle)
        self.modelSelectMenu.pack(fill='x', padx=5, pady=5,)

        self.modelSelect.trace("w", self.model_dropdown_select)

        self.menuPanelFrame = Frame(self.mainPanel, bg="lightgrey")
        self.menuPanelFrame.pack(fill='x')

        self.helpButton = Button(self.menuPanelFrame, text="Help", font=self.fontStyle, width=22, command=lambda: self.button_handler("help"))
        self.helpButton.pack(side=LEFT, padx=5, pady=5)

        self.metricsButton = Button(self.menuPanelFrame, text="Metrics", font=self.fontStyle, width=22, command=lambda: self.button_handler("metrics"))
        self.metricsButton.pack(side=LEFT, padx=5, pady=5)

        self.settingsButton = Button(self.menuPanelFrame, text="Settings", font=self.fontStyle, width=22, command=lambda: self.button_handler("settings"))
        self.settingsButton.pack(side=LEFT, padx=5, pady=5)

        """SECONDARY PANEL LAYOUT SPECIFICATIONS"""

        self.helpPanelFrame = Frame(self.secondaryPanel, bg="lightgrey")

        self.help_title = Label(self.helpPanelFrame, text="Help", font=self.fontStyle, bg="lightgrey")
        self.help_title.pack(pady=10)

        self.helpText = Text(self.helpPanelFrame, bg="lightgrey", font=self.fontStyle, bd=0, relief=FLAT)

        helpScroll = Scrollbar(self.helpPanelFrame)
        helpScroll.pack(side=RIGHT, fill=Y)
        self.helpText.pack(side=LEFT, fill='y', padx=5, pady=5)
        helpScroll.config(command=self.helpText.yview)
        self.helpText.config(yscrollcommand=helpScroll.set)

        try:
            with open('Utilities/help.txt', 'r') as file:
                data_text = file.read()
        except:
            print("[ERROR] Cannot find help.txt file in Utilities folder. Please readd file by downloading from GitHub repo at: https://github.com/CallumAltham/Facial-Emotion-Recognition")

        self.helpText.insert(END, data_text)
        self.helpText['state'] = DISABLED

        self.settingsPanelFrame = Frame(self.secondaryPanel, bg="lightgrey")
        
        self.settings_title = Label(self.settingsPanelFrame, text="Settings", font=self.fontStyle, bg="lightgrey")
        self.settings_title.pack(pady=10)

        self.fontsizeselect = StringVar(self.settingsPanelFrame)
        self.fontsizeselect.set("Font Size") # default value

        fontsizes = [8,10,12,14,16,18,20]

        self.fontsizeselectmenu = OptionMenu(self.settingsPanelFrame, self.fontsizeselect, *fontsizes)
        self.fontsizeselectmenu.config(font=self.fontStyle)
        self.fontsizeselectmenu.pack(fill='x', padx=5, pady=5)
        self.fontsizeselect.trace("w", self.fontsize_dropdown_select)

        self.fontfamilyselect = StringVar(self.settingsPanelFrame)
        self.fontfamilyselect.set("Font Family") # default value

        font_families = ['System', 'Arial', 'Arial Black', 'Calibri', 'Comic Sans MS', 'Helvetica', 'Lucia Grande', 'Times New Roman']

        self.fontfamilyselectmenu = OptionMenu(self.settingsPanelFrame, self.fontfamilyselect, *font_families)
        self.fontfamilyselectmenu.config(font=self.fontStyle)
        self.fontfamilyselectmenu.pack(fill='x', padx=5, pady=5)
        self.fontfamilyselect.trace("w", self.fontfamily_dropdown_select)

        self.fontcolourselect = StringVar(self.settingsPanelFrame)
        self.fontcolourselect.set("Font Colour") # default value

        font_colours = ["White", "Black", "Red", "Green", "Blue", "Cyan", "Yellow", "Magenta"]

        self.fontcolourselectmenu = OptionMenu(self.settingsPanelFrame, self.fontcolourselect, *font_colours)
        self.fontcolourselectmenu.config(font=self.fontStyle)
        self.fontcolourselectmenu.pack(fill='x', padx=5, pady=5)
        self.fontcolourselect.trace("w", self.fontcolour_dropdown_select)

        self.settingssaveButton = Button(self.settingsPanelFrame, text="Save Settings", font=self.fontStyle, width=60, command=lambda: self.button_handler("save-settings"))
        self.settingssaveButton.pack(side=RIGHT, padx=5, pady=5)
        self.settingssaveButton['state'] = DISABLED

        self.metricsPanelFrame = Frame(self.secondaryPanel, bg="lightgrey")

        self.metrics_title = Label(self.metricsPanelFrame, text="Metrics", font=self.fontStyle, bg="lightgrey")
        self.metrics_title.pack(pady=10)
        
        fig = Figure(figsize = (5, 5), dpi = 100)
        self.metricsCanvas = FigureCanvasTkAgg(fig, self.metricsPanelFrame)
        self.metricsCanvas.get_tk_widget().pack(pady=10, padx=10)
        self.current_fig = None

        metricsButtonsFrame = Frame(self.metricsPanelFrame, bg="lightgrey")
        metricsButtonsFrame.pack()
        
        self.metricSelect = StringVar(metricsButtonsFrame)
        self.metricSelect.set("Metric Select") # default value

        metrics = ["Confusion Matrix", "Normalized Confusion Matrix", "F-Score, Precision and Recall", "MAE and MSE"]

        self.metricsSelectMenu = OptionMenu(metricsButtonsFrame, self.metricSelect, *metrics)
        self.metricsSelectMenu.config(font=self.fontStyle)
        self.metricsSelectMenu.pack(fill='x', padx=5, pady=5)
        self.metricSelect.trace("w", self.metric_dropdown_select)

        self.saveMetricButton = Button(metricsButtonsFrame, text="Save to Disk", font=self.fontStyle, width=22, command=lambda: self.button_handler("save"))
        self.saveMetricButton.pack(padx=5, pady=5)

    """METHOD USED TO HANDLE SELECTION OF ITEMS WITHIN FONT SIZE SELECTOR DROPDOWN"""
    def fontsize_dropdown_select(self, *args):
        self.fontStyleSize = self.fontsizeselect.get()
        self.settingssaveButton['state'] = NORMAL

    """METHOD USED TO HANDLE SELECTION OF ITEMS WITHIN FONT FAMILY SELECTOR DROPDOWN"""
    def fontfamily_dropdown_select(self, *args):
        self.fontStyleFamily = self.fontfamilyselect.get()
        self.settingssaveButton['state'] = NORMAL

    """METHOD USED TO HANDLE SELECTION OF ITEMS WITHIN FONT COLOUR SELECTOR DROPDOWN"""
    def fontcolour_dropdown_select(self, *args):
        self.fontStyleColour = self.fontcolourselect.get()
        self.settingssaveButton['state'] = NORMAL

    """METHOD USED TO HANDLE SELECTION OF ITEMS WITHIN MODEL SELECTOR DROPDOWN"""
    def model_dropdown_select(self, *args):
        result = self.modelSelect.get()
        if result != "Refresh List":
            model_name = result

            model_pres = False
            try:
                file = open("Utilities/model_config.txt", "r")
                for line in file:
                    items = line.replace("\n", "").split(" ")
                    model_nme = items[0]
                    if model_nme == model_name:
                        model_pres = True
                file.close()
            except:
                print("[ERROR] Cannot access model configuration txt. Please ensure it is available at Utilities/model_config.txt")

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
                    try:
                        file = open("Utilities/model_config.txt", "a")
                        file.write(model_name + " " + height + " " + width + " " + num_classes + " " + classes + "\n")
                        file.close()
                        self.network_handler.load_model(model_name)
                    except:
                        print("[ERROR] Cannot access model configuration txt. Please ensure it is available at Utilities/model_config.txt")
        else:
            menu = self.modelSelectMenu["menu"]
            menu.delete(0, "end")
            models = []
            files = listdir("models")
            for file in files:
                if file.endswith(".model"):
                    models.append(file)
            models.append("Refresh List")
            for model in models:
                menu.add_command(label=model)

    """METHOD USED TO HANDLE SELECTION OF ITEMS WITHIN METRIC SELECTOR DROPDOWN"""    
    def metric_dropdown_select(self, *args):
        metric = self.network_handler.generate_metrics(self.metricSelect.get())
        self.current_fig = metric
        try: 
            self.metricsCanvas.get_tk_widget().pack_forget()
        except AttributeError: 
            pass  
        self.metricsCanvas = FigureCanvasTkAgg(self.current_fig, self.metricsPanelFrame)
        self.metricsCanvas.draw()
        self.metricsCanvas.get_tk_widget().pack(pady=10, padx=10)

    """METHOD USED TO START GUI LOOP AND SET MEDIA PLAYBACK FUNCTIONS AS DISABLED"""
    def start_gui(self):
        self.root.mainloop()
        
    """METHOD USED TO HANDLE FUNCTION OF INDIVIDUAL BUTTONS WITHIN GUI"""
    def button_handler(self, button):
        if button == "camera":
            self.input_flag = "camera"    

            self.imageButton['state'] = DISABLED
            self.videoButton['state'] = DISABLED
            self.cameraButton['state'] = DISABLED

            self.endButton['state'] = NORMAL
            self.modelSelectMenu['state'] = DISABLED
            try:
                self.cam_cap = camera_handler(0)
                self.update_cam_image()
            except:
                print("[ERROR] Cannot access camera, please ensure it is available and accessible")

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
                self.modelSelectMenu['state'] = DISABLED

                try:
                    self.vid_cap = video_handler(filename)
                    self.update_vid_image()
                except:
                    print("[ERROR] Cannot access file, please ensure it is available and directory is accessible")

        if button == "image":
            filename =  filedialog.askopenfilename(initialdir = "/",title = "Select Image file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

            if filename:
                self.imageButton['state'] = DISABLED
                self.videoButton['state'] = DISABLED
                self.cameraButton['state'] = DISABLED
                self.endButton['state'] = NORMAL
                self.modelSelectMenu['state'] = DISABLED
                #try:
                self.img_cap = image_handler(filename)
                
                i, height, width = self.img_cap.load_image()
                i = self.network_handler.make_prediction(i, None)

                
                self.img = Image.fromarray(i)
                self.img = ImageTk.PhotoImage(self.img)

                self.canvas.config(width=width, height=height)
                self.canvas.create_image(0, 0, anchor=NW, image=self.img)

                MsgBox = messagebox.askquestion ('Save Image To File','Do you want to save the annotated image to disk', icon = 'question')
                if MsgBox == 'yes':
                    filename = filedialog.asksaveasfilename(initialdir="/", title="Select Location To Save Annotated Image", 
                    defaultextension=".*", filetypes=(("jpeg files","*.jpg"), ("png files", "*.png"),("all files","*.*")))
                    if filename:
                        try:
                            cv2.imwrite(filename, cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
                        except:
                            print("[ERROR] Cannot write file to disk, please ensure selected directory exists and is accessible")

                #except:
                #    print("[ERROR] Cannot access file, please ensure it is available and directory is accessible")


        if button == "pause":
            self.pause_flag = True
            self.playButton['state'] = NORMAL
            self.pauseButton['state'] = DISABLED

        if button =="play":
            self.pause_flag = False
            self.playButton['state'] = DISABLED
            self.pauseButton['state'] = NORMAL

        if button == "end":

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
                    try:
                        self.current_fig.savefig(filename)
                    except:
                        print("[ERROR] Cannot write file to disk, please ensure selected directory exists and is accessible")
            else:
                MsgBox = messagebox.showwarning('No Metric Selected','Please select a metric to save to disk. A metric can be saved when an image is visible', icon = 'warning')

        if button == "save-settings":
            test = [self.main_title, self.playButton,
                    self.pauseButton, self.endButton,
                    self.imageButton,self.videoButton,
                    self.cameraButton, self.modelSelectMenu,
                    self.helpButton, self.metricsButton,
                    self.settingsButton, self.help_title,
                    self.helpText, self.settings_title,
                    self.fontsizeselectmenu, self.fontfamilyselectmenu,
                    self.fontcolourselectmenu, self.settingssaveButton, 
                    self.metrics_title, self.metricsSelectMenu, self.saveMetricButton]

            if self.fontStyleColour != "":
                for element in test:
                    element["fg"] = self.fontStyleColour
            if self.fontStyleFamily != "":
                self.fontStyle.configure(family=self.fontStyleFamily)
            if self.fontStyleSize != 0:
                self.fontStyle.configure(size=self.fontStyleSize)

            self.settingssaveButton['state'] = DISABLED

    """FUNCTION USED TO CONTINUALLY UPDATE CANVAS CONTENT TO NEW CAMERA FRAME AT 20FPS RATE"""
    def update_cam_image(self):
        frame = self.cam_cap.get_current_frame()[1]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas.config(width=image.shape[:2][1], height=image.shape[:2][0])

        if not self.saveConfirmationFlag:
            MsgBox = messagebox.askquestion('Save Video To File','Do you want to save the processed video file to disk', icon = 'question')
            if MsgBox == 'yes':
                filename = filedialog.asksaveasfilename(initialdir="/", title="Select Location To Save Annotated Video", 
                    defaultextension=".*", filetypes=(("avi files", "*.avi"),("all files","*.*")))
                if filename:
                    self.setVideoWriter(image.shape[:2][0], image.shape[:2][1], filename)
            self.saveConfirmationFlag = True

        image = self.network_handler.make_prediction(image, "video")

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

                vidimage = self.network_handler.make_prediction(vidimage, "video")

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

    """FUNCTION USED TO SET VIDEO WRITER PROPERTIES TO CONTAIN VIDEO HEIGHT, WIDTH AND FILENAME"""
    def setVideoWriter(self, height, width, filename):
        self.videoWriter = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))
        self.saveFlag = True

    """FUNCTION USED TO WRITE INDIVIDUAL VIDEO FRAME TO PREVIOUSLY CREATED VIDEO FILE"""
    def writeFrame(self, frame):
        self.videoWriter.write(frame)

    """FUNCTION USED TO RELEASE VIDEO WRITER FROM MEMORY ONCE VIDEO DATA HAS FINISHED WRITING"""
    def releaseWriter(self):
        self.videoWriter.release()

    """FUNCTION USED TO RESET GUI CANVAS ALONG WITH RESETTING ALL GUI ELEMENTS INTO INITIAL STATES"""
    def reset_canvas(self):
        self.canvas.delete("all")
        self.canvas.config(width=450, height=450)
        self.playButton['state'] = DISABLED
        self.pauseButton['state'] = DISABLED
        self.endButton['state'] = DISABLED
        self.imageButton['state'] = NORMAL
        self.videoButton['state'] = NORMAL
        self.cameraButton['state'] = NORMAL
        self.modelSelectMenu['state'] = NORMAL


if __name__ == '__main__':
    app = facial_rec()
    app.start_gui()
