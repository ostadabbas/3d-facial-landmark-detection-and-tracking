from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
import os
from tkinter import filedialog
from tkinter import messagebox
import landmark_detection_video as vLandmark
import numpy as np
import cv2

LARGE_FONT = ("Verdana", 12)

class NNSapp(Frame):
    def __init__(self, master=None):
        pass

    def Main_Window(self, root):
        root.title('Non-Nutrional Suck')
        root.geometry('300x300')
        # self.pack(fill=BOTH, expand=1)

        txt_title = Label(root, text="NNS", font="none 20 bold").pack()
        # txt_title.grid(row=0, column=0, columnspan=4, sticky=NW)
        txt_subtitle = Label(root, text="--Augmented Cognition Lab", font="none 16").pack()
        # txt_subtitle.grid(row=1, column=1, columnspan=4, sticky=NW)

        Button(root, text="Landmarks Tracking", command=self.Tracking_Window).pack()
        Button(root, text="Landmarks Processing", command=self.Processing_Window).pack()
        Button(root, text="Exit", command=root.destroy).pack()


    def Tracking_Window(self):
        tracking = Toplevel()
        tracking.grab_set()
        tracking.title('Landmarks Tracking')
        tracking.geometry('600x150')
        tracking.config()

        self.resource = Text(tracking, width=65, height=1, wrap=WORD, bg="white")
        self.resource.grid(row=0, column=0, columnspan=2, sticky=NW)
        btn_file = Button(tracking, text="Choose Video", command=self.chooseVideo)
        btn_file.grid(row=1, column=0, sticky=NW)
        btn_file = Button(tracking, text="Track Landmarks", command=self.trackLandmarks)
        btn_file.grid(row=2, column=0, sticky=NW)
        btn_file = Button(tracking, text="Close Window", command=tracking.destroy)
        btn_file.grid(row=3, column=0, sticky=NW)

    def Processing_Window(self):
        processing = Toplevel()
        processing.grab_set()
        processing.title('Landmarks Processing')
        processing.geometry('600x500')

        self.landmarks = Text(processing, width=65, height=1, wrap=WORD, bg="white").pack()
        # self.output_res.grid(row=3, column=0, columnspan=2, sticky=NW)
        # btn_file = Button(processing, text="Choose landmarks file", command=self.chooseVideo).pack()
        # btn_file.grid(row=3, column=2, sticky=NW)

    def chooseVideo(self):
        self.filepath = filedialog.askopenfilename(initialdir='.',
                                              filetypes=(("Video File", "*.mov"), ("MP4", "*.mp4"), ("AVI", "*.avi"),
                                                         ("All Files", "*.*")),
                                              title="Choose a file")
        self.resource.delete(0.0, END)
        self.resource.insert(END, self.filepath)
        print(os.getcwd())
        print(os.path.dirname(self.filepath))
        print(os.path.basename(self.filepath).split('.')[0])

    def trackLandmarks(self):
        self.target = os.path.join(os.getcwd(), os.path.basename(self.filepath).split('.')[0])
        if not os.path.exists(self.target):
            os.mkdir(self.target)
        vLandmark.tracking(self.filepath, self.target)





root = Tk()
app = NNSapp()
app.Main_Window(root)
root.mainloop()
