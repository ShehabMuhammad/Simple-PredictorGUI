import tkinter as tk
from tkinter.filedialog import askopenfilename
import os, time, cv2
from PIL import Image
import numpy as np



class Predictor():
    def __init__(self):
        self.root = tk.Tk();
        self.root.title("Predictor")
        self.button1 = tk.Button(self.root, text="Choose the image.", command=self.Predict ).pack();
        self.ld = ""
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.but = tk.Button(self.root, text="Choose the Classifier. ", command=self.Get).pack(side="bottom")
        self.width = 200;
        self.subjects = open("subjects.txt", "r").read().split("\n");
        self.height = 200;
    def Get(self):
        self.ld = askopenfilename();
        if self.ld != None and self.ld != "":
            try:
                self.recognizer.read(self.ld);
            except:
                print("There was an error parsing the Classifier, \n please be sure it's a valid file.")
            
    def Predict(self):
        if self.ld == "" or self.ld == None:
            return;
        dt = cv2.CascadeClassifier("detector.xml");
        img = cv2.imread(askopenfilename());
        if len(img) == 0: return;
        gi1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            face1 = dt.detectMultiScale(gi1);
        except:
            print("error")
            return
        for (x ,y ,w,h) in face1:
            print(f"{x},{y},{w},{h}")
            try:
                label, conf= self.recognizer.predict(gi1[y:y+h, x:x+w])
            except:
                continue;
            label_text = f" {int(conf)}% sure it's {self.subjects[label]}.";
            cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2);
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)            
            gi1 = cv2.resize(gi1, (self.width, self.height), 3);
        
        if len(face1) > 0: cv2.imshow("Predicted Image..", img );
        else: print("No faces has been detected. ")



if __name__ == '__main__':
    obj = Predictor()

        
