from flask import Flask, request, render_template, jsonify
import os
import pypyodbc
import cv2 
import numpy as np
import math

import argparse
import imutils
import sys

app = Flask(__name__)
 
@app.route("/")
def home():
    return render_template('Login.html')

@app.route('/processLogin', methods=['GET'])
def processLogin():
    emailid= request.args.get('emailid')
    password= request.args.get('password')
    conn1 = pypyodbc.connect('Driver={SQL Server};Server=LAPTOP-O3042TJS;Integrated_Security=true;Database=DeepActionParsing', autocommit=True)
    cur1 = conn1.cursor()
    sqlcmd1 = "SELECT * FROM UserTable WHERE emailid = '"+emailid+"' AND password = '"+password+"' AND isActive = 1"; 
    print(sqlcmd1)
    cur1.execute(sqlcmd1)
    row = cur1.fetchone()
    cur1.commit()
    if not row:
        return render_template('Login.html')
    return render_template('Dashboard.html')

@app.route("/Dashboard")
def Dashboard():
    return render_template('Dashboard.html')

@app.route("/VideoUpload")
def videoUpload():
    return render_template('VideoUpload.html')

@app.route("/ProcessUploadVideo",methods = ['POST'])
def processUploadFile():
    
    file = request.files['videofile']
    f = os.path.join('UPLOADED_VIDEOS', file.filename)
    
    file.save(f)
    return render_template('VideoUpload.html', processResult="Done. Video Uploaded. ")


@app.route("/ParseVideo",methods = ['GET'])
def parseVideo():
        return render_template('ParseVideo.html')
    
@app.route("/ParseVideo",methods = ['POST'])
def processParseVideo():
    vidObj = cv2.VideoCapture(os.path.join('UPLOADED_VIDEOS', "SampleVideo.mp4")) 
    cnt = 0
    isFrameAvailable = 1
    while isFrameAvailable: 
        isFrameAvailable, frameImage = vidObj.read() 
        cv2.imwrite("SampleVideo%d.jpg" % cnt, frameImage) 
        cnt += 1
    if cnt > 1:
        return render_template('ParseVideoResult.html', processResult="Success. Video Parsed. ")
    else:
        return render_template('ParseVideoResult.html', processResult="Error in Parsing")


@app.route("/uploadactivity")
def uploadactivity():
    CLASSES = open('action_recognition_kinetics.txt').read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112

    print("[INFO] loading human activity recognition model...")
    # net = cv2.dnn.readNet(args["model"])
    net = cv2.dnn.readNet('resnet-34_kinetics.onnx')

    print("[INFO] accessing video stream...")
    # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    vs = cv2.VideoCapture('example_activities.mp4' if 'example_activities.mp4' else 0)
    # vs = cv2.VideoCapture(0)
    while True:
        frames = []

        for i in range(0, SAMPLE_DURATION):
            (grabbed, frame) = vs.read()

            if not grabbed:
                print("[INFO] no frame read from stream - exiting")
                sys.exit(0)

            frame = imutils.resize(frame, width=400)
            frames.append(frame)

        blob = cv2.dnn.blobFromImages(frames, 1.0,
                                      (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        net.setInput(blob)
        outputs = net.forward()
        label = CLASSES[np.argmax(outputs)]

        for frame in frames:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.imshow("Activity Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

@app.route("/webcamactivity")
def webcamactivity():
    CLASSES = open('action_recognition_kinetics.txt').read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112

    print("[INFO] loading human activity recognition model...")
    # net = cv2.dnn.readNet(args["model"])
    net = cv2.dnn.readNet('resnet-34_kinetics.onnx')

    print("[INFO] accessing video stream...")
    # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    vs = cv2.VideoCapture(0)
    # vs = cv2.VideoCapture(0)
    while True:
        frames = []

        for i in range(0, SAMPLE_DURATION):
            (grabbed, frame) = vs.read()

            if not grabbed:
                print("[INFO] no frame read from stream - exiting")
                sys.exit(0)

            frame = imutils.resize(frame, width=400)
            frames.append(frame)

        blob = cv2.dnn.blobFromImages(frames, 1.0,
                                      (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        net.setInput(blob)
        outputs = net.forward()
        label = CLASSES[np.argmax(outputs)]

        for frame in frames:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.imshow("Activity Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break




@app.route("/DetectAction")
def DetectAction():

    hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        blur = cv2.GaussianBlur(img,(5,5),0) 
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) 
        retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
        hand = hand_cascade.detectMultiScale(thresh1, 1.3, 5) 
        mask = np.zeros(thresh1.shape, dtype = "uint8") 
        for (x,y,w,h) in hand: 
            cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2) 
            cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)
        img2 = cv2.bitwise_and(thresh1, mask)
        final = cv2.GaussianBlur(img2,(7,7),0)    
        contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        cv2.drawContours(img, contours, 0, (255,255,0), 3)
        cv2.drawContours(final, contours, 0, (255,255,0), 3)
    
        if len(contours) > 0:
            cnt=contours[0]
            hull = cv2.convexHull(cnt, returnPoints=False)

            defects = cv2.convexityDefects(cnt, hull)
            count_defects = 0
            if type(defects) is np.ndarray:
                print("Suspect Action Detected")
                for i in range(defects.shape[0]):
                    p,q,r,s = defects[i,0]
                    finger1 = tuple(cnt[p][0])
                    finger2 = tuple(cnt[q][0])
                    dip = tuple(cnt[r][0])
                    
                    a = math.sqrt((finger2[0] - finger1[0])**2 + (finger2[1] - finger1[1])**2)
                    b = math.sqrt((dip[0] - finger1[0])**2 + (dip[1] - finger1[1])**2)
                    c = math.sqrt((finger2[0] - dip[0])**2 + (finger2[1] - dip[1])**2)
                    # apply cosine rule here
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57.29
                    
                    if angle <= 90:
                        count_defects += 1
            
        #cv2.imshow('img',thresh1)
        cv2.imshow('img1',img)
        #cv2.imshow('img2',img2)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run()

