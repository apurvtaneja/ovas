from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import csv
import numpy as np
import sqlite3
import os
import pause
from Recognize import TrackImages

app = Flask(__name__)

def getImagesAndId(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f)
                  for f in os.listdir(path)]
    faces = []  # create empty face list
    Ids = []  # create empty ID list
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        grayImage = cv2.imread(imagePath,0)
        # Now we are converting the image into numpy array
        imageNp = np.array(grayImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids
    
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        conn = sqlite3.connect('StudentDetails.db')
        c = conn.cursor()
        Id= request.form['uid']
        name = request.form['name']
        c.execute("INSERT INTO student_at(Id,Name) VALUES (?, ?)",(Id,name))
        c.execute("INSERT INTO att_interval(Id) VALUES ("+ Id +")")
        conn.commit()
        c.close()
        conn.close()
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        noOfImages = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.05, 3)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # incrementing sample number
                noOfImages = noOfImages+1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(noOfImages) + ".jpg", gray[y:y+h,x:x+w])
                # display the frame
                cv2.imshow('Taking Images...', img)
            # wait for esc key or q
            if cv2.waitKey(100) and 0xFF == ord('q'):
                break
            # break if the sample number is more than 60. Meaning Images are more than 60.
            elif noOfImages >= 20:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [Id, name]
        
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        
        # It is used to Recognise Face Data
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        # Passing the Id & the Faces Received from getImagesAndId
        faces, Id = getImagesAndId("TrainingImage")
        # Training the LBPH Model
        recognizer.train(faces, np.array(Id))
        # Saving the Trained Model
        recognizer.save("TrainnedModel\Trainner.yml")
        
        return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    
    return render_template('dashboard.html')

@app.route('/marked', methods=['POST', 'GET'])
def marked():
    conn = sqlite3.connect('StudentDetails.db')
    c = conn.cursor()
    sqlite_select_query = """SELECT * from student_at"""
    c.execute(sqlite_select_query)
    records = c.fetchall()
    conn.commit()
    conn.close()
    if request.method == 'POST':
        return redirect(url_for('marked'))
    return render_template('marked.html',records=records)

@app.route('/recognize', methods=['POST', 'GET'])
def recognize():
    if request.method == 'POST':
        return redirect(url_for('recognize'))
    else:
        TrackImages(1)
    
    return render_template('recognize.html')

if __name__ == "__main__":
    app.run(debug=True)
    

    

