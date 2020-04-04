from flask import Flask, render_template, url_for, request, redirect
from datetime import datetime
import cv2
import csv
import numpy as np
import sqlite3
import os
import pause
from Recognize import TrackImages, getImagesAndId
from flask.helpers import flash
import glob

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

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
            elif noOfImages >= 42:
                break
        cam.release()
        cv2.destroyAllWindows()
        # row = [Id, name]
        
        # with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
        #     writer = csv.writer(csvFile)
        #     writer.writerow(row)
        # csvFile.close()
        
        # It is used to Recognise Face Data
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        # Passing the Id & the Faces Received from getImagesAndId
        faces, Id = getImagesAndId("TrainingImage")
        # Training the LBPH Model
        recognizer.train(faces, np.array(Id))
        # Saving the Trained Model
        recognizer.save("TrainnedModel\Trainner.yml")
        # files = glob.glob('TrainingImage\*')
        # for f in files:
        #     os.remove(f)
            
        return redirect(url_for('register'))
    return render_template('register.html')


@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    if request.method == 'POST':
        # camera = request.form['camera']
        # time_interval = request.form['time_interval']
        TrackImages(1)
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


if __name__ == "__main__":
    app.run(debug=True)
    

    

