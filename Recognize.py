import cv2, os
import numpy as np
import pandas as pd
import sqlite3
import pause

def TrackImages(at):
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainnedModel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    # StudentDetails = pd.read_csv("StudentDetails\StudentDetails.csv")
    noOfImages = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)

    conn = sqlite3.connect('StudentDetails.db')
    c = conn.cursor()
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
        noOfImages = noOfImages + 1
        #cv2.imwrite("Test\ "+str(noOfImages) + ".jpg", gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            if conf < 50:
                c.execute("SELECT Name FROM student_at WHERE Id = " + str(Id))
                idName = c.fetchone()
                # idName = StudentDetails.loc[StudentDetails['Id'] == Id]['Name'].values
                VideoTag = str(Id) + "-" + str(idName)
                c.execute('UPDATE att_interval SET at' + str(at) + ' = 1 WHERE  Id = '+str(Id))
                conn.commit()
            else:
                Id = 'Unknown'
                VideoTag = str(Id)
            if conf > 75:
                pass
                # noOfFile = len(os.listdir("ImagesUnknown")) + 1
                # cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", img[y:y + h, x:x + w])
            cv2.putText(img, str(VideoTag), (x, y + h), font, 1, (255, 255, 255), 2)
            cv2.putText(img, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 255, 255), 2)
        cv2.imshow('Recognizing Face!!!', img)
        if cv2.waitKey(100) and 0xFF == ord('q'):
            break
        # break if the sample number is more than 60. Meaning Images are more than 60.
        elif noOfImages > 20:
            break
    cam.release()
    cv2.destroyAllWindows()
    if at < 4:
        pause.seconds(2)
        TrackImages(at+1)
    else:
        sqlite_select_query = """SELECT * from att_interval"""
        c.execute(sqlite_select_query)
        records = c.fetchall()
        for row in records:
            Id_At = row[0]
            at1_At = row[1]
            at2_At = row[2]
            at3_At = row[3]
            at4_At = row[4]
            At = row[1] + row[2] + row[3] + row[4]
            if At >= 3:
                c.execute('UPDATE student_at SET Attendance  = 1 WHERE  Id = '+str(Id_At))
        print("Attendence Updated")
        conn.commit()
        conn.close()


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