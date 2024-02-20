import cv2
import face_recognition as fr
from face_recognition_system import FaceRecognitionAlgo

#Encode faces from a folder
fra = FaceRecognitionAlgo()

#The folder where all the images that can be recognised our stored. Later on we will connect this folder with the database. - Sarthak
fra.load_encoding_images("images/")

#Load Camera. 0 stands for default camera that is the laptop camera. Add up 1 to it as we will start connecting the cctv - Sarthak
cap = cv2.VideoCapture(0)
#If we have 3 cameras then load all - Sarthak

while True:
    ret, frame = cap.read()

    # Detect Faces. Below line is going to give us the frame of the face and name.
    face_location, face_names = fra.detect_known_faces(frame)

    #Face info is an object that consists the information of the name and the color of the frame. - Sarthak
    for face_loc, face_info in zip(face_location, face_names):
        name = face_info['name']
        color = face_info['color']

        #Now we are storing the location of the coordinates
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        #Code to put text.
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        #(0, 0, 200) depicts the color of the rectangle and the last 2 depicts the thickness - Sarthak
        #I have replaced the color code to a variable in the face_recognition_system. If any changes needed kindly go to line  55 & 67

    cv2.imshow("Frame", frame)

    #Since we will be looking at a realtime video hence I am puting the waitkey to 1 for frame to change. If it would have been a image then the value would be 0
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()