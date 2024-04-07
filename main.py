import cv2
import face_recognition as fr

from face_recognition_system import FaceRecognitionAlgo

#Encode faces from a folder
fra = FaceRecognitionAlgo()

#The folder where all the images that can be recognised our stored. Later on we will connect this folder with the database. - Sarthak
fra.load_encoding_images("images/")

for name, encoding in zip(fra.recognised_face_name, fra.recognised_face_array):
    fra.insert_known_face(name, encoding)

camera_configurations = [
    {"index": 1, "location": "CCTV 2"},
    {"index": 2, "location": "CCTV 3"},
]

video_capture_objects = []

for config in camera_configurations:
    video_capture_objects.append(cv2.VideoCapture(config["index"]))

#Load Camera. 0 stands for default camera that is the laptop camera. Add up 1 to it as we will start connecting the cctv - Sarthak
#cap = cv2.VideoCapture(1)basab chaudhuri
#If we have 3 cameras then load all - Sarthak

while True:
    frames = []
    for idx, cap in enumerate(video_capture_objects):
        ret, frame = cap.read()

        if ret:
            face_location, face_names = fra.detect_faces(frame, camera_configurations[idx]["location"])

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

            window_name = f"Camera {idx + 1} - {camera_configurations[idx]['location']}"

            cv2.imshow(window_name, frame)

    if frames:
        combined_frames = cv2.hconcat(frames)
        cv2.imshow("Combined Video", combined_frames)

    #Since we will be looking at a realtime video hence I am puting the waitkey to 1 for frame to change. If it would have been a image then the value would be 0
    key = cv2.waitKey(1)
    if key == 27:
        break

for cap in video_capture_objects:
    cap.release()

cv2.destroyAllWindows()