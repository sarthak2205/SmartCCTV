import datetime
import pickle
import time

import face_recognition
import cv2
import os
import glob

import mysql.connector
import numpy as np

import winsound


class FaceRecognitionAlgo:
    def __init__(self):
        self.recognised_face_array = []
        self.recognised_face_name = []
        self.unknown_face_id = 0

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

        # Directory to save unknown faces
        self.unknown_faces_dir = "videos"
        if not os.path.exists(self.unknown_faces_dir):
            os.makedirs(self.unknown_faces_dir)

        # Timer variables
        self.timer_active = False
        self.timer_start_time = None
        self.capture_delay = 10  # Delay in seconds for capturing the unknown faces - Sarthak

        self.video_writer = None
        self.is_recording = False

        # Connecting to mysql Database
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="face_recognition"
        )

        # Checking for conenction
        if self.mydb.is_connected():
            print("Connected to MySQL database")
        else:
            print("Failed to connect to MySQL database")

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.recognised_face_array.append(img_encoding)
            self.recognised_face_name.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding, face_loc in zip(face_encodings, face_locations):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.recognised_face_array, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            # # If a match was found in recognised_face_array, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = recognised_face_name[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.recognised_face_array, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.recognised_face_name[best_match_index]
                if name.startswith('u'):
                    color = (255, 0, 0)  # blue color
                else:
                    color = (0, 255, 0)  # green color

            # Unkown face detection function
            else:
                if not self.timer_active:
                    self.timer_active = True
                    self.timer_start_time = time.time()

                # Check if the timer has exceeded the delay of 5 seconds.
                if self.timer_active and (time.time() - self.timer_start_time) >= self.capture_delay:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    # Ensure that face_loc coordinates are valid
                    unknown_face_path = os.path.join("images", f"unknown_face_{timestamp}.jpg")
                    # winsound.Beep(1500, 2000)
                    cv2.imwrite(unknown_face_path, frame)
                    self.timer_active = False

                    # Video capturing
                    if not self.is_recording:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            video_filename = os.path.join("videos", f"unknown_face_{timestamp}.avi")
                            frame_width, frame_height = frame.shape[1], frame.shape[0]
                            self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0,
                                                                (frame_width, frame_height))
                            self.is_recording = True
                        except Exception as e:
                            print("Error occurred while creating video:", e)

            face_names.append({
                'name': name,
                'color': color
            })

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def detect_known_faces_previously(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding, face_loc in zip(face_encodings, face_locations):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.recognised_face_array, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            # # If a match was found in recognised_face_array, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = recognised_face_name[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.recognised_face_array, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.recognised_face_name[best_match_index]
                if name.startswith('u'):
                    color = (255, 0, 0)  # blue color
                else:
                    color = (0, 255, 0)  # green color

            # Unkown face detection function
            else:
                if not self.timer_active:
                    self.timer_active = True
                    self.timer_start_time = time.time()

                # Check if the timer has exceeded the delay of 5 seconds.
                if self.timer_active and (time.time() - self.timer_start_time) >= self.capture_delay:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    # Ensure that face_loc coordinates are valid
                    unknown_face_path = os.path.join("images", f"unknown_face_{timestamp}.jpg")
                    # winsound.Beep(1500, 2000)
                    cv2.imwrite(unknown_face_path, frame)
                    self.timer_active = False

                    # Video capturing
                    if not self.is_recording:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            video_filename = os.path.join("videos", f"unknown_face_{timestamp}.avi")
                            frame_width, frame_height = frame.shape[1], frame.shape[0]
                            self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0,
                                                                (frame_width, frame_height))
                            self.is_recording = True
                        except Exception as e:
                            print("Error occurred while creating video:", e)

            face_names.append({
                'name': name,
                'color': color
            })

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def detect_faces(self, frame, location):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.recognised_face_array, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            face_distances = face_recognition.face_distance(self.recognised_face_array, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.recognised_face_name[best_match_index]
                if name.startswith('u'):
                    color = (255, 0, 0)  # Blue color
                else:
                    color = (0, 255, 0)  # Green color

            else:
                if not self.timer_active:
                    self.timer_active = True
                    self.timer_start_time = time.time()

                if self.timer_active and (time.time() - self.timer_start_time) >= self.capture_delay:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    unknown_face_path = os.path.join("images", f"unknown_{timestamp}.jpg")
                    cv2.imwrite(unknown_face_path, frame)
                    self.timer_active = False

                    # Save unknown face information to MySQL database
                    cursor = self.mydb.cursor()
                    try:
                        cursor.execute("INSERT INTO alert (encoding, location, time) VALUES (%s, %s, %s)",
                                       (pickle.dumps(face_encoding), location, timestamp))
                        self.mydb.commit()
                        print("Unknown face alert saved to database successfully!")
                    except mysql.connector.Error as err:
                        print("Error inserting unknown face alert into database:\t", err)
                    finally:
                        cursor.close()

            face_names.append({
                'name': name,
                'color': color
            })

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def insert_known_face(self, name, encoding):
        cursor = self.mydb.cursor()
        try:
            # Check if the face already exists the database
            cursor.execute("SELECT * FROM faces WHERE name = %s", (name,))
            if cursor.fetchone() is None:
                # If the face does not exist, insert it into the database
                cursor.execute("INSERT INTO faces (name, encoding) VALUES (%s, %s)",
                               (name, pickle.dumps(encoding)))
                self.mydb.commit()
                print("Known face inserted into database successfully!")
            else:
                print("Known face already exists in the database. Skipping insertion for this.")
        except mysql.connector.Error as err:
            print("Error inserting known face into database:", err)
        finally:
            cursor.close()
