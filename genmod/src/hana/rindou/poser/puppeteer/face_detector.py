import cv2
import dlib

from hana.rindou.poser.puppeteer.head_pose_solver import HeadPoseSolver


class FaceDetector:
    def __init__(self,
                 landmark_locator_file: str,
                 image_width=640,
                 image_height=480):
        self.landmark_locator_file = landmark_locator_file
        self.face_detector = None
        self.landmark_locator = None
        self.head_pose_solver = HeadPoseSolver(image_width, image_height)

    def get_face_detector(self):
        if self.face_detector is None:
            self.face_detector = dlib.get_frontal_face_detector()
        return self.face_detector

    def get_landmark_locator(self):
        if self.landmark_locator is None:
            self.landmark_locator = dlib.shape_predictor(self.landmark_locator_file)
        return self.landmark_locator

    def detect_face(self, image):
        face_detector = self.get_face_detector()
        faces = face_detector(image)
        face_rect = faces[0]
        landmark_locator = self.get_landmark_locator()
        face_landmarks = landmark_locator(image, face_rect)
        face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)
        return face_landmarks, face_box_points, euler_angles

    def draw_face_landmarks_and_box(self, image, face_landmarks, face_box_points):
        self.draw_face_landmarks(image, face_landmarks)
        self.draw_face_box(image, face_box_points)

    def draw_face_box(self, image, face_box_points):
        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        for start, end in line_pairs:
            cv2.line(image, face_box_points[start], face_box_points[end], (255, 0, 0), thickness=2)

    def draw_face_landmarks(self, image, face_landmarks):
        for i in range(68):
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)
