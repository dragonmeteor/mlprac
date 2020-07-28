import abc
import math

import torch


class AbstractPoserPoseConverter:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def convert(self, face_landmarks, euler_angles):
        pass


class OldPoserPoseConverter(AbstractPoserPoseConverter):
    def __init__(self,
                 eye_diff_length_min: float = 4.0,
                 eye_diff_length_range: float = 3.0,
                 mouth_diff_length_min: float = 2.0,
                 mouth_diff_length_range: float = 12.0):
        self.mouth_diff_length_range = mouth_diff_length_range
        self.mouth_diff_length_min = mouth_diff_length_min
        self.eye_diff_length_range = eye_diff_length_range
        self.eye_diff_length_min = eye_diff_length_min

    def convert(self, face_landmarks, euler_angles):
        current_pose = torch.zeros(6)
        current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
        current_pose[1] = max(min(euler_angles.item(1) / 15.0, 1.0), -1.0)
        current_pose[2] = max(min(-euler_angles.item(2) / 15.0, 1.0), -1.0)

        left_eye_diff = face_landmarks.part(38) - face_landmarks.part(40)
        left_eye_diff_length = math.sqrt(left_eye_diff.x ** 2 + left_eye_diff.y ** 2)
        left_eye_params = max(0, min((left_eye_diff_length - self.eye_diff_length_min) / self.eye_diff_length_range, 1))
        current_pose[3] = 1 - left_eye_params

        right_eye_diff = face_landmarks.part(43) - face_landmarks.part(47)
        right_eye_diff_length = math.sqrt(right_eye_diff.x ** 2 + right_eye_diff.y ** 2)
        right_eye_params = max(0,
                               min((right_eye_diff_length - self.eye_diff_length_min) / self.eye_diff_length_range, 1))
        current_pose[4] = 1 - right_eye_params

        mouth_diff = face_landmarks.part(62) - face_landmarks.part(66)
        mouth_diff_length = math.sqrt(mouth_diff.x ** 2 + mouth_diff.y ** 2)
        mouth_params = max(0, min((mouth_diff_length - self.mouth_diff_length_min) / self.mouth_diff_length_range, 1))
        current_pose[5] = mouth_params * 0.95 + 0.05

        return current_pose.unsqueeze(dim=0)


class PoserPoseConverter(AbstractPoserPoseConverter):
    def __init__(self,
                 eye_min_ratio: float = 0.15,
                 eye_max_ratio: float = 0.25,
                 mouth_min_ratio: float = 0.02,
                 mouth_max_ratio: float = 0.3):
        self.mouth_max_ratio = mouth_max_ratio
        self.mouth_min_ratio = mouth_min_ratio
        self.eye_max_ratio = eye_max_ratio
        self.eye_min_ratio = eye_min_ratio

    def convert(self, face_landmarks, euler_angles):
        current_pose = torch.zeros(6)
        current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
        current_pose[1] = max(min(euler_angles.item(1) / 15.0, 1.0), -1.0)
        current_pose[2] = max(min(-euler_angles.item(2) / 15.0, 1.0), -1.0)

        left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks,
                                                                      self.eye_min_ratio,
                                                                      self.eye_max_ratio)
        current_pose[4] = 1 - left_eye_normalized_ratio

        right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks, self.eye_min_ratio,
                                                                        self.eye_max_ratio)
        current_pose[3] = 1 - right_eye_normalized_ratio

        current_pose[5] = compute_mouth_normalized_ratio(face_landmarks, self.mouth_min_ratio, self.mouth_max_ratio)

        return current_pose.unsqueeze(dim=0)


LEFT_EYE_HORIZ_POINTS = [36, 39]
LEFT_EYE_TOP_POINTS = [37, 38]
LEFT_EYE_BOTTOM_POINTS = [41, 40]

RIGHT_EYE_HORIZ_POINTS = [42, 45]
RIGHT_EYE_TOP_POINTS = [43, 44]
RIGHT_EYE_BOTTOM_POINTS = [47, 46]


def compute_left_eye_normalized_ratio(face_landmarks, min_ratio, max_ratio):
    return compute_eye_normalized_ratio(face_landmarks,
                                        LEFT_EYE_HORIZ_POINTS, LEFT_EYE_BOTTOM_POINTS, LEFT_EYE_TOP_POINTS,
                                        min_ratio, max_ratio)


def compute_right_eye_normalized_ratio(face_landmarks, min_ratio, max_ratio):
    return compute_eye_normalized_ratio(face_landmarks,
                                        RIGHT_EYE_HORIZ_POINTS, RIGHT_EYE_BOTTOM_POINTS, RIGHT_EYE_TOP_POINTS,
                                        min_ratio, max_ratio)


def compute_eye_normalized_ratio(face_landmarks, eye_horiz_points, eye_bottom_points, eye_top_points, min_ratio,
                                 max_ratio):
    left_eye_horiz_diff = face_landmarks.part(eye_horiz_points[0]) - face_landmarks.part(eye_horiz_points[1])
    left_eye_horiz_length = math.sqrt(left_eye_horiz_diff.x ** 2 + left_eye_horiz_diff.y ** 2)
    left_eye_top_point = (face_landmarks.part(eye_top_points[0]) + face_landmarks.part(eye_top_points[1])) / 2.0
    left_eye_bottom_point = (face_landmarks.part(eye_bottom_points[0]) + face_landmarks.part(
        eye_bottom_points[1])) / 2.0
    left_eye_vert_diff = left_eye_top_point - left_eye_bottom_point
    left_eye_vert_length = math.sqrt(left_eye_vert_diff.x ** 2 + left_eye_vert_diff.y ** 2)
    left_eye_ratio = left_eye_vert_length / left_eye_horiz_length
    left_eye_normalized_ratio = (min(max(left_eye_ratio, min_ratio), max_ratio) - min_ratio) / (
            max_ratio - min_ratio)
    return left_eye_normalized_ratio


MOUTH_TOP_POINTS = [61, 62, 63]
MOUTH_BOTTOM_POINTS = [67, 66, 65]
MOUTH_HORIZ_POINTS = [60, 64]


def compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio):
    mouth_top_point = (face_landmarks.part(MOUTH_TOP_POINTS[0])
                       + face_landmarks.part(MOUTH_TOP_POINTS[1])
                       + face_landmarks.part(MOUTH_TOP_POINTS[2])) / 3.0
    mouth_bottom_point = (face_landmarks.part(MOUTH_BOTTOM_POINTS[0])
                          + face_landmarks.part(MOUTH_BOTTOM_POINTS[1])
                          + face_landmarks.part(MOUTH_BOTTOM_POINTS[2])) / 3.0
    mouth_vert_diff = mouth_top_point - mouth_bottom_point
    mouth_vert_length = math.sqrt(mouth_vert_diff.x ** 2 + mouth_vert_diff.y ** 2)
    mouth_horiz_diff = face_landmarks.part(MOUTH_HORIZ_POINTS[0]) - face_landmarks.part(MOUTH_HORIZ_POINTS[1])
    mouth_horiz_length = math.sqrt(mouth_horiz_diff.x ** 2 + mouth_horiz_diff.y ** 2)
    mouth_ratio = mouth_vert_length / mouth_horiz_length
    mouth_normalized_ratio = (min(max(mouth_ratio, min_mouth_ratio), max_mouth_ratio) - min_mouth_ratio) / (
            max_mouth_ratio - min_mouth_ratio)
    return mouth_normalized_ratio
