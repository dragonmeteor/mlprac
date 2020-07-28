import cv2
import dlib

from hana.rindou.poser.puppeteer.head_pose_solver import HeadPoseSolver

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/rindou/_20190921/shape_predictor_68_face_landmarks.dat")

cv2.namedWindow("preview")
cap = cv2.VideoCapture(1)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

solver = HeadPoseSolver()
while (ret):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        shape = predictor(rgb_image, d)
        reprojectdst, eulerangles = solver.solve_head_pose(shape)

        for start, end in line_pairs:
            cv2.line(frame, reprojectdst[start], reprojectdst[end], (255, 0, 0))

        # for i in range(68):
        for i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]:
            part = shape.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0))

    cv2.imshow('preview', frame)
    ret, frame = cap.read()
    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv2.destroyWindow("frame")
