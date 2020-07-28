from tkinter import Tk, Frame, LEFT, Label, BOTH, GROOVE, Button, filedialog, PhotoImage, messagebox

import time
import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import dlib
import torch

from hana.rindou.poser.v1.combine_alpha_af_poser_gan_spec import CombineAlphaAfPoserGanSpec
from hana.rindou.poser.v1.poser_gan_apperance_flow_spec import PoserGanWithAppearanceFlowSpec
from hana.rindou.poser.v1.pumarola import Pumarola
from hana.rindou.poser.poser.morph_then_pose_gan_poser import Rindou00MorphThenPoseGanPoser
from hana.rindou.poser.poser.poser import Poser
from hana.rindou.poser.puppeteer.head_pose_solver import HeadPoseSolver
from hana.rindou.poser.puppeteer.poser_pose_converter import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from hana.rindou.poser.v2.generator.combine_two_images_generator_spec import CombineTwoImagesGeneratorSpec
from hana.rindou.poser.v2.generator.combine_two_images_module_spec import CombineTwoImageModuleSpec
from hana.rindou.poser.v2.generator.puaf_select_one_generator_spec import PuafSelectOneGeneratorSpec
from hana.rindou.poser.v2.generator.pumarola_and_appearance_flow_generator_spec import \
    PumarolaAndAppearanceFlowGeneratorSpec
from hana.rindou.poser.v2.generator.pumarola_and_appearance_flow_separate_generator_spec import \
    PumarolaAndApperanceFlowGeneratorSeparateSpec
from hana.rindou.poser.v2.generator.pumarola_generator_spec import PumarolaGeneratorSpec
from hana.rindou.poser.v2.poser.morph_pose_combine_poser import MorphPoseCombinePoser256Param6
from hana.rindou.poser.v2.poser.morph_then_pose_poser_ver2 import Rindou00MorphThenPosePoserVer2
from hana.rindou.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image


class PuppeteerApp:
    def __init__(self,
                 master,
                 poser: Poser,
                 face_detector,
                 landmark_locator,
                 video_capture,
                 torch_device: torch.device):
        self.master = master
        self.poser = poser
        self.face_detector = face_detector
        self.landmark_locator = landmark_locator
        self.video_capture = video_capture
        self.torch_device = torch_device
        self.head_pose_solver = HeadPoseSolver()

        self.master.title("Puppeteer")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        top_frame = Frame(self.master)
        top_frame.pack()

        if True:
            source_image_frame = Frame(top_frame, width=256, height=256)
            source_image_frame.pack_propagate(0)
            source_image_frame.pack(side=LEFT)

            self.source_image_label = Label(source_image_frame, text="Nothing yet!")
            self.source_image_label.pack(fill=BOTH, expand=True)

        if True:
            control_frame = Frame(top_frame, width=256, height=192)
            control_frame.pack_propagate(0)
            control_frame.pack(side=LEFT)

            self.video_capture_label = Label(control_frame, text="Nothing yet!")
            self.video_capture_label.pack(fill=BOTH, expand=True)

        if True:
            posed_image_frame = Frame(top_frame, width=256, height=256)
            posed_image_frame.pack_propagate(0)
            posed_image_frame.pack(side=LEFT, fill='y')

            self.posed_image_label = Label(posed_image_frame, text="Nothing yet!")
            self.posed_image_label.pack(fill=BOTH, expand=True)

        bottom_frame = Frame(self.master)
        bottom_frame.pack(fill='x')

        self.load_source_image_button = Button(bottom_frame, text="Load Image ...", relief=GROOVE,
                                               command=self.load_image)
        self.load_source_image_button.pack(fill='x')

        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None

        self.master.after(1000 // 60, self.update_image())

        self.source_image_label.bind('<Button-1>', self.next_image)
        self.current_image_index = 0
        self.image_file_names = [
            "data/rindou/_20190910/face_images/eudric_256.png",
            "data/rindou/_20190910/face_images/kizuna-ai.png",
            "data/rindou/_20190910/face_images/tokino-sora.png",
            "data/rindou/_20190910/face_images/hanabatake-chaika.png",
            "data/rindou/_20190910/face_images/kongou-iroha.png",
            "data/rindou/_20190910/face_images/waifu_00_256.png",
            "data/rindou/_20190910/face_images/waifu_01_256.png",
            "data/rindou/_20190910/face_images/katsuko/katsuko.png",
            "data/rindou/_20190910/face_images/kiso-azuki.png",
            "data/rindou/_20190910/face_images/suou-patora.png",
            "data/rindou/_20190910/face_images/aduchi-momo.png",
        ]

    def next_image(self, event):
        self.load_image_from_file(self.image_file_names[self.current_image_index])
        self.current_image_index = (self.current_image_index + 1) % len(self.image_file_names)

    def load_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="D:/me/workspace/hana2/data/rindou/illust_for_videos")
        if len(file_name) > 0:
            self.load_image_from_file(file_name)

    def load_image_from_file(self, file_name):
        image = PhotoImage(file=file_name)
        if image.width() != self.poser.image_size() or image.height() != self.poser.image_size():
            message = "The loaded image has size %dx%d, but we require %dx%d." \
                      % (image.width(), image.height(), self.poser.image_size(), self.poser.image_size())
            messagebox.showerror("Wrong image size!", message)
        self.source_image_label.configure(image=image, text="")
        self.source_image_label.image = image
        self.source_image_label.pack()

        self.source_image = extract_pytorch_image_from_filelike(file_name).to(self.torch_device).unsqueeze(dim=0)

    def update_image(self):
        start = time.time()

        there_is_frame, frame = self.video_capture.read()
        if not there_is_frame:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.face_detector(rgb_frame)
        euler_angles = None
        face_landmarks = None
        if len(faces) > 0:
            face_rect = faces[0]
            face_landmarks = self.landmark_locator(rgb_frame, face_rect)
            face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)
            self.draw_face_landmarks(rgb_frame, face_landmarks)
            self.draw_face_box(rgb_frame, face_box_points)
            # self.draw_face_landmarks(blank_frame, face_landmarks)
            # self.draw_face_box(blank_frame, face_box_points)

        resized_frame = cv2.flip(cv2.resize(rgb_frame, (192, 256)), 1)
        pil_image = PIL.Image.fromarray(resized_frame, mode='RGB')
        photo_image = PIL.ImageTk.PhotoImage(image=pil_image)
        self.video_capture_label.configure(image=photo_image, text="")
        self.video_capture_label.image = photo_image
        self.video_capture_label.pack()

        if euler_angles is not None and self.source_image is not None:
            self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
            self.current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
            self.current_pose[1] = max(min(-euler_angles.item(1) / 15.0, 1.0), -1.0)
            self.current_pose[2] = max(min(euler_angles.item(2) / 15.0, 1.0), -1.0)

            if self.last_pose is None:
                self.last_pose = self.current_pose
            else:
                self.current_pose = self.current_pose * 0.5 + self.last_pose * 0.5
                self.last_pose = self.current_pose

            eye_min_ratio = 0.15
            eye_max_ratio = 0.25
            left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
            self.current_pose[3] = 1 - left_eye_normalized_ratio
            right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                            eye_min_ratio,
                                                                            eye_max_ratio)
            self.current_pose[4] = 1 - right_eye_normalized_ratio

            min_mouth_ratio = 0.02
            max_mouth_ratio = 0.3
            mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
            self.current_pose[5] = mouth_normalized_ratio

            self.current_pose = self.current_pose.unsqueeze(dim=0)

            posed_image = self.poser.pose(self.source_image, self.current_pose).detach().cpu()
            numpy_image = rgba_to_numpy_image(posed_image[0])
            pil_image = PIL.Image.fromarray(np.uint8(np.rint(numpy_image * 255.0)), mode='RGBA')
            photo_image = PIL.ImageTk.PhotoImage(image=pil_image)
            self.posed_image_label.configure(image=photo_image, text="")
            self.posed_image_label.image = photo_image
            self.posed_image_label.pack()

        end = time.time()

        self.master.after(1000 // 60, self.update_image)

    def on_closing(self):
        self.video_capture.release()
        self.master.destroy()

    def draw_face_box(self, frame, face_box_points):
        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        for start, end in line_pairs:
            cv2.line(frame, face_box_points[start], face_box_points[end], (255, 0, 0), thickness=2)

    def draw_face_landmarks(self, frame, face_landmarks):
        for i in range(68):
            part = face_landmarks.part(i)
            x = part.x
            y = part.y
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), thickness=2)


if __name__ == "__main__":
    morph_gan_spec = Pumarola(
        pose_size=3,
        bone_parameter_count=0,
        discriminator_mode='reality_score_only')
    pose_gan_spec = PoserGanWithAppearanceFlowSpec(
        pose_size=3,
        bone_parameter_count=3,
        discriminator_mode='do_nothing')
    pumarola_gan_spec = Pumarola(
        pose_size=3,
        bone_parameter_count=3,
        discriminator_mode='do_nothing')
    combine_gan_spec = CombineAlphaAfPoserGanSpec(
        pose_size=3,
        bone_parameter_count=3,
        discriminator_mode='do_nothing')
    cuda = torch.device('cuda')

    poser = Rindou00MorphThenPoseGanPoser(
        morph_gan_spec=morph_gan_spec,
        # pose_gan_spec=pumarola_gan_spec,
        # pose_gan_spec=pose_gan_spec,
        pose_gan_spec=combine_gan_spec,
        morph_gan_generator_file_name="data/rindou/_20190910/pumarola_morph_baseline_01/generator_012.pt",
        # pose_gan_generator_file_name="data/rindou/_20190910/af_rotation_perceptual_00/generator_002.pt",
        # pose_gan_generator_file_name="data/rindou/_20190910/pumarola_rotation_baseline_00/generator_004.pt",
        pose_gan_generator_file_name="data/rindou/_20190910/combine_alpha_af_rotation_00/generator_002.pt",
        device=cuda)

    morph_module_spec = PumarolaGeneratorSpec(image_size=256, pose_size=3)
    morph_module_spec_unet = PumarolaGeneratorSpec(
        pose_size=3,
        body_type='unet')
    pose_module_spec = CombineTwoImagesGeneratorSpec(
        two_images_generator_spec=PumarolaAndAppearanceFlowGeneratorSpec(
            pose_size=3, requires_optimization=True),
        combine_two_images_module_spec=CombineTwoImageModuleSpec(
            pose_size=3, body_type='unet', requires_optimization=True))
    pose_module_spec_03 = CombineTwoImagesGeneratorSpec(
        two_images_generator_spec=PumarolaAndApperanceFlowGeneratorSeparateSpec(
            initial_dim=32, pose_size=3, body_type='unet', requires_optimization=True),
        combine_two_images_module_spec=CombineTwoImageModuleSpec(
            initial_dim=32, pose_size=3, body_type='unet', requires_optimization=True))

    poser_v2 = Rindou00MorphThenPosePoserVer2(
        morph_module_spec=morph_module_spec,
        morph_module_file_name="data/rindou/_20190910/pumarola_morph_baseline_01/generator_012.pt",
        pose_module_spec=pose_module_spec,
        pose_module_file_name="data/rindou/_20190928/combined_discriminator_01/generator_001.pt",
        device=cuda)

    puaf_0_module_spec = PuafSelectOneGeneratorSpec(pose_size=3, selected_index=0)
    puaf_1_module_spec = PuafSelectOneGeneratorSpec(pose_size=3, selected_index=1)
    poser_puaf_1 = Rindou00MorphThenPosePoserVer2(
        morph_module_spec=morph_module_spec,
        morph_module_file_name="data/rindou/_20190910/pumarola_morph_baseline_01/generator_012.pt",
        pose_module_spec=puaf_1_module_spec,
        pose_module_file_name="data/rindou/_20190928/puaf_rotator_01/generator_012.pt",
        device=cuda)

    combine_rotate_with_retouch = CombineTwoImageModuleSpec(
        image_size=256,
        pose_size=3,
        initial_dim=64,
        bottleneck_image_size=32,
        bottleneck_block_count=6,
        initialization_method='he',
        requires_optimization=True,
        body_type='unet')
    three_step_poser = MorphPoseCombinePoser256Param6(
        morph_module_spec=morph_module_spec,
        # morph_module_file_name="data/rindou/_20190910/pumarola_morph_baseline_01/generator_012.pt",
        # morph_module_spec=morph_module_spec_unet,
        morph_module_file_name="data/rindou/_20191104/pumarola_morph_l1_00/generator_012.pt",
        pose_module_spec=PumarolaAndAppearanceFlowGeneratorSpec(pose_size=3),
        pose_module_file_name="data/rindou/_20191104/puaf_rotator_01/generator_006.pt",
        combine_module_spec=combine_rotate_with_retouch,
        # combine_module_file_name="data/rindou/_20191009/combine_with_retouch_l1_01/generator_006.pt",
        combine_module_file_name="data/rindou/_20191104/combine_with_retouch_perceptual_01/generator_006.pt",
        # combine_module_file_name="data/rindou/_20191009/combine_with_retouch_discriminator_01/generator_006.pt",
        device=cuda)

    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor("data/rindou/_20190921/shape_predictor_68_face_landmarks.dat")

    video_capture = cv2.VideoCapture(0)

    master = Tk()
    PuppeteerApp(master, three_step_poser, face_detector, landmark_locator, video_capture, cuda)
    master.mainloop()
