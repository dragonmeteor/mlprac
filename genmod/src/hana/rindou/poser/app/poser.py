from tkinter import Frame, Label, BOTH, Tk, LEFT, HORIZONTAL, Scale, Button, GROOVE, filedialog, PhotoImage, messagebox

import PIL.Image
import PIL.ImageTk
import numpy
import torch

from hana.rindou.poser.poser.morph_then_pose_gan_poser import Rindou00MorphThenPoseGanPoser
from hana.rindou.poser.poser.poser import Poser
from hana.rindou.poser.v1.combine_alpha_af_poser_gan_spec import CombineAlphaAfPoserGanSpec
from hana.rindou.poser.v1.poser_gan_apperance_flow_spec import PoserGanWithAppearanceFlowSpec
from hana.rindou.poser.v1.pumarola import Pumarola
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


class PoserApp:
    def __init__(self,
                 master,
                 poser: Poser,
                 torch_device: torch.device):
        super().__init__()
        self.master = master
        self.poser = poser
        self.torch_device = torch_device

        self.master.title("Poser")

        source_image_frame = Frame(self.master, width=256, height=256)
        source_image_frame.pack_propagate(0)
        source_image_frame.pack(side=LEFT)

        self.source_image_label = Label(source_image_frame, text="Nothing yet!")
        self.source_image_label.pack(fill=BOTH, expand=True)

        control_frame = Frame(self.master, borderwidth=2, relief=GROOVE)
        control_frame.pack(side=LEFT, fill='y')

        self.param_sliders = []
        for param in self.poser.pose_parameters():
            slider = Scale(control_frame,
                           from_=param.lower_bound,
                           to=param.upper_bound,
                           length=256,
                           resolution=0.001,
                           orient=HORIZONTAL)
            slider.set(param.default_value)
            slider.pack(fill='x')
            self.param_sliders.append(slider)

            label = Label(control_frame, text=param.display_name)
            label.pack()

        posed_image_frame = Frame(self.master, width=256, height=256)
        posed_image_frame.pack_propagate(0)
        posed_image_frame.pack(side=LEFT)

        self.posed_image_label = Label(posed_image_frame, text="Nothing yet!")
        self.posed_image_label.pack(fill=BOTH, expand=True)

        self.load_source_image_button = Button(control_frame, text="Load Image ...", relief=GROOVE,
                                               command=self.load_image)
        self.load_source_image_button.pack(fill='x')

        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)

    def load_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="D:/me/workspace/hana2/data/rindou/illust")
        if len(file_name) > 0:
            image = PhotoImage(file=file_name)
            if image.width() != self.poser.image_size() or image.height() != self.poser.image_size():
                message = "The loaded image has size %dx%d, but we require %dx%d." \
                          % (image.width(), image.height(), self.poser.image_size(), self.poser.image_size())
                messagebox.showerror("Wrong image size!", message)
            self.source_image_label.configure(image=image, text="")
            self.source_image_label.image = image
            self.source_image_label.pack()

            self.source_image = extract_pytorch_image_from_filelike(file_name).to(self.torch_device).unsqueeze(dim=0)
            self.needs_update = True

    def update_pose(self):
        self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
        for i in range(self.pose_size):
            self.current_pose[i] = self.param_sliders[i].get()
        self.current_pose = self.current_pose.unsqueeze(dim=0)

    def update_image(self):
        self.update_pose()
        if (not self.needs_update) and self.last_pose is not None and (
                (self.last_pose - self.current_pose).abs().sum().item() < 1e-5):
            self.master.after(1000 // 30, self.update_image)
            return
        if self.source_image is None:
            self.master.after(1000 // 30, self.update_image)
            return
        self.last_pose = self.current_pose

        posed_image = self.poser.pose(self.source_image, self.current_pose).detach().cpu()
        numpy_image = rgba_to_numpy_image(posed_image[0])
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
        photo_image = PIL.ImageTk.PhotoImage(image=pil_image)

        self.posed_image_label.configure(image=photo_image, text="")
        self.posed_image_label.image = photo_image
        self.posed_image_label.pack()
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)


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
        #morph_module_file_name="data/rindou/_20190910/pumarola_morph_baseline_01/generator_012.pt",
        #morph_module_spec=morph_module_spec_unet,
        morph_module_file_name="data/rindou/_20191104/pumarola_morph_l1_00/generator_012.pt",
        pose_module_spec=PumarolaAndAppearanceFlowGeneratorSpec(pose_size=3),
        pose_module_file_name="data/rindou/_20191104/puaf_rotator_01/generator_012.pt",
        combine_module_spec=combine_rotate_with_retouch,
        #combine_module_file_name="data/rindou/_20191009/combine_with_retouch_l1_01/generator_006.pt",
        combine_module_file_name="data/rindou/_20191104/combine_with_retouch_perceptual_01/generator_006.pt",
        #combine_module_file_name="data/rindou/_20191009/combine_with_retouch_discriminator_01/generator_006.pt",
        device=cuda)

    root = Tk()
    app = PoserApp(master=root, poser=three_step_poser, torch_device=cuda)
    #app = PoserApp(master=root, poser=poser_v2, torch_device=cuda)
    #app = PoserApp(master=root, poser=poser, torch_device=cuda)
    #app = PoserApp(master=root, poser=poser_puaf_1, torch_device=cuda)
    root.mainloop()
