import csv
import os

import numpy
import torch

import PIL.Image

from hana.rindou.poser.dataset.three_step_data import load_three_step_data_tsv
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec
from hana.rindou.util import torch_load, extract_pytorch_image_from_filelike, rgba_to_numpy_image
from pytasuku import Workspace
from pytasuku.workspace import do_nothing


class TwoImagesGeneratorTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 three_step_tasks_prefix: str,
                 module_spec: PoserGanModuleSpec,
                 module_file_name: str,
                 pose_size: int = 6,
                 bone_parameters_count: int = 3,
                 batch_size: int = 32,
                 device=torch.device('cpu')):
        self.workspace = workspace
        self.prefix = prefix
        self.three_step_tasks_prefix = three_step_tasks_prefix
        self.module_spec = module_spec
        self.module_file_name = module_file_name
        self.pose_size = pose_size
        self.bone_parameters_count = bone_parameters_count
        self.batch_size = batch_size
        self.device = device

        for dataset_name in ["train", "validation", "test"]:
            self.define_dataset_tasks(dataset_name)

        self.workspace.create_command_task(
            self.prefix + "/images",
            [self.images_done_file_name(x) for x in ["train", "validation", "test"]],
            do_nothing)
        self.workspace.create_command_task(
            self.prefix + "/data",
            [self.data_file_name(x) for x in ["train", "validation", "test"]],
            do_nothing)

    def define_dataset_tasks(self, dataset_name):
        self.workspace.create_file_task(
            self.images_done_file_name(dataset_name),
            [
                self.three_step_render_done_file_name(dataset_name),
                self.three_step_data_file_name(dataset_name),
            ],
            lambda: self.generate_images(dataset_name))
        self.workspace.create_file_task(
            self.data_file_name(dataset_name),
            [
                self.three_step_data_file_name(dataset_name),
            ],
            lambda: self.generate_data_file(dataset_name)
        )

    def images_done_file_name(self, dataset_name: str):
        return self.prefix + "/" + dataset_name + "/images_done.txt"

    def three_step_render_done_file_name(self, dataset_name: str):
        return self.three_step_tasks_prefix + "/" + dataset_name + "/render_done.txt"

    def three_step_data_file_name(self, dataset_name: str):
        return self.three_step_tasks_prefix + "/" + dataset_name + "/data.tsv"

    def load_image(self, file_name):
        print("Loading %s ..." % file_name)
        with open(file_name, "rb") as file:
            return extract_pytorch_image_from_filelike(file)

    def generate_images(self, dataset_name):
        three_step_data = load_three_step_data_tsv(self.three_step_data_file_name(dataset_name), self.pose_size)
        prefix_length = len(self.three_step_tasks_prefix + "/" + dataset_name + "/")
        module = self.module_spec.get_module().to(self.device)
        module.train(False)
        module.load_state_dict(torch_load(self.module_file_name))

        done_count = 0
        for example in three_step_data:
            target_image_file_name = example[-2]
            target_image_file_name_no_prefix = target_image_file_name[prefix_length:]
            comps = os.path.splitext(target_image_file_name_no_prefix)

            image_file_name_0 = self.prefix + "/" + dataset_name + "/" + comps[0] + "_0" + comps[1]
            image_file_name_1 = self.prefix + "/" + dataset_name + "/" + comps[0] + "_1" + comps[1]

            morphed_image_file_name = example[2]
            pose = torch.tensor(example[3][:3], device=self.device).unsqueeze(0)
            morphed_image = self.load_image(morphed_image_file_name).to(self.device).unsqueeze(0)
            outputs = module(morphed_image, pose)
            numpy_0 = rgba_to_numpy_image(outputs[0][0].detach().cpu())
            numpy_1 = rgba_to_numpy_image(outputs[1][0].detach().cpu())

            print("Saving %s ..." % image_file_name_0)
            pil_image_0 = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_0 * 255.0)), mode='RGBA')
            os.makedirs(os.path.dirname(image_file_name_0), exist_ok=True)
            pil_image_0.save(image_file_name_0)

            print("Saving %s ..." % image_file_name_1)
            pil_image_1 = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_1 * 255.0)), mode='RGBA')
            os.makedirs(os.path.dirname(image_file_name_1), exist_ok=True)
            pil_image_1.save(image_file_name_1)

            done_count += 1
            print(
                "%d of %d done! (%f%%)" % (
                done_count, len(three_step_data), (done_count * 100.0 / len(three_step_data))))

        with open(self.images_done_file_name(dataset_name), 'wt') as fout:
            fout.write("DONE!\n")

    def data_file_name(self, dataset_name):
        return self.prefix + "/" + dataset_name + "/data.tsv"

    def generate_data_file(self, dataset_name: str):
        three_step_data = load_three_step_data_tsv(self.three_step_data_file_name(dataset_name), self.pose_size)
        prefix_length = len(self.three_step_tasks_prefix + "/" + dataset_name + "/")

        os.makedirs(os.path.dirname(self.data_file_name(dataset_name)), exist_ok=True)
        with open(self.data_file_name(dataset_name), 'wt', newline='') as fout:
            tsv_writer = csv.writer(fout, delimiter='\t')
            for example in three_step_data:
                source_image_file_name = example[2]
                target_image_file_name = example[-2]
                target_image_file_name_no_prefix = target_image_file_name[prefix_length:]
                comps = os.path.splitext(target_image_file_name_no_prefix)

                image_file_name_0 = self.prefix + "/" + dataset_name + "/" + comps[0] + "_0" + comps[1]
                image_file_name_1 = self.prefix + "/" + dataset_name + "/" + comps[0] + "_1" + comps[1]
                tsv_writer.writerow([image_file_name_0, image_file_name_1] + example[-3] +
                                    [source_image_file_name, target_image_file_name])
