import numpy
import torch
import os
import PIL.Image

from hana.rindou.poser.dataset.three_step_data import load_three_step_data_tsv
from hana.rindou.poser.poser.poser import Poser
from hana.rindou.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks


class PoserResultGenerationTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 poser: Poser,
                 three_step_data_tasks_prefix: str,
                 device: torch.device=torch.device('cpu')):
        self.device = device
        self.three_step_data_tasks_prefix = three_step_data_tasks_prefix
        self.poser = poser
        self.prefix = prefix
        self.workspace = workspace

        self.pose_done_tasks = PoseDoneTasks(self)

    def compute(self):
        os.makedirs(os.path.dirname(self.pose_done_tasks.file_name), exist_ok=True)

        three_step_data = load_three_step_data_tsv(self.three_step_data_file_name(), 6)
        prefix_length = len(self.three_step_data_tasks_prefix + "/")

        done_count = 0
        for example in three_step_data:
            target_image_file_name = example[-2]
            target_image_file_name_no_prefix = target_image_file_name[prefix_length:]
            output_file_name = self.prefix + "/" + target_image_file_name_no_prefix

            rest_image_file_name = example[0]
            rest_image = self.load_image(rest_image_file_name).to(self.device).unsqueeze(0)
            pose = torch.tensor(example[3], device=self.device).unsqueeze(0)
            output = self.poser.pose(rest_image, pose)

            print("Saving %s ..." % output_file_name)
            numpy_0 = rgba_to_numpy_image(output.detach().squeeze().cpu())
            pil_image_0 = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_0 * 255.0)), mode='RGBA')
            os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
            pil_image_0.save(output_file_name)

            done_count += 1
            print(
                "%d of %d done! (%f%%)" % (
                    done_count, len(three_step_data), (done_count * 100.0 / len(three_step_data))))

        with open(self.pose_done_tasks.file_name, "wt") as fout:
            fout.write("DONE!!!\n")

    def three_step_data_file_name(self):
        return self.three_step_data_tasks_prefix + "/data.tsv"

    def load_image(self, file_name):
        print("Loading %s ..." % file_name)
        with open(file_name, "rb") as file:
            return extract_pytorch_image_from_filelike(file)


class PoseDoneTasks(NoIndexFileTasks):
    def __init__(self, poser_result_tasks: PoserResultGenerationTasks):
        super().__init__(
            workspace=poser_result_tasks.workspace,
            prefix=poser_result_tasks.prefix,
            command_name="pose_done",
            define_tasks_immediately=False)
        self.poser_result_tasks = poser_result_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/pose_done.txt"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [],
            self.poser_result_tasks.compute)
