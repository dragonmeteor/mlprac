from typing import List

import numpy

from hana.rindou.poser.dataset.three_step_data import load_three_step_data_tsv
from hana.rindou.util import extract_numpy_image_from_filelike
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from skimage.metrics import structural_similarity as ssim

from pytasuku.workspace import do_nothing


class EvalResultTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 three_step_data_tasks_prefix: str,
                 result_tasks_prefix: str,
                 evaluation_dependencies: List[str]):
        self.evaluation_dependencies = evaluation_dependencies
        self.result_tasks_prefix = result_tasks_prefix
        self.three_step_data_tasks_prefix = three_step_data_tasks_prefix
        self.prefix = prefix
        self.workspace = workspace

        self.msre_tasks = MsreTasks(self)
        self.ssim_tasks = SsimTasks(self)

        self.eval_command_task_name = self.prefix + "/eval"
        self.workspace.create_command_task(
            self.eval_command_task_name,
            [self.msre_tasks.file_name, self.ssim_tasks.file_name],
            do_nothing)


    def compute_mrse(self):
        self.sum_mrse = 0.0

        def process(target_image_file_name, output_image_file_name):
            target_image = extract_numpy_image_from_filelike(target_image_file_name)
            output_image = extract_numpy_image_from_filelike(output_image_file_name)
            mrse = numpy.mean((target_image - output_image) ** 2)
            self.sum_mrse += mrse

        count = self.iterate_through_dataset(process)
        avg_mrse = self.sum_mrse / count

        with open(self.msre_tasks.file_name, "wt") as fout:
            fout.write("%f\n" % avg_mrse)

    def compute_ssim(self):
        self.sum_ssim = 0.0

        def process(target_image_file_name, output_image_file_name):
            target_image = extract_numpy_image_from_filelike(target_image_file_name)
            output_image = extract_numpy_image_from_filelike(output_image_file_name)
            ss = ssim(target_image, output_image, data_range=1.0, multichannel=True)
            self.sum_ssim += ss

        count = self.iterate_through_dataset(process)
        avg_ssim = self.sum_ssim / count

        with open(self.ssim_tasks.file_name, "wt") as fout:
            fout.write("%f\n" % avg_ssim)

    def three_step_data_file_name(self):
        return self.three_step_data_tasks_prefix + "/data.tsv"

    def iterate_through_dataset(self, func):
        three_step_data = load_three_step_data_tsv(self.three_step_data_file_name(), 6)
        prefix_length = len(self.three_step_data_tasks_prefix + "/")

        done_count = 0
        for example in three_step_data:
            target_image_file_name = example[-2]
            target_image_file_name_no_prefix = target_image_file_name[prefix_length:]
            result_image_file_name = self.result_tasks_prefix + "/" + target_image_file_name_no_prefix
            func(target_image_file_name, result_image_file_name)

            done_count += 1
            print(
                "%d of %d done! (%f%%)" % (
                    done_count, len(three_step_data), (done_count * 100.0 / len(three_step_data))))

        return len(three_step_data)


class MsreTasks(NoIndexFileTasks):
    def __init__(self, eval_result_tasks: EvalResultTasks):
        super().__init__(
            eval_result_tasks.workspace,
            eval_result_tasks.prefix,
            "msre",
            False)
        self.eval_result_tasks = eval_result_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/msre.txt"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            self.eval_result_tasks.evaluation_dependencies,
            self.eval_result_tasks.compute_mrse)


class SsimTasks(NoIndexFileTasks):
    def __init__(self, eval_result_tasks: EvalResultTasks):
        super().__init__(
            eval_result_tasks.workspace,
            eval_result_tasks.prefix,
            "ssim",
            False)
        self.eval_result_tasks = eval_result_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/ssim.txt"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            self.eval_result_tasks.evaluation_dependencies,
            self.eval_result_tasks.compute_ssim)
