import os

import torch
from torch.optim import Adam

from data.anime_face.data_loader import anime_face_data_loader
from gans.pggan import PgGanGenerator
from gans.pggan_tasks import PgGanTasks, STABILIZE_PHASE_NAME, TRANSITION_PHASE_NAME
from gans.util import torch_save, torch_load
from gans.zero_gp_loss import ZeroGpLoss
from pytasuku import Workspace


def convert_avs_to_avi(avs_file, avi_file):
    file = open("temp.vdub", "w")
    file.write("VirtualDub.Open(\"%s\");" % avs_file)
    file.write("VirtualDub.video.SetCompression(\"cvid\", 0, 10000, 0);")
    file.write("VirtualDub.SaveAVI(\"%s\");" % avi_file)
    file.write("VirtualDub.Close();")
    file.close()

    os.system("C:\\ProgramData\\chocolatey\\lib\\virtualdub\\tools\\vdub64.exe /i temp.vdub")

    os.remove("temp.vdub")

class TrainingVideoTasks:
    def __init__(self, workspace: Workspace, dir: str, pg_gan_tasks: PgGanTasks, fps=20, subtitle_y=760):
        self.dir = dir
        self.pg_gan_tasks = pg_gan_tasks
        self.workspace = workspace

        self.avs_file_name = self.dir + "/video.avs"
        self.avi_file_name = self.dir + "/video.avi"

        self.sample_image_per_phase = self.pg_gan_tasks.sample_per_save_point \
                                      // self.pg_gan_tasks.sample_per_sample_image
        self.fps = fps
        self.subtitle_y = subtitle_y

    def avs_file_dependencies(self):
        output = []
        size = 4
        while size <= self.pg_gan_tasks.output_image_size:
            if size > 4:
                for i in range(self.pg_gan_tasks.save_point_per_phase):
                    for j in range(self.sample_image_per_phase):
                        output.append(self.pg_gan_tasks.sample_images_file_name(TRANSITION_PHASE_NAME, size, i, j))

            for i in range(self.pg_gan_tasks.save_point_per_phase):
                for j in range(self.sample_image_per_phase):
                    output.append(self.pg_gan_tasks.sample_images_file_name(STABILIZE_PHASE_NAME, size, i, j))
            size *= 2
        return output

    def create_avs_file(self):
        os.makedirs(os.path.dirname(self.avs_file_name), exist_ok=True)
        file = open(self.avs_file_name, "w")

        size = 4
        while size <= self.pg_gan_tasks.output_image_size:
            if size > 4:
                for i in range(self.pg_gan_tasks.save_point_per_phase):
                    for j in range(self.sample_image_per_phase):
                        file_name = os.path.relpath(
                            self.pg_gan_tasks.sample_images_file_name(TRANSITION_PHASE_NAME, size, i, j),
                            self.dir)
                        file.write(
                            "Subtitle(ImageSource(\"%s\", start=0, end=0, fps=%d), \"Transition Phase %dx%d to %dx%d\\n%d images shown\", size=32, align=2, lsp=1, y=%d)" %
                            (file_name, self.fps, size // 2, size // 2, size, size,
                             (i * self.sample_image_per_phase + j) * self.pg_gan_tasks.sample_per_sample_image,
                             self.subtitle_y))
                        file.write(" + \\\n")

            for i in range(self.pg_gan_tasks.save_point_per_phase):
                for j in range(self.sample_image_per_phase):
                    file_name = os.path.relpath(
                        self.pg_gan_tasks.sample_images_file_name(STABILIZE_PHASE_NAME, size, i, j),
                        self.dir)
                    file.write(
                        "Subtitle(ImageSource(\"%s\", start=0, end=0, fps=%d), \"Stabilize Phase %dx%d\\n%d images shown\", size=32, align=2, lsp=1, y=%d)" %
                        (file_name, self.fps, size, size,
                         (i * self.sample_image_per_phase + j) * self.pg_gan_tasks.sample_per_sample_image,
                         self.subtitle_y))
                    if size == self.pg_gan_tasks.output_image_size \
                            and i == self.pg_gan_tasks.save_point_per_phase - 1 \
                            and j == self.sample_image_per_phase - 1:
                        file.write("\n")
                    else:
                        file.write(" + \\\n")
            size *= 2

        file.close()

    def create_avi_file(self):
        convert_avs_to_avi(self.avs_file_name, self.avi_file_name)

    def define_tasks(self):
        self.workspace.create_file_task(self.avs_file_name, self.avs_file_dependencies(),
                                        lambda: self.create_avs_file())
        self.workspace.create_file_task(self.avi_file_name, [self.avs_file_name], lambda: self.create_avi_file())


class InterpolationVideoTasks:
    def __init__(self, workspace: Workspace, dir: str, pg_gan_tasks: PgGanTasks,
                 latent_vector_set_count: int = 10,
                 frames_per_segment: int = 60,
                 fps=30):
        self.workspace = workspace
        self.dir = dir
        self.pg_gan_tasks = pg_gan_tasks

        self.latent_vector_set_count = latent_vector_set_count
        assert self.latent_vector_set_count >= 2

        self.frames_per_segment = frames_per_segment

        self.image_count = self.frames_per_segment * (self.latent_vector_set_count - 1)

        self.avs_file_name = self.dir + "/video.avs"
        self.avi_file_name = self.dir + "/video.avi"

        self.fps = fps

    def latent_vectors_file_name(self, index):
        return self.dir + "/latent_vectors_%03d.pt" % index

    def save_latent_vectors_file(self, index):
        latent_vectors = self.pg_gan_tasks.sample_latent_vectors(self.pg_gan_tasks.sample_image_count)
        torch_save(latent_vectors, self.latent_vectors_file_name(index))

    def image_file_name(self, index):
        return (self.dir + "/images/image_%03d.png") % index

    def create_image_file(self, index):
        segment_index = index / self.frames_per_segment
        frame_index = index % self.frames_per_segment
        latent_vectors_0 = torch_load(self.latent_vectors_file_name(segment_index))
        latent_vectors_1 = torch_load(self.latent_vectors_file_name(segment_index + 1))
        self.pg_gan_tasks.load_rng_state(self.pg_gan_tasks.rng_state_file_name("stabilize", 64, 6))
        alpha = frame_index * 1.0 / self.frames_per_segment
        latent_vectors = latent_vectors_0 * (1.0 - alpha) + latent_vectors_1 * alpha

        G = PgGanGenerator(self.pg_gan_tasks.output_image_size).to(self.pg_gan_tasks.device)
        state_dict = torch_load(self.pg_gan_tasks.final_generator_file_name)
        G.load_state_dict(state_dict)
        G.to(self.pg_gan_tasks.device)

        self.pg_gan_tasks.generate_sample_images_from_latent_vectors(G, latent_vectors,
                                                                     self.pg_gan_tasks.batch_size[
                                                                         self.pg_gan_tasks.output_image_size],
                                                                     self.image_file_name(index))

    def create_avs_file(self):
        os.makedirs(os.path.dirname(self.avs_file_name), exist_ok=True)
        file = open(self.avs_file_name, "w")

        for i in range(self.image_count):
            file.write("ImageSource(\"%s\", start=0, end=0, fps=%d)" % (
                os.path.relpath(self.image_file_name(i), self.dir), self.fps))
            if i == self.image_count-1:
                file.write("\n")
            else:
                file.write(" + \\\n")

        file.close()

    def create_avi_file(self):
        convert_avs_to_avi(self.avs_file_name, self.avi_file_name)

    def define_tasks(self):
        def latent_vectors_func(index):
            def save_it():
                self.save_latent_vectors_file(index)

            return save_it

        latent_vectors_files = []
        for i in range(self.latent_vector_set_count):
            self.workspace.create_file_task(self.latent_vectors_file_name(i),
                                            [],
                                            latent_vectors_func(i))
            latent_vectors_files.append(self.latent_vectors_file_name(i))
        self.workspace.create_command_task(self.dir + "/latent_vectors", latent_vectors_files)

        def image_func(index):
            def save_it():
                self.create_image_file(index)

            return save_it

        image_files = []
        for i in range(self.image_count):
            self.workspace.create_file_task(self.image_file_name(i),
                                            [self.pg_gan_tasks.final_generator_file_name] + latent_vectors_files,
                                            image_func(i))
            image_files.append(self.image_file_name(i))
        self.workspace.create_command_task(self.dir + "/images", image_files)

        self.workspace.create_file_task(self.avs_file_name, [], lambda: self.create_avs_file())
        self.workspace.create_file_task(self.avi_file_name, [self.avs_file_name] + image_files,
                                        lambda: self.create_avi_file())


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')
    pg_gan_tasks = PgGanTasks(
        workspace=workspace,
        dir="data/anime_face_pggan",
        output_image_size=64,
        loss_spec=ZeroGpLoss(grad_loss_weight=100.0, device=cuda),
        data_loader_func=anime_face_data_loader,
        device=cuda,
        generator_learning_rate=1e-4,
        discriminator_learning_rate=3e-4,
        generator_betas=(0.5, 0.9),
        discriminator_betas=(0.5, 0.9))
    pg_gan_tasks.define_tasks()

    TrainingVideoTasks(workspace, pg_gan_tasks.dir + "/training_video", pg_gan_tasks).define_tasks()

    InterpolationVideoTasks(workspace, pg_gan_tasks.dir + "/interpolation", pg_gan_tasks).define_tasks()
