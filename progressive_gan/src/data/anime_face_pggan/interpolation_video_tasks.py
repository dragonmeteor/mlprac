import os

from data.anime_face_pggan.video_util import convert_avs_to_avi
from gans.karras_2017_pggan import PgGanGenerator
from gans.pggan_tasks import PgGanTasks
from gans.util import torch_save, torch_load
from pytasuku import Workspace


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