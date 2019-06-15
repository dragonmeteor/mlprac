import os

from data.anime_face_style_gan.video_util import convert_avs_to_avi
from gans.style_gan_tasks import StyleGanTasks
from gans.util import torch_save, torch_load, load_rng_state
from pytasuku import Workspace


class InterpolationVideoTasks:
    def __init__(self, workspace: Workspace, dir: str, style_gan_tasks: StyleGanTasks,
                 latent_vector_set_count: int = 10,
                 frames_per_segment: int = 60,
                 fps=30):
        self.workspace = workspace
        self.dir = dir
        self.style_gan_tasks = style_gan_tasks

        self.latent_vector_set_count = latent_vector_set_count
        assert self.latent_vector_set_count >= 2

        self.frames_per_segment = frames_per_segment

        self.image_count = self.frames_per_segment * (self.latent_vector_set_count - 1)

        self.avs_file_name = self.dir + "/video.avs"
        self.avi_file_name = self.dir + "/video.avi"

        self.fps = fps

    def latent_vector_file_name(self, index):
        return self.dir + "/latent_vector_%03d.pt" % index

    def save_latent_vectors_file(self, index):
        latent_vectors = self.style_gan_tasks.sample_latent_vectors(self.style_gan_tasks.sample_image_count)
        torch_save(latent_vectors, self.latent_vector_file_name(index))

    def image_file_name(self, index):
        return (self.dir + "/images/image_%03d.png") % index

    def create_image_file(self, index):
        segment_index = index / self.frames_per_segment
        frame_index = index % self.frames_per_segment
        latent_vectors_0 = torch_load(self.latent_vector_file_name(segment_index))
        latent_vectors_1 = torch_load(self.latent_vector_file_name(segment_index + 1))
        load_rng_state(self.style_gan_tasks.stabilize_phases[-1].rng_state_tasks.get_file_name(
            self.style_gan_tasks.save_point_per_phase))
        alpha = frame_index * 1.0 / self.frames_per_segment
        latent_vectors = latent_vectors_0 * (1.0 - alpha) + latent_vectors_1 * alpha

        M = self.style_gan_tasks.style_gan_spec.mapping_module()
        M.load_state_dict(torch_load(self.style_gan_tasks.finished_mapping_module_tasks.file_name))
        M = M.to(self.style_gan_tasks.device)
        G = self.style_gan_tasks.style_gan_spec.generator_module_stabilize(self.style_gan_tasks.output_image_size).to(
            self.style_gan_tasks.device)
        G.load_state_dict(torch_load(self.style_gan_tasks.finished_generator_module_tasks.file_name))
        G = G.to(self.style_gan_tasks.device)

        noise_image = self.style_gan_tasks.load_noise_image()

        self.style_gan_tasks.save_sample_images_from_input_data(M, G,
                                                                latent_vectors,
                                                                noise_image,
                                                                self.style_gan_tasks.batch_size[
                                                                    self.style_gan_tasks.output_image_size],
                                                                self.image_file_name(index))

    def create_avs_file(self):
        os.makedirs(os.path.dirname(self.avs_file_name), exist_ok=True)
        file = open(self.avs_file_name, "w")

        for i in range(self.image_count):
            file.write("ImageSource(\"%s\", start=0, end=0, fps=%d)" % (
                os.path.relpath(self.image_file_name(i), self.dir), self.fps))
            if i == self.image_count - 1:
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
            self.workspace.create_file_task(self.latent_vector_file_name(i),
                                            [],
                                            latent_vectors_func(i))
            latent_vectors_files.append(self.latent_vector_file_name(i))
        self.workspace.create_command_task(self.dir + "/latent_vectors", latent_vectors_files)

        def image_func(index):
            def save_it():
                self.create_image_file(index)

            return save_it

        image_files = []
        for i in range(self.image_count):
            self.workspace.create_file_task(self.image_file_name(i),
                                            [
                                                self.style_gan_tasks.finished_generator_module_tasks.file_name,
                                                self.style_gan_tasks.finished_mapping_module_tasks.file_name
                                            ] + latent_vectors_files,
                                            image_func(i))
            image_files.append(self.image_file_name(i))
        self.workspace.create_command_task(self.dir + "/images", image_files)

        self.workspace.create_file_task(self.avs_file_name, [], lambda: self.create_avs_file())
        self.workspace.create_file_task(self.avi_file_name, [self.avs_file_name] + image_files,
                                        lambda: self.create_avi_file())
