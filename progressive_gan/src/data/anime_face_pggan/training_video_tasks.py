import os

from data.anime_face_pggan.video_util import convert_avs_to_avi
from gans.pggan_tasks import PgGanTasks, TRANSITION_PHASE_NAME, STABILIZE_PHASE_NAME
from pytasuku import Workspace


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