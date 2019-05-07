from pytasuku import Workspace
import data.anime_face.tasks
import data.anime_face_pggan.tasks

def define_tasks(workspace: Workspace):
    data.anime_face.tasks.define_tasks(workspace)
    data.anime_face_pggan.tasks.define_tasks(workspace)