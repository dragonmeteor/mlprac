from typing import List
from path import Path
import logging

from pytasuku import Workspace


def create_delete_all_task(workspace: Workspace, name: str, files: List[str]):
    def delete_all():
        for file in files:
            path = Path(file)
            if path.exists():
                logging.info("Removing %s ..." % file)
                path.remove()

    workspace.create_command_task(name, [], delete_all)
