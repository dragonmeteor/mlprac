from pytasuku import *
import tasks
import sys
import logging

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python src/run.py <task-name-1> <task-name-2> ...")
        sys.exit(0)

    logging.basicConfig(level=logging.INFO)
    workspace = Workspace()
    tasks.define_tasks(workspace)

    workspace.start_session()
    for arg in sys.argv[1:]:
        workspace.run(arg)
    workspace.end_session()
