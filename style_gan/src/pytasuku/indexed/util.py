import os

def delete_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
        print("[delete] " + file_name)
    else:
        print("[not exist] " + file_name)