import os

def remove_files_from_dir(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            remove_files_from_dir(f"{path}/{d}")
            if os.path.isdir(f"{path}/{d}") == True:
                os.removedirs(f"{path}/{d}")
        for f in files:
            os.remove(f"{path}/{f}")
    if os.path.isdir(path) == True:
        os.removedirs(path)