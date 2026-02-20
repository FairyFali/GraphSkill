import os

def get_folders_in_directory(directory_path):
    """
    Returns a list of folder names in the specified directory_path.
    """
    folders = []
    # listdir returns files/folders in the directory (not recursively)
    for item in os.listdir(directory_path):
        full_path = os.path.join(directory_path, item)
        if os.path.isdir(full_path):
            folders.append(item)
    return folders