import os

def find_question_files(base_folder, task_name):
    """
    Recursively walks through the given folder and returns a list of files
    that end with '_{task_name}_quetions.json'.
    """
    question_files = []
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith(f'_{task_name}_questions.json'):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))

                # Here we'll collect them in a list:
                question_files.append(os.path.join(root, file_name))
                # info_files.append(os.path.join(root, root.split('/')[-1]+".json")) # extract the dataset name from '{dataset_name}_shortest_path_questions.json' file
    return question_files

def find_txt_files(base_folder):
    """
    Recursively walks through the given folder and returns a list of files
    that do NOT end with '.txt.gz'.
    """
    txt_files = []
    
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith('.txt'):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))
                
                # Here we'll collect them in a list:
                txt_files.append(os.path.join(root, file_name))
    
    return txt_files

def check_file_existence(file_name_list):
    file_existence_flag = True
    for file in file_name_list:
        if not os.path.exists(file):
            file_existence_flag = False
            print(f'The file at {file} does not exits')
    return file_existence_flag

def find_files(base_folder, endswith: str):
    """
    Recursively walks through the given folder and returns a list of files
    that end with '_{task_name}_quetions.json'.
    """
    question_files = []
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith(endswith):
                # If you only want to print them, you could do:
                # print(os.path.join(root, file_name))

                # Here we'll collect them in a list:
                question_files.append(os.path.join(root, file_name))
                # info_files.append(os.path.join(root, root.split('/')[-1]+".json")) # extract the dataset name from '{dataset_name}_shortest_path_questions.json' file
    return question_files

def strip_metadata(files: list[str]) -> list[str]:
    """Remove complexity/label metadata JSON files from *files*."""
    blocked = ("complexity.json", "labels.json", "label.json")
    return [f for f in files if not f.endswith(blocked)]