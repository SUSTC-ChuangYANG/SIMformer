import os
import pickle as pk 



def make_dir(folder_name, path="./"):
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created")
    else:
        print(f"Folder '{folder_path}' already exists")
    return folder_path


def load_data(data_path):
    with open(data_path, "rb") as tar:
        data = pk.load(tar)
    return data

def save_data(data, out_path): 
    with open(out_path, 'wb') as f:
        pk.dump(data, f)