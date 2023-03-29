import os
import torch


def find_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device", device)
    return device


def make_folders():

    root = '.'
    fig_folder = f"{root}/figures"
    backup_folder = f"{root}/backup"
    
    os.listdir(root)
    for f in fig_folder, backup_folder:
        os.makedirs(f, exist_ok=True)
    
    print("Figures folder:", fig_folder)
    print("Backup folder:", backup_folder)
    return fig_folder, backup_folder
