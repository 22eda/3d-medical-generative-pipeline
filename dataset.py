import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset

class RadiotherapyDataset(Dataset):
    def __init__(self, csv_file, base_dir="/net/pr2/projects/plgrid/plggaimed/ZT/KARDIOMEGALIA/data", is_train=True):
        self.base_dir = base_dir
        self.df = pd.read_csv(csv_file)
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def _normalize_hu(self, ct_data):
        # Thorax / Soft tissue windowing: [-1000, 600] HU and Z-score normalization
        ct_clamped = np.clip(ct_data, -1000.0, 600.0)
        mean = np.mean(ct_clamped)
        std = np.std(ct_clamped) + 1e-8
        ct_norm = (ct_clamped - mean) / std
        return ct_norm.astype("float32")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["Filename"]
        label = int(row["label_lvedv"])
        
        sub_folder = "positive" if label == 1 else "negative"
        ct1_path = os.path.join(self.base_dir, sub_folder, filename)
        
        # Load planning CT (CT1)
        ct1_nii = nib.load(ct1_path)
        ct1_data = self._normalize_hu(ct1_nii.get_fdata())
        ct1_tensor = torch.from_numpy(ct1_data).unsqueeze(0) # Shape: [1, D, H, W]
        
        # Load subsequent fraction CT (CTn) if available, otherwise use zeros for unsupervised/generative setup
        newest_filename = filename.replace("first_", "newest_")
        newest_path = os.path.join(self.base_dir, sub_folder, newest_filename)
        
        if os.path.exists(newest_path):
            ctn_nii = nib.load(newest_path)
            ctn_data = self._normalize_hu(ctn_nii.get_fdata())
            ctn_tensor = torch.from_numpy(ctn_data).unsqueeze(0)
        else:
            ctn_tensor = torch.zeros_like(ct1_tensor)
            
        return {
            "ct1": ct1_tensor,
            "ctn": ctn_tensor,
            "filename": filename
        }
