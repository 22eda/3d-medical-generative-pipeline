import torch
import os
import nibabel as nib
import numpy as np
from dataset import RadiotherapyDataset
from model_unet import UNet3D
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_csv = "/net/pr2/projects/plgrid/plggaimed/ZT/KARDIOMEGALIA/data/splits/1731_test.csv"
test_dataset = RadiotherapyDataset(csv_file=test_csv, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UNet3D(in_channels=1, out_channels=1).to(device)
if os.path.exists("best_unet_model.pth"):
    model.load_state_dict(torch.load("best_unet_model.pth"))
    print("Best model weights loaded successfully!")
model.eval()

os.makedirs("predictions", exist_ok=True)
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        ct1 = batch["ct1"].to(device)
        if torch.sum(ct1) == 0:
            continue
        pred_ctn = model(ct1)
        pred_np = pred_ctn[0, 0].cpu().numpy()
        
        nii_out = nib.Nifti1Image(pred_np, np.eye(4))
        nib.save(nii_out, "predictions/predicted_ctn.nii.gz")
        print("Successfully generated and saved: predictions/predicted_ctn.nii.gz")
        break
