import os
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader

from dataset import RadiotherapyDataset
from model_unet import UNet3D

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")
    
    train_csv = "/net/pr2/projects/plgrid/plggaimed/ZT/KARDIOMEGALIA/data/splits/1731_train.csv"
    test_csv = "/net/pr2/projects/plgrid/plggaimed/ZT/KARDIOMEGALIA/data/splits/1731_test.csv"
    
    train_dataset = RadiotherapyDataset(csv_file=train_csv, is_train=True)
    test_dataset = RadiotherapyDataset(csv_file=test_csv, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Train Samples: {len(train_dataset)} | Test Samples: {len(test_dataset)}")
    
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    criterion = nn.L1Loss() # Mean Absolute Error (MAE) for sharp medical image reconstruction
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 20
    best_loss = float('inf')
    
    print("\n" + "=" * 80)
    print("3D UNet Generative Radiotherapy Training Started...")
    print("=" * 80)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            ct1 = batch["ct1"].to(device)
            ctn = batch["ctn"].to(device)
            
            # Skip samples without valid pair
            if torch.sum(ctn) == 0:
                continue
                
            optimizer.zero_grad()
            pred_ctn = model(ct1)
            loss = criterion(pred_ctn, ctn)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * ct1.size(0)
            
        epoch_train_loss = train_loss / len(train_dataset)
        
        # Validation / Test
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                ct1 = batch["ct1"].to(device)
                ctn = batch["ctn"].to(device)
                
                if torch.sum(ctn) == 0:
                    continue
                    
                pred_ctn = model(ct1)
                loss = criterion(pred_ctn, ctn)
                val_loss += loss.item() * ct1.size(0)
                
                # Save the first test prediction as a NIfTI file for Napari inspection
                if epoch == epochs and i == 0:
                    os.makedirs("predictions", exist_ok=True)
                    pred_np = pred_ctn[0, 0].cpu().numpy()
                    nii_out = nib.Nifti1Image(pred_np, np.eye(4))
                    nib.save(nii_out, "predictions/predicted_ctn.nii.gz")
                    
        epoch_val_loss = val_loss / len(test_dataset)
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            save_flag = "-> [BEST MODEL SAVED]"
        else:
            save_flag = ""
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss (L1): {epoch_train_loss:.4f} | Val Loss (L1): {epoch_val_loss:.4f} {save_flag}")
        print("-" * 80)
        
    print(f"\nTraining Completed! Best Validation Loss: {best_loss:.4f}")
    print("Predicted NIfTI file saved to 'predictions/predicted_ctn.nii.gz' for Napari inspection.")

if __name__ == "__main__":
    train()
