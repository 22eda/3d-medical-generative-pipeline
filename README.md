# Generative Modeling of Anatomical Variations for Quantifying Dose Uncertainty

This repository contains the implementation, training scripts, and initial results for the generative modeling project focused on anatomical variations in CT imaging.

## Project Overview
The primary objective of this project is to leverage generative architectures to synthesize anatomical variations, aiding in the quantification of dose uncertainty in medical radiotherapy planning. By training models on CT datasets, we aim to produce realistic anatomical variations that assist in radiotherapy dose estimation.

## Technical Stack
* **Framework:** PyTorch
* **Data Processing:** `nibabel` for NIfTI medical image handling.
* **Visualization:** `napari` for interactive 3D volume inspection and comparative analysis.
* **Architecture:** U-Net based generative models for image synthesis.
* **Platform:** Athena HPC Cluster (SLURM workload manager).

## Workflow & Implementation
1. **Data Pipeline:** We utilize NIfTI data sourced from the `KARDIOMEGALIA` dataset (`/net/pr2/projects/plgrid/plggaimed/ZT/KARDIOMEGALIA/data/positive/`). Data loading logic is implemented in `dataset.py` to ensure high-fidelity tensor representation.
2. **Training:** Model training is orchestrated via SLURM scripts (`train_generative.py` and `train_gen.slurm`), utilizing GPU acceleration on the Athena cluster.
3. **Inference & Validation:** The `test_save.py` script generates predictions (e.g., `predicted_ctn.nii.gz`), matching them against ground-truth files like `oldest_0002.nii.gz` and `first_0002.nii.gz`.

## Initial Results & Analysis
The visual output compares the model's generated CT scan (`Tahmini CTn`) with the ground truth (`Gerçek CTn`) and the computed absolute difference map.

### Visual Comparison (Difference Map - Magma Colormap)
![Fark Haritası](results/Mutlak%20Fark%20Haritası%20(Pred%20-%20GT).tif)
<img width="518" height="526" alt="Ekran görüntüsü 2026-08-12 004709" src="https://github.com/user-attachments/assets/fd6fb5f0-4e33-4edf-ab7c-92ed138d1157" />


**Interpretation:**
The difference map demonstrates the model's structural accuracy:
* **Low-Error Regions (Dark/Purple):** The model effectively captures the overarching anatomical structure, including the ribcage and lung parenchyma.
* **High-Contrast Regions (Yellow/Bright):** Minor deviations are observed in high-contrast vascular and airway structures, which is expected in 3D generative medical imaging tasks due to subtle pixel-level shifts. 
* **Conclusion:** The model shows high capability in preserving global anatomical validity while maintaining structural consistency, serving as a robust foundation for further refinement of high-frequency details.

## Repository Structure
* `/results` or `/visualizations`: Contains generated result samples, `.tif` outputs, and difference heatmaps.
* `model_unet.py`: Core U-Net architecture definition.
* `test_save.py`: Script for generating and saving predictions.
* `diff_viz.py`: Custom script utilizing `nibabel` and `napari` for comparative visualization and difference mapping.
