# Frame-Level Driver Drowsiness Detection (FL3D) with SNR Enhancement

This project implements a driver drowsiness detection system using the FL3D dataset. It features an advanced image enhancement pipeline to boost signal-to-noise ratio (SNR) and a deep learning model for robust binary classification of driver alertness.

## Results


## Project Structure

```
.
├── download_data.py      # Download FL3D dataset
├── data_utils.py         # Dataset organization and loading utilities
├── enhancement.py        # Image enhancement pipeline
├── model.py              # Drowsiness detection model (ResNet18)
├── train.py              # Training script
├── visualize_results.py  # Visualization and evaluation
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the dataset:
   ```bash
   python download_data.py
   ```

## Image Enhancement Pipeline

- **Denoising:** Non-Local Means
- **Contrast Enhancement:** CLAHE
- **Sharpness Enhancement:** Unsharp masking

## Training

Train for 30 epochs on either raw or enhanced images:

- **Raw images:**
  ```bash
  python train.py --data_dir data/classification_frames --save_dir checkpoints --epochs 30
  ```
- **Enhanced images:**
  ```bash
  python train.py --data_dir data/classification_frames --save_dir checkpoints --epochs 30 --use_enhanced
  ```
- Models are saved as `checkpoints/best_model_raw.pth` and `checkpoints/best_model_enhanced.pth`.

## Evaluation & Visualization

- Evaluate and compare both models on the **validation and test sets**.
- Visualize classification metrics and side-by-side image results:
  ```bash
  python visualize_results.py --raw_model checkpoints/best_model_raw.pth --enhanced_model checkpoints/best_model_enhanced.pth
  ```
- Metrics and visualizations are saved in the `visualization_results/` directory.

## Metrics

- **Image Quality:**
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
- **Classification:**
  - Accuracy, Precision, Recall, F1-score (on val/test sets)

## Model Architecture

- ResNet18 backbone with a custom binary classification head (sigmoid activation)
- Class-weighted binary cross-entropy loss to address class imbalance

## License

This project is licensed under the MIT License. See the LICENSE file for details.
