# Driver Drowsiness Detection with SNR Enhancement

This project implements a driver drowsiness detection system using the Frame-Level Driver Drowsiness Detection (FL3D) dataset. The system includes an image enhancement pipeline to improve signal-to-noise ratio (SNR) and a deep learning model for drowsiness classification.

## Project Structure

```
.
├── download_data.py      # Script to download FL3D dataset
├── data_utils.py        # Dataset organization and loading utilities
├── enhancement.py       # Image enhancement pipeline
├── model.py            # Drowsiness detection model architecture
├── train.py            # Training script
└── requirements.txt    # Project dependencies
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

## Usage

### Image Enhancement

The enhancement pipeline includes:

- Denoising using Non-Local Means
- Contrast enhancement using CLAHE
- Sharpness enhancement using unsharp masking

### Training

Train the model on raw images:

```bash
python train.py --data_dir /path/to/dataset --save_dir checkpoints
```

Train the model on enhanced images:

```bash
python train.py --data_dir /path/to/dataset --save_dir checkpoints --use_enhanced
```

## Evaluation Metrics

The project tracks several metrics:

- Image Quality: PSNR, SSIM
- Classification: Accuracy, Precision, Recall, F1-score

## Model Architecture

The baseline model uses ResNet18 as a backbone with a custom classification head for binary drowsiness detection.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
