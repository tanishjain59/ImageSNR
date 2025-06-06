import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class ImageEnhancer:
    def __init__(self, config=None):
        self.config = config or {
            'denoise': True,
            'contrast': True,
            'sharpen': True
        }
    
    def denoise(self, image):
        """Apply Non-Local Means denoising"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def enhance_contrast(self, image):
        """Apply CLAHE for contrast enhancement"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    def sharpen(self, image):
        """Apply unsharp masking"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    def enhance(self, image):
        """Apply the full enhancement pipeline"""
        enhanced = image.copy()
        
        if self.config['denoise']:
            enhanced = self.denoise(enhanced)
        
        if self.config['contrast']:
            enhanced = self.enhance_contrast(enhanced)
            
        if self.config['sharpen']:
            enhanced = self.sharpen(enhanced)
            
        return enhanced
    
    def calculate_metrics(self, original, enhanced):
        """Calculate PSNR and SSIM between original and enhanced images"""
        psnr_value = psnr(original, enhanced)
        # Determine win_size for SSIM
        min_dim = min(original.shape[0], original.shape[1])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3  # minimum allowed by skimage
        ssim_value = ssim(original, enhanced, channel_axis=-1, win_size=win_size)
        return {
            'psnr': psnr_value,
            'ssim': ssim_value
        } 