import albumentations as A
from albumentations import Compose, OneOf, GaussianBlur, Downscale, GaussNoise
from sinc_filter import SincFilterTransform
from jpeg_compression import JpegCompression
from PIL import Image
import numpy as np

combined_augmentations = Compose([
    # First Order Operation
    GaussianBlur(blur_limit=(1, 3), p=1.0), # generalized isotropic Gaussian filter
    OneOf([
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, p=1.0), # downsampling - using bicubic
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=1, p=1.0)  # downsampling - using bilinear
    ], p=1.0),
    GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0), # Gaussian noise
    JpegCompression(quality_lower=99, quality_upper=100, p=1.0), # JPEG compression
    # Second Order Operation
    GaussianBlur(blur_limit=(1, 3), p=1.0), # blur
    OneOf([
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, p=1.0), # downsampling - using bicubic
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=1, p=1.0)  # downsampling - using bilinear
    ], p=1.0),
    GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0), # noise
    JpegCompression(quality_lower=99, quality_upper=100, p=1.0), # JPEG compression
    SincFilterTransform(size=5),
], p=1)

# Load the image
image = Image.open('DataAugmentation/input.jpg')
image = np.array(image)

# Apply the augmentations
augmented = combined_augmentations(image=image)

# Get the augmented image
augmented_image = augmented['image']

# Save the augmented image
Image.fromarray(augmented_image).save('DataAugmentation/Result/output.jpg')