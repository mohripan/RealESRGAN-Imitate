import albumentations as A
from albumentations import Compose, OneOf, GaussianBlur, Downscale, GaussNoise
from sinc_filter import SincFilterTransform

# For the resizing, I'll include only bicubic and bilinear since Albumentations does not have "area" method.

# First Order Operation
first_order_augmentations = Compose([
    GaussianBlur(blur_limit=(1, 3), p=1.0), # generalized isotropic Gaussian filter
    OneOf([
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, p=1.0), # downsampling - using bicubic
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=1, p=1.0)  # downsampling - using bilinear
    ], p=1.0),
    GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0) # Gaussian noise
])

# Second Order Operation
second_order_augmentations = Compose([
    GaussianBlur(blur_limit=(1, 3), p=1.0), # blur
    OneOf([
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, p=1.0), # downsampling - using bicubic
        Downscale(scale_min=0.25, scale_max=0.25, interpolation=1, p=1.0)  # downsampling - using bilinear
    ], p=1.0),
    GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0), # noise
    SincFilterTransform(size=5),
])

combined_augmentations = Compose([
    first_order_augmentations,
    second_order_augmentations,
])

# Load an image
image = A.load_image("cc2.jpg")

# Apply data augmentation
augmented_image = combined_augmentations(image=image)

# Save the augmented image
A.save_image(augmented_image, "augmented_image.jpg")