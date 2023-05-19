import albumentations as A

# Define the augmentations to be used for first-order data augmentation
first_order_augmentations = A.Compose([
    A.GaussianBlur(kernel_size=5),
    A.Resize(size=(256, 256)),
    A.AddGaussianNoise(mean=0, std=0.1)
])

# Define the augmentations to be used for second-order data augmentation
second_order_augmentations = A.Compose([
    A.GaussianBlur(kernel_size=5),
    A.Resize(size=(256, 256)),
    A.AddGaussianNoise(mean=0, std=0.1),
    A.SincFilter()
])

# Define the data augmentation pipeline
data_augmentation_pipeline = A.Compose([
    first_order_augmentations,
    second_order_augmentations
])

# Load an image
image = A.load_image("image.jpg")

# Apply data augmentation
augmented_image = data_augmentation_pipeline(image=image)

# Save the augmented image
A.save_image(augmented_image, "augmented_image.jpg")