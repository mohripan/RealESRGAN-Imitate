import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Load an image from file
image = Image.open('Image/input.jpg')

# Determine cropping needed to make image dimensions divisible by downscale factor
downscale_factor = 2
width, height = image.size
crop_height = (height % downscale_factor)
crop_width = (width % downscale_factor)

# Crop the image
transform = transforms.Compose([
    transforms.ToTensor(),
])
image = transform(image).unsqueeze(0)

# Remove the most outer part of the image
image = image[:, :, :height-crop_height, :width-crop_width]

# Perform pixel unshuffle
unshuffled_image = F.pixel_unshuffle(image, downscale_factor)

print(unshuffled_image.shape)