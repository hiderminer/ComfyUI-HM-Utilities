import torch
import torch.nn.functional as F
from typing import Tuple


class NormalizeImageWithRectangle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize_image"
    CATEGORY = "image/processing"
    
    def normalize_image(self, image: torch.Tensor, width: int, height: int) -> Tuple[torch.Tensor]:
        batch_size, img_height, img_width, channels = image.shape
        
        # Calculate scaling factor to fit image within target rectangle while preserving aspect ratio
        scale_x = width / img_width
        scale_y = height / img_height
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions after scaling
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image while preserving aspect ratio
        # Convert to BHWC -> BCHW for interpolation
        image_chw = image.permute(0, 3, 1, 2)
        resized_image = F.interpolate(
            image_chw, 
            size=(new_height, new_width), 
            mode='bilinear', 
            align_corners=False
        )
        # Convert back to BHWC
        resized_image = resized_image.permute(0, 2, 3, 1)
        
        # Create output tensor with target dimensions
        if channels == 3:
            # If input is RGB, output RGBA with alpha channel
            output = torch.zeros((batch_size, height, width, 4), dtype=image.dtype, device=image.device)
            # Set alpha to 0 (transparent) for padding areas
            output[:, :, :, 3] = 0.0
        else:
            # If input already has alpha channel, preserve it
            output = torch.zeros((batch_size, height, width, channels), dtype=image.dtype, device=image.device)
            if channels == 4:
                # Set alpha to 0 (transparent) for padding areas
                output[:, :, :, 3] = 0.0
        
        # Calculate padding to center the resized image
        pad_x = (width - new_width) // 2
        pad_y = (height - new_height) // 2
        
        # Place resized image in the center of output tensor
        if channels == 3:
            # Copy RGB channels and set alpha to 1 for image area
            output[:, pad_y:pad_y+new_height, pad_x:pad_x+new_width, :3] = resized_image
            output[:, pad_y:pad_y+new_height, pad_x:pad_x+new_width, 3] = 1.0
        else:
            # Copy all channels including alpha
            output[:, pad_y:pad_y+new_height, pad_x:pad_x+new_width, :] = resized_image
        
        return (output,)