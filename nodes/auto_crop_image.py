import torch
import numpy as np
from typing import Tuple


class AutoCropImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "auto_crop"
    CATEGORY = "image/processing"
    
    def auto_crop(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size, height, width, channels = image.shape
        result_images = []
        
        for i in range(batch_size):
            img = image[i]
            
            if channels == 4:
                alpha = img[:, :, 3]
            else:
                alpha = torch.ones((height, width), dtype=image.dtype, device=image.device)
            
            non_zero_indices = torch.nonzero(alpha > 0)
            
            if len(non_zero_indices) == 0:
                result_images.append(img)
                continue
            
            y_min = non_zero_indices[:, 0].min().item()
            y_max = non_zero_indices[:, 0].max().item()
            x_min = non_zero_indices[:, 1].min().item()
            x_max = non_zero_indices[:, 1].max().item()
            
            cropped_img = img[y_min:y_max+1, x_min:x_max+1, :]
            result_images.append(cropped_img)
        
        # Find the maximum dimensions to pad all images to the same size
        max_h = max(img.shape[0] for img in result_images)
        max_w = max(img.shape[1] for img in result_images)
        
        padded_images = []
        for img in result_images:
            h, w = img.shape[:2]
            pad_h = max_h - h
            pad_w = max_w - w
            
            # Pad with zeros (transparent pixels)
            padded_img = torch.nn.functional.pad(img, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
            padded_images.append(padded_img)
        
        return (torch.stack(padded_images),)