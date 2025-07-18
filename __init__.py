from .nodes.auto_crop_image import AutoCropImage
from .nodes.normalize_image_with_rectangle import NormalizeImageWithRectangle

NODE_CLASS_MAPPINGS = {
    "AutoCropImage": AutoCropImage,
    "NormalizeImageWithRectangle": NormalizeImageWithRectangle
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropImage": "Auto Crop Image",
    "NormalizeImageWithRectangle": "Normalize Image With Rectangle"
}