from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(size=224):
    return  [Compose([
                Resize((size, size), interpolation=BICUBIC),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])]
    
