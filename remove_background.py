import cv2
import torch
import numpy as np
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import fire


class BgRemover:
    def __init__(self):
        self.bg_model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        self.bg_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bg_model.to(self.device)

    def get_background_mask(self, input_image):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.bg_model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # create a binary (black and white) mask of the profile foreground
        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        bin_mask = np.where(mask, background, 255).astype(np.uint8)

        return bin_mask

    def make_transparent_background(self, input_image):
        bin_mask = self.get_background_mask(input_image)
        # split the image into channels
        b, g, r = cv2.split(np.array(pic).astype('uint8'))
        # add an alpha channel with and fill all with transparent pixels (max 255)
        a = np.ones(mask.shape, dtype='uint8') * 255
        # merge the alpha channel back
        alpha_im = cv2.merge([b, g, r, a], 4)
        # create a transparent background
        bg = np.zeros(alpha_im.shape)
        # setup the new mask
        new_mask = np.stack([mask, mask, mask, mask], axis=2)
        # copy only the foreground color pixels from the original image where mask is set
        foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

        return foreground
      
    def make_colored_background(self, input_image, color=0):
        solid_image = Image.new('RGB', input_image.size, color)
        mask = Image.fromarray(self.get_background_mask(input_image))
        return Image.composite(solid_image, input_image, mask)
      
def batch_remove(load_path: str, save_path: str, name_format: str = '{}', color: str = 'black'):
    bg_remover = BgRemover()
    load_path = Path(load_path)
    save_path = Path(save_path)
    files_to_load = [*load_path.glob('*.jpg'), *load_path.glob('*.png')]
    for f in tqdm(files_to_load):
        image = Image.open(f)
        new_image = bg_remover.make_colored_background(image, color)
        new_image.save(save_path/name_format.format(f.name))
    
if __name__ == "__main__":
    fire.Fire(batch_remove)
