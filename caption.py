from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import fire


class Captioner:
    def __init__(self):
        self.model = self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_git(self):
        self.processor = AutoProcessor.from_pretrained("microsoft/git-large-r-textcaps")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-textcaps")
      
    def load_blip(self):
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
    def caption(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        inputs = inputs.to(torch.float16)

        generated_ids = self.model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_caption


def batch_caption(load_path: str, save_path: str, model: str = "blip", save_ext: str = "txt")
    captioner = Captioner()
    if model == "blip":
        captioner.load_blip()
    elif model == "git":
        captioner.load_git()
    else:
        raise ValueError(f"unknown model: {model}")
    load_path = Path(load_path)
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    files_to_load = [*load_path.glob('*.jpg'), *load_path.glob('*.png')]
    for f in tqdm(files_to_load):
        image = Image.open(f)
        caption = captioner.caption(image)
        caption_fname = save_path / f"{f.stem}.{save_ext}"
        with open(caption_fname, 'w') as save_f:
            print(caption, file=save_f)

if __name__ == "__main__":
    fire.Fire(batch_caption)
