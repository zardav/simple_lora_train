from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
import fire


def batch_make(load_dir: str, save_dir: str, images_per_prompt: int = 4, replacement: Tuple[str, str] = ("", ""), batch_size = 4):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    load_dir = Path(load_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    negative_prompt = "blurry, low quality, low resolution. mutated hands"
    prompt_files = list(load_dir.glob('*.txt'))
    for file_path in tqdm(prompt_files):
        with open(file_path) as f:
            prompt = f.readline().strip()
        images = []
        while len(images) < images_per_prompt:
            n_images = min(batch_size, images_per_prompt-len(images))
            images += pipe(prompt, guidance_scale=7, negative_prompt=negative_prompt, num_images_per_prompt=n_images).images
        for i, image in enumerate(images):
            save_name = f"{file_path.name}-{i}"
            with open(save_dir/f"{save_name}.txt", 'w') as f:
                print(prompt, file=f)
            image.save(save_dir/f"{save_name}.png")
            
if __name__ == "__main__":
    fire.Fire(batch_make)
