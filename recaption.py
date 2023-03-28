from pathlib import Path
from tqdm import tqdm
import re


def apply_replacements(str1, replacements):
    for _from, _to in replacements:
        str1 = re.sub(_from, _to, str1)
    return str1


def batch_recaption(load_dir: str, save_dir: str, replacements: list, caption_ext: str = 'txt'):
    load_dir = Path(load_dir)
    save_dir = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    files_to_load = [*load_dir.glob('*.jpg'), *load_dir.glob('*.png')]
    for f in tqdm(files_to_load):
        target_img_path = save_dir/f.name
        target_img_path.symlink_to(f)
        orig_text_path = load_dir / f"{f.stem}.{caption_ext}"
        target_text_path = save_dir / f"{f.stem}.{caption_ext}"
        
        text = orig_text_path.read_text()
        new_text = apply_replacements(text, replacements)
        target_text_path.write_text(new_text)
        
