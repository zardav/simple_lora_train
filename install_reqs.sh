git clone https://github.com/Linaqruf/kohya-trainer
sudo apt -q install liblz4-tool aria2
cd kohya-trainer && pip -q install -r requirements.txt && cd ..
pip -q install -U -I --no-deps https://github.com/camenduru/stable-diffusion-webui-colab/releases/download/0.0.16/xformers-0.0.16+814314d.d20230118-cp38-cp38-linux_x86_64.whl
pip install transformers fire tqdm diffusers opencv-python Pillow
