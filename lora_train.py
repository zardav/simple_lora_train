import tempfile
import yaml
import json
from .utils import run_process


def lora_train(project_name, project_dir, concepts_json_path, train_config_yaml, tmp_path=None):
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    tmp_path = Path(tmp_path)

    train_data_dir = tmp_path / "train_data"
    reg_data_dir = tmp_path / "reg_data"
    train_data_dir.mkdir(exist_ok=True)
    reg_data_dir.mkdir(exist_ok=True)
    
    with open(concepts_json_path) as f:
        d = json.load(f)
    concepts = d['concepts']
    regs = d['regs']
    

    for con in concepts:
        link_from = train_data_dir / f"{con.get('repeats', 1)}_{con['name']}"
        link_to = Path(con['data_dir'])
        link_from.symlink_to(link_to)

    for reg in regs:
        link_from = reg_data_dir / f"{reg.get('repeats', 1)}_{reg['name']}"
        link_to = Path(reg['data_dir']})
        link_from.symlink_to(link_to)
        
    network_weights = ""
    max_epoch = 0
    for p in project_dir.glob('*.safetensors'):
        cur_epoch = int(p.name.split('-')[-1].split('.')[0])
        if cur_epoch > max_epoch:
            max_epoch = cur_epoch
            network_weights = str(p)
    if network_weights:
        print(f"will load network weights from {network_weights}")
    else:
        print(f"creating new network weights")
    
    with open(train_config_yaml) as f:
        train_config = yaml.safe_load(f)
    
    model_path = Path(train_config['model_path']).absolute()
    v2 = train_config['v2']
    
    if not model_path.exists():
        run_process(f"wget {train_config['model_url']} -O {model_path}", shell=True)
    if v2:
        run_process(f"wget https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml {project_dir}/{project_name}.yaml", shell=True)
    
    train_command=f"""accelerate launch --num_cpu_threads_per_process=8 train_network.py \
      {"--v2" if v2 else ""} \
      {"--v_parameterization" if v2 and train_config['v_parameterization'] else ""} \
      --network_dim={train_config['network_dim']} \
      --network_alpha={train_config['network_alpha']} \
      --network_module={train_config['network_module']} \
      {"--network_weights=" + network_weights if network_weights else ""} \
      {"--network_train_unet_only" if train_config['network_train_on'] == "unet_only" else ""} \
      {"--network_train_text_encoder_only" if train_config['network_train_on'] == "text_encoder_only" else ""} \
      --learning_rate={train_config['learning_rate']} \
      {"--unet_lr=" + format(train_config['unet_lr']) if train_config['unet_lr'] !=0 else ""} \
      {"--text_encoder_lr=" + format(train_config['text_encoder_lr']) if train_config['text_encoder_lr'] !=0 else ""} \
      {"--no_metadata" if train_config['no_metadata'] else ""} \
      {"--training_comment=" + train_config['training_comment'] if train_config['training_comment'] and not train_config['no_metadata'] else ""} \
      --lr_scheduler={train_config['lr_scheduler']} \
      {"--lr_scheduler_num_cycles=" + format(train_config['lr_scheduler_num_cycles']) if train_config['lr_scheduler'] == "cosine_with_restarts" else ""} \
      {"--lr_scheduler_power=" + format(train_config['lr_scheduler_power']) if train_config['lr_scheduler'] == "polynomial" else ""} \
      --pretrained_model_name_or_path={model_path} \
      {"--vae=" + train_config['vae'] if train_config['vae'] else ""} \
      {"--caption_extension=" + train_config['caption_extension'] if train_config['caption_extension'] else ""} \
      --train_data_dir={train_data_dir} \
      --reg_data_dir={reg_data_dir} \
      --output_dir={project_dir} \
      --prior_loss_weight={train_config['prior_loss_weight']} \
      {"--resume="+str(train_config['resume_dir']) if Path(train_config['resume_dir']).joinpath("pytorch_model.bin").exists() else ""} \
      {"--output_name=" + project_name if project_name else ""} \
      --mixed_precision={train_config['mixed_precision']} \
      --save_precision={train_config['save_precision']} \
      {"--save_every_n_epochs=" + format(train_config['save_n_epochs_type_value']) if train_config['save_n_epochs_type']=="save_every_n_epochs" else ""} \
      {"--save_n_epoch_ratio=" + format(train_config['save_n_epochs_type_value']) if train_config['save_n_epochs_type']=="save_n_epoch_ratio" else ""} \
      --save_model_as={train_config['save_model_as']} \
      --resolution={train_config['resolution']} \
      {"--enable_bucket" if train_config['enable_bucket'] else ""} \
      {"--min_bucket_reso=" + format(train_config['min_bucket_reso']) if train_config['enable_bucket'] else ""} \
      {"--max_bucket_reso=" + format(train_config['max_bucket_reso']) if train_config['enable_bucket'] else ""} \
      {"--cache_latents" if train_config['cache_latents'] else ""} \
      --train_batch_size={train_config['train_batch_size']} \
      --max_token_length={train_config['max_token_length']} \
      {"--use_8bit_adam" if train_config['use_8bit_adam'] else ""} \
      --max_train_epochs={train_config['num_epochs']} \
      {"--seed=" + format(train_config['seed']) if train_config['seed'] > 0 else ""} \
      {"--gradient_checkpointing" if train_config['gradient_checkpointing'] else ""} \
      {"--gradient_accumulation_steps=" + format(train_config['gradient_accumulation_steps']) } \
      {"--clip_skip=" + format(train_config['clip_skip']) if v2 == False else ""} \
      --logging_dir={tmp_path/'logs'} \
      --log_prefix={project_name} \
      {"--shuffle_caption" if train_config['shuffle_caption'] else ""} \
      --xformers"""
    if not Path("kohya-trainer").exists():
        run_process("git clone https://github.com/Linaqruf/kohya-trainer", shell=True)
    run_process(train_command, shell=True, cwd="kohya-trainer")
    
