import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import HAAQI_Net as haaqi_net
import yaml
import argparse
import torch
import torchaudio
from BEATs import BEATs, BEATsConfig

def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)
    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg

def load_BEATs(beats_model_path):
    # load the pre-trained checkpoints
    checkpoint = torch.load(beats_model_path)

    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    
    return BEATs_model

def main(config_path):
    # Arguments
    parser = argparse.ArgumentParser(description="Combine_Net")
    config = yaml_config_hook(config_path)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # Load BEATs model
    beats_model = load_BEATs(args.beats_model_path)

    # Load HAAQI-Net model
    model = getattr(haaqi_net, args.model)(args.input_size, args.hidden_size, args.num_layers, \
        args.dropout, args.linear_output, args.act_fn)        
    print(f'Loading the model from training dic:{args.model_checkpoint_dir}')   
    ckpt = torch.load(args.model_checkpoint_dir+f'best_loss.pth')['model']
    model.load_state_dict(ckpt) 
    model.eval()

    # Load data
    waveform_data, org_sr_data = torchaudio.load('/path/to/your/wav/file.wav')
    waveform_data = torchaudio.functional.resample(waveform_data, orig_freq=org_sr_data, new_freq=16000)
    
    # Extract features
    representation_data, _= beats_model.extract_features(waveform_data)

    # Inference
    listener_audiogram = torch.FloatTensor([[20, 20, 20, 30, 95, 85, 75, 30]]) # example of listener audiogram
    haaqi_score = model(representation_data.permute(0,2,1), listener_audiogram)

    print(f'HAAQI score: {haaqi_score[1].item():.4f}')


if __name__ == "__main__":
    config_path = "hyper.yaml"
    print(config_path)

    main(config_path)
