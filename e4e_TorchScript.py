import argparse

import torch
from torchvision import utils
from models.encoders import psp_encoders
from configs.paths_config import model_paths
from torch import nn

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--resave",
        type=bool,
        default=False,
        help="resave torch model",
    )

    args = parser.parse_args()
    
    opts = {'stylegan_size':1024}
    encoder = psp_encoders.Encoder4Editing(50, 'ir_se', opts).to(device)
    
    if args.resave:
        checkpoint = torch.load(args.ckpt)
        encoder.load_state_dict(get_keys(checkpoint, 'encoder'), strict=True)
        torch.save(encoder.state_dict(), "e4e.pt",_use_new_zipfile_serialization=False)
    else:
        checkpoint = torch.load(args.ckpt)
        encoder.load_state_dict(checkpoint)
        encoder.eval()
        #encoder.eval().half()
        sample_i = torch.randn(1, 3, 256, 256, device=device)
        #sample_i = torch.randn(1, 3, 256, 256, device=device).half()
        traced_script_module_encoder = torch.jit.trace(encoder, (sample_i))
        traced_script_module_encoder.save('encoder.pt')
    
    
