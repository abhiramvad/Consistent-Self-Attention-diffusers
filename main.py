"""
Example usage
"""
import torch
import numpy as np
import prompts
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from utils import set_attention_processor
from consistent_self_attention import SpatialAttnProcessor2_0
from style_template import styles
from prompts import *
from inference import *
from globals import *

# load model 
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float32, use_safetensors=False)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
pipe = pipe.to(device)
unet = pipe.unet
set_attention_processor(unet)

learn_id_preservation(general_prompt,prompt_array)
generate_new_images(styles["Digital/Oil Painting"],general_prompt,negative_prompt,new_prompt_array)

