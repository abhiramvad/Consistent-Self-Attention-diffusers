import torch
from style_template import styles
from utils import setup_seed
from globals import *

def learn_id_preservation(general_prompt,prompt_array):
    style_name = "Comic book"
    setup_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prompts = [general_prompt+","+prompt for prompt in prompt_array]
    id_prompts = prompts[:id_length]
    real_prompts = prompts[id_length:]
    write = True
    cur_step = 0
    attn_count = 0
    id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    id_images = pipe(id_prompts, num_inference_steps = num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images

    write = False
    # for id_image in id_images:
    #     # display(id_image)
    #     continue
        
def generate_new_images(style_name,general_prompt,negative_prompt,new_prompt_array):
    new_prompts = [general_prompt+","+prompt for prompt in new_prompt_array]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    new_images = []
    for new_prompt in new_prompts :
        cur_step = 0
        new_prompt = apply_style_positive(style_name, new_prompt)
        new_images.append(pipe(new_prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images[0])
    for ind,new_image in enumerate(new_images):
        # display(new_image)
        print("new image ",ind)
        print(new_image)  

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive) 

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

