global write
global  sa16,sa32, sa64
global height,width
global pipe,guidance_scale
global attn_procs,unet
global sd_model_path 
global attn_count, total_count, id_length, total_length,cur_step, cur_model_type,num_steps
global mask256,mask1024,mask4096
global models_dict
global seed
global DEFAULT_STYLE_NAME

###
write = False
### strength of consistent self-attention: the larger, the stronger
sa32 = 0.5
sa64 = 0.5
### Res. of the Generated Comics. Please Note: SDXL models may do worse in a low-resolution! 
height = 768
width = 768
attn_procs = {}
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
device="cuda"
guidance_scale = 5.0
seed = 2047
sa32 = 0.5
sa64 = 0.5
id_length = 4
num_steps = 50
DEFAULT_STYLE_NAME = "(No style)"
sd_model_path  = "stabilityai/stable-diffusion-xl-base-1.0"
