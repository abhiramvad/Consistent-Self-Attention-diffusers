
import torch
import numpy as np
import random

from consistent_self_attention import SpatialAttnProcessor2_0
from diffusers import AttnProcessor
from globals import *


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_attention_processor(unet,id_length):
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks") :
                attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
            else:    
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(attn_procs)

def cal_attn_mask(total_length,id_length,sa16,sa32,sa64,device="cuda",dtype= torch.float16):
    bool_matrix256 = torch.rand((1, total_length * 256),device = device,dtype = dtype) < sa16
    bool_matrix1024 = torch.rand((1, total_length * 1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * 4096),device = device,dtype = dtype) < sa64
    bool_matrix256 = bool_matrix256.repeat(total_length,1)
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix256[i:i+1,id_length*256:] = False
        bool_matrix1024[i:i+1,id_length*1024:] = False
        bool_matrix4096[i:i+1,id_length*4096:] = False
        bool_matrix256[i:i+1,i*256:(i+1)*256] = True
        bool_matrix1024[i:i+1,i*1024:(i+1)*1024] = True
        bool_matrix4096[i:i+1,i*4096:(i+1)*4096] = True
    mask256 = bool_matrix256.unsqueeze(1).repeat(1,256,1).reshape(-1,total_length * 256)
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,1024,1).reshape(-1,total_length * 1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,4096,1).reshape(-1,total_length * 4096)
    return mask256,mask1024,mask4096