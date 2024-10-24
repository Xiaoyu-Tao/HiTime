import os
import torch
import torch.nn as nn

class ts_MLP(nn.Module):
    """ Patch MLP for ts, a simple test network
    """
    def __init__(self, seq_len, in_chans, patch_size=16, embed_dim=768, drop_rate=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_features = embed_dim

        self.patch_emb = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # MLP
        main_net = [
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        ]*2
        main_net.append(nn.Linear(embed_dim, embed_dim))
        
        self.main_net = nn.Sequential(*main_net)
        
        
    def forward(self, x):
        # Input: (B, L, C)
        x = self.patch_emb(x.permute(0,2,1)).permute(0,2,1) # (B, L, C), 
        batch_size, seq_len, _ = x.size()
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)
        x = self.main_net(x)
        return x
    
def create_ts_MLP(seq_len, in_chans, precision="fp16"):
    model = ts_MLP(
        seq_len=seq_len, 
        in_chans=in_chans, 
        patch_size=16, 
        embed_dim=768, 
        drop_rate=0.1
    )  
    model_path = "/data/tingyue/tingyue/TS2LLM/data/tsEncoder/ts_MLP/checkpoints.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")    
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
#     print(incompatible_keys)
    
#     if precision == "fp16":
# #         model.to("cuda") 
#         convert_weights_to_fp16(model)
    return model