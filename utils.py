import torch
import torch.nn as nn

def make_data_parallel(model:nn.Module):
    
    if torch.cuda.is_available():
        # empty cache
        torch.cuda.empty_cache()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPU.")
        return nn.DataParallel(model)
    else:
        print("Using single GPU (or on CPU).")
        return model