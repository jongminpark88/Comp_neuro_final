from model.dependencies import *

def gen_topk_random(sbj_tsnr, k, dim, largest):
    random_idx = torch.rand_like(sbj_tsnr).argsort(dim=dim)
    sbj_tsnr_rndm = torch.take_along_dim(sbj_tsnr, random_idx, dim=dim)
    _, k_random_idx = torch.topk(sbj_tsnr_rndm, k=k, largest=largest, dim=dim)
    k_idx = torch.take_along_dim(random_idx, k_random_idx, dim=dim)
    return k_idx

def get_least_used_gpu(num_gpus):
    pynvml.nvmlInit()
    min_memory = float('inf')
    best_gpu = 0
    
    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory = info.used

        if used_memory < min_memory:
            min_memory = used_memory
            best_gpu = i

    return best_gpu