from model.dependencies import *

def gen_topk_random(sbj_tsnr, k, dim, largest):
    random_idx = torch.rand_like(sbj_tsnr).argsort(dim=dim)
    sbj_tsnr_rndm = torch.take_along_dim(sbj_tsnr, random_idx, dim=dim)
    _, k_random_idx = torch.topk(sbj_tsnr_rndm, k=k, largest=largest, dim=dim)
    k_idx = torch.take_along_dim(random_idx, k_random_idx, dim=dim)
    return k_idx