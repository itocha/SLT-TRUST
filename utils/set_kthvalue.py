import numpy as np
import torch

from utils.slt_modules import PartialFrozenConv2d, PartialFrozenLinear


# According to this issue https://github.com/neuralmagic/sparseml/pull/828,
# it is better to use torch.sort() than torch.kthvalue() if we want to get kthvalue.
def get_kthvalue(param, k):
    sorted_param, _ = param.flatten().sort()
    try:
        return sorted_param[k-1]
    except:
        if k == 0:
            return -1
        else:
            raise ValueError

def set_kthvalue(model, algo, device): #NOTE: this function only works correctly when dtype=torch.float32
    with torch.no_grad():
        t_float32_max   = torch.finfo(torch.float32).max * torch.ones(1).to(device, non_blocking=True)
        if algo == 'local_ep':
            for module in filter(lambda x: isinstance(x, (PartialFrozenConv2d, PartialFrozenLinear)), model.modules()):
                k = (module.score.numel() * module.local_sparsity).int()
                mod_score = module.ternary_frozen_mask * t_float32_max # [-fp32_max, 0, fp32_max]
                mod_score *= t_float32_max                             # [-inf, 0, inf]
                mod_score += module.score.abs()

                module.kthvalue.data[0] = get_kthvalue(mod_score, k)
        elif algo == 'global_ep':
            all_scores = [
                (
                    module.ternary_frozen_mask * t_float32_max * t_float32_max
                    + module.score.abs()
                ).flatten().clone().detach()
                for module in filter(lambda x: isinstance(x, (PartialFrozenConv2d, PartialFrozenLinear)), model.modules())
                ]
            for module in filter(lambda x: isinstance(x, (PartialFrozenConv2d, PartialFrozenLinear)), model.modules()):
                global_sparsity = module.global_sparsity
                break

            all_score = torch.cat(all_scores)

            k = (all_score.numel()*global_sparsity).int()
            kthvalue = get_kthvalue(all_score, k)

            for module in filter(lambda x: isinstance(x, (PartialFrozenConv2d, PartialFrozenLinear)), model.modules()):
                module.kthvalue.data[0] = kthvalue
