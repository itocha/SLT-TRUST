import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        out = torch.where(
            scores < threshold, zeros.to(scores.device), ones.to(scores.device)
        )
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

class SparseModule(nn.Module):
    def init_param_(
        self,
        param,
        init_mode=None,
        scale=None,
        sparse_value=None,
        gain="relu",
        args=None,
    ):
        if init_mode == "kaiming_normal":
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity=gain)
            param.data *= scale
        elif init_mode == "uniform":
            nn.init.uniform_(param, a=-1, b=1)
            param.data *= scale
        elif init_mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity=gain)
            param.data *= scale
        elif init_mode == "kaiming_normal_SF":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain(gain)
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            param.data.normal_(0, std)
        elif init_mode == "signed_constant":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain(gain)
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale
        elif init_mode == "signed_kaiming_constant":
            fan = nn.init._calculate_correct_fan(param, "fan_in")
            gain = nn.init.calculate_gain(gain)
            scale_fan = fan * (1 - sparse_value)
            std = gain / math.sqrt(scale_fan)
            nn.init.kaiming_normal_(param)  # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale  # scale value is defined in defualt as 1.0
        elif init_mode == "signed_xavier_uniform_constant_SF":
            gain = nn.init.calculate_gain(gain)
            nn.init.xavier_uniform_(param, gain)
            std = torch.std(param)
            scaled_std = std * math.sqrt(1 / (1 - sparse_value))
            nn.init.kaiming_normal_(param)
            param.data = param.data.sign() * scaled_std
            param.data *= scale
        else:
            raise NotImplementedError

class SLT_Linear(SparseModule):
    def __init__(self, in_ch, out_ch, args):
        super().__init__()

        self.sparsity = args.pruning_rate
        self.init_mode_weight = args.init_mode_weight
        self.init_mode_score = args.init_mode_score
        self.init_scale_weight = args.init_scale_weight
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.weight.requires_grad = False
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity

        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.sparsity[0],
            args=args,
        )

        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.sparsity[0],
            args=args,
        )

        self.weight_zeros = torch.zeros(self.weight.size())
        self.weight_ones = torch.ones(self.weight.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
        self.args = args

    def reset_parameters(self):
        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.sparsity[0],
            args=self.args,
        )
        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.sparsity[0],
            args=self.args,
        )

    def forward(self, x, threshold):
        subnet = GetSubnet.apply(torch.abs(self.weight_score),
            threshold,
            self.weight_zeros,
            self.weight_ones,
        )
        pruned_weight = self.weight * subnet
        ret = F.linear(x, pruned_weight, None)
        return ret

class SLT_Conv2d(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, args=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.ones(out_channels, in_channels // groups, *self.kernel_size))
        self.weight.requires_grad = False

        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = args.conv_sparsity if hasattr(args, 'conv_sparsity') else args.pruning_rate

        self.init_mode_weight = args.init_mode_weight
        self.init_mode_score = args.init_mode_score
        self.init_scale_weight = args.init_scale_weight
        self.init_scale_score = args.init_scale_score

        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.weight_score.sparsity[0],
            args=args,
        )

        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.weight_score.sparsity[0],
            args=args,
        )

        self.weight_zeros = torch.zeros(self.weight.size())
        self.weight_ones = torch.ones(self.weight.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False
        self.args = args

    def reset_parameters(self):
        self.init_param_(
            self.weight,
            init_mode=self.init_mode_weight,
            scale=self.init_scale_weight,
            sparse_value=self.weight_score.sparsity[0],
            args=self.args,
        )
        self.init_param_(
            self.weight_score,
            init_mode=self.init_mode_score,
            scale=self.init_scale_score,
            sparse_value=self.weight_score.sparsity[0],
            args=self.args,
        )

    def forward(self, x, threshold):
        subnet = GetSubnet.apply(
            torch.abs(self.weight_score),
            threshold,
            self.weight_zeros,
            self.weight_ones,
        )

        pruned_weight = self.weight * subnet

        return F.conv2d(
            x,
            pruned_weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

    def apply_post_pruning(self, post_pruning_ratio):
        """Apply post pruning to the current mask"""
        if post_pruning_ratio <= 0:
            return

        mask_flat = self.mask.flatten()
        non_zero_indices = torch.where(mask_flat == 1)[0]
        num_to_prune = int(len(non_zero_indices) * post_pruning_ratio)
        if num_to_prune > 0:
            indices_to_prune = torch.randperm(len(non_zero_indices), device=mask_flat.device)[:num_to_prune]
            prune_indices = non_zero_indices[indices_to_prune]
            mask_flat[prune_indices] = 0
            self.mask = mask_flat.reshape(self.mask.shape)

class PartialFrozenLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            sparsity,
            algo           = 'global_ep',
            scale_method   = None,
            bias           = False,
            device         = None,
            dtype          = None,
            ):
        super(PartialFrozenLinear, self).__init__(in_features, out_features, bias, device, dtype)

        self.algo = algo
        self.scale_method = scale_method
        self.score = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('mask',                torch.Tensor(self.weight.size()))
        self.register_buffer('ternary_frozen_mask', torch.zeros_like(self.weight))
        self.register_buffer('local_sparsity',      torch.Tensor([sparsity]))
        self.register_buffer('global_sparsity',     torch.Tensor([sparsity]))
        self.register_buffer('scale',               torch.Tensor([1]))
        self.register_buffer('kthvalue',            torch.Tensor([0]))
        self.score.is_score = True
        self.weight.requires_grad = False

    def forward(self, input):
        mask = GetSupermask.apply(self.score.abs(), self.kthvalue.data[0], self.ternary_frozen_mask)
        self.mask = mask
        with torch.no_grad():
            if self.algo == 'global_ep':
                self.local_sparsity = 1 - (mask.sum() / mask.numel())
            self.scale = get_scale(self.local_sparsity, self.global_sparsity, self.scale_method)
        masked_weight = mask * self.weight * self.scale
        return F.linear(input, masked_weight, self.bias)


class PartialFrozenConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            sparsity,
            algo           = 'local_ep',
            scale_method   = None,
            stride         = 1,
            padding        = 0,
            dilation       = 1,
            groups         = 1,
            bias           = False,
            padding_mode   = 'zeros',
            device         = None,
            dtype          = None,
            post_pruning_ratio = 0.0
            ):
        super(PartialFrozenConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.algo         = algo
        self.scale_method = scale_method
        self.score = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('mask',                torch.Tensor(self.weight.size()))
        self.register_buffer('ternary_frozen_mask', torch.zeros_like(self.weight))
        self.register_buffer('local_sparsity',      torch.Tensor([sparsity]))
        self.register_buffer('global_sparsity',     torch.Tensor([sparsity]))
        self.register_buffer('scale',               torch.Tensor([1]))
        self.register_buffer('kthvalue',            torch.Tensor([0]))
        self.score.is_score = True
        self.weight.requires_grad = False
        self.post_pruning_ratio = post_pruning_ratio

    def forward(self, input):
        mask = GetSupermask.apply(self.score.abs(), self.kthvalue.data[0], self.ternary_frozen_mask)
        self.mask = mask
        with torch.no_grad():
            if self.algo == 'global_ep':
                self.local_sparsity = 1 - (mask.sum() / mask.numel())
            self.scale = get_scale(self.local_sparsity, self.global_sparsity, self.scale_method)
        masked_weight = mask * self.weight * self.scale
        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class GetSupermask(Function):
    @staticmethod
    def forward(ctx, score, kthvalue, ternary_frozen_mask):
        ones   = torch.ones_like(score).to(score.device)
        zeros  =  0 * ones
        m_ones = -1 * ones

        mask  = torch.gt(score, kthvalue).to(score.device)

        mask  = torch.where(
            ternary_frozen_mask == m_ones, zeros, mask)
        mask  = torch.where(
            ternary_frozen_mask == ones, ones, mask)

        return mask

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None

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
        t_float32_max   = torch.finfo(torch.float32).max * torch.ones(1).to(device)
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


def set_initial_value(param, init_method, a=0, mode='fan_in', nonlinearity='relu'):
    if init_method == 'signed_kaiming_constant':
        fan = nn.init._calculate_correct_fan(param, mode)
        gain = nn.init.calculate_gain(nonlinearity)
        std = gain / (fan ** (1/2))
        param.data = std * param.data.sign()

    elif init_method == 'kaiming_normal':
        nn.init.kaiming_normal_(param, a=a, mode=mode, nonlinearity=nonlinearity)
    elif init_method == 'kaiming_uniform':
        nn.init.kaiming_uniform_(param, a=a, mode=mode, nonlinearity=nonlinearity)
    elif init_method == 'trunc_normal':
        trunc_normal_(param, std=.02)
    elif init_method == 'signed_trunc_constant':
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.
        def gauss_pdf(x):
            return (1 / math.sqrt(2*math.pi)) * math.exp(-(x**2)/2)
        mean =  0
        std  = .02
        a    = -2.
        b    =  2.
        alpha = (a - mean) / std
        beta  = (b - mean) / std
        Z     = norm_cdf(beta) - norm_cdf(alpha)
        modified_var = (std**2) * (1 - (beta * gauss_pdf(beta) - alpha * gauss_pdf(alpha)) / Z - ((gauss_pdf(beta) - gauss_pdf(alpha)) / Z)**2)
        modified_std = modified_var**(1/2)
        param.data = modified_std * param.data.sign()
    else:
        raise NotImplementedError

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

def get_supermask(mask, ternary_frozen_mask):
    ones   = torch.ones_like(mask).to(mask.device)
    zeros  =  0 * ones
    m_ones = -1 * ones

    mask  = torch.where(
        ternary_frozen_mask == m_ones, zeros, mask)
    mask  = torch.where(
        ternary_frozen_mask == ones,  ones,  mask)

    return mask

def get_scale(local_sparsity, global_sparsity, scale_method):
    if scale_method == 'dynamic_scaled':
        return 1.0 / torch.sqrt(1.0 - local_sparsity)
    elif scale_method == 'static_scaled':
        return 1.0 / torch.sqrt(1.0 - global_sparsity)
    else:
        return 1.0

def calculate_sparsities(sparsity, epoch, max_epoch_half):
    return [value * (epoch / max_epoch_half) for value in sparsity]

def percentile(t, q):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()

def get_threshold(model, epoch=None, args=None):
    if not args.slt and not args.partial_frozen_slt:
        return None

    max_epoch_half = args.epochs // 2
    # Handle both tensor and list cases for pruning_rate
    if hasattr(args.pruning_rate, 'item'):  # It's a tensor
        pruning_rate_val = args.pruning_rate.item()
    else:
        pruning_rate_val = args.pruning_rate[0] if isinstance(args.pruning_rate, (list, tuple)) else args.pruning_rate
    sparsity_value = pruning_rate_val * (min(epoch, max_epoch_half) / max_epoch_half)
    local = torch.cat(
        [
            p.detach().flatten()
            for name, p in model.named_parameters()
            if hasattr(p, "is_score") and p.is_score
        ]
    )
    threshold = percentile(
        local.abs(),
        sparsity_value * 100,
    )
    return threshold

def initialize_params(
        model, w_init_method, s_init_method=None, m_init_method=None,
        p_ratio=0, r_ratio=0, r_method = 'sparsity_distribution', mask_file=None,
        nonlinearity='relu', algo=None):
    if mask_file != None:
        print(f'Use {mask_file}.')
        mask_data = torch.load(mask_file)
        total = 0
        count = 0
        for n, m in model.named_modules():
            if isinstance(m, (PartialFrozenLinear, PartialFrozenConv2d)):
                total += 1
                if not n + '.weight' in mask_data:
                    n = 'module.' + n
                if mask_data[n + '.weight'].numel() == m.weight.numel():
                    count += 1
        if total == count:
            assert m_init_method != None
            frozen_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - (p_ratio+r_ratio))
            is_same_size = True
            frozen_dist_count = 0
        elif count == 0:
            is_same_size = False
        else:
            raise ValueError

        p_dist = []
        r_dist = []
        for n, m in model.named_modules():
            if isinstance(m, (PartialFrozenLinear, PartialFrozenConv2d)):
                if is_same_size:
                    if not n + '.weight' in mask_data:
                        n = 'module.' + n
                    mask_ones_rate  = ((mask_data[n + '.mask'] == 1).sum()/m.weight.numel()).to(frozen_dist[0].device)
                    mask_zeros_rate = 1 - mask_ones_rate
                    print(f'{mask_zeros_rate=}, {mask_ones_rate=}')
                    free_rate = (1 - frozen_dist[frozen_dist_count])
                    free_ones_rate  = free_rate / 2
                    free_zeros_rate = free_rate / 2
                    print(f'0: {free_zeros_rate=}, {free_ones_rate=}')
                    if free_ones_rate > mask_ones_rate:
                        free_zeros_rate += free_ones_rate - mask_ones_rate
                        free_ones_rate = mask_ones_rate
                        print(f'1: {free_zeros_rate=}, {free_ones_rate=}')
                    if free_zeros_rate > mask_zeros_rate:
                        free_ones_rate += free_zeros_rate - mask_zeros_rate
                        free_zeros_rate = mask_zeros_rate
                        print(f'2: {free_zeros_rate=}, {free_ones_rate=}')
                    p_dist.append(mask_zeros_rate - free_zeros_rate)
                    r_dist.append(mask_ones_rate - free_ones_rate)
                    print(f'{p_dist[-1]=}, {p_dist[-1]=}')
                    eps = 1e-4
                    assert abs(free_ones_rate + free_zeros_rate - free_rate) < eps,          f'{free_ones_rate} + {free_zeros_rate} != {free_rate}.'
                    assert (p_dist[-1] + r_dist[-1] - frozen_dist[frozen_dist_count]) < eps, f'{p_dist[-1]} + {r_dist[-1]} != {frozen_dist[frozen_dist_count]}.'
                    frozen_dist_count += 1
                else:
                    raise NotImplementedError
        p_dist = torch.tensor(p_dist)
        r_dist = torch.tensor(r_dist)
    elif m_init_method != None:
        if m_init_method == 'layer_wise':
            p_dist = torch.tensor(p_ratio)
            r_dist = torch.tensor(r_ratio)
        elif r_method == None or r_method == 'density_distribution':
            assert p_ratio[0] + r_ratio[0] <= 1
            # print('r_method: density_distribution')
            p_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - p_ratio[0])
            r_dist = 1 - get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=r_ratio[0])
        elif r_method == 'sparsity_distribution':
            if isinstance(p_ratio, list):
                p_ratio = p_ratio[0]
                r_ratio = r_ratio[0]
            else:
                p_ratio = p_ratio
            assert p_ratio + r_ratio <= 1
            # print('r_method: sparsity_distribution')
            frozen_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - (p_ratio+r_ratio))
            p_dist      = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - p_ratio)
            r_dist      = frozen_dist - p_dist
            assert torch.logical_and(r_dist >= 0, r_dist <= 1).all()
        else:
            raise ValueError

    module_count = 0

    for m in model.modules():
        if isinstance(m, (PartialFrozenLinear, PartialFrozenConv2d)):
            set_initial_value(m.weight, init_method=w_init_method, nonlinearity=nonlinearity)
            set_initial_value(m.score,  init_method=s_init_method, nonlinearity=nonlinearity)
            if m.bias != None:
                raise NotImplementedError

            if m_init_method != None:
                assert p_dist[module_count] + r_dist[module_count] <= 1
                rand_indices = torch.randperm(m.weight.numel(), device=m.weight.device)
                n_p_params   = (m.weight.numel() * p_dist[module_count]).int()
                n_r_params   = (m.weight.numel() * r_dist[module_count]).int()

                ternary_frozen_mask = m.ternary_frozen_mask.flatten()
                ternary_frozen_mask[rand_indices[:n_p_params]] = -1
                ternary_frozen_mask[rand_indices[n_p_params:n_p_params+n_r_params]] = 1
                m.ternary_frozen_mask = ternary_frozen_mask.reshape(m.ternary_frozen_mask.size()).clone().detach()
                module_count += 1

        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            set_initial_value(m.weight, init_method=w_init_method, nonlinearity=nonlinearity)
            if m.bias != None:
                set_initial_value(m.bias, init_method=w_init_method, nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight != None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_sparsity_distribution(
    model, loss=None, dataloader=None, s_dist_method=None, density=0):

    masked_parameters = []
    for m in filter(lambda x: isinstance(x, (PartialFrozenLinear, PartialFrozenConv2d)), model.modules()):
        # masked_parameters.append(m.ternary_frozen_mask)
        masked_parameters.append(m.weight)

    # print(f'Sparsity distribution : {s_dist_method}')
    if s_dist_method == 'snip':
        raise NotImplementedError
        edges = get_snip(
            model=model, loss=loss, dataloader=dataloader,
            masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'grasp':
        raise NotImplementedError
        edges = get_grasp(
            model=model, loss=loss, dataloader=dataloader,
            masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'synflow':
        raise NotImplementedError
        edges = get_synflow(
            model=model, loss=loss, dataloader=dataloader,
            masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'erk':
        sparsities, edges = get_erk(masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'igq':
        sparsities, edges = get_igq(masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'epl':
        sparsities, edges = get_epl(masked_parameters=masked_parameters, density=density)
    else:
        raise ValueError

    return sparsities

def get_erk(masked_parameters, density):
    # We have to enforce custom sparsities and then find the correct scaling
    # factor.
    is_eps_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_eps_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
        divisor = 0
        rhs     = 0
        raw_probabilities = {}
        print(f'Loop : ')
        for p in masked_parameters:
            n_param = p.numel()
            n_zeros = int(p.numel() * (1 - density))
            if id(p) in dense_layers:
                print(f'{id(p)} is a dense layer')
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros
            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                n_ones = n_param - n_zeros
                rhs += n_ones
                assert id(p) not in raw_probabilities
                print(f'{id(p)} raw_prob : {(sum(p.size()) / p.numel())}')
                raw_probabilities[id(p)] = (sum(p.size()) / p.numel())
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[id(p)] * n_param
        print()
        # All layer is dense
        if divisor == 0:
            is_eps_valid = True
            break
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        eps = rhs / divisor
        print(f'eps : {rhs} / {divisor} = {eps}')
        # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob     = np.max(list(raw_probabilities.values()))
        print(f'raw_prob_list : {list(raw_probabilities.values())}')
        print(f'max_prob : {max_prob}')
        max_prob_one = max_prob * eps
        print(f'max_prob_one : {max_prob_one}')
        if max_prob_one >= 1:
            is_eps_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    print(f'Sparsity of layer {mask_name} had to be set to 0')
                    dense_layers.add(mask_name)
            print()
        else:
            is_eps_valid = True

    sparsities = []
    edges      = []
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for p in masked_parameters:
        n_param = p.numel()
        if id(p) in dense_layers:
            sparsities.append(0.)
            edges.append(p.numel())
        else:
            probability_one = eps * raw_probabilities[id(p)]
            sparsities.append(1. - probability_one)
            print(f'{p.numel()} * {probability_one} = {p.numel() * probability_one}')
            edges.append(int(p.numel() * probability_one))

    return torch.tensor(sparsities), torch.tensor(edges)

def get_epl(masked_parameters, density):
    layers = set(range(len(masked_parameters)))
    n_params_lst = []
    edges = []
    for p in masked_parameters:
        n_params_lst.append(p.numel())
        edges.append(0)
    total = sum(n_params_lst) * density
    dense_layers = set()

    while total != 0:
        for k in layers:
            edges[k] += total / len(layers)

        total = 0
        for k in layers:
            if edges[k] > n_params_lst[k]:
                total += edges[k] - n_params_lst[k]
                edges[k] = n_params_lst[k]
                dense_layers.add(k)
        layers = layers - dense_layers

    sparsities = []
    for i in range(len(edges)):
        edges[i] = int(edges[i])
        density = edges[i] / n_params_lst[i]
        sparsities.append(1 - density)

    return torch.tensor(sparsities), torch.tensor(edges)

def get_igq(masked_parameters, density):
    def bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high):
        lengths_low          = [Length / (f_low / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity_low = 1 - sum(lengths_low) / sum(Lengths)
        if abs(overall_sparsity_low - target_sparsity) < tolerance:
            return [1 - length / Length for length, Length in zip(lengths_low, Lengths)]

        lengths_high          = [Length / (f_high / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity_high = 1 - sum(lengths_high) / sum(Lengths)
        if abs(overall_sparsity_high - target_sparsity) < tolerance:
            return [1 - length / Length for length, Length in zip(lengths_high, Lengths)]

        force            = float(f_low + f_high) / 2
        lengths          = [Length / (force / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity = 1 - sum(lengths) / sum(Lengths)
        f_low            = force if overall_sparsity < target_sparsity else f_low
        f_high           = force if overall_sparsity > target_sparsity else f_high
        return bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high)

    edges  = []
    counts = []
    for p in masked_parameters:
        counts.append(p.numel())
    tolerance = 1./sum(counts)
    areas     = [1./count for count in counts]
    sparsities = bs_force_igq(
        areas=areas, Lengths=counts, target_sparsity=1-density,
        tolerance=tolerance, f_low=0, f_high=1e20)
    for i, p in enumerate(masked_parameters):
        edges.append(int(p.numel() * (1 - sparsities[i])))

    return torch.tensor(sparsities), torch.tensor(edges)

def calc_global_sparsity(model, args):
    total_zeros = 0
    total_params = 0

    if args.magnitude_pruning:
        # For magnitude pruning, check weight sparsity
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Linear)) and 'fc2' not in name:
                weight = module.weight.data
                zeros = (weight == 0).sum().item()
                total = weight.numel()
                sparsity = zeros / total
                # print(f"{name} weight sparsity: {sparsity:.4f}")
                total_zeros += zeros
                total_params += total
    else:
        # For SLT/partial frozen SLT, check mask sparsity
        for module in model.modules():
            if hasattr(module, 'mask'):
                mask = module.mask
                zeros = (mask == 0).sum().item()
                total = mask.numel()
                sparsity = zeros / total
                # print(f"{module.__class__.__name__} mask sparsity: {sparsity:.4f}")
                total_zeros += zeros
                total_params += total

    if total_params > 0:
        global_sparsity = total_zeros / total_params
    else:
        global_sparsity = 0.0
    #
    # if total_params > 0:
    #     print(f"==> Model total sparsity: {global_sparsity:.4f}")
    # print("-" * 40)

    return global_sparsity
