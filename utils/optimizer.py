import torch.optim as optim

def get_optimizer(optimizer_name, lr, momentum, weight_decay, model, filter_bias_and_bn=False):
    update_params_w_wd  = []
    update_params_wo_wd = []
    # print(f'Params to update:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f'{name}')
            if filter_bias_and_bn and (param.ndim <= 1 or name.endswith(".bias")):
                update_params_wo_wd.append(param)
            else:
                update_params_w_wd.append(param)
    if filter_bias_and_bn:
        update_params = [
            {'params': update_params_w_wd,  'weight_decay': weight_decay},
            {'params': update_params_wo_wd, 'weight_decay': 0.}
            ]
        weight_decay = 0.
    else:
        update_params = update_params_w_wd

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(update_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(update_params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer
