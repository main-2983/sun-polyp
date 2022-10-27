import torch


def label_assignment(preds: list[torch.Tensor], target: torch.Tensor=None,
                     assign_func=None, **kwargs):
    if assign_func is None:
        return [target] * len(preds)
    else:
        targets = assign_func(preds, target, **kwargs)
        return targets


@torch.no_grad()
def strategy_1(preds: list[torch.Tensor], target: torch.Tensor=None, num_outs=3):
    """ This function output `num_outs` aux target for aux heads base on the
    lead head prediction mask, which is `pred`.
    Strategy 1 take pred as target mask for aux heads, and target for lead head
    """
    targets = []
    targets.append(target)
    aux_targets = []
    for i in range(num_outs):
        pred = preds[0]
        pred = pred.sigmoid()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        aux_targets.append(pred)
    targets.extend(aux_targets)
    return targets


@torch.no_grad()
def strategy_2(preds: list[torch.Tensor], target: torch.Tensor=None, num_outs=3,
               cur_ep=None, total_eps=20, frac=0.8):
    ep_to_change = int(total_eps * frac)
    if cur_ep <= ep_to_change:
        targets = [target] * len(preds)
        return targets
    else:
        targets = []
        targets.append(target)
        aux_targets = []
        for i in range(num_outs):
            pred = preds[0]
            pred = pred.sigmoid()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            aux_targets.append(pred)
        targets.extend(aux_targets)
        return targets


@torch.no_grad()
def strategy_3(preds: list[torch.Tensor], target:torch.Tensor=None, num_outs=3):
    targets = []
    targets.append(target)
    aux_targets = []
    for i in range(num_outs):
        pred = preds[i]
        pred = pred.sigmoid()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        aux_targets.append(pred)
    targets.extend(aux_targets)
    return targets