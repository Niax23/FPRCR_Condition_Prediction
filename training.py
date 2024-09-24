import torch
from tqdm import tqdm
import numpy as np
from data_utils import convert_log_into_label
from torch.nn.functional import cross_entropy


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def calc_trans_loss(trans_pred, trans_lb, ignore_index, lbsm=0.0):
    batch_size, maxl, num_c = trans_pred.shape
    trans_pred = trans_pred.reshape(-1, num_c)
    trans_lb = trans_lb.reshape(-1)

    losses = cross_entropy(
        trans_pred, trans_lb, reduction='none',
        ignore_index=ignore_index, label_smoothing=lbsm
    )
    losses = losses.reshape(batch_size, maxl)
    loss = torch.mean(torch.sum(losses, dim=-1))
    return loss


def calc_loss(pred, labels):
    loss = 0
    for idx in range(5):
        logits = pred[idx]
        # print(logits.shape)
        targets = labels[:, idx]
        loss += cross_entropy(logits, targets)
    return loss


def train_uspto_condition(loader, model, optimizer, device, warmup=False):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for data in tqdm(loader):
        reac_fps, prod_fps, labels = data

        reac_fps = reac_fps.to(device)
        prod_fps = prod_fps.to(device)
        labels = labels.to(device)

        res = model(
            prod_fps, reac_fps
        )

        loss = calc_loss(res, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_uspto_condition(loader, model, device):
    model, accs, gt = model.eval(), [], []
    for data in tqdm(loader):
        reac_fps, prod_fps, labels = data

        reac_fps = reac_fps.to(device)
        prod_fps = prod_fps.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            res = model(
                prod_fps, reac_fps
            )
            result = convert_log_into_label(res, mod='softmax')

        accs.append(result)
        gt.append(labels)

    accs = torch.cat(accs, dim=0)
    gt = torch.cat(gt, dim=0)

    keys = ['catalyst', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    results, overall = {}, None
    for idx, k in enumerate(keys):
        results[k] = accs[:, idx] == gt[:, idx]
        if idx == 0:
            overall = accs[:, idx] == gt[:, idx]
        else:
            overall &= (accs[:, idx] == gt[:, idx])

    results['overall'] = overall
    results = {k: v.float().mean().item() for k, v in results.items()}
    return results
