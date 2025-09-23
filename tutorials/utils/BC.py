from typing import Any, Optional, Union, List
import numpy as np
import torch as th
from collections import Counter
import os
from tqdm import tqdm
import math
try:
    from nuplan.planning.simulation.planner.utils.smpc_utils import to_tensor_var
except ImportError:
    def to_tensor_var(x, use_cuda: bool):
        t = th.as_tensor(x)
        return t.cuda(non_blocking=True) if use_cuda and th.cuda.is_available() else t


# --------------------------------------------------------------------------
# Long-tail helpers
# --------------------------------------------------------------------------
def compute_l1_counts(dataset, l1_dim: int, l1_ubd: float) -> th.Tensor:
    cnt = Counter({0: 0, 1: 0, 2: 0})
    for _, ac, *_ in dataset:
        ac = th.as_tensor(ac)
        l1 = ac[:l1_dim]
        cls = (l1 > 1e-3).int() + (l1 > (l1_ubd * 0.99)).int()
        cnt[0] += int((cls == 0).sum())
        cnt[1] += int((cls == 1).sum())
        cnt[2] += int((cls == 2).sum())
    return th.tensor([cnt[0], cnt[1], cnt[2]], dtype=th.float32)


class LDAMLoss(th.nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30.0, beta=None, use_drw=False):
        super().__init__()
        if not isinstance(cls_num_list, th.Tensor):
            cls_num_list = th.tensor(cls_num_list, dtype=th.float32)
        else:
            cls_num_list = cls_num_list.to(dtype=th.float32)
        cls_num_list = th.clamp(cls_num_list, min=1.0)
        m_list = 1.0 / th.sqrt(th.sqrt(cls_num_list))
        m_list = max_m * (m_list / m_list.max())
        self.register_buffer("m_list", m_list)
        self.s = s
        self.use_drw = use_drw
        self.warmup_epochs = 160
        if beta is not None:
            eff_num = 1.0 - th.pow(th.tensor(beta, dtype=th.float32), cls_num_list)
            weights = (1.0 - beta) / th.clamp(eff_num, min=1e-12)
            weights = weights * (len(cls_num_list) / weights.sum())
            self.register_buffer("class_weights", weights.to(dtype=th.float32))
        else:
            self.class_weights = None
        self.drw_active = False

    def set_phase(self, phase: int):
        self.drw_active = (phase >= 2)

    def forward(self, logits, targets):
        targets = targets.long()
        margins = th.zeros_like(logits)
        gt_margin = self.m_list[targets]
        margins.scatter_(1, targets.reshape(-1, 1), gt_margin.reshape(-1, 1))
        logits_adj = self.s * (logits - margins)
        if self.use_drw and self.class_weights is not None and self.drw_active:
            return th.nn.functional.cross_entropy(
                logits_adj, targets, weight=self.class_weights
            )
        else:
            return th.nn.functional.cross_entropy(logits_adj, targets)

def l1_mask_balanced(
    logits_l1: th.Tensor,     # (B, P, 3) raw logits for L1
    targets: th.Tensor,       # (B, P) in {0,1,2}
    max_ratio_head: float=1.5,# keep at most r * (#tails) from class 1
    min_tail_keep: int=16,    # if tails are scarce, ensure at least this many samples
    rng: np.random.Generator=None
) -> th.Tensor:
    """
    Returns boolean mask (B,P) selecting samples for L1 loss this batch.
    - Keep all class 0 and 2.
    - Keep at most r * tails from class 1.
    - If no tails present, select `min_tail_keep` hardest non‑1 candidates via margin.
    """
    if rng is None:
        rng = np.random.default_rng()

    B, P = targets.shape
    flat_t = targets.reshape(-1)
    mask = th.zeros_like(flat_t, dtype=th.bool)

    idx0 = (flat_t == 0).nonzero(as_tuple=False).squeeze(1)
    idx1 = (flat_t == 1).nonzero(as_tuple=False).squeeze(1)
    idx2 = (flat_t == 2).nonzero(as_tuple=False).squeeze(1)

    # Keep all tails
    tail_idx = th.cat([idx0, idx2], dim=0)
    if tail_idx.numel() > 0:
        mask[tail_idx] = True

    # Limit class‑1 amount based on tails
    tails = int(tail_idx.numel())
    if tails > 0:
        k1 = int(max_ratio_head * tails)
        if idx1.numel() > 0:
            if idx1.numel() <= k1:
                mask[idx1] = True
            else:
                pick = th.from_numpy(
                    rng.choice(idx1.cpu().numpy(), size=k1, replace=False)
                ).to(idx1.device)
                mask[pick] = True
    else:
        # No tails in this batch: mine "hard non‑1" using small margin to a tail logit
        # Compute per‑position margin (= class1_logit - max(logit_c0, logit_c2))
        with th.no_grad():
            flat_logits = logits_l1.reshape(-1, 3)  # (B*P,3)
            cls1 = flat_logits[:, 1]
            tails_max = th.maximum(flat_logits[:, 0], flat_logits[:, 2])
            margin = cls1 - tails_max  # smaller (or negative) margin -> “hard” for class1

        # take top `min_tail_keep` with smallest margin (most tail‑like)
        k = min(min_tail_keep, margin.numel())
        if k > 0:
            hard_idx = th.topk(-margin, k=k, largest=True).indices  # negative => small margin
            mask[hard_idx] = True

    # If mask ended up empty (degenerate), keep everything to avoid NaNs
    if not mask.any():
        mask = th.ones_like(mask, dtype=th.bool)
    return mask.view(B, P)

def l1_mask_majority(targets: th.Tensor, keep_majority: float = 0.15,
                     rng: np.random.Generator = None) -> th.Tensor:
    if rng is None:
        rng = np.random.default_rng()
    with th.no_grad():
        flat = targets.reshape(-1)
        idx_maj = (flat == 1).nonzero(as_tuple=False).squeeze(1)
        idx_min = (flat != 1).nonzero(as_tuple=False).squeeze(1)
        k = int(keep_majority * int(idx_maj.numel()))
        mask = th.zeros_like(flat, dtype=th.bool)
        if idx_min.numel() > 0:
            mask[idx_min] = True
        if k > 0 and idx_maj.numel() > 0:
            pick = th.from_numpy(rng.choice(idx_maj.cpu().numpy(), size=k, replace=False)).to(idx_maj.device)
            mask[pick] = True
        return mask.view_as(targets)

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
class ImbalancedMetrics:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.confusion = th.zeros(self.num_classes, self.num_classes)

    def update(self, preds: th.Tensor, targs: th.Tensor):
        p = preds.reshape(-1).cpu().long()
        t = targs.reshape(-1).cpu().long()
        for pi, ti in zip(p, t):
            if 0 <= ti < self.num_classes and 0 <= pi < self.num_classes:
                self.confusion[ti, pi] += 1

    def summary(self):
        cm = self.confusion
        tp = th.diag(cm)
        acc = tp.sum() / cm.sum().clamp_min(1)
        prec = tp / (cm.sum(0).clamp_min(1))
        rec = tp / (cm.sum(1).clamp_min(1))
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
        macro_f1 = f1.mean()
        return {
            "accuracy": acc.item(),
            "macro_f1": macro_f1.item(),
            "per_class_f1": f1.tolist(),
            "confusion": cm.tolist(),
        }


# --------------------------------------------------------------------------
# BC trainer
# --------------------------------------------------------------------------
class BC:
    def __init__(self, policy, device, optimizer, optim_lr, rng,
                 demonstrations, logger, normalize, normalize_obs,
                 config, ca_dual_dim, l1_dual_dim, batch_size=32,
                 dagger_mode=False, ismlp=False, joint_dual_pred=False,
                 keep_majority=0.15, num_tvs=3, policy_type='RAIDNET'):
        self.policy = policy
        self.device = th.device(device)
        self.logger = logger
        self.config = config
        self.rng = rng
        self.normalize = normalize
        self.normalize_obs = normalize_obs
        self.ca_dual_dim = policy[1].output_dim
        self.l1_dual_dim = policy[0].output_dim
        self.joint_dual_pred = joint_dual_pred
        self.keep_majority = keep_majority
        self.num_tvs = num_tvs
        self.policy_type = policy_type

        weights = th.DoubleTensor(demonstrations.dataset_weights)
        sampler = th.utils.data.WeightedRandomSampler(weights, demonstrations.max_size)
        self.train_loader = th.utils.data.DataLoader(demonstrations, batch_size=batch_size,
                                                     sampler=sampler, pin_memory=True)
        self.demonstrations = demonstrations
        self.use_cuda = th.cuda.is_available() and (str(self.device).startswith('cuda'))

        if not joint_dual_pred:
            assert isinstance(self.policy, list) and len(self.policy) == 2
            if optimizer.lower() == 'adam':
                self.optimizer = [th.optim.AdamW(m.parameters(), lr=optim_lr) for m in self.policy]
            else:
                self.optimizer = [th.optim.RMSprop(m.parameters(), lr=optim_lr) for m in self.policy]
            self.lr_sched = [th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2,
                                                                               eta_min=optim_lr*0.1)
                             for opt in self.optimizer]
        else:
            if optimizer.lower() == 'adam':
                self.optimizer = th.optim.AdamW(self.policy.parameters(), lr=optim_lr)
            else:
                self.optimizer = th.optim.RMSprop(self.policy.parameters(), lr=optim_lr)
            self.lr_sched = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2,
                                                                              eta_min=optim_lr*0.1)

        self.l1_ubd = getattr(self.policy[0], 'lmbd_ubd', 1000.0) if not joint_dual_pred else getattr(self.policy, 'lmbd_ubd', 1000.0)
        self.l1_counts = compute_l1_counts(self.demonstrations, self.l1_dual_dim, self.l1_ubd).to(self.device)
        self.ldam = LDAMLoss(self.l1_counts, max_m=0.8, s=30.0, beta=0.9999, use_drw=True).to(self.device)

        self.pos_weight_hi = 1.0 / max(1e-3, 0.10) - 1.0
        self.pos_weight_lo = 3.5
        self.ca_bce = None

        self.metrics_l1 = ImbalancedMetrics(3, ['L1_c0', 'L1_c1', 'L1_c2'])
        self.metrics_ca = ImbalancedMetrics(2, ['CA_0', 'CA_1'])

    def _build_l1_targets(self, l1_duals: th.Tensor) -> th.Tensor:
        return (l1_duals > 1e-3).int() + (l1_duals > (self.l1_ubd * 0.99)).int()

    def _per_class_acc(self, preds: th.Tensor, targets: th.Tensor, num_classes: int = 3) -> th.Tensor:
        """Return tensor [acc_c0, acc_c1, acc_c2] (nan if class not present in batch)."""
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        accs = []
        for c in range(num_classes):
            mask = (targets == c)
            if mask.any():
                accs.append((preds[mask] == c).float().mean())
            else:
                accs.append(th.tensor(float('nan')))
        return th.stack(accs)

    def train(self, n_epochs: int, model_name: Optional[str] = None, pred_mode: Optional[List[str]] = None) -> dict:
        training_log = {}
        save_root = os.path.join(self.config['root_dir'], self.config['model_save_dir'], model_name or 'bc')
        os.makedirs(save_root, exist_ok=True)
        self.file_path = os.path.join(save_root, model_name or 'bc')
        drw_switch = int(0.2 * n_epochs)

        for epoch in range(n_epochs):
            phase = 1 if epoch < drw_switch else 2
            self.ldam.set_phase(phase)

            # Anneal CA pos_weight
            t = min(1.0, epoch / max(1, n_epochs - 1))
            pos_w = self.pos_weight_hi * (1 - t) + self.pos_weight_lo * t
            self.ca_bce = th.nn.BCEWithLogitsLoss(
                pos_weight=th.full((self.policy[1].output_dim,), pos_w, device=self.device)
            )

            # ---------- epoch accumulators (running metrics) ----------
            # L1
            l1_total_correct = 0
            l1_total = 0
            l1_counts = th.zeros(3, dtype=th.long)         # #samples per class in epoch
            l1_correct_per_class = th.zeros(3, dtype=th.long)

            # CA
            ca_total_correct = 0
            ca_total = 0
            ca_pred_pos = 0
            ca_targ_pos = 0
            ca_true_pos = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True)
            for i, (ob_batch, ac_batch, *_) in enumerate(pbar):
                B = ob_batch.shape[0]
                obs = to_tensor_var(ob_batch, use_cuda=self.use_cuda)
                acts = to_tensor_var(ac_batch, use_cuda=self.use_cuda)
                if 'RAIDNET' in self.policy_type:
                    obs_reshaped = obs.reshape(B, self.num_tvs, -1)
                else:
                    obs_reshaped = obs

                # ----- L1 -----
                self.optimizer[0].zero_grad()
                logits_l1 = self.policy[0](obs_reshaped)               # (B, P, 3)
                l1_duals  = acts[:, :self.policy[0].output_dim]        # (B, P)
                targets_l1 = (l1_duals > 1e-3).int() + (l1_duals > (self.l1_ubd * 0.99)).int()

                # mask = l1_mask_majority(targets_l1, keep_majority=self.keep_majority)
                mask = l1_mask_balanced(
                    logits_l1=logits_l1,
                    targets=targets_l1,
                    max_ratio_head=1.5,     # try 1.0–2.0
                    min_tail_keep=32        # ensure some tail-like samples even in bad batches
                )
                if mask.sum() == 0:
                    mask = th.ones_like(targets_l1, dtype=th.bool)

                loss_l1 = self.ldam(logits_l1[mask], targets_l1[mask].long())
                loss_l1.backward()
                self.optimizer[0].step()
                self.lr_sched[0].step(epoch + i / max(1, len(self.train_loader)))

                with th.no_grad():
                    preds_l1 = logits_l1.argmax(dim=-1)
                    # epoch running totals
                    l1_total_correct += (preds_l1 == targets_l1).sum().item()
                    l1_total += targets_l1.numel()
                    for c in range(3):
                        m = (targets_l1 == c)
                        l1_counts[c] += m.sum().item()
                        if m.any():
                            l1_correct_per_class[c] += (preds_l1[m] == c).sum().item()
                    # per‑class acc this batch
                    l1_acc_batch = []
                    for c in (0,1,2):
                        m = (targets_l1 == c)
                        if m.any():
                            l1_acc_batch.append((preds_l1[m] == c).float().mean().item())
                        else:
                            l1_acc_batch.append(float('nan'))
                    # balanced accuracy ignoring NaNs
                    valid = [a for a in l1_acc_batch if a == a]
                    bal_acc = float(np.mean(valid)) if valid else 0.0

                # ----- CA -----
                self.optimizer[1].zero_grad()
                logits_ca = self.policy[1](obs_reshaped)               # (B, CA_dim)
                targets_ca = acts[:, self.policy[0].output_dim:]       # (B, CA_dim) in {0,1}
                loss_ca = self.ca_bce(logits_ca, targets_ca.float())
                loss_ca.backward()
                self.optimizer[1].step()
                self.lr_sched[1].step(epoch + i / max(1, len(self.train_loader)))

                with th.no_grad():
                    preds_ca = (th.sigmoid(logits_ca) > 0.5).long()
                    ca_total_correct += (preds_ca == targets_ca).sum().item()
                    ca_total += targets_ca.numel()
                    ca_pred_pos += preds_ca.sum().item()
                    ca_targ_pos += targets_ca.sum().item()
                    ca_true_pos += ((preds_ca == 1) & (targets_ca == 1)).sum().item()

                # ---------- compute running (epoch‑to‑date) metrics ----------
                l1_acc = (l1_total_correct / l1_total) if l1_total > 0 else 0.0
                per_class_acc = []
                for c in range(3):
                    denom = int(l1_counts[c].item())
                    if denom > 0:
                        per_class_acc.append(l1_correct_per_class[c].item() / denom)
                    else:
                        per_class_acc.append(float('nan'))

                # nice short strings for the bar
                # ---------- compute running (epoch‑to‑date) metrics ----------
                l1_acc = (l1_total_correct / l1_total) if l1_total > 0 else 0.0
                ca_acc = (ca_total_correct / ca_total) if ca_total > 0 else 0.0
                ca_recall = (ca_true_pos / ca_targ_pos) if ca_targ_pos > 0 else 0.0
                ca_precision = (ca_true_pos / ca_pred_pos) if ca_pred_pos > 0 else 0.0

                # ---------- PER‑BATCH metrics for display ----------
                # L1 per‑batch per‑class accuracy & counts
                with th.no_grad():
                    # counts per class in THIS batch
                    batch_counts = th.bincount(targets_l1.reshape(-1), minlength=3)  # (3,)
                    # correct per class in THIS batch
                    batch_correct_per_c = th.zeros(3, dtype=th.long, device=targets_l1.device)
                    for c in range(3):
                        m = (targets_l1 == c)
                        if m.any():
                            batch_correct_per_c[c] = (preds_l1[m] == c).sum()

                    # per‑class acc THIS batch
                    batch_per_class_acc = []
                    for c in range(3):
                        denom = int(batch_counts[c].item())
                        batch_per_class_acc.append(
                            float(batch_correct_per_c[c].item()) / denom if denom > 0 else float('nan')
                        )

                    # strings for tqdm
                    pc_str = ",".join("-" if math.isnan(a) else f"{a:.2f}" for a in batch_per_class_acc)
                    cnt_str = '['+",".join(str(int(x)) for x in batch_counts.tolist())+']'

                # CA per‑batch #predicted 1s / #target 1s
                with th.no_grad():
                    batch_pred_pos = int((th.sigmoid(logits_ca) > 0.5).long().sum().item())
                    batch_targ_pos = int(targets_ca.sum().item())

                # ---------- update progress bar (running acc/recall + per‑batch details) ----------
                pbar.set_postfix({
                    "L1 balAcc": f"{bal_acc*100:.1f}%",
                    "L1 acc": f"{'/'.join('-' if a!=a else f'{a*100:.1f}' for a in l1_acc_batch)} ({l1_acc*100:.1f})%",
                    # "L1_acc": f"{l1_acc*100:.1f}%",   # running
                    # "L1_cls": pc_str,                 # per‑batch per‑class acc c0,c1,c2
                    "L1_cnt": cnt_str,                # per‑batch class counts
                    "CA_acc": f"{ca_acc*100:.1f}%",   # running
                    "CA prec": f"{ca_precision*100:.1f}%",  # per‑batch
                    "CA_rec": f"{ca_recall*100:.1f}%",# running
                    "CA#1":  f"{batch_pred_pos}/{batch_targ_pos}"  # per‑batch
                })

            # ---- logger at epoch end (optional) ----
            if self.logger:
                # you can log l1_acc, per_class_acc, ca_acc, ca_recall here as scalars
                self.logger.log_scalar(l1_acc, 'l1_acc', epoch)
                for c, a in enumerate(per_class_acc):
                    if not math.isnan(a):
                        self.logger.log_scalar(a, f'l1_acc_c{c}', epoch)
                self.logger.log_scalar(ca_acc, 'ca_acc', epoch)
                self.logger.log_scalar(ca_recall, 'ca_recall', epoch)
                self.logger.log_scalar(ca_precision, 'ca_precision', epoch)
                self.logger.log_scalar(ca_pred_pos, 'ca_pred_ones', epoch)
                self.logger.log_scalar(ca_targ_pos, 'ca_targ_ones', epoch)
                self.logger.flush()

            if (epoch + 1) % max(1, self.config.get('model_save_period', 50)) == 0:
                self.save(self.file_path, self.config, iter=epoch + 1)
            self.metrics_l1.reset()
            self.metrics_ca.reset()

        training_log.update({'Epochs': n_epochs})
        return training_log

    def save(self, model_save_dir: str, config: dict, iter: Optional[int] = None):
        th.save({'model_state_dict': self.policy[0].state_dict(),
                 'optimizer_state_dict': self.optimizer[0].state_dict(),
                 'config': config},
                f"{model_save_dir}_L1_{iter or 'final'}.pt")
        th.save({'model_state_dict': self.policy[1].state_dict(),
                 'optimizer_state_dict': self.optimizer[1].state_dict(),
                 'config': config},
                f"{model_save_dir}_CA_{iter or 'final'}.pt")
