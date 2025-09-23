from tutorials.policies import RAID_NET, MLP
from tutorials.raidnet import RAID_NET_V2

import torch as th
import argparse
import yaml
import numpy as np
from matplotlib.patches import Rectangle
import pdb
import os
import time
from itertools import product
from tutorials.utils.replay_buffer import ReplayBuffer
from tutorials.utils.BC import BC, l1_mask_balanced
from tutorials.utils.logger import Logger
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import gzip
try:
    from nuplan.planning.simulation.planner.utils.smpc_utils import to_tensor_var
except ImportError:
    def to_tensor_var(x, use_cuda: bool):
        t = th.as_tensor(x)
        return t.cuda(non_blocking=True) if use_cuda and th.cuda.is_available() else t

import shutil
from contextlib import contextmanager

def forward_in_batches(model, x, batch_size=1024, device=None, return_device='cpu'):
    """
    Run `model(x)` in smaller batches to avoid OOM and concatenate outputs along dim=0.
    - x: (B, ...) tensor
    - batch_size: per-chunk size
    - device: device to run the model on (defaults to model's device)
    - return_device: 'cpu' (default) or a torch.device string for the final tensor
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    outs = []
    with th.inference_mode():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size].to(device, non_blocking=True)
            yb = model(xb)
            outs.append(yb.detach().to('cpu'))
            del xb, yb

    out = th.cat(outs, dim=0)
    if was_training:
        model.train()
    return out.to(return_device)


def evaluate(smpc_config,config,policy,device,policy_type,l1_dual_dim,ca_dual_dim,l1_num,pred_mode,other_models):
    #Load expert dataset
    with gzip.open(config['eval_data_dir'],'rb') as file:
        expert_data = pickle.load(file)
    replay_buffer = ReplayBuffer(config['max_replay_buffer_size'],training_dataset=False)

    #remove empty list in expert_data
    expert_data['observation'] = [obs for obs in expert_data['observation'] if len(obs)>0]
    expert_data['optimal_duals'] = [acs for acs in expert_data['optimal_duals'] if len(acs)>0]
    expert_data['dual_class'] = [acs for acs in expert_data['dual_class'] if len(acs)>0]

    observation = np.squeeze(np.concatenate([obs for obs in expert_data["observation"]]),axis=1)
    optimal_duals = np.concatenate([acs for acs in expert_data["optimal_duals"]])

    replay_buffer.obs = observation; replay_buffer.acs = optimal_duals; replay_buffer.opt_duals = optimal_duals; replay_buffer.terminals = np.zeros_like(observation); replay_buffer.next_obs = np.zeros_like(observation); replay_buffer.rews = np.zeros_like(observation) 
    replay_buffer.smpc_params_dim = expert_data['smpc_params_dim']

    flattened_dual_class = []
    for data in expert_data["dual_class"]:
        flattened_dual_class.extend(data)
    replay_buffer.dual_classes = np.array(flattened_dual_class)
    feature_stat = np.load(config['feature_stat_path']) #open npz file
    feature_mean = feature_stat['feature_mean']
    feature_cov = feature_stat['feature_cov']
    replay_buffer.normalize4evaluation(l1_num, feature_mean=feature_mean, feature_cov=feature_cov, target_mean=None, target_cov=None, l1_pred_mode=pred_mode[0])
    replay_buffer.set_weights()

    print('Loading pretrained model...')
    checkpoint = []
    checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06_L1_100.pt'))
    checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06_CA_100.pt'))
    #L1
    policy[0].load_state_dict(checkpoint[0]['model_state_dict'])
    #CA
    policy[1].load_state_dict(checkpoint[1]['model_state_dict'])

    print('EVALUATION STARTED'.center(80,'*'))
    bc_learner = BC(policy=policy, optimizer=config['optimizer'], optim_lr=config['lr'], demonstrations=replay_buffer, rng=np.random.default_rng(0), device=device, batch_size=config['batch_size'], logger=None, normalize=False, config=config, normalize_obs=False, l1_dual_dim=l1_dual_dim, ca_dual_dim=ca_dual_dim, joint_dual_pred=config['joint_dual_pred'])
    metrics = {}

    #TODO: Evaluate the policy on the evaluation_data (single pass on the entire dataset. no minibatch)
    
    #We can use the BC learner's helper functions such as l1_target, per_class_accuracy, loss functions, etc to help us compute the following 
    # L1 Duals:
    # 1) Normalized Confunsion Matrix (Multi-class)
    # 2) Normalized L1 Loss
    # 3) Per-class accuracy
    # 4) Overall accuracy

    # CA Duals:
    # 1) Normalized Confunsion Matrix
    # 2) Normalized CA Loss
    # 3) Precision
    # 4) Recall
    # 5) F1 Score
    # 6) Average overprediction rate (avg. (predicted ones / actual ones in target) )
    # Store them in a dict form with appropriate keys
    # ---------- EVALUATION (single pass; no minibatch) ----------
    policy[0].eval()
    policy[1].eval()

    # tensors
    obs = to_tensor_var(replay_buffer.obs, use_cuda=(device.type == 'cuda')).to(device)
    acts = to_tensor_var(replay_buffer.acs, use_cuda=(device.type == 'cuda')).to(device)
    B = obs.shape[0]
    print(f"Evaluating on {B} samples...")
    obs_reshaped = obs.view(B, smpc_config['num_tvs'], -1)
    nbins = 100

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

    # Forward
    with th.no_grad():
        logits_l1 = policy[0](obs_reshaped)                 # (B, P_l1, 3)
        logits_ca = policy[1](obs_reshaped)                 # (B, P_ca)

        l1_duals  = acts[:, :policy[0].output_dim]        # (B, P)
        targets_l1 = (l1_duals > 1e-3).int() + (l1_duals > (smpc_config['l1_lmbd'] * 0.99)).int()

        # mask = l1_mask_majority(targets_l1, keep_majority=self.keep_majority)
        mask = l1_mask_balanced(
            logits_l1=logits_l1,
            targets=targets_l1,
            max_ratio_head=1.5,     # try 1.0–2.0
            min_tail_keep=32        # ensure some tail-like samples even in bad batches
        )
        if mask.sum() == 0:
            mask = th.ones_like(targets_l1, dtype=th.bool)

        loss_l1 = bc_learner.ldam(logits_l1[mask], targets_l1[mask].long())
        #Compute normalized loss per sample
        # wrong_pred_l1 = th.zeros(policy[0].output_dim,logits_l1.shape[-1]).to(logits_l1.device)
        # wrong_pred_l1[:,1] = 1.0
        # sample_target_l1 = th.ones(policy[0].output_dim).to(logits_l1.device)
        # Multi-class classification cross-entropy loss
        # max_loss_l1 = np.log(logits_l1.shape[-1]) #Cross-entropy loss for a random guess is log(C) where C is the number of classes
        loss = th.nn.functional.cross_entropy(logits_l1.reshape(-1,logits_l1.shape[-1]), targets_l1.flatten().long(),reduction='none').cpu()
        loss_hist_l1, loss_bin_edges_l1 = np.histogram(loss, bins=np.linspace(0, 1, nbins))
        if other_models:
            logits_l1_mlp = other_models['mlp'][0](obs)
            loss_l1_mlp = th.nn.functional.cross_entropy(logits_l1_mlp.reshape(-1,logits_l1_mlp.shape[-1]), targets_l1.flatten().long(),reduction='none').cpu()
            lost_hist_l1_mlp = np.histogram(loss_l1_mlp, bins=np.linspace(0, 1, nbins))

            #Running out of memory because obs_reshape batch size it too big
            # --- usage ---
            logits_l1_raidnet_v1 = forward_in_batches(other_models['RAIDNET_V1'][0], obs_reshaped, batch_size=1024,return_device='cuda:0')
            # logits_l1_raidnet_v1 = other_models['RAIDNET_V1'][0](obs_reshaped)
            loss_l1_raidnet_v1 = th.nn.functional.cross_entropy(logits_l1_raidnet_v1.reshape(-1,logits_l1_raidnet_v1.shape[-1]), targets_l1.flatten().long(),reduction='none').cpu()
            lost_hist_l1_raidnet_v1 = np.histogram(loss_l1_raidnet_v1, bins=np.linspace(0, 1, nbins))
            
        #Compute max possible loss
        l1_loss_per_sample  = 1

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
        targets_ca = acts[:, policy[0].output_dim:]       # (B, CA_dim) in {0,1}
        # t = min(1.0, epoch / max(1, n_epochs - 1))
        t = 0
        pos_w = bc_learner.pos_weight_hi * (1 - t) + bc_learner.pos_weight_lo * t
        ca_bce = th.nn.BCEWithLogitsLoss(reduction='none',
                pos_weight=th.full((policy[1].output_dim,), pos_w, device=logits_ca.device),
            )
        loss_ca = ca_bce(logits_ca, targets_ca.float()).sum()

        #Compute normalized loss per sample
        wrong_pred    = th.zeros(policy[1].output_dim).to(logits_ca.device)
        sample_target = th.ones(policy[1].output_dim).to(logits_ca.device)
        max_loss_ca = th.sum(ca_bce(wrong_pred,sample_target),dim=-1).cpu()
        loss = th.sum(ca_bce(logits_ca, targets_ca.float()),dim=-1).cpu()
        loss_hist_ca, loss_bin_edges_ca = np.histogram((loss/max_loss_ca),bins=np.linspace(0,1,nbins))

        if other_models:
            logits_ca_mlp = other_models['mlp'][1](obs)
            loss_ca_mlp = th.sum(ca_bce(logits_ca_mlp, targets_ca.float()),dim=-1).cpu()
            lost_hist_ca_mlp = np.histogram((loss_ca_mlp/max_loss_ca), bins=np.linspace(0, 1, nbins))

            #recall for mlp
            preds_ca_mlp = (th.sigmoid(logits_ca_mlp) > 0.5).long()
            ca_true_pos_mlp = ((preds_ca_mlp == 1) & (targets_ca == 1)).sum().item()
            ca_targ_pos_mlp = targets_ca.sum().item()
            ca_recall_mlp = (ca_true_pos_mlp / ca_targ_pos_mlp) if ca_targ_pos_mlp > 0 else 0.0

            # logits_ca_raidnet_v1 = other_models['RAIDNET_V1'][1](obs_reshaped)
            logits_ca_raidnet_v1 = forward_in_batches(other_models['RAIDNET_V1'][1], obs_reshaped, batch_size=1024, return_device='cuda:0')
            loss_ca_raidnet_v1 = th.sum(ca_bce(logits_ca_raidnet_v1, targets_ca.float()),dim=-1).cpu()
            lost_hist_ca_raidnet_v1 = np.histogram((loss_ca_raidnet_v1/max_loss_ca), bins=np.linspace(0, 1, nbins))
            #recall for v1
            preds_ca_v1 = (th.sigmoid(logits_ca_raidnet_v1) > 0.5).long()
            ca_true_pos_v1 = ((preds_ca_v1 == 1) & (targets_ca == 1)).sum().item()
            ca_targ_pos_v1 = targets_ca.sum().item()
            ca_recall_v1 = (ca_true_pos_v1 / ca_targ_pos_v1) if ca_targ_pos_v1 > 0 else 0.0

            #CA precision
            ca_pred_pos_mlp = preds_ca_mlp.sum().item()
            ca_precision_mlp = (ca_true_pos_mlp / ca_pred_pos_mlp) if ca_pred_pos_mlp > 0 else 0.0
            ca_pred_pos_v1 = preds_ca_v1.sum().item()
            ca_precision_v1 = (ca_true_pos_v1 / ca_pred_pos_v1) if ca_pred_pos_v1 > 0 else 0.0

           # accuracty
            ca_total_correct_mlp = (preds_ca_mlp == targets_ca).sum().item()
            ca_total_mlp = targets_ca.numel()
            ca_acc_mlp = (ca_total_correct_mlp / ca_total_mlp) if ca_total_mlp > 0 else 0.0
            ca_total_correct_v1 = (preds_ca_v1 == targets_ca).sum().item()
            ca_total_v1 = targets_ca.numel()
            ca_acc_v1 = (ca_total_correct_v1 / ca_total_v1) if ca_total_v1 > 0 else 0.0

        preds_ca = (th.sigmoid(logits_ca) > 0.5).long()
        ca_total_correct += (preds_ca == targets_ca).sum().item()
        ca_total += targets_ca.numel()
        ca_pred_pos += preds_ca.sum().item()
        ca_pred_pos_per_sample = preds_ca.sum(dim=-1)
        ca_targ_pos += targets_ca.sum().item()
        ca_targ_pos_per_sample = targets_ca.sum(dim=-1)
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
        ca_overpredictions = (ca_pred_pos_per_sample/logits_ca.shape[-1]) -(ca_targ_pos_per_sample/logits_ca.shape[-1])
        ca_overprediction_avg = ca_overpredictions.mean().item() if ca_overpredictions.numel() > 0 else 0.0
        ca_f1 = (2 * ca_precision * ca_recall / (ca_precision + ca_recall)) if (ca_precision + ca_recall) > 0 else 0.0

        # L1 per‑batch per‑class accuracy & counts
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
        pc_str = ",".join(f"{a:.2f}" for a in batch_per_class_acc)
        cnt_str = '['+",".join(str(int(x)) for x in batch_counts.tolist())+']'

        # CA per‑batch #predicted 1s / #target 1s
        with th.no_grad():
            batch_pred_pos = int((th.sigmoid(logits_ca) > 0.5).long().sum().item())
            batch_targ_pos = int(targets_ca.sum().item())

    # Compute confusion matrices
    l1_confusion_matrix = ConfusionMatrixDisplay.from_predictions(targets_l1.cpu().ravel(), preds_l1.cpu().ravel(),labels=[0,1,2],normalize='true',cmap='Blues',im_kw={'vmax': 1.},values_format='.3g')
    ca_confusion_matrix = ConfusionMatrixDisplay.from_predictions(targets_ca.cpu().ravel(), preds_ca.cpu().ravel(),labels=[0,1],normalize='true',cmap='Blues',im_kw={'vmax': 1.},values_format='.3g')

    # Package metrics
    metrics = {
        # L1 (multi-class)
        'l1_confusion_matrix': l1_confusion_matrix,
        'l1_loss': loss_l1,                          # total loss
        'l1_loss_bin_edges': loss_bin_edges_l1,  # per sample loss
        'l1_loss_hist': loss_hist_l1,                # histogram of normalized loss
        'l1_per_class_accuracy': per_class_acc,
        'l1_overall_accuracy': l1_acc,
        'l1_class_names': ['L1_c0', 'L1_c1', 'L1_c2'],

        # CA (binary)
        'ca_confusion_matrix': ca_confusion_matrix,
        'ca_loss': loss_ca,                          # total loss
        'ca_loss_bin_edges': loss_bin_edges_ca, # per sample loss
        'ca_loss_hist': loss_hist_ca,                # histogram of normalized loss
        'ca_precision': ca_precision,
        'ca_recall': ca_recall,
        'ca_overprediction': ca_overpredictions, #global
        'ca_overprediction_avg': ca_overprediction_avg, #average overpredictions per sample
        'ca_f1': ca_f1,
        'ca_class_names': ['CA_0', 'CA_1'],
    }
    if other_models:
        baseline_metrics = {}
        baseline_metrics['l1_loss_hist_mlp'] = lost_hist_l1_mlp[0]
        baseline_metrics['l1_loss_hist_raidnet_v1'] = lost_hist_l1_raidnet_v1[0]
        baseline_metrics['l1_loss_bin_edges_mlp'] = lost_hist_l1_mlp[1]
        baseline_metrics['l1_loss_bin_edges_raidnet_v1'] = lost_hist_l1_raidnet_v1[1]
        baseline_metrics['ca_loss_hist_mlp'] = lost_hist_ca_mlp[0]
        baseline_metrics['ca_loss_hist_raidnet_v1'] = lost_hist_ca_raidnet_v1[0]
        baseline_metrics['ca_loss_bin_edges_mlp'] = lost_hist_ca_mlp[1]
        baseline_metrics['ca_loss_bin_edges_raidnet_v1'] = lost_hist_ca_raidnet_v1[1]
        baseline_metrics['ca_recall_mlp'] = ca_recall_mlp
        baseline_metrics['ca_recall_raidnet_v1'] = ca_recall_v1
        baseline_metrics['ca_precision_mlp'] = ca_precision_mlp
        baseline_metrics['ca_precision_raidnet_v1'] = ca_precision_v1
        baseline_metrics['ca_acc_mlp'] = ca_acc_mlp
        baseline_metrics['ca_acc_raidnet_v1'] = ca_acc_v1
    else:
        baseline_metrics = None
    return metrics, baseline_metrics

def overlay_histograms(ax, bin_edges, series, labels, colors,
                       alpha=0.65, edgecolor='black', linewidth=0.6,
                       jitter_frac=0.05,  # small x-offset so bars don't perfectly coincide
                       zbase=10, draw_outlines=True):
    """
    Overplot multiple histograms (same bins) so all bars remain visible.
    Draws largest bars first and smallest last (front-most per bin).
    """
    assert all(len(s) == len(bin_edges) - 1 for s in series), "All hist series must share the same bins."

    n_models = len(series)
    widths = np.diff(bin_edges)
    lefts = bin_edges[:-1]

    # Draw per-bin in order: largest -> smallest
    for left, width, i in zip(lefts, widths, range(len(widths))):
        heights = [s[i] for s in series]
        order = np.argsort(heights)[::-1]  # largest first
        for rank, m in enumerate(order):
            h = heights[m]
            if h <= 0:
                continue
            # small symmetric jitter so edges are visible
            offset = (rank - (n_models - 1) / 2.0) * jitter_frac * width
            rect = Rectangle((left + offset, 0.0), width, h,
                             facecolor=colors[m], edgecolor=edgecolor,
                             linewidth=linewidth, alpha=alpha,
                             zorder=zbase + rank)
            ax.add_patch(rect)

    # thin line at the top of bars so overlapping heights are readable
    if draw_outlines:
        xs = np.repeat(bin_edges, 2)[1:-1]
        for m, (lab, col, hist) in enumerate(zip(labels, colors, series)):
            ys = np.repeat(hist, 2)
            ax.plot(xs, ys, lw=1.0, color=col, alpha=0.95, label=lab, zorder=zbase + n_models + m + 1)
    else:
        for lab, col in zip(labels, colors):
            ax.bar([], [], color=col, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=lab)



@contextmanager
def classy_mathtext():
    """Use mathtext with a clean sans-serif style (no LaTeX dependency)."""
    with plt.rc_context({
        "text.usetex": False,            # don't call external LaTeX
        "mathtext.fontset": "stixsans",  # nice sans-serif math
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],  # bundled font
        "axes.unicode_minus": False,     # fix minus sign with sans fonts
    }):
        yield


def print_and_plot_metrics(metrics: dict, baselines: dict = None):
    """
    Print metrics (unchanged) and overlay histograms for RAID-Net V2 + baselines.
    """
    # ----------------- PRINTS -----------------
    print("L1 Loss:", metrics['l1_loss'].item())
    print("L1 Per-Class Accuracy:", metrics['l1_per_class_accuracy'])
    print("L1 Overall Accuracy:", metrics['l1_overall_accuracy'])
    print('-' * 40)
    print("CA Loss:", metrics['ca_loss'].item())
    print("CA Precision:", metrics['ca_precision'])
    print("CA Recall:", metrics['ca_recall'])
    print("CA F1 Score:", metrics['ca_f1'])
    print("CA Overprediction Rate:", metrics['ca_overprediction_avg'])

    if baselines:
        print('MLP Baseline CA Recall:', baselines['ca_recall_mlp'])
        print('MLP Baseline CA Precision:', baselines['ca_precision_mlp'])
        print('MLP Baseline CA Accuracy:', baselines['ca_acc_mlp'])
        print('RAID-Net V1 Baseline CA Recall:', baselines['ca_recall_raidnet_v1'])
        print('RAID-Net V1 Baseline CA Precision:', baselines['ca_precision_raidnet_v1'])
        print('RAID-Net V1 Baseline CA Accuracy:', baselines['ca_acc_raidnet_v1'])

    # Save confusion matrices
    _ = metrics['l1_confusion_matrix'].figure_.savefig('../nuplan/evaluation/l1_confusion_matrix.png')
    _ = metrics['ca_confusion_matrix'].figure_.savefig('../nuplan/evaluation/ca_confusion_matrix.png')

    # ----------------- L1 HISTOGRAM -----------------
    with classy_mathtext():
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},
                               figsize=(6.8, 8))

        l1_series = [metrics['l1_loss_hist']]
        l1_labels = [r'$\pi^{\text{RAIDN V2}}$']
        l1_colors = ['#355C7D']

        if baselines:
            if 'l1_loss_hist_mlp' in baselines:
                l1_series.append(baselines['l1_loss_hist_mlp'])
                l1_labels.append(r'$\pi^{\text{MLP}}$')
                l1_colors.append('#E67E22')
            if 'l1_loss_hist_raidnet_v1' in baselines:
                l1_series.append(baselines['l1_loss_hist_raidnet_v1'])
                l1_labels.append(r'$\pi^{\text{RAIDN V1}}$')
                l1_colors.append('#27AE60')

        overlay_histograms(
            ax=ax[0],
            bin_edges=metrics['l1_loss_bin_edges'],
            series=l1_series, labels=l1_labels, colors=l1_colors,
            alpha=0.65, jitter_frac=0.05, draw_outlines=True
        )
        ax[0].set_ylabel(r'#')
        ax[0].set_xlabel(r'$\mathrm{CE}(\pi_g,\tilde{g}^{\star})$')
        ax[0].set_xlim(metrics['l1_loss_bin_edges'][0],
                       metrics['l1_loss_bin_edges'][-1])
        ymax_l1 = max(s.max() for s in l1_series)
        ax[0].set_ylim(0, max(1, int(ymax_l1 * 1.12)))
        ax[0].legend(frameon=False)

        metrics['l1_confusion_matrix'].plot(ax=ax[1], cmap='Blues',
                                            im_kw={'vmax': 1.},
                                            values_format='.3g')
        plt.tight_layout()
        plt.savefig('../nuplan/evaluation/l1_metrics.png')
        plt.show()

    # ----------------- CA HISTOGRAM -----------------
    with classy_mathtext():
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},
                               figsize=(6.8, 8))

        ca_series = [metrics['ca_loss_hist']]
        ca_labels = [r'$\pi^{\text{RAIDN V2}}$']
        ca_colors = ['#355C7D']

        if baselines:
            if 'ca_loss_hist_mlp' in baselines:
                ca_series.append(baselines['ca_loss_hist_mlp'])
                ca_labels.append(r'$\pi^{\text{MLP}}$')
                ca_colors.append('#E67E22')
            if 'ca_loss_hist_raidnet_v1' in baselines:
                ca_series.append(baselines['ca_loss_hist_raidnet_v1'])
                ca_labels.append(r'$\pi^{\text{RAIDN V1}}$')
                ca_colors.append('#27AE60')

        overlay_histograms(
            ax=ax[0],
            bin_edges=metrics['ca_loss_bin_edges'],
            series=ca_series, labels=ca_labels, colors=ca_colors,
            alpha=0.65, jitter_frac=0.05, draw_outlines=True
        )
        ax[0].set_ylabel(r'#')
        ax[0].set_xlabel(
            r'$\text{Normalized CA }\ell(\pi_{\mu},\tilde{\mu}^{\star})$'
        )
        if baselines:
            #Find the last bin edge that contains any non-zero value from baselines
            last_nonzero_bin = 0
            for s in ca_series[1:]:
                nonzero_bins = np.where(s > 0)[0]
                if len(nonzero_bins) > 0:
                    last_nonzero_bin = max(last_nonzero_bin, nonzero_bins[-1])
            ax[0].set_xlim(metrics['ca_loss_bin_edges'][0],
                           metrics['ca_loss_bin_edges'][last_nonzero_bin + 1] + 1e-3)
        else:
            ax[0].set_xlim(metrics['ca_loss_bin_edges'][0],
                           metrics['ca_loss_bin_edges'][-1])
        ymax_ca = max(s.max() for s in ca_series)
        ax[0].set_ylim(0, max(1, int(ymax_ca * 1.12)))
        ax[0].legend(frameon=False)

        metrics['ca_confusion_matrix'].plot(ax=ax[1], cmap='Blues',
                                            im_kw={'vmax': 1.},
                                            values_format='.3g')
        plt.tight_layout()
        plt.savefig('../nuplan/evaluation/ca_metrics.png')
        plt.show()



def main(smpc_config,config):
    #Define the configuration for RAIDNET
    eval_other_models = True
    n_modes = [smpc_config['num_modes'] for _ in range(smpc_config['num_tvs'])]
    mode_map = dict(enumerate(product(*[range(n_modes[k]) for k in range(smpc_config['num_tvs'])])))
    observation_dim = smpc_config['num_tvs'] * (smpc_config['num_modes'] * (3*config['N']) + 2)
    ca_num = len(mode_map)*(config['N']-1)*smpc_config['num_tvs']
    l1_num = sum(n_modes)*(config['N']-1)*2
    num_layers = config['num_layers']
    hidden_dim = config['hidden_dim']
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu") 

    l1_dual_dim = [config['N']-1, n_modes, smpc_config['num_tvs']]
    ca_dual_dim = [config['N']-1, len(mode_map), smpc_config['num_tvs']]

    raidnet_config = {'num_tvs': smpc_config['num_tvs'], 'num_heads': config['num_heads'],'dropout_prob':config['dropout_prob']}
    
    #Initialize RAIDNET
    pred_mode = ['both duals','tertiary','binary']
    l1_policy = RAID_NET_V2(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, l1_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=l1_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['l1','tertiary','binary'])
    ca_policy = RAID_NET_V2(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, ca_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=ca_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['ca','binary','binary'])

    if eval_other_models: 
        with open('/home/mpc/nuplan-devkit/tutorials/mlp_training_config.yaml', 'r') as f:
            mlp_config = yaml.load(f,Loader=yaml.FullLoader)
        l1_policy_mlp = MLP(observation_dim, l1_num, hidden_layers=mlp_config['num_layers'], hidden_size=mlp_config['hidden_dim'],device=device,tertiary=True)
        ca_policy_mlp = MLP(observation_dim, ca_num, hidden_layers=mlp_config['num_layers'], hidden_size=mlp_config['hidden_dim'],device=device)
        l1_policy_mlp.to(device)
        ca_policy_mlp.to(device)

        checkpoint = []
        checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/MLP_NuPlan_N14_N_TV3_512_3_15-09-2025_10-43-29/MLP_NuPlan_N14_N_TV3_15-09-2025_10-43-29_L1_100.pt'))
        checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/MLP_NuPlan_N14_N_TV3_512_3_15-09-2025_10-43-29/MLP_NuPlan_N14_N_TV3_15-09-2025_10-43-29_CA_100.pt'))
        #L1
        l1_policy_mlp.load_state_dict(checkpoint[0]['model_state_dict'])
        #CA
        ca_policy_mlp.load_state_dict(checkpoint[1]['model_state_dict'])

        l1_policy_raidnet_v1 = RAID_NET(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, l1_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=l1_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['l1','tertiary','binary'])
        ca_policy_raidnet_v1 = RAID_NET(raidnet_config,int(observation_dim/(smpc_config['num_tvs'])), observation_dim, ca_num, config['N']-1, smpc_config['num_tvs'], num_layers//2, hidden_dim//2,lambda_dim=ca_num, lambda_ubd=smpc_config['l1_lmbd'], pred_mode=['ca','binary','binary'])
        l1_policy_raidnet_v1.to(device)
        ca_policy_raidnet_v1.to(device)

        checkpoint = []
        checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40_L1_4.pt'))
        checkpoint.append(th.load('/home/mpc/nuplan-devkit/nuplan/nn_models/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40_CA_4.pt'))
        #L1
        l1_policy_raidnet_v1.load_state_dict(checkpoint[0]['model_state_dict'])
        #CA
        ca_policy_raidnet_v1.load_state_dict(checkpoint[1]['model_state_dict'])

        other_models = {'mlp':[l1_policy_mlp,ca_policy_mlp],'RAIDNET_V1': [l1_policy_raidnet_v1,ca_policy_raidnet_v1]}
    else:
        other_models = None
    l1_policy.to(device)
    ca_policy.to(device)
    policy = [l1_policy, ca_policy]

    policy_type = 'RAIDNET'
    checkpoint = None

    metrics, baseline_metrics = evaluate(smpc_config,config,policy,device,policy_type,l1_dual_dim=l1_dual_dim,ca_dual_dim=ca_dual_dim,l1_num=l1_num,pred_mode=pred_mode,other_models=other_models)
    print_and_plot_metrics(metrics,baseline_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpc_config', required=False,type=str, default='/home/mpc/nuplan-devkit/nuplan/planning/simulation/planner/smpc_config.yaml')
    parser.add_argument('--config', required=False,type=str, default='/home/mpc/nuplan-devkit/tutorials/training_config.yaml')
    args = parser.parse_args()
    with open(args.smpc_config, 'r') as f:
        smpc_config = yaml.load(f,Loader=yaml.FullLoader)
    with open(args.config, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    main(smpc_config,config)