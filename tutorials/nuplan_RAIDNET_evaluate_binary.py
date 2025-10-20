from tutorials.policies import RAID_NET, MLP
from tutorials.raidnet import RAID_NET_V2

import torch as th
import argparse
import yaml
import numpy as np
from matplotlib.patches import Rectangle
from itertools import product
from tutorials.utils.replay_buffer import ReplayBuffer
from tutorials.utils.BC import BC
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pickle, gzip, os

try:
    from nuplan.planning.simulation.planner.utils.smpc_utils import to_tensor_var
except ImportError:
    def to_tensor_var(x, use_cuda: bool):
        t = th.as_tensor(x)
        return t.cuda(non_blocking=True) if use_cuda and th.cuda.is_available() else t

from contextlib import contextmanager

NUPLAN_ROOT_DIR = os.envrion['NUPLAN_ROOT_DIR']
# ============================ Utilities ============================

def forward_in_batches(model, x, batch_size=1024, device=None, return_device='cpu'):
    """Run `model(x)` in smaller batches to avoid OOM and concatenate outputs along dim=0."""
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


# DO NOT MODIFY THIS (per your request)
def tertiary2binary(logits, targets):
    logits = (logits[:, 0] + logits[:, 2])
    bce = th.nn.BCEWithLogitsLoss()
    # target = 2 -> 0, 1 -> 1, 0 -> 0
    targets = (targets == 2).long() | (targets == 1).long()
    loss_ca = bce(logits, targets.float()).sum()
    return loss_ca, logits, targets


def l1_make_targets_ternary(l1_duals: th.Tensor, lmbd: float) -> th.Tensor:
    """Map L1 dual magnitudes to {0,1,2} as you do in training."""
    return (l1_duals > 1e-3).int() + (l1_duals > (lmbd * 0.99)).int()


def l1_ternary_logits_to_binary_logits_via_fn(logits3_flat: th.Tensor,
                                              targets3_flat: th.Tensor):
    """
    Correct binary logit for {1,2} vs {0} is logsumexp(z1,z2) - z0.
    We pack that into col0 so your tertiary2binary (col0+col2) returns the same value.
    """
    z0, z1, z2 = logits3_flat[:, 0], logits3_flat[:, 1], logits3_flat[:, 2]
    ell_bin = th.logsumexp(th.stack([z1, z2], dim=1), dim=1) - z0
    logits3_for_fn = th.stack([ell_bin,
                               th.zeros_like(ell_bin),
                               th.zeros_like(ell_bin)], dim=1)
    return tertiary2binary(logits3_for_fn, targets3_flat)


def per_sample_norm_bce_hist(bin_logits_flat: th.Tensor,
                             bin_targets_flat: th.Tensor,
                             B: int, P: int, nbins: int = 100):
    """
    Turn elementwise BCE into a **per-sample** normalized loss histogram (like CA).
    Returns (hist, bin_edges, norm_losses_numpy).
    """
    logits = bin_logits_flat.view(B, P)
    targets = bin_targets_flat.view(B, P).float()

    bce = th.nn.BCEWithLogitsLoss(reduction='none')
    loss_elem = bce(logits, targets)  # (B,P)
    max_elem = bce(th.zeros_like(logits), th.ones_like(logits))

    loss_sample = loss_elem.sum(dim=1).detach().cpu().numpy()
    max_sample = max_elem.sum(dim=1).detach().cpu().numpy()
    norm = loss_sample / np.maximum(max_sample, 1e-12)

    hist, edges = np.histogram(norm, bins=np.linspace(0, 1, nbins))
    return hist, edges, norm


def overlay_histograms(ax, bin_edges, series, labels, colors,
                       alpha=0.65, edgecolor='black', linewidth=0.6,
                       jitter_frac=0.05, zbase=10, draw_outlines=True):
    """Overlay histograms (same bins) so bars stay visible."""
    assert all(len(s) == len(bin_edges) - 1 for s in series), "All hist series must share the same bins."
    n_models = len(series)
    widths = np.diff(bin_edges)
    lefts = bin_edges[:-1]

    for left, width, i in zip(lefts, widths, range(len(widths))):
        heights = [s[i] for s in series]
        order = np.argsort(heights)[::-1]  # largest first
        for rank, m in enumerate(order):
            h = heights[m]
            if h <= 0:
                continue
            offset = (rank - (n_models - 1) / 2.0) * jitter_frac * width
            rect = Rectangle((left + offset, 0.0), width, h,
                             facecolor=colors[m], edgecolor=edgecolor,
                             linewidth=linewidth, alpha=alpha,
                             zorder=zbase + rank)
            ax.add_patch(rect)

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
    """Mathtext with sans-serif style (no LaTeX dependency)."""
    with plt.rc_context({
        "text.usetex": False,
        "mathtext.fontset": "stixsans",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
    }):
        yield


# ============================ Evaluation ============================

def evaluate(smpc_config, config, policy, device, policy_type,
             l1_dual_dim, ca_dual_dim, l1_num, pred_mode, other_models):

    # ---- load & prep data ----
    with gzip.open(config['eval_data_dir'], 'rb') as file:
        expert_data = pickle.load(file)

    expert_data['observation']   = [x for x in expert_data['observation']   if len(x) > 0]
    expert_data['optimal_duals'] = [x for x in expert_data['optimal_duals'] if len(x) > 0]
    expert_data['dual_class']    = [x for x in expert_data['dual_class']    if len(x) > 0]

    observation   = np.squeeze(np.concatenate(expert_data['observation']), axis=1)
    optimal_duals = np.concatenate(expert_data['optimal_duals'])

    rb = ReplayBuffer(config['max_replay_buffer_size'], training_dataset=False)
    rb.obs, rb.acs = observation, optimal_duals
    rb.opt_duals   = optimal_duals
    rb.terminals   = np.zeros_like(observation)
    rb.next_obs    = np.zeros_like(observation)
    rb.rews        = np.zeros_like(observation)
    rb.smpc_params_dim = expert_data['smpc_params_dim']
    rb.dual_classes = np.array([d for lst in expert_data['dual_class'] for d in lst])

    feat = np.load(config['feature_stat_path'])
    rb.normalize4evaluation(l1_num, feature_mean=feat['feature_mean'], feature_cov=feat['feature_cov'],
                            target_mean=None, target_cov=None, l1_pred_mode=pred_mode[0])
    rb.set_weights()

    # ---- load weights ----
    print('Loading pretrained model...')
    l1_ckpt = th.load('../nuplan/nn_models/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06_L1_100.pt')
    ca_ckpt = th.load('../nuplan/nn_models/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06/RAIDNET_V2_NuPlan_N14_N_TV3_15-09-2025_11-26-06_CA_100.pt')
    policy[0].load_state_dict(l1_ckpt['model_state_dict'])
    policy[1].load_state_dict(ca_ckpt['model_state_dict'])

    print('EVALUATION STARTED'.center(80, '*'))
    bc_learner = BC(policy=policy, optimizer=config['optimizer'], optim_lr=config['lr'],
                    demonstrations=rb, rng=np.random.default_rng(0), device=device,
                    batch_size=config['batch_size'], logger=None, normalize=False,
                    config=config, normalize_obs=False, l1_dual_dim=l1_dual_dim,
                    ca_dual_dim=ca_dual_dim, joint_dual_pred=config['joint_dual_pred'])

    # ---- tensors ----
    policy[0].eval(); policy[1].eval()
    obs  = to_tensor_var(rb.obs,  use_cuda=(device.type == 'cuda')).to(device)
    acts = to_tensor_var(rb.acs,  use_cuda=(device.type == 'cuda')).to(device)
    B    = obs.shape[0]
    print(f"Evaluating on {B} samples...")
    obs_reshaped = obs.view(B, smpc_config['num_tvs'], -1)
    P_l1 = policy[0].output_dim
    nbins = 100

    # ---- forward ----
    with th.no_grad():
        # L1 (RAID-Net V2) → ternary → binary using your fn
        logits_l1_ternary = policy[0](obs_reshaped)               # (B, P_l1, 3)
        l1_duals = acts[:, :P_l1]                                 # (B, P_l1)
        targets_l1_ternary = l1_make_targets_ternary(l1_duals, smpc_config['l1_lmbd'])
        flat_logits3  = logits_l1_ternary.reshape(-1, 3)
        flat_targets3 = targets_l1_ternary.reshape(-1)

        l1_loss_total_scalar, l1_bin_logits_flat, l1_bin_targets_flat = \
            l1_ternary_logits_to_binary_logits_via_fn(flat_logits3, flat_targets3)

        # per-sample normalized BCE histogram (same normalization as CA)
        loss_hist_l1, loss_bin_edges_l1, _ = \
            per_sample_norm_bce_hist(l1_bin_logits_flat, l1_bin_targets_flat, B, P_l1, nbins)

        # binary predictions & accuracy
        preds_l1_bin = (th.sigmoid(l1_bin_logits_flat) > 0.5).long()
        l1_total_correct = int((preds_l1_bin == l1_bin_targets_flat).sum().item())
        l1_total = int(l1_bin_targets_flat.numel())
        l1_acc = (l1_total_correct / l1_total) if l1_total > 0 else 0.0

        # per-class accuracy
        l1_counts = th.zeros(2, dtype=th.long)
        l1_correct_per_class = th.zeros(2, dtype=th.long)
        for c in (0, 1):
            m = (l1_bin_targets_flat == c)
            l1_counts[c] += int(m.sum())
            if m.any():
                l1_correct_per_class[c] += int((preds_l1_bin[m] == c).sum())
        per_class_acc = [
            (l1_correct_per_class[c].item() / max(1, l1_counts[c].item())) for c in (0, 1)
        ]

        # confusion matrix (binary)
        l1_confusion_matrix = ConfusionMatrixDisplay.from_predictions(
            l1_bin_targets_flat.detach().cpu().numpy(),
            preds_l1_bin.detach().cpu().numpy(),
            labels=[0, 1], normalize='true', cmap='Blues',
            im_kw={'vmax': 1.}, values_format='.3g'
        )

        # ----- CA forward + metrics -----
        logits_ca = policy[1](obs_reshaped)                        # (B, P_ca)
        targets_ca = acts[:, P_l1:]                                # (B, P_ca)
        pos_w = bc_learner.pos_weight_hi  # t=0
        ca_bce = th.nn.BCEWithLogitsLoss(reduction='none',
                    pos_weight=th.full((policy[1].output_dim,), pos_w, device=logits_ca.device))
        loss_ca_total = ca_bce(logits_ca, targets_ca.float()).sum()

        wrong_pred    = th.zeros(policy[1].output_dim, device=logits_ca.device)
        sample_target = th.ones(policy[1].output_dim,  device=logits_ca.device)
        max_loss_ca = th.sum(ca_bce(wrong_pred, sample_target), dim=-1).cpu()   # (B,)
        loss_ca_per_sample = th.sum(ca_bce(logits_ca, targets_ca.float()), dim=-1).cpu()
        loss_hist_ca, loss_bin_edges_ca = np.histogram((loss_ca_per_sample / max_loss_ca),
                                                       bins=np.linspace(0, 1, nbins))
        preds_ca = (th.sigmoid(logits_ca) > 0.5).long()
        ca_total_correct = int((preds_ca == targets_ca).sum().item())
        ca_total = int(targets_ca.numel())
        ca_acc = (ca_total_correct / ca_total) if ca_total > 0 else 0.0

        ca_true_pos = int(((preds_ca == 1) & (targets_ca == 1)).sum().item())
        ca_pred_pos = int(preds_ca.sum().item())
        ca_targ_pos = int(targets_ca.sum().item())
        ca_precision = (ca_true_pos / ca_pred_pos) if ca_pred_pos > 0 else 0.0
        ca_recall    = (ca_true_pos / ca_targ_pos) if ca_targ_pos > 0 else 0.0
        ca_f1 = (2 * ca_precision * ca_recall / (ca_precision + ca_recall)) if (ca_precision + ca_recall) > 0 else 0.0
        ca_overpredictions = (preds_ca.sum(dim=-1) / logits_ca.shape[-1]) - (targets_ca.sum(dim=-1) / logits_ca.shape[-1])
        ca_overprediction_avg = float(ca_overpredictions.mean().item())

        # ----- Baselines (optional) -----
        if other_models:
            # L1 MLP
            logits_l1_mlp_ternary = other_models['mlp'][0](obs)  # (B, P_l1, 3)
            flat_logits3_mlp  = logits_l1_mlp_ternary.reshape(-1, 3)
            flat_targets3     = targets_l1_ternary.reshape(-1)
            _, bin_logits_flat_mlp, bin_targets_flat_mlp = \
                l1_ternary_logits_to_binary_logits_via_fn(flat_logits3_mlp, flat_targets3)
            hist_l1_mlp, edges_l1_mlp, _ = \
                per_sample_norm_bce_hist(bin_logits_flat_mlp, bin_targets_flat_mlp, B, P_l1, nbins)

            # L1 RAIDNET_V1
            logits_l1_v1_ternary = forward_in_batches(other_models['RAIDNET_V1'][0],
                                                      obs_reshaped, batch_size=1024, return_device='cuda:0')
            flat_logits3_v1  = logits_l1_v1_ternary.reshape(-1, 3)
            _, bin_logits_flat_v1, bin_targets_flat_v1 = \
                l1_ternary_logits_to_binary_logits_via_fn(flat_logits3_v1, flat_targets3)
            hist_l1_v1, edges_l1_v1, _ = \
                per_sample_norm_bce_hist(bin_logits_flat_v1, bin_targets_flat_v1, B, P_l1, nbins)

            # CA baselines
            logits_ca_mlp = other_models['mlp'][1](obs)
            loss_ca_mlp = th.sum(ca_bce(logits_ca_mlp, targets_ca.float()), dim=-1).cpu()
            lost_hist_ca_mlp = np.histogram((loss_ca_mlp / max_loss_ca), bins=np.linspace(0, 1, nbins))

            logits_ca_v1 = forward_in_batches(other_models['RAIDNET_V1'][1], obs_reshaped,
                                              batch_size=1024, return_device='cuda:0')
            loss_ca_v1 = th.sum(ca_bce(logits_ca_v1, targets_ca.float()), dim=-1).cpu()
            lost_hist_ca_v1 = np.histogram((loss_ca_v1 / max_loss_ca), bins=np.linspace(0, 1, nbins))

    # ---- Pack metrics ----
    metrics = {
        # L1 (binary)
        'l1_confusion_matrix': l1_confusion_matrix,
        'l1_loss': l1_loss_total_scalar,
        'l1_loss_bin_edges': loss_bin_edges_l1,
        'l1_loss_hist': loss_hist_l1,
        'l1_per_class_accuracy': per_class_acc,
        'l1_overall_accuracy': l1_acc,
        'l1_class_names': ['L1_0', 'L1_1'],

        # CA (binary)
        'ca_confusion_matrix': ConfusionMatrixDisplay.from_predictions(
            targets_ca.detach().cpu().ravel(),
            preds_ca.detach().cpu().ravel(),
            labels=[0, 1], normalize='true', cmap='Blues', im_kw={'vmax': 1.}, values_format='.3g'
        ),
        'ca_loss': loss_ca_total,
        'ca_loss_bin_edges': loss_bin_edges_ca,
        'ca_loss_hist': loss_hist_ca,
        'ca_precision': ca_precision,
        'ca_recall': ca_recall,
        'ca_overprediction': ca_overpredictions,
        'ca_overprediction_avg': ca_overprediction_avg,
        'ca_f1': ca_f1,
        'ca_class_names': ['CA_0', 'CA_1'],
    }

    if other_models:
        baseline_metrics = {
            'l1_loss_hist_mlp': hist_l1_mlp,
            'l1_loss_hist_raidnet_v1': hist_l1_v1,
            'l1_loss_bin_edges_mlp': edges_l1_mlp,
            'l1_loss_bin_edges_raidnet_v1': edges_l1_v1,
            'ca_loss_hist_mlp': lost_hist_ca_mlp[0],
            'ca_loss_hist_raidnet_v1': lost_hist_ca_v1[0],
            'ca_loss_bin_edges_mlp': lost_hist_ca_mlp[1],
            'ca_loss_bin_edges_raidnet_v1': lost_hist_ca_v1[1],
        }
    else:
        baseline_metrics = None

    return metrics, baseline_metrics


# ============================ Plotting ============================

def print_and_plot_metrics(metrics: dict, baselines: dict = None):
    """Print metrics; overlay histograms for RAID-Net V2 + baselines; make labels & CM numbers larger/bold."""
    # ---------- PRINTS ----------
    print("L1 Loss:", float(metrics['l1_loss'].item()))
    print("L1 Per-Class Accuracy:", metrics['l1_per_class_accuracy'])
    print("L1 Overall Accuracy:", metrics['l1_overall_accuracy'])
    print('-' * 40)
    print("CA Loss:", float(metrics['ca_loss'].item()))
    print("CA Precision:", metrics['ca_precision'])
    print("CA Recall:", metrics['ca_recall'])
    print("CA F1 Score:", metrics['ca_f1'])
    print("CA Overprediction Rate:", metrics['ca_overprediction_avg'])

    # Save raw confusion matrices (separate pngs) before we re-draw them with bigger fonts
    _ = metrics['l1_confusion_matrix'].figure_.savefig('../nuplan/evaluation/l1_confusion_matrix.png')
    _ = metrics['ca_confusion_matrix'].figure_.savefig('../nuplan/evaluation/ca_confusion_matrix.png')

    label_fs = 16     # axis label fontsize (bigger)
    tick_fs  = 12     # tick fontsize
    cm_num_fs = 14    # confusion matrix number fontsize

    # ---------- L1 HIST + CM ----------
    with classy_mathtext():
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},
                               figsize=(7.5, 9.0))

        l1_series = [metrics['l1_loss_hist']]
        l1_labels = [r'$\pi^{\text{class}}$']
        l1_colors = ['#355C7D']

        if baselines:
            if 'l1_loss_hist_mlp' in baselines:
                l1_series.append(baselines['l1_loss_hist_mlp'])
                l1_labels.append(r'$\pi^{\text{MLP}}$')
                l1_colors.append('#E67E22')
            if 'l1_loss_hist_raidnet_v1' in baselines:
                l1_series.append(baselines['l1_loss_hist_raidnet_v1'])
                l1_labels.append(r'$\pi^{\text{RAIDN}}$')
                l1_colors.append('#27AE60')

        overlay_histograms(
            ax=ax[0],
            bin_edges=metrics['l1_loss_bin_edges'],
            series=l1_series, labels=l1_labels, colors=l1_colors,
            alpha=0.65, jitter_frac=0.05, draw_outlines=True
        )
        ax[0].set_ylabel(r'#', fontsize=label_fs)
        ax[0].set_xlabel(r'$\text{Normalized } \mathrm{BCE}(\tilde{g},\tilde{g}^{\star})$', fontsize=label_fs)
        ax[0].tick_params(axis='both', labelsize=tick_fs)
        ax[0].set_xlim(metrics['l1_loss_bin_edges'][0], metrics['l1_loss_bin_edges'][-1])
        ymax_l1 = max(s.max() for s in l1_series)
        ax[0].set_ylim(0, max(1, int(ymax_l1 * 1.12)))
        ax[0].legend(frameon=False, loc='upper right', fontsize=16)

        # Re-plot CM so we can style numbers
        disp = metrics['l1_confusion_matrix']
        disp.plot(ax=ax[1], cmap='Blues', im_kw={'vmax': 1.}, values_format='.3g', colorbar=True)
        ax[1].set_xlabel('Predicted label', fontsize=label_fs)
        ax[1].set_ylabel('True label', fontsize=label_fs)
        ax[1].tick_params(axis='both', labelsize=tick_fs)

        # make the numbers bigger and bold
        for text in ax[1].texts:
            text.set_fontsize(cm_num_fs)
            text.set_fontweight('bold')

        plt.tight_layout()
        os.makedirs('../nuplan/evaluation', exist_ok=True)
        plt.savefig('../nuplan/evaluation/l1_metrics.png', dpi=150)
        plt.show()

    # ---------- CA HIST + CM ----------
    with classy_mathtext():
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},
                               figsize=(7.5, 9.0))

        ca_series = [metrics['ca_loss_hist']]
        ca_labels = [r'$\pi^{\text{class}}$']
        ca_colors = ['#355C7D']

        if baselines:
            if 'ca_loss_hist_mlp' in baselines:
                ca_series.append(baselines['ca_loss_hist_mlp'])
                ca_labels.append(r'$\pi^{\text{MLP}}$')
                ca_colors.append('#E67E22')
            if 'ca_loss_hist_raidnet_v1' in baselines:
                ca_series.append(baselines['ca_loss_hist_raidnet_v1'])
                ca_labels.append(r'$\pi^{\text{RAIDN}}$')
                ca_colors.append('#27AE60')

        overlay_histograms(
            ax=ax[0],
            bin_edges=metrics['ca_loss_bin_edges'],
            series=ca_series, labels=ca_labels, colors=ca_colors,
            alpha=0.65, jitter_frac=0.05, draw_outlines=True
        )
        ax[0].set_ylabel(r'#', fontsize=label_fs)
        ax[0].set_xlabel(r'$\text{Normalized } \mathrm{BCE}(\tilde{\mu},\tilde{\mu}^{\star})$', fontsize=label_fs)
        ax[0].tick_params(axis='both', labelsize=tick_fs)

        ymax_ca = max(s.max() for s in ca_series)
        ax[0].set_ylim(0, max(1, int(ymax_ca * 1.12)))
        ax[0].legend(frameon=False, fontsize=16)

        disp = metrics['ca_confusion_matrix']
        disp.plot(ax=ax[1], cmap='Blues', im_kw={'vmax': 1.}, values_format='.3g', colorbar=True)
        ax[1].set_xlabel('Predicted label', fontsize=label_fs)
        ax[1].set_ylabel('True label', fontsize=label_fs)
        ax[1].tick_params(axis='both', labelsize=tick_fs)
        for text in ax[1].texts:
            text.set_fontsize(cm_num_fs)
            text.set_fontweight('bold')

        plt.tight_layout()
        os.makedirs('../nuplan/evaluation', exist_ok=True)
        plt.savefig('../nuplan/evaluation/ca_metrics.png', dpi=150)
        plt.show()


# ============================ Main ============================

def main(smpc_config, config):
    eval_other_models = True

    n_modes = [smpc_config['num_modes'] for _ in range(smpc_config['num_tvs'])]
    mode_map = dict(enumerate(product(*[range(n_modes[k]) for k in range(smpc_config['num_tvs'])])))
    observation_dim = smpc_config['num_tvs'] * (smpc_config['num_modes'] * (3 * config['N']) + 2)
    ca_num = len(mode_map) * (config['N'] - 1) * smpc_config['num_tvs']
    l1_num = sum(n_modes) * (config['N'] - 1) * 2
    num_layers = config['num_layers']
    hidden_dim = config['hidden_dim']
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    l1_dual_dim = [config['N'] - 1, n_modes, smpc_config['num_tvs']]
    ca_dual_dim = [config['N'] - 1, len(mode_map), smpc_config['num_tvs']]

    raidnet_config = {'num_tvs': smpc_config['num_tvs'],
                      'num_heads': config['num_heads'],
                      'dropout_prob': config['dropout_prob']}

    # RAID-Net V2
    pred_mode = ['both duals', 'tertiary', 'binary']
    l1_policy = RAID_NET_V2(raidnet_config, int(observation_dim / smpc_config['num_tvs']),
                            observation_dim, l1_num, config['N'] - 1, smpc_config['num_tvs'],
                            num_layers // 2, hidden_dim // 2,
                            lambda_dim=l1_num, lambda_ubd=smpc_config['l1_lmbd'],
                            pred_mode=['l1', 'tertiary', 'binary'])
    ca_policy = RAID_NET_V2(raidnet_config, int(observation_dim / smpc_config['num_tvs']),
                            observation_dim, ca_num, config['N'] - 1, smpc_config['num_tvs'],
                            num_layers // 2, hidden_dim // 2,
                            lambda_dim=ca_num, lambda_ubd=smpc_config['l1_lmbd'],
                            pred_mode=['ca', 'binary', 'binary'])

    l1_policy.to(device)
    ca_policy.to(device)
    policy = [l1_policy, ca_policy]

    # Baselines
    if eval_other_models:
        with open('mlp_training_config.yaml', 'r') as f:
            mlp_config = yaml.load(f, Loader=yaml.FullLoader)
        l1_policy_mlp = MLP(observation_dim, l1_num,
                            hidden_layers=mlp_config['num_layers'],
                            hidden_size=mlp_config['hidden_dim'],
                            device=device, tertiary=True)
        ca_policy_mlp = MLP(observation_dim, ca_num,
                            hidden_layers=mlp_config['num_layers'],
                            hidden_size=mlp_config['hidden_dim'],
                            device=device)
        l1_policy_mlp.to(device); ca_policy_mlp.to(device)

        ckpt = []
        ckpt.append(th.load(NUPLAN_ROOT_DIR+'/nuplan/nn_models/MLP_NuPlan_N14_N_TV3_512_3_15-09-2025_10-43-29/MLP_NuPlan_N14_N_TV3_15-09-2025_10-43-29_L1_100.pt'))
        ckpt.append(th.load(NUPLAN_ROOT_DIR+'/nuplan/nn_models/MLP_NuPlan_N14_N_TV3_512_3_15-09-2025_10-43-29/MLP_NuPlan_N14_N_TV3_15-09-2025_10-43-29_CA_100.pt'))
        l1_policy_mlp.load_state_dict(ckpt[0]['model_state_dict'])
        ca_policy_mlp.load_state_dict(ckpt[1]['model_state_dict'])

        l1_policy_v1 = RAID_NET(raidnet_config, int(observation_dim / smpc_config['num_tvs']),
                                observation_dim, l1_num, config['N'] - 1, smpc_config['num_tvs'],
                                num_layers // 2, hidden_dim // 2,
                                lambda_dim=l1_num, lambda_ubd=smpc_config['l1_lmbd'],
                                pred_mode=['l1', 'tertiary', 'binary'])
        ca_policy_v1 = RAID_NET(raidnet_config, int(observation_dim / smpc_config['num_tvs']),
                                observation_dim, ca_num, config['N'] - 1, smpc_config['num_tvs'],
                                num_layers // 2, hidden_dim // 2,
                                lambda_dim=ca_num, lambda_ubd=smpc_config['l1_lmbd'],
                                pred_mode=['ca', 'binary', 'binary'])
        l1_policy_v1.to(device); ca_policy_v1.to(device)

        ckpt = []
        ckpt.append(th.load(NUPLAN_ROOT_DIR+'/nuplan/nn_models/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40_L1_4.pt'))
        ckpt.append(th.load(NUPLAN_ROOT_DIR+'/nuplan/nn_models/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40/RAIDNET_V1_NuPlan_N14_N_TV3_15-09-2025_11-48-40_CA_4.pt'))
        l1_policy_v1.load_state_dict(ckpt[0]['model_state_dict'])
        ca_policy_v1.load_state_dict(ckpt[1]['model_state_dict'])

        other_models = {'mlp': [l1_policy_mlp, ca_policy_mlp],
                        'RAIDNET_V1': [l1_policy_v1, ca_policy_v1]}
    else:
        other_models = None

    metrics, baseline_metrics = evaluate(smpc_config, config, policy, device, 'RAIDNET',
                                         l1_dual_dim=l1_dual_dim, ca_dual_dim=ca_dual_dim,
                                         l1_num=l1_num, pred_mode=pred_mode,
                                         other_models=other_models)
    print_and_plot_metrics(metrics, baseline_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpc_config', required=False, type=str,
                        default=NUPLAN_ROOT_DIR+'/nuplan/planning/simulation/planner/smpc_config_eval.yaml')
    parser.add_argument('--config', required=False, type=str,
                        default=NUPLAN_ROOT_DIR+'/tutorials/training_config.yaml')
    args = parser.parse_args()
    with open(args.smpc_config, 'r') as f:
        smpc_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(smpc_config, config)
