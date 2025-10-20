from tutorials.policies import RAID_NET, MLP               # V1 + MLP (your MLP class supports joint binary)
from tutorials.raidnet import RAID_NET_V2                  # V2

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

NUPLAN_ROOT_DIR = os.environ['NUPLAN_ROOT_DIR']
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


# DO NOT MODIFY THIS (per your request) — not used for joint, but kept here.
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


# ============================ Evaluation (JOINT) ============================

def evaluate_joint(smpc_config, config, models, device, l1_num, ca_num):
    """
    Evaluate joint models: each returns flat logits of shape (B, l1_num + ca_num), all **binary**.
    - models: dict with keys {'V2', 'V1', 'MLP'} and nn.Module values
    Returns metrics dict with joint histograms and V2 L1 confusion matrix.
    """

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
    # normalize just like training (we only need feature stats)
    rb.normalize4evaluation(l1_num, feature_mean=feat['feature_mean'], feature_cov=feat['feature_cov'],
                            target_mean=None, target_cov=None)
    rb.set_weights()

    # ---- tensors ----
    obs  = to_tensor_var(rb.obs,  use_cuda=(device.type == 'cuda')).to(device)
    acts = to_tensor_var(rb.acs,  use_cuda=(device.type == 'cuda')).to(device)
    B    = obs.shape[0]
    print(f"Evaluating on {B} samples...")

    # Some models (V1/V2) expect (B, num_tvs, -1); MLP expects flat obs
    num_tvs = smpc_config['num_tvs']
    obs_reshaped = obs.view(B, num_tvs, -1)

    # ----- build targets -----
    # L1 binary targets from tertiary rule: 1 iff class==1; 0 otherwise.
    l1_duals = acts[:, :l1_num]
    targets_l1_ternary = l1_make_targets_ternary(l1_duals, smpc_config['l1_lmbd'])
    targets_l1_bin = (targets_l1_ternary == 2).long() | (targets_l1_ternary == 1).long()
    targets_ca     = acts[:, l1_num:l1_num + ca_num].long()       # (B, ca_num)

    # ----- eval all models -----
    NBINS = 100
    bin_edges = np.linspace(0.0, 1.0, NBINS)
    bce_elem = th.nn.BCEWithLogitsLoss(reduction='none')

    histograms = {}         # name -> (hist, bin_edges)
    v2_cm_disp = None       # confusion matrix display object for V2 L1
    v2_cm_ready = False

    for name, model in models.items():
        model = model.to(device).eval()
        with th.no_grad():
            if name == 'MLP':
                logits_all = forward_in_batches(model, obs, batch_size=1024, return_device='cpu')       # (B, l1+ca)
            else:
                logits_all = forward_in_batches(model, obs_reshaped, batch_size=1024, return_device='cpu')

        assert logits_all.shape[1] == (l1_num + ca_num), f"{name}: expected (B, {l1_num+ca_num}) got {tuple(logits_all.shape)}"

        logits_l1 = logits_all[:, :l1_num]
        logits_ca = logits_all[:, l1_num:l1_num + ca_num]

        # per-element BCE
        loss_elem_l1 = bce_elem(logits_l1, targets_l1_bin.float())
        loss_elem_ca = bce_elem(logits_ca, targets_ca.float())

        # per-element max loss (logit=0 for target=1)
        max_elem_l1 = bce_elem(th.zeros_like(logits_l1), th.ones_like(logits_l1))
        max_elem_ca = bce_elem(th.zeros_like(logits_ca), th.ones_like(logits_ca))

        # per-sample sums across (L1 + CA), normalized
        loss_per_sample = (loss_elem_l1.sum(dim=1) + loss_elem_ca.sum(dim=1)).cpu().numpy()
        max_per_sample  = (max_elem_l1.sum(dim=1) + max_elem_ca.sum(dim=1)).cpu().numpy()
        norm = loss_per_sample / np.maximum(max_per_sample, 1e-12)

        hist, _ = np.histogram(norm, bins=bin_edges)
        histograms[name] = (hist, bin_edges)

        # replace the CM block with this
        if (name == 'V2') and (not v2_cm_ready):
            preds_all = (th.sigmoid(th.cat([logits_l1, logits_ca], dim=1)) > 0.5).long().cpu().numpy().ravel()
            targs_all = th.cat([targets_l1_bin, targets_ca], dim=1).cpu().numpy().ravel()

            v2_cm_disp = ConfusionMatrixDisplay.from_predictions(
                targs_all, preds_all, labels=[0, 1], normalize='true',
                cmap='Blues', im_kw={'vmax': 1.}, values_format='.3g'
            )
            v2_cm_ready = True

    metrics = {
        'joint_loss_hists': histograms,          # dict: name -> (hist, bin_edges)
        'v2_l1_confusion': v2_cm_disp,           # ConfusionMatrixDisplay
    }
    return metrics


# ============================ Plotting (JOINT) ============================

def plot_joint_metrics(metrics: dict):
    """One figure: top = joint normalized BCE histogram (L1+CA) for V2/V1/MLP; bottom = V2 L1 confusion matrix."""
    histograms = metrics['joint_loss_hists']
    v2_cm_disp = metrics['v2_l1_confusion']

    # order + styles
    order = [k for k in ['V2', 'V1', 'MLP'] if k in histograms]
    labels = {
        'V2':  r'$\pi^{\text{class}}$',
        'V1':  r'$\pi^{\text{RAIDN}}$',
        'MLP': r'$\pi^{\text{MLP}}$',
    }
    colors = {
        'V2': '#355C7D',
        'V1': '#27AE60',
        'MLP': '#E67E22',
    }

    series = [histograms[n][0] for n in order]
    bin_edges = next(iter(histograms.values()))[1]  # all share the same bins

    label_fs = 16
    tick_fs  = 12
    cm_num_fs = 14

    with classy_mathtext():
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},
                               figsize=(7.8, 9.2))

        # TOP: joint loss hist (L1+CA)
        overlay_histograms(
            ax=ax[0],
            bin_edges=bin_edges,
            series=series,
            labels=[labels[n] for n in order],
            colors=[colors[n] for n in order],
            alpha=0.65, jitter_frac=0.05, draw_outlines=True
        )
        ax[0].set_ylabel(r'#', fontsize=label_fs)
        ax[0].set_xlabel(r'$\mathrm{Normalized\ BCE}([\tilde{\mu},\,\tilde{g}],\ [\tilde{\mu}^\star,\,\tilde{g}^\star])$', fontsize=label_fs)

        ax[0].tick_params(axis='both', labelsize=tick_fs)
        ymax = max(s.max() for s in series)
        ax[0].set_ylim(0, max(1, int(ymax * 1.12)))
        ax[0].legend(frameon=False, loc='upper right', fontsize=16)

        # BOTTOM: V2 L1 CM (normalized)
        if v2_cm_disp is not None:
            disp = v2_cm_disp
            disp.plot(ax=ax[1], cmap='Blues', im_kw={'vmax': 1.}, values_format='.3g', colorbar=True)
            ax[1].set_xlabel('Predicted label', fontsize=label_fs)
            ax[1].set_ylabel('True label', fontsize=label_fs)
            ax[1].tick_params(axis='both', labelsize=tick_fs)
            for text in ax[1].texts:
                text.set_fontsize(cm_num_fs)
                text.set_fontweight('bold')

        plt.tight_layout()
        os.makedirs('../nuplan/evaluation', exist_ok=True)
        plt.savefig('../nuplan/evaluation/joint_metrics.png', dpi=600)
        plt.show()


# ============================ Main ============================

def main(smpc_config, config):
    # problem sizes
    n_modes = [smpc_config['num_modes'] for _ in range(smpc_config['num_tvs'])]
    mode_map = dict(enumerate(product(*[range(n_modes[k]) for k in range(smpc_config['num_tvs'])])))
    observation_dim = smpc_config['num_tvs'] * (smpc_config['num_modes'] * (3 * config['N']) + 2)
    ca_num = len(mode_map) * (config['N'] - 1) * smpc_config['num_tvs']
    l1_num = sum(n_modes) * (config['N'] - 1) * 2
    total_out_dim = l1_num + ca_num
    num_layers = config['num_layers']
    hidden_dim = config['hidden_dim']
    # device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    device = th.device('cpu')

    raidnet_config = {'num_tvs': smpc_config['num_tvs'],
                      'num_heads': config['num_heads'],
                      'dropout_prob': config['dropout_prob']}

    per_tv_in = int(observation_dim / smpc_config['num_tvs'])
    Nm1       = config['N'] - 1
    nlayers   = num_layers // 2
    hdim      = hidden_dim // 2

    # ---------- Build JOINT models (all output flat (B, l1_num+ca_num), binary) ----------
    # RAID-NET V2 (joint)
    v2 = RAID_NET_V2(
        raidnet_config, per_tv_in, observation_dim, total_out_dim,
        Nm1, smpc_config['num_tvs'], nlayers, hdim,
        lambda_dim=total_out_dim, lambda_ubd=smpc_config['l1_lmbd'],
        pred_mode=['both duals', 'binary', 'binary']
    ).to(device)

    # RAID-NET V1 (joint)
    v1 = RAID_NET(
        raidnet_config, per_tv_in, observation_dim, total_out_dim,
        Nm1, smpc_config['num_tvs'], nlayers, hdim,
        lambda_dim=total_out_dim, lambda_ubd=smpc_config['l1_lmbd'],
        pred_mode=['both duals', 'binary', 'binary']
    ).to(device)

    # MLP (joint) — your MLP class returns flat logits when tertiary=False
    mlp = MLP(
        observation_dim, total_out_dim,
        hidden_layers=config.get('num_layers', 4),
        hidden_size=config.get('hidden_dim', 512),
        device=device, tertiary=False
    ).to(device)

    # ---------- Load JOINT checkpoints from config ----------
    # Expect: config['v2_joint_ckpt'], config['v1_joint_ckpt'], config['mlp_joint_ckpt']
    ckpt_paths = {
        'V2':  config.get('v2_joint_ckpt',  None),
        'V1':  config.get('v1_joint_ckpt',  None),
        'MLP': config.get('mlp_joint_ckpt', None),
    }
    for name, path in ckpt_paths.items():
        if path is None or (not os.path.isfile(path)):
            raise FileNotFoundError(f'Missing checkpoint path for {name}: set config["{name.lower()}_joint_ckpt"]')
    v2.load_state_dict(th.load(ckpt_paths['V2'])['model_state_dict'])
    v1.load_state_dict(th.load(ckpt_paths['V1'])['model_state_dict'])
    mlp.load_state_dict(th.load(ckpt_paths['MLP'])['model_state_dict'])

    models = {'V2': v2, 'V1': v1, 'MLP': mlp}

    # ---------- Evaluate joint ----------
    metrics = evaluate_joint(
        smpc_config=smpc_config,
        config=config,
        models=models,
        device=device,
        l1_num=l1_num,
        ca_num=ca_num
    )

    # ---------- Plot ----------
    plot_joint_metrics(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpc_config', required=False, type=str,
                        default=NUPLAN_ROOT_DIR +'/nuplan/planning/simulation/planner/smpc_config_eval.yaml')
    parser.add_argument('--config', required=False, type=str,
                        default=NUPLAN_ROOT_DIR +'/tutorials/training_config.yaml')
    args = parser.parse_args()
    with open(args.smpc_config, 'r') as f:
        smpc_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['v2_joint_ckpt'] = NUPLAN_ROOT_DIR +'/nuplan/nn_models/results/RAIDNET_V2_JOINT_NuPlan_N14_N_TV3_10-10-2025_02-25-31_JOINT_300.pt'
    config['v1_joint_ckpt'] = NUPLAN_ROOT_DIR +'/nuplan/nn_models/results/RAIDNET_V1_NuPlan_N14_N_TV3_10-10-2025_03-06-17_JOINT_300.pt'
    config['mlp_joint_ckpt'] = NUPLAN_ROOT_DIR +'/nuplan/nn_models/results/MLP_NuPlan_N14_N_TV3_10-10-2025_03-05-45_JOINT_300.pt'
    main(smpc_config, config)
