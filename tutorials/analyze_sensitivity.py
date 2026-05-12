"""
Sensitivity analysis for SHIELD parameters (ε, ζ, λ).
Reads sensitivity_*.pkl.gz files from nuplan/expert_data/sensitivity/
and produces:
  1. LaTeX sensitivity table (per parameter sweep)
  2. Pareto plot: screening aggressiveness vs feasibility, marker size = speedup
"""
import os, gzip, pickle, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

DATADIR = '/home/mpc/nuplan-devkit/nuplan/expert_data/sensitivity'
OUTDIR  = '/home/mpc/nuplan-devkit/nuplan/expert_data/sensitivity'

# ── Sweeps definition ─────────────────────────────────────────────────────────
SWEEPS = {
    'eps': {
        'label': r'$\varepsilon$',
        'values': [0.001, 0.005, 0.01, 0.05, 0.10],
        'nominal': 0.01,
        'fmt': lambda v: f'{v:.3f}',
        'file_fmt': lambda v: f'{v:.2f}' if v >= 0.01 else f'{v:.3f}',
    },
    'l1': {
        'label': r'$\lambda$',
        'values': [10, 50, 100, 150, 200],
        'nominal': 100,
        'fmt': lambda v: f'{int(v)}',
    },
    # ζ (tightening) sweep omitted: requires retraining RAID-Net per value.
}


def load_pkl(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def compute_metrics(d):
    """Extract per-scenario metrics from a sensitivity pkl."""
    n = len(d['scenario_id'])

    constr_keep, gain_keep, solve_time, feasible, collide = [], [], [], [], []
    for i in range(n):
        # reduced_constr_keep: list of per-timestep arrays (each is flat keep vector)
        ck = d.get('reduced_constr_keep', [[]])[i]
        if ck:
            rates = [np.mean(step) for step in ck if step is not None and len(step) > 0]
            constr_keep.append(np.mean(rates) if rates else np.nan)
        else:
            constr_keep.append(np.nan)

        gk = d.get('reduced_gain_keep', [[]])[i]
        if gk:
            rates = [np.mean(step) for step in gk if step is not None and len(step) > 0]
            gain_keep.append(np.mean(rates) if rates else np.nan)
        else:
            gain_keep.append(np.nan)

        st = d.get('reduced_computation_time', [[]])[i]
        if st:
            vals = []
            for v in st:
                if v is None: continue
                if isinstance(v, dict):
                    vals.append(float(v.get('solve_time', v.get('total', 0.0))))
                else:
                    vals.append(float(v))
            vals = [v for v in vals if not np.isnan(v)]
            solve_time.append(np.mean(vals) if vals else np.nan)
        else:
            solve_time.append(np.nan)

        inf_ = d.get('reduced_infeasibility', [[]])[i]
        feas = 1.0 - np.mean([v for v in inf_ if v is not None]) if inf_ else 1.0
        feasible.append(feas)

        col = d.get('reduced_collisions', [[]])[i]
        collide.append(np.mean([v for v in col if v is not None]) if col else 0.0)

    def s(arr):
        a = np.array([x for x in arr if not np.isnan(x)])
        return (np.mean(a), np.std(a)) if len(a) > 0 else (np.nan, np.nan)

    return {
        'n': n,
        'constr_keep': s(constr_keep),   # (mean, std) fraction
        'gain_keep':   s(gain_keep),
        'solve_time':  s(solve_time),
        'feasibility': np.mean(feasible) * 100,
        'collision':   np.mean(collide) * 100,
    }


def load_sweep(sweep_name):
    """Load all pkls for a sweep, return dict: value → metrics."""
    results = {}
    for v in SWEEPS[sweep_name]['values']:
        file_fmt = SWEEPS[sweep_name].get('file_fmt', SWEEPS[sweep_name]['fmt'])
        tag = f'{sweep_name}_{file_fmt(v)}'
        path = os.path.join(DATADIR, f'sensitivity_{tag}.pkl.gz')
        if not os.path.exists(path):
            print(f'  [MISSING] {path}')
            continue
        d = load_pkl(path)
        results[v] = compute_metrics(d)
        print(f'  {tag}: n={results[v]["n"]}, '
              f'constr={results[v]["constr_keep"][0]*100:.1f}%, '
              f'solve={results[v]["solve_time"][0]:.3f}s')
    return results


# ── Nominal solve time (for speedup computation) ─────────────────────────────
NOMINAL_SOLVE_TIME = None  # filled from eps=0.01 or tightening=2.3 or l1=100

def get_nominal_solve_time(all_results):
    # Use the nominal row from eps sweep (eps=0.01)
    r = all_results.get('eps', {}).get(0.01) or \
        all_results.get('tightening', {}).get(2.3) or \
        all_results.get('l1', {}).get(100)
    return r['solve_time'][0] if r else 0.554  # Table I baseline


# ── LaTeX table ───────────────────────────────────────────────────────────────
def make_latex_table(all_results, nominal_solve):
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{Parameter sensitivity of SHIELD. Bold = nominal value. '
                 r'Metrics averaged over 20 scenarios.}')
    lines.append(r'\label{tab:sensitivity}')
    lines.append(r'\begin{tabular}{lrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Param & Constr Keep (\%) & ADF Keep (\%) & '
                 r'Solve Time (s) & Feasibility (\%) & Collision (\%) \\')
    lines.append(r'\midrule')

    for sweep_name, info in SWEEPS.items():
        results = all_results.get(sweep_name, {})
        if not results:
            continue
        lines.append(rf'\multicolumn{{6}}{{l}}{{\textit{{Sweep {info["label"]}}}}}\\ ')
        for v in info['values']:
            m = results.get(v)
            if m is None:
                continue
            bold = v == info['nominal']
            vfmt = info['fmt'](v)
            ck = f"{m['constr_keep'][0]*100:.1f} \\pm {m['constr_keep'][1]*100:.1f}"
            gk = f"{m['gain_keep'][0]*100:.1f} \\pm {m['gain_keep'][1]*100:.1f}"
            st = f"{m['solve_time'][0]:.3f} \\pm {m['solve_time'][1]:.3f}"
            fe = f"{m['feasibility']:.1f}"
            co = f"{m['collision']:.1f}"
            if bold:
                vfmt = r'\textbf{' + vfmt + r'}'
            row = f'{vfmt} & ${ck}$ & ${gk}$ & ${st}$ & {fe} & {co} \\\\'
            lines.append(row)
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ── Pareto plot ───────────────────────────────────────────────────────────────
def make_pareto_plot(all_results, nominal_solve, outpath):
    n_sweeps = len(SWEEPS)
    fig, axes = plt.subplots(1, n_sweeps, figsize=(5 * n_sweeps, 4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))

    for ax, (sweep_name, info) in zip(axes, SWEEPS.items()):
        results = all_results.get(sweep_name, {})
        if not results:
            ax.set_visible(False)
            continue

        xs, ys, sizes, labels, is_nom = [], [], [], [], []
        for v in info['values']:
            m = results.get(v)
            if m is None:
                continue
            screen_rate = 1.0 - m['constr_keep'][0]  # fraction screened
            feas = m['feasibility']
            speedup = nominal_solve / m['solve_time'][0] if m['solve_time'][0] > 0 else 1.0
            xs.append(screen_rate * 100)
            ys.append(feas)
            sizes.append(max(30, min(600, speedup * 20)))
            labels.append(info['fmt'](v))
            is_nom.append(v == info['nominal'])

        for i, (x, y, sz, lbl, nom) in enumerate(zip(xs, ys, sizes, labels, is_nom)):
            marker = '*' if nom else 'o'
            ec = 'red' if nom else 'white'
            ax.scatter(x, y, s=sz, c=[colors[i]], marker=marker,
                       edgecolors=ec, linewidths=1.5, zorder=3)
            ax.annotate(lbl, (x, y), textcoords='offset points',
                        xytext=(5, 4), fontsize=8,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.set_xlabel('Screening Rate (1 − Constraint Keep) [%]', fontsize=9)
        ax.set_title(f'Sweep {info["label"]}', fontsize=10)
        ax.set_ylim(95, 101)
        ax.grid(True, alpha=0.3)
        ax.axhline(100, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

    axes[0].set_ylabel('Feasibility Rate [%]', fontsize=9)

    # Shared legend for marker size = speedup
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=10, label='Nominal setting'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Other setting\n(size ∝ speedup)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle('SHIELD Parameter Sensitivity: Screening Rate vs. Feasibility',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved Pareto plot to {outpath}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)

    print('Loading sensitivity results...')
    all_results = {}
    for sweep_name in SWEEPS:
        print(f'  Sweep: {sweep_name}')
        all_results[sweep_name] = load_sweep(sweep_name)

    nominal_solve = get_nominal_solve_time(all_results)
    print(f'\nNominal solve time: {nominal_solve:.3f} s')

    # LaTeX table
    latex = make_latex_table(all_results, nominal_solve)
    table_path = os.path.join(OUTDIR, 'sensitivity_table.tex')
    with open(table_path, 'w') as f:
        f.write(latex)
    print(f'\nSaved LaTeX table to {table_path}')
    print('\n' + latex)

    # Pareto plot
    plot_path = os.path.join(OUTDIR, 'sensitivity_pareto.png')
    make_pareto_plot(all_results, nominal_solve, plot_path)
