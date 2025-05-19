import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

from dgswp import data_gen

# === Setup ===
palette = {
    "SWD": "steelblue",             # stays distinct
    "ASWD": "darkorange",          # also distinct

    # SWGG family: purples
    "min-SWGG (random search)": "#a678b4",   # soft purple
    "min-SWGG (optim)": "#6a3d9a",          # deeper purple

    # DGSWP family: greens
    "DGSWP (linear)": "#66c2a5",        # light teal-green
    "DGSWP (NN)": "#1b9e77",            # darker green-teal
}

# === Load all 4 datasets ===
filepaths = [
    "results_gf/gf_gaussian_2d.csv",
    "results_gf/gf_gaussian_500d.csv",
    "results_gf/gf_swiss_roll.csv",
    "results_gf/gf_two_moons.csv",
]

params = {"text.usetex": True,
      "font.family": "serif",               # Match LaTeX default serif font (Computer Modern)
      "font.serif": ["Computer Modern Roman"],
      "text.latex.preamble": r"\usepackage{amsmath}",
      'legend.fontsize': 24,
      'axes.labelsize': 24,
      'axes.titlesize': 24,
      'mathtext.fontset': 'cm',
      'mathtext.rm': 'serif', 
      "ytick.left" : False,
      'lines.linewidth': 4.
      }
matplotlib.rcParams.update(params)

# === Create Figure Layout ===
fig = plt.figure(figsize=(18, 8))
gs = GridSpec(2, 6, figure=fig)
list_axes = [
    fig.add_subplot(gs[0, 1:3]),
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 4:6]),
    fig.add_subplot(gs[0, 3]),
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 4:6]),
    fig.add_subplot(gs[1, 3]),
]

for i, filepath in enumerate(filepaths):
    df = pd.read_csv(filepath)
    ax = list_axes[2 * i]

    # Plot each method
    for method in palette.keys():
        df_method = df[df['method'] == method]
        grouped = df_method.groupby('iteration')['log10_W2']
        
        median = grouped.median()
        lower = grouped.quantile(0.25)
        upper = grouped.quantile(0.75)

        color = palette.get(method, "gray")

        ax.plot(median.index, median.values, label=method, color=color, linestyle="-")
        ax.fill_between(median.index, lower, upper, color=color, alpha=0.2)

    # ax.set_title(f"Dataset {i+1}")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"$\log_{10}(W_2^2)$")
    ax.grid(True)
    ax.set_xlim([0, df['iteration'].max()])

    ax = list_axes[2 * i + 1]
    ax.set_xticks([])
    ax.set_yticks([])
    if filepath != "results_gf/gf_gaussian_500d.csv":
        d = 2
        n = 50
        dataset_name = filepath[len("results_gf/gf_"):filepath.rfind(".")]
        if dataset_name.startswith("gaussian_"):
            dataset_name = "gaussian"
            n = 500
        _, target = data_gen(n_samples_per_distrib=n, d=d,
                             name=dataset_name, random_state=0)
        ax.scatter(target[:, 0], target[:, 1], marker="x", color="C0", linewidth=1.)
    else:
        ax.text(.5, .5, 'Gaussian 500d',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes, fontsize=24)

# === Legend ===
handles, labels = list_axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave space for legend
plt.savefig("fig/fig_gf.pdf", bbox_inches='tight')
