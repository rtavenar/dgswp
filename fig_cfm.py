import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set plotting parameters
params = {"text.usetex": True,
      "font.family": "serif",               # Match LaTeX default serif font (Computer Modern)
      "font.serif": ["Computer Modern Roman"],
      "text.latex.preamble": r"\usepackage{amsmath}",
    'legend.fontsize': 21,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    "ytick.left": False,
    'lines.linewidth': 4.
}
matplotlib.rcParams.update(params)

# Load data
df = pd.read_csv('results_cfm/fid_results_with_runs.csv')

# Plotting
plt.figure(figsize=(8, 6))

# Define colors for each model
model_colors = {
    'I-CFM': 'C0',
    'OT-CFM': 'C1',
    "DGSWP-CFM (linear)": "#66c2a5",        # light teal-green
    "DGSWP-CFM (NN)": "#1b9e77",            # darker green-teal
}

# Plot lines for each model/integrator combination
for model in df['Model'].unique():
    for integrator in df['Integrator'].unique():
        if 'Euler' not in integrator:
            continue
        sub_df = df[(df['Model'] == model) & (df['Integrator'] == integrator)]
        # Group by step to average over runs
        grouped = sub_df.groupby('Step')['FID'].mean()
        plt.plot(grouped.index, grouped.values, label=model,
                 color=model_colors[model])

plt.xlabel('Training Step')
plt.ylabel('FID')
plt.grid(True, linestyle=':', linewidth=1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncols=2, frameon=False)
plt.xlim([df['Step'].min(), df['Step'].max()])
plt.xticks(ticks=[100_000, 200_000, 300_000, 400_000], 
           labels=["100k", "200k", "300k", "400k"])
plt.tight_layout()
plt.savefig('fig_fid_plot.pdf')
