import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ===========================================
# SPARSITY TEST - Figura 1: Fixed Charges
# ===========================================

# X values (decreasing order as shown)
x_sparsity = [0.99, 0.98, 0.97, 0.96, 0.95]
x_tick_labels = ['0.99', '0.98', '0.97', '0.96', '0.95']

# Experiment 2: SINAPSI 0.5 a 1, 0.5 a -1, cariche fisse
data_exp2 = {
    'MSNPS(GPU)': {
        'avg': [23.138, 46.580, 70.173, 94.233, 117.202],
        'std': [3.724, 7.486, 6.823, 3.591, 3.590]
    },
    'OOPSNPS': {
        'avg': [376.320, 782.652, 1243.465, 1813.882, 2465.712],
        'std': [37.380, 46.082, 58.860, 85.022, 124.824]
    },
    'MSNPS(CPU)': {
        'avg': [1005.227, 1978.463, 3295.093, 4772.798, 6158.401],
        'std': [29.112, 71.147, 28.092, 35.696, 38.131]
    }
}

# Colors and markers
colors = {
    'MSNPS(GPU)': '#1F77B4',   # Blue
    'MSNPS(CPU)': '#D62728',   # Red
    'OOPSNPS': '#2CA02C'       # Green
}

markers = {
    'MSNPS(GPU)': 'o',    # Circle
    'OOPSNPS': 's',       # Square
    'MSNPS(CPU)': '^'     # Triangle
}

line_styles = {
    'MSNPS(GPU)': '-',
    'OOPSNPS': '-',
    'MSNPS(CPU)': '-'
}

# Ordine della legenda
legend_order = ['MSNPS(CPU)', 'OOPSNPS', 'MSNPS(GPU)']

# ===========================================
# FIGURA 1: Fixed Charges
# ===========================================

fig1, ax1 = plt.subplots(figsize=(10, 7))

# Plot each line with error bars
for label in data_exp2.keys():
    y_values = data_exp2[label]['avg']
    std_values = data_exp2[label]['std']
    
    color = colors[label]
    marker = markers[label]
    line_style = line_styles[label]
    
    # Line with markers
    ax1.plot(x_sparsity, y_values, 
            color=color, 
            marker=marker, 
            markersize=10, 
            linewidth=2.5, 
            linestyle=line_style,
            label=label,
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=0.8,
            alpha=0.9)
    
    # Error bars
    ax1.errorbar(x_sparsity, y_values, yerr=std_values,
                fmt='none', 
                ecolor='black', 
                elinewidth=2.0,
                capsize=6,
                capthick=2.0,
                alpha=0.7,
                zorder=5)

# Set y-axis scale (log)
ax1.set_yscale('log')

# Labels
ax1.set_xlabel('Sparsity percentage in $\\mathbf{sM_{\\Pi}}$', fontsize=26, fontweight='bold')
ax1.set_ylabel('Average time per step (ms)', fontsize=26, fontweight='bold')

# Set x-ticks and labels
ax1.set_xticks(x_sparsity)
ax1.set_xticklabels(x_tick_labels, fontsize=20)

# Set y-ticks fontsize
ax1.tick_params(axis='y', labelsize=20)  # <--- AGGIUNTO: imposta fontsize dei tick dell'asse y
ax1.tick_params(axis='x', labelsize=20)  # <--- AGGIUNTO: anche per l'asse x per coerenza

# Invert x-axis (decreasing)
ax1.invert_xaxis()

# Grid
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# Style
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Set limits
ax1.set_ylim(10, 20000)
ax1.set_xlim(max(x_sparsity) + 0.01, min(x_sparsity) - 0.01)

# Legend in desired order
handles, labels = ax1.get_legend_handles_labels()
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in legend_order]
ordered_labels = legend_order

ax1.legend(ordered_handles, ordered_labels,
          frameon=True, fancybox=False, edgecolor='gray', 
          fontsize=18, loc='upper left', ncol=1,
          markerscale=1.5, labelspacing=0.7,
          handlelength=2.5, handletextpad=0.8, borderpad=0.5)

plt.tight_layout()
fig1.savefig('sparsity_test_fixed_charges.pdf', format='pdf', bbox_inches='tight')
fig1.savefig('sparsity_test_fixed_charges.png', dpi=600, bbox_inches='tight')
print("Graph saved as sparsity_test_fixed_charges.pdf and sparsity_test_fixed_charges.png")


# ===========================================
# FIGURA 2: Synapses
# ===========================================

# Experiment 3: sinapsi con 0.5 a 1 e 0.5 a -1
data_exp3 = {
    'MSNPS(GPU)': {
        'avg': [22.654, 46.021, 69.502, 93.555, 116.872],
        'std': [2.720, 3.291, 3.298, 3.285, 3.302]
    },
    'OOPSNPS': {
        'avg': [52.353, 58.095, 62.550, 67.662, 71.912],
        'std': [7.612, 10.890, 14.455, 17.979, 21.778]
    },
    'MSNPS(CPU)': {
        'avg': [1168.840, 2370.166, 3566.823, 5022.058, 6805.111],
        'std': [10.608, 22.039, 31.605, 173.683, 263.380]
    }
}

fig2, ax2 = plt.subplots(figsize=(10, 7))

# Plot each line with error bars
for label in data_exp3.keys():
    y_values = data_exp3[label]['avg']
    std_values = data_exp3[label]['std']
    
    color = colors[label]
    marker = markers[label]
    line_style = line_styles[label]
    
    # Line with markers
    ax2.plot(x_sparsity, y_values, 
            color=color, 
            marker=marker, 
            markersize=10, 
            linewidth=2.5, 
            linestyle=line_style,
            label=label,
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=0.8,
            alpha=0.9)
    
    # Error bars
    ax2.errorbar(x_sparsity, y_values, yerr=std_values,
                fmt='none', 
                ecolor='black', 
                elinewidth=2.0,
                capsize=6,
                capthick=2.0,
                alpha=0.7,
                zorder=5)

# Set y-axis scale (log)
ax2.set_yscale('log')

# Labels
ax2.set_xlabel('Sparsity percentage in $\\mathbf{sM_{\\Pi}}$', fontsize=26, fontweight='bold')
ax2.set_ylabel('Average time per step (ms)', fontsize=26, fontweight='bold')

# Set x-ticks and labels
ax2.set_xticks(x_sparsity)
ax2.set_xticklabels(x_tick_labels, fontsize=20)

# Set y-ticks fontsize
ax2.tick_params(axis='y', labelsize=20)  # <--- AGGIUNTO: imposta fontsize dei tick dell'asse y
ax2.tick_params(axis='x', labelsize=20)  # <--- AGGIUNTO: anche per l'asse x per coerenza

# Invert x-axis (decreasing)
ax2.invert_xaxis()

# Grid
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# Style
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Set limits
ax2.set_ylim(10, 20000)
ax2.set_xlim(max(x_sparsity) + 0.01, min(x_sparsity) - 0.01)

# Legend in desired order
handles, labels = ax2.get_legend_handles_labels()
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in legend_order]
ordered_labels = legend_order

ax2.legend(ordered_handles, ordered_labels,
          frameon=True, fancybox=False, edgecolor='gray', 
          fontsize=18, loc='upper left', ncol=1,
          markerscale=1.5, labelspacing=0.7,
          handlelength=2.5, handletextpad=0.8, borderpad=0.5)

plt.tight_layout()
fig2.savefig('sparsity_test_synapses.pdf', format='pdf', bbox_inches='tight')
fig2.savefig('sparsity_test_synapses.png', dpi=600, bbox_inches='tight')
print("Graph saved as sparsity_test_synapses.pdf and sparsity_test_synapses.png")
