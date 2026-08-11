import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def create_scientific_line_chart(x_data, y_data_dict, xlabel, ylabel, filename,
                                 colors=None, markers=None, line_styles=None,
                                 x_tick_labels=None, log_scale=False):
    """
    Creates and saves a line chart of scientific quality.
    
    Parameters:
    -----------
    x_data : list
        X-axis values
    y_data_dict : dict
        Dictionary with format: {'label': [y_values], ...}
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    filename : str
        Name of the output file (without extension)
    colors : dict or None
        Dictionary with format: {'label': 'color', ...}
    markers : dict or None
        Dictionary with format: {'label': 'marker_style', ...}
    line_styles : dict or None
        Dictionary with format: {'label': 'line_style', ...}
    x_tick_labels : list or None
        Custom labels for x-axis ticks
    log_scale : bool
        If True, use logarithmic y-axis (default: False)
    """
    
    rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'mathtext.fontset': 'stix',
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'axes.linewidth': 1.0,
    })
    
    # Default colors: same color for same system
    if colors is None:
        colors = {
            'MSNPS(GPU)': '#1F77B4',   # Blue
            'MSNPS(CPU)': '#D62728',   # Red
            'OOPSNPS': '#2CA02C'   # Green
        }
    
    # Default markers
    if markers is None:
        markers = {
            'MSNPS(GPU)': 'o',    # Circle
            'OOPSNPS': 's',   # Square
            'MSNPS(CPU)': '^'     # Triangle
        }
    
    # Default line styles
    if line_styles is None:
        line_styles = {
            'MSNPS(GPU)': '-',    # Solid line
            'OOPSNPS': '-',   # Solid line
            'MSNPS(CPU)': '-'     # Solid line
        }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each line
    for label, y_values in y_data_dict.items():
        # Extract base system name for color/marker lookup
        base_system = label.split(' - ')[0] if ' - ' in label else label
        
        color = colors.get(label, colors.get(base_system, 'black'))
        marker = markers.get(label, markers.get(base_system, 'o'))
        line_style = line_styles.get(label, line_styles.get(base_system, '-'))
        
        # Use only the system name for the legend (remove " - TEST" or " - TRAIN")
        legend_label = base_system
        
        ax.plot(x_data, y_values, 
                color=color, 
                marker=marker, 
                markersize=8, 
                linewidth=2.2, 
                linestyle=line_style,
                label=legend_label,  # Usa solo il nome del sistema
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=0.5,
                alpha=0.9)
    
    # Set y-axis scale
    if log_scale:
        ax.set_yscale('log')
        ylabel_text = f'{ylabel} (log scale)'
    else:
        ax.set_yscale('linear')
        ylabel_text = ylabel
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel_text, fontsize=20, fontweight='bold')
    
    # Set x-ticks and labels
    if x_tick_labels is not None:
        ax.set_xticks(x_data)
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    else:
        ax.set_xticks(x_data)
    
    # Grid
    if log_scale:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
    else:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend = ax.legend(frameon=True, 
                      fancybox=False, 
                      edgecolor='gray', 
                      fontsize=18,
                      loc='upper left',
                      ncol=1,
                      markerscale=1.3,
                      labelspacing=0.7,
                      handlelength=2.2,
                      handletextpad=0.7,
                      borderpad=0.5)
    
    legend.get_frame().set_linewidth(1.0)
    
    # Adjust limits
    ax.set_xlim(min(x_data) - 0.2, max(x_data) + 0.2)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
    
    print(f"Graph saved as {filename}.pdf and {filename}.png")
    
    return fig, ax


def create_scientific_bar_chart(categories, values, xlabel, ylabel, filename, 
                                colors=None, edgecolor='black', std_devs=None):
    """
    Creates and saves a bar chart of scientific quality for categorical data.
    
    Parameters:
    -----------
    categories : list
        Category names
    values : list
        Bar heights
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    filename : str
        Name of the output file (without extension)
    colors : list or None
        Colors for bars
    edgecolor : str
        Color of bar edges
    std_devs : list or None
        Standard deviations for error bars (optional)
    """
    
    rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'mathtext.fontset': 'stix',
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'axes.linewidth': 1.0,
    })
    
    if colors is None:
        colors = ['#4682B4', '#CD5C5C', '#2E8B57']
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    bars = ax.bar(categories, values, 
                  color=colors, 
                  edgecolor=edgecolor, 
                  linewidth=0.8,
                  alpha=0.8,
                  width=0.5)
    
    # Add error bars if std_devs is provided
    if std_devs is not None:
        ax.errorbar(categories, values, yerr=std_devs, 
                   fmt='none', ecolor='black', capsize=5, capthick=1.5,
                   elinewidth=1.5, zorder=5)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if ax.get_yscale() == 'log':
            text_y = height * 1.1
        else:
            text_y = height + height * 0.01
        ax.text(bar.get_x() + bar.get_width()/2., text_y,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=15)
    
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y', which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ymin = min(values) * 0.5
    ymax = max(values) * 2.0
    ax.set_ylim(ymin, ymax)
    ax.margins(x=0.08)
    
    plt.tight_layout()
    
    fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
    
    print(f"Graph saved as {filename}.pdf and {filename}.png")
    
    return fig, ax


def create_grouped_bar_chart(categories, values_dict, xlabel, ylabel, filename,
                             colors=None, log_scale=True, bar_width=0.25):
    """
    Creates and saves a grouped bar chart for comparing systems across sizes.
    
    Parameters:
    -----------
    categories : list
        List of category names (e.g., ['50-50', '1000-100', '5000-2500'])
    values_dict : dict
        Dictionary with format: {'System Name': [values_for_each_category], ...}
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    filename : str
        Name of the output file (without extension)
    colors : dict or None
        Dictionary with format: {'System Name': 'color', ...}
    log_scale : bool
        If True, use logarithmic y-axis (default: True)
    bar_width : float
        Width of each bar (default: 0.25)
    """
    
    rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'mathtext.fontset': 'stix',
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'axes.linewidth': 1.0,
    })
    
    # Default colors
    if colors is None:
        colors = {
            'MSNPS(GPU)': '#1F77B4',   # Blue
            'MSNPS(CPU)': '#D62728',   # Red
            'OOPSNPS': '#2CA02C'   # Green
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Number of systems and categories
    n_systems = len(values_dict)
    n_categories = len(categories)
    
    # Position of bars on x-axis
    x_pos = np.arange(n_categories)
    
    # Plot bars for each system
    for i, (system_name, values) in enumerate(values_dict.items()):
        color = colors.get(system_name, 'gray')
        
        # Calculate offset for each group
        offset = (i - n_systems/2 + 0.5) * bar_width
        
        bars = ax.bar(x_pos + offset, values, 
                      bar_width, 
                      label=system_name,
                      color=color, 
                      edgecolor='black', 
                      linewidth=0.8,
                      alpha=0.85)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if log_scale:
                text_y = height * 1.15  # 15% above in log scale
            else:
                text_y = height + max(values) * 0.02
            
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Set y-axis scale
    if log_scale:
        ax.set_yscale('log')
        ylabel_text = f'{ylabel} (log scale)'
    else:
        ax.set_yscale('linear')
        ylabel_text = ylabel
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel_text, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=12)
    
    # Grid
    if log_scale:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y', which='both')
    else:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend = ax.legend(frameon=True, 
                      fancybox=False, 
                      edgecolor='gray', 
                      fontsize=13,
                      loc='upper left',
                      ncol=1,
                      markerscale=1.0,
                      labelspacing=0.7,
                      handlelength=1.5,
                      handletextpad=0.7,
                      borderpad=0.5)
    
    legend.get_frame().set_linewidth(1.0)
    
    # Set y-axis limits for log scale
    if log_scale:
        all_values = [val for values in values_dict.values() for val in values]
        ymin = min(all_values) * 0.8
        ymax = max(all_values) * 1.5
        ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
    
    print(f"Graph saved as {filename}.pdf and {filename}.png")
    
    return fig, ax


# ===========================================
# BAR CHART 1: Average times per system WITH STD
# ===========================================
system_colors = ['#D62728',  '#2CA02C','#1F77B4']
systems = ['MSNPS(CPU)', 'OOPSNPS', 'MSNPS(GPU)']
execution_times = [270.861, 41.067, 6.168]
std_deviations = [ 2.582, 1.397, 0.031]  # Standard deviations in the same order

fig, ax = create_scientific_bar_chart(
    categories=systems,
    values=execution_times,
    xlabel='System',
    ylabel='Average time per step (ms)',
    filename='system_comparison',
    colors=system_colors,
    std_devs=std_deviations
)

# ===========================================
# LINE CHART: Performance vs K
# ===========================================
BK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K = [794 + b * 1014 for b in BK]

y_data_k = {
    'MSNPS(GPU)': [0.713, 1.271, 2.147, 2.941, 3.725, 4.488, 5.233, 5.963, 6.668, 7.165],
    'OOPSNPS': [14.128, 17.298, 20.490, 23.453, 26.705, 29.749, 32.896, 36.079, 39.119, 42.318],
    'MSNPS(CPU)': [12.145, 35.973, 63.572, 111.022, 142.226, 176.574, 215.771, 261.857, 287.502, 305.829]
}

colors_k = {
    'MSNPS(GPU)': '#1F77B4',
    'OOPSNPS': '#2CA02C',
    'MSNPS(CPU)': '#D62728'
}

markers_k = {
    'MSNPS(GPU)': 's',
    'OOPSNPS': 's',
    'MSNPS(CPU)': 's'
}

line_styles_k = {
    'MSNPS(GPU)': '-',
    'OOPSNPS': '-',
    'MSNPS(CPU)': '-'
}

scientific_labels = [
    '1.81E3',   # 1808
    '2.82E3',   # 2822
    '3.84E3',   # 3836
    '4.85E3',   # 4850
    '5.86E3',   # 5864
    '6.88E3',   # 6878
    '7.89E3',   # 7892
    '8.91E3',   # 8906
    '9.92E3',   # 9920
    '1.09E4'    # 10934
]

fig, ax = create_scientific_line_chart(
    x_data=K,
    y_data_dict=y_data_k,
    xlabel='Number of neurons',
    ylabel='Average time per step (ms)',
    filename='performance_comparison',
    colors=colors_k,
    markers=markers_k,
    line_styles=line_styles_k,
    log_scale=False
)

# ===========================================
# GROUPED BAR CHART: Performance vs Size
# ===========================================
size_categories = ['50-50', '1000-100', '5000-2500']

values_size = {
    'MSNPS(GPU)': [16.721, 44.524, 137.759],
    'OOPSNPS': [17.371, 69.383, 375.115],
    'MSNPS(CPU)': [43.850, 319.506, 2051.867]
}

colors_size = {
    'MSNPS(GPU)': '#1F77B4',   # Blue
    'OOPSNPS': '#2CA02C',  # Green
    'MSNPS(CPU)': '#D62728'    # Red
}

fig, ax = create_grouped_bar_chart(
    categories=size_categories,
    values_dict=values_size,
    xlabel='Problem Size',
    ylabel='Time (ms)',
    filename='performance_size_grouped_bar',
    colors=colors_size,
    log_scale=True,
    bar_width=0.22
)

# ===========================================
# LINE CHART: Performance vs Number of Neurons (from Spike Probability)
# ===========================================

sparsity_values = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

x_neurons = [(5408 + 3380 * (1 - sp)) * 8 for sp in sparsity_values]

y_data_spike = {
    'MSNPS(GPU)': [6.812, 6.813, 6.813, 6.700, 6.617, 6.513, 6.428, 6.339, 6.233, 6.143, 6.044, 5.946],
    'MSNPS(CPU)': [298.416, 298.722, 298.752, 294.298, 293.158, 288.256, 285.325, 281.799, 277.763, 272.597, 270.101, 264.488],
    'OOPSNPS': [41.032, 41.026, 41.013, 42.909, 43.839, 43.982, 43.990, 43.725, 43.552, 43.331, 42.945, 42.669]
}

colors_spike = {
    'MSNPS(GPU)': '#1F77B4',   # Blue
    'MSNPS(CPU)': '#D62728',   # Red
    'OOPSNPS': '#2CA02C'       # Green
}

markers_spike = {
    'MSNPS(GPU)': 'o',
    'MSNPS(CPU)': '^',
    'OOPSNPS': 's'
}

line_styles_spike = {
    'MSNPS(GPU)': '-',
    'MSNPS(CPU)': '-',
    'OOPSNPS': '-'
}

fig, ax = create_scientific_line_chart(
    x_data=x_neurons,
    y_data_dict=y_data_spike,
    xlabel='Number of synapses',
    ylabel='Average time per step (ms)',
    filename='performance_spike_probability',
    colors=colors_spike,
    markers=markers_spike,
    line_styles=line_styles_spike,
    x_tick_labels=None,
    log_scale=False
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ===========================================
# LINE CHART: Sparsity Test - Filled Error Bands
# ===========================================

# Dati dalla tabella
sparsity_levels = ['SP0.99', 'SP0.98', 'SP0.97', 'SP0.96', 'SP0.95']
x_sparsity = [0.99, 0.98, 0.97, 0.96, 0.95]

# Dati AVG e STD per ciascun sistema
data_sparsity = {
    'MSNPS(GPU)': {
        'avg': [22.660, 46.026, 69.510, 93.565, 116.875],
        'std': [2.746, 3.275, 3.276, 3.304, 3.330]
    },
    'OOPSNPS': {
        'avg': [47.676, 55.269, 56.466, 57.060, 57.490],
        'std': [7.628, 16.602, 27.391, 39.039, 47.713]
    },
    'MSNPS(CPU)': {
        'avg': [1168.727, 2371.391, 3567.153, 4757.560, 5961.190],
        'std': [10.105, 21.665, 27.026, 29.902, 38.730]
    }
}

# Colori: stesso colore per sistema
colors_sparsity = {
    'MSNPS(GPU)': '#1F77B4',   # Blue
    'MSNPS(CPU)': '#D62728',   # Red
    'OOPSNPS': '#2CA02C'       # Green
}

# Simboli diversi per ciascun sistema
markers_sparsity = {
    'MSNPS(GPU)': 'o',    # Circle
    'OOPSNPS': 's',       # Square
    'MSNPS(CPU)': '^'     # Triangle up
}

# Stili linea
line_styles_sparsity = {
    'MSNPS(GPU)': '-',
    'OOPSNPS': '-',
    'MSNPS(CPU)': '-'
}

# Impostazioni stile scientifico
rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'mathtext.fontset': 'stix',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.linewidth': 1.0,
})

# Creazione del grafico
fig, ax = plt.subplots(figsize=(10, 7))

# Plot per ciascun sistema con bande di errore
for label, system_data in data_sparsity.items():
    color = colors_sparsity[label]
    marker = markers_sparsity[label]
    avg_values = system_data['avg']
    std_values = system_data['std']
    
    # Linea principale con marker
    ax.plot(x_sparsity, avg_values, 
            color=color, 
            marker=marker, 
            markersize=10, 
            linewidth=2.5, 
            linestyle='-',
            label=label,
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=0.8,
            alpha=0.9,
            zorder=3)
    
    # Banda di errore riempita (più visibile delle barre)
    upper = [avg + std for avg, std in zip(avg_values, std_values)]
    lower = [avg - std for avg, std in zip(avg_values, std_values)]
    
    # Crea la banda di errore con trasparenza
    ax.fill_between(x_sparsity, lower, upper, 
                    color=color, 
                    alpha=0.25, 
                    zorder=2)
    
    # Aggiungi anche le linee sottili per i bordi della banda
    ax.plot(x_sparsity, upper, 
            color=color, 
            linestyle=':', 
            linewidth=0.8, 
            alpha=0.5,
            zorder=1)
    ax.plot(x_sparsity, lower, 
            color=color, 
            linestyle=':', 
            linewidth=0.8, 
            alpha=0.5,
            zorder=1)

# Set y-axis scale (logaritmico)
ax.set_yscale('log')

# Labels
ax.set_xlabel('Sparsity Level', fontsize=20, fontweight='bold')
ax.set_ylabel('Average Time per Step (ms) - Log Scale', fontsize=20, fontweight='bold')

# Set x-ticks
ax.set_xticks(x_sparsity)
ax.set_xticklabels(['0.99', '0.98', '0.97', '0.96', '0.95'], fontsize=14)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend = ax.legend(frameon=True, 
                  fancybox=False, 
                  edgecolor='gray', 
                  fontsize=16,
                  loc='upper left',
                  ncol=1,
                  markerscale=1.5,
                  labelspacing=0.7,
                  handlelength=2.5,
                  handletextpad=0.8,
                  borderpad=0.5)

legend.get_frame().set_linewidth(1.0)

# Adjust limits
ax.set_xlim(0.94, 1.0)
# Lascia che matplotlib calcoli automaticamente i limiti y
# oppure imposta manualmente per vedere meglio:
# ax.set_ylim(10, 20000)

plt.tight_layout()

# Save
fig.savefig('sparsity_test_bands.pdf', format='pdf', bbox_inches='tight')
fig.savefig('sparsity_test_bands.png', dpi=600, bbox_inches='tight')

print("Graph saved as sparsity_test_bands.pdf and sparsity_test_bands.png")

# # ===========================================
# # LINE CHART: Sparsity Test - All Synapses at 1
# # ===========================================

# # Dati dalla tabella
# sparsity_levels = ['SP0.99', 'SP0.98', 'SP0.97', 'SP0.96', 'SP0.95']
# # Valori numerici per l'asse x (es. 0.99, 0.98, 0.97, 0.96, 0.95)
# x_sparsity = [0.99, 0.98, 0.97, 0.96, 0.95]

# # Dati AVG e STD per ciascun sistema
# data_sparsity = {
#     'MSNPS(GPU)': {
#         'avg': [22.660, 46.026, 69.510, 93.565, 116.875],
#         'std': [2.746, 3.275, 3.276, 3.304, 3.330]
#     },
#     'OOPSNPS': {
#         'avg': [47.676, 55.269, 56.466, 57.060, 57.490],
#         'std': [7.628, 16.602, 27.391, 39.039, 47.713]
#     },
#     'MSNPS(CPU)': {
#         'avg': [1168.727, 2371.391, 3567.153, 4757.560, 5961.190],
#         'std': [10.105, 21.665, 27.026, 29.902, 38.730]
#     }
# }

# # Colori: stesso colore per sistema (come nei grafici precedenti)
# colors_sparsity = {
#     'MSNPS(GPU)': '#1F77B4',   # Blue
#     'MSNPS(CPU)': '#D62728',   # Red
#     'OOPSNPS': '#2CA02C'       # Green
# }

# # Simboli diversi per ciascun sistema
# markers_sparsity = {
#     'MSNPS(GPU)': 'o',    # Circle
#     'OOPSNPS': 's',       # Square
#     'MSNPS(CPU)': '^'     # Triangle up
# }

# # Stili linea
# line_styles_sparsity = {
#     'MSNPS(GPU)': '-',
#     'OOPSNPS': '-',
#     'MSNPS(CPU)': '-'
# }

# # Creazione del grafico con log scale
# rcParams.update({
#     'font.family': 'serif',
#     'font.size': 14,
#     'mathtext.fontset': 'stix',
#     'figure.dpi': 600,
#     'savefig.dpi': 600,
#     'axes.linewidth': 1.0,
# })

# fig, ax = plt.subplots(figsize=(10, 7))

# # Plot per ciascun sistema con barre di errore
# for label, system_data in data_sparsity.items():
#     color = colors_sparsity[label]
#     marker = markers_sparsity[label]
#     line_style = line_styles_sparsity[label]
    
#     avg_values = system_data['avg']
#     std_values = system_data['std']
    
#     # Linea con marker
#     ax.plot(x_sparsity, avg_values, 
#             color=color, 
#             marker=marker, 
#             markersize=10, 
#             linewidth=2.5, 
#             linestyle=line_style,
#             label=label,
#             markerfacecolor=color,
#             markeredgecolor='black',
#             markeredgewidth=0.8,
#             alpha=0.9)
    
#     # Barre di errore (STD)
#     ax.errorbar(x_sparsity, avg_values, yerr=std_values,
#                 fmt='none', 
#                 ecolor=color, 
#                 elinewidth=2.0,
#                 capsize=6,
#                 capthick=2.0,
#                 alpha=0.7,
#                 zorder=5)

# # Set y-axis scale (logaritmico)
# ax.set_yscale('log')

# # Labels
# ax.set_xlabel('Sparsity Level', fontsize=20, fontweight='bold')
# ax.set_ylabel('Average Time per Step (ms) - Log Scale', fontsize=20, fontweight='bold')

# # Set x-ticks
# ax.set_xticks(x_sparsity)
# ax.set_xticklabels(['0.99', '0.98', '0.97', '0.96', '0.95'], fontsize=14)

# # Grid
# ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# # Style
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Legend
# legend = ax.legend(frameon=True, 
#                   fancybox=False, 
#                   edgecolor='gray', 
#                   fontsize=18,
#                   loc='upper left',
#                   ncol=1,
#                   markerscale=1.5,
#                   labelspacing=0.7,
#                   handlelength=2.5,
#                   handletextpad=0.8,
#                   borderpad=0.5)

# legend.get_frame().set_linewidth(1.0)

# # Adjust limits
# ax.set_xlim(0.94, 1.0)
# ax.set_ylim(10, 20000)  # Per adattare tutti i dati

# plt.tight_layout()

# # Save
# fig.savefig('sparsity_test_log.pdf', format='pdf', bbox_inches='tight')
# fig.savefig('sparsity_test_log.png', dpi=600, bbox_inches='tight')

# print("Graph saved as sparsity_test_log.pdf and sparsity_test_log.png")


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rcParams

# # ===========================================
# # LINE CHART: Sparsity Test - STD as Separate Lines
# # ===========================================

# # Dati dalla tabella
# sparsity_levels = ['SP0.99', 'SP0.98', 'SP0.97', 'SP0.96', 'SP0.95']
# x_sparsity = [0.99, 0.98, 0.97, 0.96, 0.95]

# # Dati AVG e STD per ciascun sistema
# data_sparsity = {
#     'MSNPS(GPU)': {
#         'avg': [22.660, 46.026, 69.510, 93.565, 116.875],
#         'std': [2.746, 3.275, 3.276, 3.304, 3.330]
#     },
#     'OOPSNPS': {
#         'avg': [47.676, 55.269, 56.466, 57.060, 57.490],
#         'std': [7.628, 16.602, 27.391, 39.039, 47.713]
#     },
#     'MSNPS(CPU)': {
#         'avg': [1168.727, 2371.391, 3567.153, 4757.560, 5961.190],
#         'std': [10.105, 21.665, 27.026, 29.902, 38.730]
#     }
# }

# # Colori: stesso colore per sistema
# colors_sparsity = {
#     'MSNPS(GPU)': '#1F77B4',   # Blue
#     'MSNPS(CPU)': '#D62728',   # Red
#     'OOPSNPS': '#2CA02C'       # Green
# }

# # Simboli diversi per ciascun sistema
# markers_sparsity = {
#     'MSNPS(GPU)': 'o',    # Circle
#     'OOPSNPS': 's',       # Square
#     'MSNPS(CPU)': '^'     # Triangle up
# }

# # Simboli per le linee STD (vuoti)
# markers_std = {
#     'MSNPS(GPU)': 'o',    # Circle
#     'OOPSNPS': 's',       # Square
#     'MSNPS(CPU)': '^'     # Triangle up
# }

# # Stili linea
# line_styles_sparsity = {
#     'MSNPS(GPU)': '-',    # Solid per AVG
#     'OOPSNPS': '-',       # Solid per AVG
#     'MSNPS(CPU)': '-'     # Solid per AVG
# }

# line_styles_std = {
#     'MSNPS(GPU)': '--',    # Dashed per STD
#     'OOPSNPS': '--',       # Dashed per STD
#     'MSNPS(CPU)': '--'     # Dashed per STD
# }

# # Impostazioni stile scientifico
# rcParams.update({
#     'font.family': 'serif',
#     'font.size': 14,
#     'mathtext.fontset': 'stix',
#     'figure.dpi': 600,
#     'savefig.dpi': 600,
#     'axes.linewidth': 1.0,
# })

# # Creazione del grafico
# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot per ciascun sistema
# for label, system_data in data_sparsity.items():
#     color = colors_sparsity[label]
#     marker = markers_sparsity[label]
#     avg_values = system_data['avg']
#     std_values = system_data['std']
    
#     # Calcola AVG + STD e AVG - STD
#     avg_plus_std = [avg + std for avg, std in zip(avg_values, std_values)]
#     avg_minus_std = [avg - std for avg, std in zip(avg_values, std_values)]
    
#     # LINEA 1: AVG (linea solida, marker pieni)
#     ax.plot(x_sparsity, avg_values, 
#             color=color, 
#             marker=marker, 
#             markersize=10, 
#             linewidth=2.5, 
#             linestyle=line_styles_sparsity[label],
#             label=f'{label} (AVG)',
#             markerfacecolor=color,
#             markeredgecolor='black',
#             markeredgewidth=0.8,
#             alpha=0.9,
#             zorder=3)
    
#     # LINEA 2: AVG + STD (linea tratteggiata, marker vuoti)
#     ax.plot(x_sparsity, avg_plus_std, 
#             color=color, 
#             marker=markers_std[label], 
#             markersize=8, 
#             linewidth=1.5, 
#             linestyle=line_styles_std[label],
#             label=f'{label} (AVG+STD)',
#             markerfacecolor='white',
#             markeredgecolor=color,
#             markeredgewidth=1.5,
#             alpha=0.7,
#             zorder=2)
    
#     # LINEA 3: AVG - STD (linea tratteggiata, marker vuoti)
#     ax.plot(x_sparsity, avg_minus_std, 
#             color=color, 
#             marker=markers_std[label], 
#             markersize=8, 
#             linewidth=1.5, 
#             linestyle=line_styles_std[label],
#             label=f'{label} (AVG-STD)',
#             markerfacecolor='white',
#             markeredgecolor=color,
#             markeredgewidth=1.5,
#             alpha=0.7,
#             zorder=2)
    
#     # Aggiungi area ombreggiata leggera tra AVG+STD e AVG-STD
#     ax.fill_between(x_sparsity, avg_minus_std, avg_plus_std, 
#                     color=color, 
#                     alpha=0.1,
#                     zorder=1)

# # Set y-axis scale (logaritmico)
# ax.set_yscale('log')

# # Labels
# ax.set_xlabel('Sparsity Level', fontsize=20, fontweight='bold')
# ax.set_ylabel('Average Time per Step (ms) - Log Scale', fontsize=20, fontweight='bold')

# # Set x-ticks
# ax.set_xticks(x_sparsity)
# ax.set_xticklabels(['0.99', '0.98', '0.97', '0.96', '0.95'], fontsize=14)

# # Grid
# ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# # Style
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Legend - prendo solo le prime 3 entries per evitare troppe righe
# handles, labels = ax.get_legend_handles_labels()
# # Prendo solo AVG per sistema (le linee solide)
# unique_labels = []
# unique_handles = []
# seen = set()
# for handle, label in zip(handles, labels):
#     if '(AVG)' in label and label not in seen:
#         unique_labels.append(label.replace(' (AVG)', ''))
#         unique_handles.append(handle)
#         seen.add(label)

# # Legend
# legend = ax.legend(unique_handles, unique_labels,
#                   frameon=True, 
#                   fancybox=False, 
#                   edgecolor='gray', 
#                   fontsize=16,
#                   loc='upper left',
#                   ncol=1,
#                   markerscale=1.5,
#                   labelspacing=0.7,
#                   handlelength=2.5,
#                   handletextpad=0.8,
#                   borderpad=0.5)

# legend.get_frame().set_linewidth(1.0)

# # Adjust limits
# ax.set_xlim(0.94, 1.0)

# plt.tight_layout()

# # Save
# fig.savefig('sparsity_test_std_lines.pdf', format='pdf', bbox_inches='tight')
# fig.savefig('sparsity_test_std_lines.png', dpi=600, bbox_inches='tight')

# print("Graph saved as sparsity_test_std_lines.pdf and sparsity_test_std_lines.png")

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rcParams

# # ===========================================
# # LINE CHART: Sparsity Test - STD as Normal Values
# # ===========================================

# # Dati dalla tabella
# x_sparsity = [0.99, 0.98, 0.97, 0.96, 0.95]

# # Dati AVG e STD per ciascun sistema
# data_sparsity = {
#     'MSNPS(GPU)': {
#         'avg': [22.660, 46.026, 69.510, 93.565, 116.875],
#         'std': [2.746, 3.275, 3.276, 3.304, 3.330]
#     },
#     'OOPSNPS': {
#         'avg': [47.676, 55.269, 56.466, 57.060, 57.490],
#         'std': [7.628, 16.602, 27.391, 39.039, 47.713]
#     },
#     'MSNPS(CPU)': {
#         'avg': [1168.727, 2371.391, 3567.153, 4757.560, 5961.190],
#         'std': [10.105, 21.665, 27.026, 29.902, 38.730]
#     }
# }

# # Colori: stesso colore per sistema ma con trasparenza diversa per STD
# colors_sparsity = {
#     'MSNPS(GPU)': '#1F77B4',   # Blue
#     'MSNPS(CPU)': '#D62728',   # Red
#     'OOPSNPS': '#2CA02C'       # Green
# }

# # Simboli per AVG (pieni)
# markers_avg = {
#     'MSNPS(GPU)': 'o',    # Circle
#     'OOPSNPS': 's',       # Square
#     'MSNPS(CPU)': '^'     # Triangle up
# }

# # Simboli per STD (diversi o vuoti)
# markers_std = {
#     'MSNPS(GPU)': 'D',    # Diamond
#     'OOPSNPS': '*',       # Star
#     'MSNPS(CPU)': 'v'     # Triangle down
# }

# # Stili linea
# line_styles_avg = {
#     'MSNPS(GPU)': '-',    # Solid
#     'OOPSNPS': '-',       # Solid
#     'MSNPS(CPU)': '-'     # Solid
# }

# line_styles_std = {
#     'MSNPS(GPU)': '--',   # Dashed
#     'OOPSNPS': '--',      # Dashed
#     'MSNPS(CPU)': '--'    # Dashed
# }

# rcParams.update({
#     'font.family': 'serif',
#     'font.size': 14,
#     'mathtext.fontset': 'stix',
#     'figure.dpi': 600,
#     'savefig.dpi': 600,
#     'axes.linewidth': 1.0,
# })

# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot AVG per ciascun sistema
# for label, system_data in data_sparsity.items():
#     color = colors_sparsity[label]
#     avg_values = system_data['avg']
    
#     # AVG - linea solida con marker pieni
#     ax.plot(x_sparsity, avg_values, 
#             color=color, 
#             marker=markers_avg[label], 
#             markersize=10, 
#             linewidth=2.5, 
#             linestyle=line_styles_avg[label],
#             label=f'{label} (AVG)',
#             markerfacecolor=color,
#             markeredgecolor='black',
#             markeredgewidth=0.8,
#             alpha=1.0,
#             zorder=3)

# # Plot STD per ciascun sistema (come se fossero valori normali)
# for label, system_data in data_sparsity.items():
#     color = colors_sparsity[label]
#     std_values = system_data['std']
    
#     # STD - linea tratteggiata con marker diversi
#     ax.plot(x_sparsity, std_values, 
#             color=color, 
#             marker=markers_std[label], 
#             markersize=10, 
#             linewidth=2.5, 
#             linestyle=line_styles_std[label],
#             label=f'{label} (STD)',
#             markerfacecolor='white',
#             markeredgecolor=color,
#             markeredgewidth=1.5,
#             alpha=0.7,
#             zorder=2)

# # Set y-axis scale (logaritmico)
# ax.set_yscale('log')

# # Labels
# ax.set_xlabel('Sparsity Level', fontsize=20, fontweight='bold')
# ax.set_ylabel('Value (ms) - Log Scale', fontsize=20, fontweight='bold')
# ax.set_title('Average and Standard Deviation vs Sparsity', fontsize=22, fontweight='bold')

# # Set x-ticks
# ax.set_xticks(x_sparsity)
# ax.set_xticklabels(['0.99', '0.98', '0.97', '0.96', '0.95'], fontsize=14)

# # Grid
# ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')

# # Style
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Legend
# legend = ax.legend(frameon=True, 
#                   fancybox=False, 
#                   edgecolor='gray', 
#                   fontsize=14,
#                   loc='upper left',
#                   ncol=1,
#                   markerscale=1.5,
#                   labelspacing=0.7,
#                   handlelength=2.5,
#                   handletextpad=0.8,
#                   borderpad=0.5)

# legend.get_frame().set_linewidth(1.0)

# # Adjust limits
# ax.set_xlim(0.94, 1.0)

# plt.tight_layout()

# # Save
# fig.savefig('sparsity_test_avg_std_as_values.pdf', format='pdf', bbox_inches='tight')
# fig.savefig('sparsity_test_avg_std_as_values.png', dpi=600, bbox_inches='tight')

# print("Graph saved as sparsity_test_avg_std_as_values.pdf and sparsity_test_avg_std_as_values.png")

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rcParams
# from matplotlib.lines import Line2D

# # ===========================================
# # SPARSITY TEST - Entrambi gli esperimenti affiancati
# # ===========================================

# x_sparsity = [0.99, 0.98, 0.97, 0.96, 0.95]

# # Dati Esperimento 1: Cariche Fisse
# data_fixed = {
#     'MSNPS(GPU)': {'avg': [23.138, 46.580, 70.173, 94.233, 117.202], 'std': [3.724, 7.486, 6.823, 3.591, 3.590]},
#     'OOPSNPS': {'avg': [376.320, 782.652, 1243.465, 1813.882, 2465.712], 'std': [37.380, 46.082, 58.860, 85.022, 124.824]},
#     'MSNPS(CPU)': {'avg': [1005.227, 1978.463, 3295.093, 4772.798, 6158.401], 'std': [29.112, 71.147, 28.092, 35.696, 38.131]}
# }

# # Dati Esperimento 2: Sinapsi
# data_synapses = {
#     'MSNPS(GPU)': {'avg': [22.654, 46.021, 69.502, 93.555, 116.872], 'std': [2.720, 3.291, 3.298, 3.285, 3.302]},
#     'OOPSNPS': {'avg': [52.353, 58.095, 62.550, 67.662, 71.912], 'std': [7.612, 10.890, 14.455, 17.979, 21.778]},
#     'MSNPS(CPU)': {'avg': [1168.840, 2370.166, 3566.823, 5022.058, 6805.111], 'std': [10.608, 22.039, 31.605, 173.683, 263.380]}
# }

# colors = {'MSNPS(GPU)': '#1F77B4', 'MSNPS(CPU)': '#D62728', 'OOPSNPS': '#2CA02C'}
# markers_avg = {'MSNPS(GPU)': 'o', 'OOPSNPS': 's', 'MSNPS(CPU)': '^'}
# markers_std = {'MSNPS(GPU)': 'D', 'OOPSNPS': '*', 'MSNPS(CPU)': 'v'}

# rcParams.update({
#     'font.family': 'serif',
#     'font.size': 12,
#     'mathtext.fontset': 'stix',
#     'figure.dpi': 600,
#     'savefig.dpi': 600,
#     'axes.linewidth': 1.0,
# })

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# # Funzione per plottare i dati
# def plot_data(ax, data):
#     for label, system_data in data.items():
#         color = colors[label]
#         avg_values = system_data['avg']
#         std_values = system_data['std']
        
#         # AVG
#         ax.plot(x_sparsity, avg_values, 
#                 color=color, marker=markers_avg[label], markersize=8,
#                 linewidth=2.5, linestyle='-',
#                 markerfacecolor=color, markeredgecolor='black',
#                 markeredgewidth=0.8, alpha=1.0, zorder=3)
        
#         # STD
#         ax.plot(x_sparsity, std_values, 
#                 color=color, marker=markers_std[label], markersize=8,
#                 linewidth=2.5, linestyle='--',
#                 markerfacecolor='white', markeredgecolor=color,
#                 markeredgewidth=1.5, alpha=0.7, zorder=2)
    
#     ax.set_yscale('log')
#     ax.set_xlabel('Sparsity Level', fontsize=16, fontweight='bold')
#     ax.set_ylabel('Time (ms) - Log Scale', fontsize=16, fontweight='bold')
#     ax.set_title(title, fontsize=16, fontweight='bold')
#     ax.set_xticks(x_sparsity)
#     ax.set_xticklabels(['0.99', '0.98', '0.97', '0.96', '0.95'], fontsize=12)
#     ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_xlim(0.94, 1.0)

# # Plot entrambi
# plot_data(ax1, data_fixed)
# plot_data(ax2, data_synapses)

# # Legenda unica
# legend_elements = []
# for label in data_fixed.keys():
#     color = colors[label]
#     legend_elements.append(Line2D([0], [0], color=color, marker=markers_avg[label],
#                                   linestyle='-', linewidth=2.5, markersize=8,
#                                   markerfacecolor=color, markeredgecolor='black',
#                                   markeredgewidth=0.8, label=f'{label} (AVG)'))
#     legend_elements.append(Line2D([0], [0], color=color, marker=markers_std[label],
#                                   linestyle='--', linewidth=2.5, markersize=8,
#                                   markerfacecolor='white', markeredgecolor=color,
#                                   markeredgewidth=1.5, label=f'{label} (STD)'))

# fig.legend(handles=legend_elements, frameon=True, fancybox=False, 
#            edgecolor='gray', fontsize=11, loc='upper center', 
#            ncol=6, markerscale=1.2, labelspacing=0.5,
#            handlelength=2.0, handletextpad=0.6, borderpad=0.5,
#            bbox_to_anchor=(0.5, 1.02))

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# fig.savefig('sparsity_test_comparison.pdf', format='pdf', bbox_inches='tight')
# fig.savefig('sparsity_test_comparison.png', dpi=600, bbox_inches='tight')
# print("Graph saved as sparsity_test_comparison.pdf and sparsity_test_comparison.png")

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rcParams

# def create_scientific_line_chart_with_error(x_data, y_data_dict, std_data_dict, xlabel, ylabel, filename,
#                                             colors=None, markers=None, line_styles=None,
#                                             x_tick_labels=None, log_scale=False, 
#                                             ax=None, show_legend=True, title=None):
#     """
#     Creates and saves a line chart of scientific quality with error bars.
    
#     Parameters:
#     -----------
#     x_data : list
#         X-axis values
#     y_data_dict : dict
#         Dictionary with format: {'label': [y_values], ...}
#     std_data_dict : dict
#         Dictionary with format: {'label': [std_values], ...}
#     xlabel : str
#         Label for x-axis
#     ylabel : str
#         Label for y-axis
#     filename : str
#         Name of the output file (without extension)
#     colors : dict or None
#         Dictionary with format: {'label': 'color', ...}
#     markers : dict or None
#         Dictionary with format: {'label': 'marker_style', ...}
#     line_styles : dict or None
#         Dictionary with format: {'label': 'line_style', ...}
#     x_tick_labels : list or None
#         Custom labels for x-axis ticks
#     log_scale : bool
#         If True, use logarithmic y-axis (default: False)
#     ax : matplotlib.axes.Axes or None
#         If provided, plot on this axis
#     show_legend : bool
#         If True, show legend
#     title : str or None
#         Title for the subplot
#     """
    
#     rcParams.update({
#         'font.family': 'serif',
#         'font.size': 14,
#         'mathtext.fontset': 'stix',
#         'figure.dpi': 600,
#         'savefig.dpi': 600,
#         'axes.linewidth': 1.0,
#     })
    
#     # Default colors: same color for same system
#     if colors is None:
#         colors = {
#             'MSNPS(GPU)': '#1F77B4',   # Blue
#             'MSNPS(CPU)': '#D62728',   # Red
#             'OOPSNPS': '#2CA02C'       # Green
#         }
    
#     # Default markers
#     if markers is None:
#         markers = {
#             'MSNPS(GPU)': 'o',    # Circle
#             'OOPSNPS': 's',       # Square
#             'MSNPS(CPU)': '^'     # Triangle
#         }
    
#     # Default line styles
#     if line_styles is None:
#         line_styles = {
#             'MSNPS(GPU)': '-',    # Solid line
#             'OOPSNPS': '-',       # Solid line
#             'MSNPS(CPU)': '-'     # Solid line
#         }
    
#     # Create axis if not provided
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         save_fig = True
#     else:
#         save_fig = False
    
#     # Plot each line with error bars
#     for label in y_data_dict.keys():
#         y_values = y_data_dict[label]
#         std_values = std_data_dict[label]
        
#         # Extract base system name for color/marker lookup
#         base_system = label.split(' - ')[0] if ' - ' in label else label
        
#         color = colors.get(label, colors.get(base_system, 'black'))
#         marker = markers.get(label, markers.get(base_system, 'o'))
#         line_style = line_styles.get(label, line_styles.get(base_system, '-'))
        
#         # Use only the system name for the legend
#         legend_label = base_system
        
#         # Line with markers
#         ax.plot(x_data, y_values, 
#                 color=color, 
#                 marker=marker, 
#                 markersize=8, 
#                 linewidth=2.2, 
#                 linestyle=line_style,
#                 label=legend_label,
#                 markerfacecolor=color,
#                 markeredgecolor='black',
#                 markeredgewidth=0.5,
#                 alpha=0.9)
        
#         # Error bars
#         ax.errorbar(x_data, y_values, yerr=std_values,
#                     fmt='none', 
#                     ecolor=color, 
#                     elinewidth=1.5,
#                     capsize=5,
#                     capthick=1.5,
#                     alpha=0.6,
#                     zorder=5)
    
#     # Set y-axis scale
#     if log_scale:
#         ax.set_yscale('log')
#         ylabel_text = f'{ylabel}'
#     else:
#         ax.set_yscale('linear')
#         ylabel_text = ylabel
    
#     # Labels
#     ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
#     ax.set_ylabel(ylabel_text, fontsize=16, fontweight='bold')
    
#     # Set x-ticks and labels
#     if x_tick_labels is not None:
#         ax.set_xticks(x_data)
#         ax.set_xticklabels(x_tick_labels, fontsize=12)
#     else:
#         ax.set_xticks(x_data)
#         ax.set_xticklabels([f'{x:.2f}' for x in x_data], fontsize=12)
    
#     # Title
#     if title is not None:
#         ax.set_title(title, fontsize=16, fontweight='bold')
    
#     # Grid
#     if log_scale:
#         ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
#     else:
#         ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
#     # Style
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Legend
#     if show_legend:
#         legend = ax.legend(frameon=True, 
#                           fancybox=False, 
#                           edgecolor='gray', 
#                           fontsize=14,
#                           loc='upper left',
#                           ncol=1,
#                           markerscale=1.3,
#                           labelspacing=0.7,
#                           handlelength=2.2,
#                           handletextpad=0.7,
#                           borderpad=0.5)
#         legend.get_frame().set_linewidth(1.0)
    
#     # Adjust limits - keep x axis in original order (decreasing)
#     ax.set_xlim(min(x_data) - 0.01, max(x_data) + 0.01)
    
#     # For log scale, set y limits
#     if log_scale:
#         all_values = []
#         for values in y_data_dict.values():
#             all_values.extend(values)
#         ax.set_ylim(min(all_values) * 0.5, max(all_values) * 2.0)
    
#     plt.tight_layout()
    
#     # Save if we created a new figure
#     if save_fig:
#         fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
#         fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
#         print(f"Graph saved as {filename}.pdf and {filename}.png")
#         return fig, ax
#     else:
#         return ax

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rcParams

# def create_scientific_line_chart_with_error(x_data, y_data_dict, std_data_dict, xlabel, ylabel, filename,
#                                             colors=None, markers=None, line_styles=None,
#                                             x_tick_labels=None, log_scale=False, 
#                                             ax=None, show_legend=True, title=None):
#     """
#     Creates and saves a line chart of scientific quality with error bars.
#     """
    
#     rcParams.update({
#         'font.family': 'serif',
#         'font.size': 14,
#         'mathtext.fontset': 'stix',
#         'figure.dpi': 600,
#         'savefig.dpi': 600,
#         'axes.linewidth': 1.0,
#     })
    
#     # Default colors: same color for same system
#     if colors is None:
#         colors = {
#             'MSNPS(GPU)': '#1F77B4',   # Blue
#             'MSNPS(CPU)': '#D62728',   # Red
#             'OOPSNPS': '#2CA02C'       # Green
#         }
    
#     # Default markers
#     if markers is None:
#         markers = {
#             'MSNPS(GPU)': 'o',    # Circle
#             'OOPSNPS': 's',       # Square
#             'MSNPS(CPU)': '^'     # Triangle
#         }
    
#     # Default line styles
#     if line_styles is None:
#         line_styles = {
#             'MSNPS(GPU)': '-',    # Solid line
#             'OOPSNPS': '-',       # Solid line
#             'MSNPS(CPU)': '-'     # Solid line
#         }
    
#     # Create axis if not provided
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         save_fig = True
#     else:
#         save_fig = False
    
#     # Plot each line with error bars
#     for label in y_data_dict.keys():
#         y_values = y_data_dict[label]
#         std_values = std_data_dict[label]
        
#         # Extract base system name for color/marker lookup
#         base_system = label.split(' - ')[0] if ' - ' in label else label
        
#         color = colors.get(label, colors.get(base_system, 'black'))
#         marker = markers.get(label, markers.get(base_system, 'o'))
#         line_style = line_styles.get(label, line_styles.get(base_system, '-'))
        
#         # Use only the system name for the legend
#         legend_label = base_system
        
#         # Line with markers
#         ax.plot(x_data, y_values, 
#                 color=color, 
#                 marker=marker, 
#                 markersize=8, 
#                 linewidth=2.2, 
#                 linestyle=line_style,
#                 label=legend_label,
#                 markerfacecolor=color,
#                 markeredgecolor='black',
#                 markeredgewidth=0.5,
#                 alpha=0.9)
        
#         # Error bars
#         ax.errorbar(x_data, y_values, yerr=std_values,
#                     fmt='none', 
#                     ecolor=color, 
#                     elinewidth=1.5,
#                     capsize=5,
#                     capthick=1.5,
#                     alpha=0.6,
#                     zorder=5)
    
#     # Set y-axis scale
#     if log_scale:
#         ax.set_yscale('log')
#         ylabel_text = f'{ylabel}'
#     else:
#         ax.set_yscale('linear')
#         ylabel_text = ylabel
    
#     # Labels
#     ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
#     ax.set_ylabel(ylabel_text, fontsize=16, fontweight='bold')
    
#     # Set x-ticks and labels
#     if x_tick_labels is not None:
#         ax.set_xticks(x_data)
#         ax.set_xticklabels(x_tick_labels, fontsize=12)
#     else:
#         ax.set_xticks(x_data)
#         ax.set_xticklabels([f'{x:.2f}' for x in x_data], fontsize=12)
    
#     # INVERTI L'ASSE X (decrescente)
#     ax.invert_xaxis()  # <--- AGGIUNTO: inverte l'asse x
    
#     # Title
#     if title is not None:
#         ax.set_title(title, fontsize=16, fontweight='bold')
    
#     # Grid
#     if log_scale:
#         ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
#     else:
#         ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
#     # Style
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Legend - SOLO se show_legend è True
#     if show_legend:
#         legend = ax.legend(frameon=True, 
#                           fancybox=False, 
#                           edgecolor='gray', 
#                           fontsize=14,
#                           loc='upper left',
#                           ncol=1,
#                           markerscale=1.3,
#                           labelspacing=0.7,
#                           handlelength=2.2,
#                           handletextpad=0.7,
#                           borderpad=0.5)
#         legend.get_frame().set_linewidth(1.0)
    
#     # Adjust limits - keep x axis in original order (decreasing)
#     ax.set_xlim(max(x_data) + 0.01, min(x_data) - 0.01)  # <--- MODIFICATO: invertito per decrescente
    
#     # For log scale, set y limits
#     if log_scale:
#         all_values = []
#         for values in y_data_dict.values():
#             all_values.extend(values)
#         ax.set_ylim(min(all_values) * 0.5, max(all_values) * 2.0)
    
#     plt.tight_layout()
    
#     # Save if we created a new figure
#     if save_fig:
#         fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
#         fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
#         print(f"Graph saved as {filename}.pdf and {filename}.png")
#         return fig, ax
#     else:
#         return ax

# # ===========================================
# # SPARSITY TEST - Three Experiments in Subplot 1x3
# # ===========================================
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rcParams
# from matplotlib.lines import Line2D

# # ===========================================
# # SPARSITY TEST - Due Esperimenti in Subplot 1x2
# # ===========================================

# # X values (decreasing order as shown)
# x_sparsity = [0.99, 0.98, 0.97, 0.96, 0.95]
# x_tick_labels = ['0.99', '0.98', '0.97', '0.96', '0.95']

# # Experiment 2: SINAPSI 0.5 a 1, 0.5 a -1, cariche fisse
# data_exp2 = {
#     'MSNPS(GPU)': {
#         'avg': [23.138, 46.580, 70.173, 94.233, 117.202],
#         'std': [3.724, 7.486, 6.823, 3.591, 3.590]
#     },
#     'OOPSNPS': {
#         'avg': [376.320, 782.652, 1243.465, 1813.882, 2465.712],
#         'std': [37.380, 46.082, 58.860, 85.022, 124.824]
#     },
#     'MSNPS(CPU)': {
#         'avg': [1005.227, 1978.463, 3295.093, 4772.798, 6158.401],
#         'std': [29.112, 71.147, 28.092, 35.696, 38.131]
#     }
# }

# # Experiment 3: sinapsi con 0.5 a 1 e 0.5 a -1
# data_exp3 = {
#     'MSNPS(GPU)': {
#         'avg': [22.654, 46.021, 69.502, 93.555, 116.872],
#         'std': [2.720, 3.291, 3.298, 3.285, 3.302]
#     },
#     'OOPSNPS': {
#         'avg': [52.353, 58.095, 62.550, 67.662, 71.912],
#         'std': [7.612, 10.890, 14.455, 17.979, 21.778]
#     },
#     'MSNPS(CPU)': {
#         'avg': [1168.840, 2370.166, 3566.823, 5022.058, 6805.111],
#         'std': [10.608, 22.039, 31.605, 173.683, 263.380]
#     }
# }

# # Colors and markers (consistent across all plots)
# colors = {
#     'MSNPS(GPU)': '#1F77B4',   # Blue
#     'MSNPS(CPU)': '#D62728',   # Red
#     'OOPSNPS': '#2CA02C'       # Green
# }

# markers = {
#     'MSNPS(GPU)': 'o',    # Circle
#     'OOPSNPS': 's',       # Square
#     'MSNPS(CPU)': '^'     # Triangle
# }

# line_styles = {
#     'MSNPS(GPU)': '-',
#     'OOPSNPS': '-',
#     'MSNPS(CPU)': '-'
# }

# # Create figure with 1 row, 2 columns
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Prepare data for each subplot
# experiments = [
#     (data_exp2, 'Fixed Charges\n(0.5→1, 0.5→-1)'),
#     (data_exp3, 'Synapses\n(0.5→1, 0.5→-1)')
# ]

# # Plot each experiment
# for idx, (data, title) in enumerate(experiments):
#     ax = axes[idx]
    
#     # Extract y_data_dict and std_data_dict
#     y_data_dict = {}
#     std_data_dict = {}
#     for label, values in data.items():
#         y_data_dict[label] = values['avg']
#         std_data_dict[label] = values['std']
    
#     # Plot each line with error bars
#     for label in y_data_dict.keys():
#         y_values = y_data_dict[label]
#         std_values = std_data_dict[label]
        
#         color = colors[label]
#         marker = markers[label]
#         line_style = line_styles[label]
        
#         # Line with markers
#         ax.plot(x_sparsity, y_values, 
#                 color=color, 
#                 marker=marker, 
#                 markersize=8, 
#                 linewidth=2.2, 
#                 linestyle=line_style,
#                 label=label,
#                 markerfacecolor=color,
#                 markeredgecolor='black',
#                 markeredgewidth=0.5,
#                 alpha=0.9)
        
#         # Error bars
#         ax.errorbar(x_sparsity, y_values, yerr=std_values,
#                     fmt='none', 
#                     ecolor=color, 
#                     elinewidth=1.5,
#                     capsize=5,
#                     capthick=1.5,
#                     alpha=0.6,
#                     zorder=5)
    
#     # Set y-axis scale (log)
#     ax.set_yscale('log')
    
#     # Labels - sMΠ in bold usando \mathbf
#     ax.set_xlabel('Sparsity Percentage in $\\mathbf{sM_{\\Pi}}$', fontsize=16, fontweight='bold')
#     if idx == 0:
#         ax.set_ylabel('Average Time per Step (ms)', fontsize=16, fontweight='bold')
    
#     # Set x-ticks and labels
#     ax.set_xticks(x_sparsity)
#     ax.set_xticklabels(x_tick_labels, fontsize=12)
    
#     # Invert x-axis (decreasing)
#     ax.invert_xaxis()
    
#     # Title
#     ax.set_title(title, fontsize=16, fontweight='bold')
    
#     # Grid
#     ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
    
#     # Style
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Set y-axis limits for consistency
#     ax.set_ylim(10, 20000)
#     ax.set_xlim(max(x_sparsity) + 0.01, min(x_sparsity) - 0.01)

# # Adjust layout with more space at the bottom for legend
# plt.tight_layout(rect=[0, 0.08, 1, 1])

# # Add common legend at the bottom in the desired order: MSNPS(CPU) - OOPSNPS - MSNPS(GPU)
# legend_order = ['MSNPS(CPU)', 'OOPSNPS', 'MSNPS(GPU)']
# legend_elements = []
# for label in legend_order:
#     color = colors[label]
#     marker = markers[label]
#     legend_elements.append(Line2D([0], [0], color=color, marker=marker,
#                                   linestyle='-', linewidth=2, markersize=10,
#                                   markerfacecolor=color, markeredgecolor='black',
#                                   markeredgewidth=0.5, label=label))

# fig.legend(handles=legend_elements, frameon=True, fancybox=False, 
#            edgecolor='gray', fontsize=16, loc='lower center', 
#            ncol=3, markerscale=1.5, labelspacing=0.7,
#            handlelength=2.5, handletextpad=0.8, borderpad=0.5,
#            bbox_to_anchor=(0.5, -0.08))

# # Save
# plt.savefig('sparsity_test_two_experiments.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('sparsity_test_two_experiments.png', dpi=600, bbox_inches='tight')
# print("Graph saved as sparsity_test_two_experiments.pdf and sparsity_test_two_experiments.png")

# ===========================================
# LINE CHART: Performance vs K - CON LEGGENDA RIORDINATA
# ===========================================
BK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K = [794 + b * 1014 for b in BK]

# Dati originali
y_data_k = {
    'MSNPS(GPU)': [0.713, 1.271, 2.147, 2.941, 3.725, 4.488, 5.233, 5.963, 6.668, 7.165],
    'OOPSNPS': [14.128, 17.298, 20.490, 23.453, 26.705, 29.749, 32.896, 36.079, 39.119, 42.318],
    'MSNPS(CPU)': [12.145, 35.973, 63.572, 111.022, 142.226, 176.574, 215.771, 261.857, 287.502, 305.829]
}

# STD per MSNPS(GPU) [0.026	0.019	0.019	0.018	0.019	0.018	0.018	0.019	0.020	0.020]
# STD per OOPSNPS [0.346	1.551	2.012	0.311	2.433	2.331	3.879	3.872	4.203	5.423]
# STD per MSNPS(CPU) [0.359	0.580	0.611	0.737	0.759	1.020	1.191	0.808	0.977	1.301]

colors_k = {
    'MSNPS(GPU)': '#1F77B4',   # Blue
    'OOPSNPS': '#2CA02C',      # Green
    'MSNPS(CPU)': '#D62728'    # Red
}

# Marker come nel grafico all_experiments
markers_k = {
    'MSNPS(GPU)': 'o',    # Circle (come in all_experiments)
    'OOPSNPS': 's',       # Square (come in all_experiments)
    'MSNPS(CPU)': '^'     # Triangle (come in all_experiments)
}

line_styles_k = {
    'MSNPS(GPU)': '-',
    'OOPSNPS': '-',
    'MSNPS(CPU)': '-'
}

# Crea il grafico con la funzione originale
fig, ax = create_scientific_line_chart(
    x_data=K,
    y_data_dict=y_data_k,
    xlabel='Number of neurons',
    ylabel='Average time per step (ms)',
    filename='performance_comparison_temp',  # Temporaneo
    colors=colors_k,
    markers=markers_k,
    line_styles=line_styles_k,
    log_scale=False
)

# Riordina la legenda
handles, labels = ax.get_legend_handles_labels()
# Ordine desiderato
legend_order = ['MSNPS(CPU)', 'OOPSNPS', 'MSNPS(GPU)']
# Crea un dizionario per lookup rapido
handle_dict = dict(zip(labels, handles))
# Riordina
ordered_handles = [handle_dict[label] for label in legend_order]
ordered_labels = legend_order

# Rimuovi la legenda esistente e creane una nuova
ax.legend(ordered_handles, ordered_labels, 
          frameon=True, fancybox=False, edgecolor='gray', 
          fontsize=18, loc='upper left', ncol=1,
          markerscale=1.3, labelspacing=0.7,
          handlelength=2.2, handletextpad=0.7, borderpad=0.5)

# Salva con il nome corretto
plt.tight_layout()
fig.savefig('performance_comparison.pdf', format='pdf', bbox_inches='tight')
fig.savefig('performance_comparison.png', dpi=600, bbox_inches='tight')
print("Graph saved as performance_comparison.pdf and performance_comparison.png")

# ===========================================
# LINE CHART: Performance vs K - CON LEGGENDA RIORDINATA E STD
# ===========================================
BK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K = [794 + b * 1014 for b in BK]

y_data_k = {
    'MSNPS(GPU)': [0.713, 1.271, 2.147, 2.941, 3.725, 4.488, 5.233, 5.963, 6.668, 7.165],
    'OOPSNPS': [14.128, 17.298, 20.490, 23.453, 26.705, 29.749, 32.896, 36.079, 39.119, 42.318],
    'MSNPS(CPU)': [12.145, 35.973, 63.572, 111.022, 142.226, 176.574, 215.771, 261.857, 287.502, 305.829]
}

yerr_k = {
    'MSNPS(GPU)': [0.026, 0.019, 0.019, 0.018, 0.019, 0.018, 0.018, 0.019, 0.020, 0.020],
    'OOPSNPS': [0.346, 1.551, 2.012, 0.311, 2.433, 2.331, 3.879, 3.872, 4.203, 5.423],
    'MSNPS(CPU)': [0.359, 0.580, 0.611, 0.737, 0.759, 1.020, 1.191, 0.808, 0.977, 1.301]
}

colors_k = {
    'MSNPS(GPU)': '#1F77B4',
    'OOPSNPS': '#2CA02C',
    'MSNPS(CPU)': '#D62728'
}

markers_k = {
    'MSNPS(GPU)': 'o',
    'OOPSNPS': 's',
    'MSNPS(CPU)': '^'
}

line_styles_k = {
    'MSNPS(GPU)': '-',
    'OOPSNPS': '-',
    'MSNPS(CPU)': '-'
}

# CREA IL GRAFICO SENZA ERRORI
fig, ax = create_scientific_line_chart(
    x_data=K,
    y_data_dict=y_data_k,
    xlabel='Number of neurons',
    ylabel='Average time per step (ms)',
    filename='performance_comparison_temp',  # Temporaneo
    colors=colors_k,
    markers=markers_k,
    line_styles=line_styles_k,
    log_scale=False
)

# AGGIUNGI LE BARRE DI ERRORE MANUALMENTE - STILE ELEGANTE
for label in y_data_k.keys():
    y_values = y_data_k[label]
    y_err = yerr_k[label]
    color = colors_k[label]
    ax.errorbar(K, y_values, yerr=y_err,
                fmt='none',
                ecolor='black',
                elinewidth=1.8,            # Più sottili
                capsize=5,
                capthick=2.5,
                alpha=0.6,                 # Leggermente più trasparenti
                zorder=5)

# ============================================================
# AGGIUNGI MARGINE ORIZZONTALE - RISOLVE IL TAGLIO DELL'ULTIMO VALORE
# ============================================================
# Calcola l'intervallo dei dati
x_range = max(K) - min(K)
# Aggiungi il 10% di margine a sinistra e a destra
margin = x_range * 0.10
ax.set_xlim(min(K) - margin, max(K) + margin)

# Riordina la legenda
handles, labels = ax.get_legend_handles_labels()
legend_order = ['MSNPS(CPU)', 'OOPSNPS', 'MSNPS(GPU)']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in legend_order]
ordered_labels = legend_order

ax.legend(ordered_handles, ordered_labels,
          frameon=True, fancybox=False, edgecolor='gray',
          fontsize=18, loc='upper left', ncol=1,
          markerscale=1.3, labelspacing=0.7,
          handlelength=2.2, handletextpad=0.7, borderpad=0.5)

# Salva
plt.tight_layout()
fig.savefig('performance_comparison2.pdf', format='pdf', bbox_inches='tight')
fig.savefig('performance_comparison2.png', dpi=600, bbox_inches='tight')
print("Graph saved as performance_comparison2.pdf and performance_comparison2.png")