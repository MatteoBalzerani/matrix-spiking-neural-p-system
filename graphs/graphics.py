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
                                colors=None, edgecolor='black'):
    """
    Creates and saves a bar chart of scientific quality for categorical data.
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
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if ax.get_yscale() == 'log':
            text_y = height * 1.1
        else:
            text_y = height + height * 0.01
        ax.text(bar.get_x() + bar.get_width()/2., text_y,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=18)
    
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
# BAR CHART 1: Average times per system
# ===========================================
system_colors = ['#1F77B4', '#D62728', '#2CA02C']

systems = ['MSNPS(GPU)', 'MSNPS(CPU)', 'OOPSNPS']
execution_times = [6.168, 270.861,41.067]

fig, ax = create_scientific_bar_chart(
    categories=systems,
    values=execution_times,
    xlabel='System',
    ylabel='Average time per step (ms)',
    filename='system_comparison',
    colors=system_colors
)

# ===========================================
# LINE CHART: Performance vs K
# ===========================================
BK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K = [794 + b * 1014 for b in BK]

# NOTA: Ho cambiato le chiavi del dizionario per usare solo il nome del sistema
# La funzione ora estrae automaticamente il nome base per la legenda
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
    ylabel='Time (ms)',
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

# Create the grouped bar chart WITH LOG SCALE
fig, ax = create_grouped_bar_chart(
    categories=size_categories,
    values_dict=values_size,
    xlabel='Problem Size',
    ylabel='Time (ms)',
    filename='performance_size_grouped_bar',
    colors=colors_size,
    log_scale=True,  # Logarithmic scale for better comparison
    bar_width=0.22
)