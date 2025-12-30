import typing
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)}, font_scale=1.5)
sns.set_palette(sns.color_palette("Paired"))

def create_scatterplots(accuraciesTL: dict):
    # Create figure and axis
    fig, ax = plt.subplots()
    
    paired_palette = sns.color_palette("Paired")
    
    # Plot each network
    for idx, (network_name, network_data) in enumerate(accuraciesTL.items()):
        accuracies, median_perturbations = network_data
        accuracies_tl = accuracies[1:]
        median_perturbations_tl = median_perturbations[1:]
        
        labels = [f'TL{i}' for i in range(1, len(accuracies))]
        
        color = paired_palette[idx % len(paired_palette)]
        
        ax.plot(accuracies_tl, median_perturbations_tl, linewidth=1.5, label=network_name, color=color, marker='x', markersize=10)
        
        # Add numbered markers for each TL point showing which retraining layer it corresponds to
        for i, (acc, map_val) in enumerate(zip(accuracies_tl, median_perturbations_tl)):
            ax.text(acc+0.0015, map_val+0.005, str(i+1), fontsize=16, ha='center', va='center',
                   color=color, weight='bold')
    
    ax.set_xlabel('Clean Accuracy')
    ax.set_ylabel('Median minimum adversarial perturbation')
    ax.set_title('Transfer Learning MNIST to EMNIST')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.close()

    return fig

def main():
    accuraciesTL = {
        # First accuracy/MAP corresponds to source model
        'Arbitrary small CNN' : ([0.9763, 0.7881,0.8444,0.8448], [0.355, 0.035, 0.050, 0.035]),
        'RELU_4_1024' : ([0.9842, 0.7159,0.7699,0.7997,0.8498], [0.395, 0.105, 0.115, 0.065, 0.050]), 
        'CNN Madry et al.' : ([0.9903, 0.8402,0.8602,0.8696,0.8777], [0.381,0.231,0.256,0.101,0.071]),
        'CNN Yang et al.' : ([0.9929, 0.7574, 0.8189, 0.8584,0.8695,0.867, 0.869, 0.8748], [0.401, 0.196, 0.201, 0.251, 0.256, 0.266, 0.071,0.076])
    }

    base_path = Path("/home/s3665534/VERONA/examples/Networks/MNIST")

    scatterplot = create_scatterplots(accuraciesTL)
    scatterplot.savefig(base_path / "scatter_accuracy.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()