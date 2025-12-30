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
            ax.text(acc, map_val+0.0008, str(i+1), fontsize=16, ha='center', va='center',
                   color=color, weight='bold')
    
    ax.set_xlabel('Clean Accuracy')
    ax.set_ylabel('Median minimum adversarial perturbation')
    ax.set_title('Additional hidden layers ResNet-34')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.close()

    return fig

def main():
    accuracies_CIFAR = {
        'ResNet-18' : ([0.1843, 0.1877, 0.3866, 0.46, 0.5858, 0.613, 0.6129], [0.015, 0.014, 0.010, 0.008, 0.005,0.002,0.002]),
        'ResNet-34' : ([0.1967,0.1925,0.3954,0.4652,0.5863,0.6154,0.6042], [0.018,0.018, 0.011, 0.008, 0.004, 0.002,0.001]),
        'ResNet-50' : ([0.0995,0.1032,0.2888,0.4747,0.5696,0.6002,0.6061], [0.020, 0.025, 0.009, 0.005, 0.003,0.002,0.002]),
        'WideResNet-28-10' :  ([0.0948,0.1601,0.4025,0.5092,0.5987], [0.025, 0.018, 0.010, 0.007, 0.003]),
        'WideResNet-34-10' : ([0.1052,0.1733,0.3895,0.5093,0.5901], [0.028, 0.018, 0.012, 0.007, 0.002])
    }

    accuracies_ResNet34 = {
        'ResNet-34' : ([0.1967,0.1925,0.3954,0.4652,0.5863,0.6154,0.6042], [0.018,0.018, 0.011, 0.008, 0.004, 0.002,0.001]),
        'One hidden layer' : ([0.2579,0.2508,0.3933,0.4501,0.5723,0.596,0.6029], [0.017 ,0.017, 0.011, 0.009, 0.004,0.002, 0.001]),
        'Two hidden layers' : ([0.2995, 0.3069, 0.4423, 0.5491, 0.5831, 0.6195, 0.5974],[0.016, 0.016, 0.010, 0.008,0.004, 0.002, 0.002]),
        'Three hidden layers' : ([0.3096, 0.301, 0.4189, 0.5427, 0.5664, 0.6121, 0.6128], [0.015,0.016, 0.011,0.008 ,0.005,0.002,0.002])
    }
    base_path = Path("/home/s3665534/VERONA/examples/Networks/CIFAR")

    scatterplot = create_scatterplots(accuracies_ResNet34)
    scatterplot.savefig(base_path / "scatter_accuracy.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()