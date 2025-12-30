# bachelorthesis_davidwunsch
This repository contains all scripts that were needed to perform the experiments and create the plots for the thesis "Transfer Learning of Robustness for Image Classification Neural Networks". 

## Adversarial Training Box scripts
The folder "adversarial-training-box-scripts" contains the scripts that were used with the adversarial-training-box (https://github.com/Aaron99B/adversarial-training-box). It contains the scripts used for conventional and adversarial training and transfer learning. The scripts to create the adversarially trained models were cifar-pgd-training.py, pgd-training.py and emnist-pgd-training.py. The scripts for conventional training are cifar-standard-training.py, standard-training.py and emnist-standard-training.py. The scripts for transfer learning using conventional retraining are transfer_learn.py and transfer_learn_cifar.py. The scripts for transfer learning using adversarial retraining are transfer_learn_adversarial.py and transfer_learn_adversarial_cifar.py.

## VERONA scripts

The folder "verona-scripts" contains the scripts that were used with ADA-VERONA (https://github.com/ADA-research/VERONA). With this script the robustness distributions for my experiments were created. The scripts are heavily based on the example script for pytorch models, which can be found under the mentioned GitHub.

## Plotting scripts
The folder "plotting-scripts" contains the scripts that were used for creation of plots. The plotting script stem originally from ADA-VERONA and were modified to also plot the accuracies and add the median minimum adversarial perturbations and accuracies as text in the plot.  
