sbatch das5/exp-hyperp-label-propagation.job expander "" propagate
sbatch das5/exp-hyperp-label-propagation.job rbf 1 propagate
sbatch das5/exp-hyperp-label-propagation.job rbf 0.5 propagate
sbatch das5/exp-hyperp-label-propagation.job rbf 0.1 propagate
sbatch das5/exp-hyperp-label-propagation.job rbf 0.01 propagate
sbatch das5/exp-hyperp-label-propagation.job expander "" spread
sbatch das5/exp-hyperp-label-propagation.job expander "" nearest
sbatch das5/exp-hyperp-label-propagation.job rbf 1 nearest
