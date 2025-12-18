# Automated Off-Policy Estimator Selection via Supervised Learning

## Installation
1. First of all, clone this repository, then enter in the repository folder

2. Create the conda environment with all the dependencies needed: 
    - if cuda-based NVIDIA GPUs are available, run
    ```shell script
    conda env create -f environment_nvidia.yml
    ```
    - otherwise run the following command (no GPUs support)
    ```shell script
    conda env create -f environment.yml
    ```

3. Download the datasets used in the experiments reported in the paper, that are 
    - the Open Bandit Dataset, available [here](https://research.zozo.com/data.html)
    - the CIFAR10 Dataset, available [here](https://www.cs.toronto.edu/~kriz/cifar.html)

4. Create the directories that will contain the datasets used in the experiments:
    - Open Bandit Dataset:
    ```shell script
    mkdir -p real_datasets/open_bandit_dataset
    ```
    Copy the downloaded dataset in the created directory.

    - CIFAR-10
    ```shell script
    mkdir -p real_datasets/cifar10 
    ```
    Unzip and copy the downloaded dataset in the created directory. Be sure that the data path is **_real_datasets/cifar10/cifar-10-batches-py/_**

5. Finally, run the code to generate the data, optimise and train the model, and execute the experiments. 

In the following the commands for all the paper experiments are reported. For a more in depth explanation of each parameter you can take a look to the code.

### Data Generation
To generate the synthetic data used to train the AutoOPE model, run this command:
```shell script
cd src/black_box/data/
PYTHONPATH=../../ python3 binary_rwd_dataset_generation.py 
```

### Model Optimization
- To optimize, train and test the AutoOPE model on the generated synthetic dataset, run this command:
```shell script
cd src/black_box/estimator_selection/
PYTHONPATH=../../ python3 optimization.py --val_perc 0.25
```


## Experiments
Below are reported the command line arguments used for each experiment:


- **Open Bandit Dataset experiment**
```shell script
cd src/black_box/evaluation/
PYTHONPATH=../../ python3 real_data_evaluation_run.py \
--dataset obd \
--subsampling_ratio 0.05 \
--n_data_generation 20  \
--outer_n_jobs 20 
```

- **UCI Datasets experiments**
  
The parameter ```<NAME>``` is set to a different value based on the dataset. The complete list of all the values is the following: 'letter', 'optdigits', 'page-blocks', 'pendigits', 'satimage', 'vehicle', 'yeast', 'breast-cancer'.
```shell script
cd src/black_box/evaluation/
PYTHONPATH=../../ python3 real_data_evaluation_run.py \
--dataset <NAME> \
--subsampling_ratio 0.9 \
--n_data_generation 50  \
--outer_n_jobs 50 
```

- **Cifar-10 experiment**
```shell script
cd src/common/evaluation/
PYTHONPATH=../../ python3 real_data_evaluation_run.py \
--dataset cifar10 \
--subsampling_ratio 0.66666 \
--n_data_generation 20  \
--outer_n_jobs 20 
```

- **Synthetic experiments**
    - $(\beta_1, \beta_2) = (2, -2)$
    ```shell script
    cd src/black_box/evaluation/
    PYTHONPATH=../../ python3 synthetic_data_evaluation_run.py \
    --beta_1 -2 \
    --beta_2 2 \
    --outer_n_jobs_gt 100 \
    --inner_n_jobs 10 \
    --outer_n_jobs 100 
    ```

    - $(\beta_1, \beta_2) = (3, 7)$
    ```shell script
    cd src/black_box/evaluation/
    PYTHONPATH=../../ python3 synthetic_data_evaluation_run.py \
    --beta_1 3 \
    --beta_2 7 \
    --outer_n_jobs_gt 100 \
    --inner_n_jobs 10 \
    --outer_n_jobs 100 
    ```

## Additional Scaling and Ablation Experiments
- To reproduce the scaling experiments reported in the paper, run this command:
```shell script
cd src/black_box/estimator_selection/
PYTHONPATH=../../ python3 incremental_train_size_optimization.py \
```

- To reproduce the ablation experiments on the features types reported in the paper, run this command with ```<FEATURE_TYPE>``` chosen from 'policy_dep', 'policy_indep', 'estimator_dep':
```shell script
cd src/black_box/estimator_selection/
PYTHONPATH=../../ python3 optimization.py \
--features_subset <FEATURE_TYPE>
```

- To reproduce the ablation experiments on the dataset diversity reported in the paper, run this command with ```<DATA_TYPE>``` chosen from 'KL', 'actions':
```shell script
cd src/black_box/estimator_selection/
PYTHONPATH=../../ python3 optimization.py \
--features_subset <DATA_TYPE>
```
