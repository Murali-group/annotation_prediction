# FastSinkSource
This is the main repository for the paper "Accurate and Efficient Gene Function Prediction using a Multi-Bacterial Network".

## Installation
These scripts requires Python 3 due to the use of obonet to build the GO DAG.

Required packages: `networkx`, `numpy`, `scipy`, `pandas`, `sklearn`, `obonet`, `pyyaml`, `tqdm`

To install the required packages:
```
pip3 install -r requirements.txt
```

Optional: use a virtual environment
```
virtualenv -p /usr/bin/python3 py3env
source py3env/bin/activate
pip install -r requirements.txt
```

## Download Datasets
The networks and GO term annotations for the 200 bacterial species with the most EXPC and COMP annotations are available here: http://bioinformatics.cs.vt.edu/~jeffl/supplements/2020-fastsinksource/

## Run the FastSinkSource Pipeline
### Generate predictions
The script will automatically generate predictions from each of the given methods with `should_run: [True]` in the config file. The default number of predictions stored is 10. To write more, use either the `--num-pred-to-write` or `--factor-pred-to-write options` (see python run_eval_algs.py --help). For example:
```
python run_eval_algs.py  --config config.yaml --num-pred-to-write -1
```

### Cross Validation
The relevant options are below. See `python run_eval_algs.py --help` for more details.
  - `cross_validation_folds`
    - Number of folds to use for cross validation. Specifying this parameter will also run CV
  - `cv_seed`
    - Can be used to specify the seed to use when generating the CV splits. 
    
Example:
```
python run_eval_algs.py  --config config.yaml --cross-validation-folds 5 --only-eval
```

#### Plot
After CV has finished, to visualize the results, use the `plot.py` script. For example:
```
python plot.py --config config.yaml --box --measure fmax
```

## Cite
If you use FastSinkSource or other methods in this package, please cite:

Jeffrey Law, Shiv D. Kale, and T. M. Murali. [Accurate and Efficient Gene Function Prediction using a Multi-Bacterial Network](https://doi.org/10.1101/646687), _bioRxiv_ (2020). doi.org/10.1101/646687
