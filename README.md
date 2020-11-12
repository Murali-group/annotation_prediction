# annotation-prediction
Pipeline for running and evaluating algorithms for gene/protein annotation prediction, 
e.g., to the Gene Ontology (GO) and to the Human Phenotype Ontology(HPO).


## Installation

- Required Python packages: `networkx`, `numpy`, `scipy`, `pandas`, `sklearn`, `pyyaml`, `tqdm`, `rpy2`
- Required R packages: PPROC

```
conda create -n fastsinksource python=3.7 r=3.6
conda activate fastsinksource
pip install -r requirements.txt
```
To install the R packages:
```
R -e "install.packages('https://cran.r-project.org/src/contrib/PRROC_1.3.1.tar.gz', type = 'source')"
conda install -c bioconda bioconductor-clusterprofiler
```

## Usage 
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

Jeffrey N. Law, Shiv D. Kale, and T. M. Murali. [Accurate and Efficient Gene Function Prediction using a Multi-Bacterial Network](https://doi.org/10.1093/bioinformatics/btaa885), _Bioinformatics (2020). 
