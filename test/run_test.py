
from pathlib import Path

import pandas as pd
import numpy as np

from rf_tools import RFPipeline




if __name__ == "__main__":
    
    # test on mock-data
    n_rows = 100
    n_cols = 10 # number of metric variables
    mode = "regression" # or "regression"
    
    import os
    path_output = Path(os.getcwd())
    
    columns = [f"num{i}" for i in range (n_cols)]
    
    # generate mock-data
    feats = pd.DataFrame(np.random.rand(n_rows, n_cols), columns=columns)
    
    # generate mock prediction-labels
    if mode == "classification":
        labels = pd.Series(np.random.randint(0, 2, size=(n_rows, )))
    else:
        labels = pd.Series(np.random.rand(n_rows))
    
    # add some categorical variables
    feats["cat1"] = np.random.randint(0,2, size=(n_rows, )).astype(str)
    feats["cat2"] = np.random.randint(0,4, size=(n_rows, )).astype(str)
    feats["cat3"] = np.random.randint(0,4, size=(n_rows, )).astype(str)
    feats["cat4"] = np.random.randint(0,3, size=(n_rows, )).astype(str)
    
    # generate random missingness
    mask = np.random.randint(0, 10, size=feats.shape) > 8
    feats[mask] = np.nan
    
    # run pipeline
    model = RFPipeline(
        feats, 
        labels, 
        n_folds=3, 
        n_imputations=10, 
        path_output=path_output, 
        categorical_vars = ["cat1", "cat2", "cat3", "cat4"], 
        mode=mode
    )