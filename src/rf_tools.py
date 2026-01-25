
from sklearn.experimental import enable_iterative_imputer #noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance

from scipy.stats import pearsonr

import pandas as pd
import numpy as np

from tqdm import tqdm

from pathlib import Path
import pickle as pkl

def _impute_missing(
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        cat_vars: list[str] | None, 
        M: int = 5, 
        random_state: int | None = 42
    ) -> tuple[list, list, list]:
    
    """
    Perform multiple imputation (MICE) on numeric and one-hot encoded categorical variables.
    Returns M imputed datasets (train/test) on mixed metric/categorical dataframes. 
    
    The imputer is fit on the training data and used to transform the test data to prevent leakage. 
    
    Parameters:
        - train_df (pd.DataFrame)
        - test_df (pd.DataFrame)
        - cat_vars (list): List of categorical variables for one-hot encoding
        - M (int): Number of Imputations to perform. Higher M allows better estimation
        of uncertainty. 
        - random_state (int): Pass a number for reproducibility, None for randomness. 
        Note: Each imputation adds +1 to random_state to ensure different datasets are
        created. 
    
    Returns:
        - imputed_train_list (list): List of imputed training DataFrames (n=M)
        - imputed_test_list (list): List of imputed test DataFrames (n=M)
        - imputers (list): List of imputers fitted during imputation
    
    Note: The current implementation of cat_vars imputation is not optimal and 
    will be switched out for imputation using MissForest (https://github.com/stekhoven/missForest)
    """
    
    cat_vars = [] if cat_vars is None else cat_vars
    num_vars = [c for c in train_df.columns if c not in cat_vars]

    # Separate numerics and categoricals
    train_num = train_df[num_vars]
    test_num = test_df[num_vars]
    train_cat = train_df[cat_vars].copy()
    test_cat = test_df[cat_vars].copy()

    sentinel = "missing"
    train_cat_filled = train_cat.fillna(sentinel).astype(str)
    test_cat_filled = test_cat.fillna(sentinel).astype(str)

    # Fit encoder on train only
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat_encoded = pd.DataFrame(
        enc.fit_transform(train_cat_filled),
        columns=enc.get_feature_names_out(cat_vars),
        index=train_df.index
    )
    test_cat_encoded = pd.DataFrame(
        enc.transform(test_cat_filled),
        columns=enc.get_feature_names_out(cat_vars),
        index=test_df.index
    )

    # Convert sentinel-encoded rows to NaN
    for var in cat_vars:
        sentinel_col = f"{var}_{sentinel}"
        if sentinel_col not in train_cat_encoded.columns:
            continue

        var_cols = [c for c in train_cat_encoded.columns if c.startswith(f"{var}_")]

        # Mark sentinel rows
        train_rows = train_cat_encoded[sentinel_col] == 1
        test_rows = test_cat_encoded[sentinel_col] == 1

        train_cat_encoded.loc[train_rows, var_cols] = np.nan
        test_cat_encoded.loc[test_rows, var_cols] = np.nan

        # Drop sentinel column
        train_cat_encoded.drop(columns=[sentinel_col], inplace=True)
        test_cat_encoded.drop(columns=[sentinel_col], inplace=True)

    # Combine numeric + encoded categorical
    X_train_combined = pd.concat([train_num, train_cat_encoded], axis=1)
    X_test_combined = pd.concat([test_num, test_cat_encoded], axis=1)

    imputed_train_list = []
    imputed_test_list = []
    imputers = []

    for m in tqdm(range(M), desc="     Imputing datasets"):
        imp = IterativeImputer(
            random_state=random_state + m,
            max_iter=10,
            sample_posterior=True
        )

        X_train_imp = imp.fit_transform(X_train_combined)
        X_test_imp = imp.transform(X_test_combined)

        X_train_imp_df = pd.DataFrame(X_train_imp, columns=X_train_combined.columns, index=train_df.index)
        X_test_imp_df = pd.DataFrame(X_test_imp, columns=X_test_combined.columns, index=test_df.index)

        # Collapse imputed one-hot categorical back to category labels
        X_cat_train_imp = pd.DataFrame(index=train_df.index)
        X_cat_test_imp = pd.DataFrame(index=test_df.index)

        for var in cat_vars:
            cols = [c for c in X_train_imp_df.columns if c.startswith(f"{var}_")]
            if not cols:
                X_cat_train_imp[var] = np.nan
                X_cat_test_imp[var] = np.nan
                continue

            # Use column names to derive category labels
            categories = [c.split(f"{var}_", 1)[1] for c in cols]

            # TRAIN
            train_sub = X_train_imp_df[cols].to_numpy()
            argmax_idx_train = np.argmax(train_sub, axis=1)
            recovered_train = np.array(categories)[argmax_idx_train].astype(object)
            recovered_train[recovered_train == sentinel] = np.nan

            X_cat_train_imp[var] = recovered_train

            # TEST
            test_sub = X_test_imp_df[cols].to_numpy()
            argmax_idx_test = np.argmax(test_sub, axis=1)
            recovered_test = np.array(categories)[argmax_idx_test].astype(object)
            recovered_test[recovered_test == sentinel] = np.nan
            
            X_cat_test_imp[var] = recovered_test

        # Label encode recovered categorical columns (convert to integers)
        X_cat_train_encoded = X_cat_train_imp.copy()
        X_cat_test_encoded = X_cat_test_imp.copy()
        for col in cat_vars:
            labels = pd.Series(X_cat_train_encoded[col].dropna().unique()).sort_values()
            label_map = {label: i for i, label in enumerate(labels)}
            X_cat_train_encoded[col] = X_cat_train_encoded[col].map(label_map)
            X_cat_test_encoded[col] = X_cat_test_encoded[col].map(label_map)  # unseen categories → NaN

        # Combine with numeric
        X_train_final = pd.concat([X_train_imp_df[num_vars], X_cat_train_encoded], axis=1)[train_df.columns]
        X_test_final = pd.concat([X_test_imp_df[num_vars], X_cat_test_encoded], axis=1)[test_df.columns]

        imputed_train_list.append(X_train_final)
        imputed_test_list.append(X_test_final)
        imputers.append(imp)
        
    return imputed_train_list, imputed_test_list, imputers

def _init_model(
        mode: str, 
        config_params: dict | None = None
    ) -> RandomForestClassifier | RandomForestRegressor:
    
    base_config = {
        "n_estimators": 100,
        "criterion": "gini" if mode == "classification" else "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt" if mode == "classification" else 1.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": None,
        "random_state": None,
        "verbose": 0,
        "warm_start": False,
    }
    
    # resolve config based on base_config and parsed config_params
    config = {**base_config, **(config_params or {})}
    
    # define model based on mode
    model_class = (
        RandomForestClassifier
        if mode == "classification"
        else RandomForestRegressor
    )
    
    return model_class(**config)

def _performance_metrics(
        model: RandomForestClassifier | RandomForestRegressor, 
        x_test: pd.DataFrame, 
        y_test: pd.DataFrame, 
        mode: str
    ) -> dict:
    
    ''' Calculates Performance Metrics based on model-type '''
    
    if mode == "classification":
        
        # get predictions and prediction probabilities
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, -1]
        
        # true positive, false positive, false negative, true positive
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        def safe_div(n, d): return n / d if d else 0.0
    
        auc = roc_auc_score(y_test, y_prob) # Area Under the Curve
        sens = safe_div(tp, tp + fn) # Sensitivity
        spec = safe_div(tn, tn + fp) # Specificity
        ppv = safe_div(tp, tp + fp) # Positive-Predictive Value
        f1 = f1_score(y_test, y_pred, pos_label=1) # F1-Score
        acc = safe_div(tp + tn, tp + tn + fp + fn) # Accuracy
        
        return pd.Series(
            {"AUC": auc, "Sensitivity": sens, "Specificity": spec, 
             "F1-Score": f1, "PPV": ppv, "Accuracy": acc}
        )
    
    elif mode == "regression":
        
        # get predictions and correlate
        y_pred = model.predict(x_test)
        r = pearsonr(y_pred, y_test)
        
        return pd.Series(
            {"Pearson": r.statistic, "p-value": r.pvalue}
        )

def _save_outputs(
        path_output: Path, 
        performance_metrics: dict,
        models: dict, 
        feature_importances: dict, 
        n_folds: int, 
        n_imputations: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]: 
    '''
    Saves outputs generated during RFPipeline. 
    
    Per Full experiment run (saved to path_output/*):
        - Summary Performance Metrics (.xlsx)
        - Summary Feature Importances (.xlsx)
    
    Per imputation (saved to path_output/fold-x/m-x/*)
        - Performance Metrics (.xlsx)
        - Models (.pkl)
        - Feature importances (.xlsx)
    
    Returns: 
        - summary_metrics (pd.DataFrame)
        - summary_importances (pd.DataFrame)
    '''
    # iterate over k folds
    for fold in range(n_folds):
        
        fold_key = f"fold-{fold}"
        
        # iterate over m imputations
        for m in range(n_imputations):
            m_key = f"m-{m}"
            
            # save each imputation result to imputation-specific dir
            output_dir = path_output / fold_key / m_key
            output_dir.mkdir(exist_ok=True, parents=True)
            
            performance_metrics[fold_key][m_key].to_excel(output_dir / "performance-metrics.xlsx")
            feature_importances[fold_key][m_key].to_excel(output_dir / "feature-importances.xlsx")
            
            # save imputation-specific model
            with open(output_dir / "model.pkl", "wb") as f:
                pkl.dump(models[fold_key][m_key], f)
    
    # get results across all folds and imputations into single pd.DataFrame
    rows = []
    
    # get fold (key) and fold_dict (= dict of imputation-specific results)
    for fold, fold_dict in performance_metrics.items():
        
        # get imputation (key) and series (= results)
        for imputation, series in fold_dict.items():
            
            # format to {"fold": "fold-x", "imputation": "m-x", "AUC": ...}
            row = {
                "fold": fold,
                "imputation": imputation,
                **series.to_dict()
            }
            
            rows.append(row)

    summary_metrics = pd.DataFrame(rows)
    summary_metrics.to_excel(path_output / "summary_performance-metrics.xlsx")
    
    # get feature importances across all folds/imputations into single pd.DataFrame
    cols = []
    for fold, fold_dict in feature_importances.items():
        for imputation, col in fold_dict.items():
            cols.append(col)
    
    # Concatenate into dataframe and calculate mean importance across all runs
    summary_importances = pd.concat(
        cols, axis=1
    ).mean(axis=1).sort_values(ascending=False)
    
    summary_importances.to_excel(path_output / "summary_feature-importances.xlsx")
    
    return summary_metrics, summary_importances
    
def RFPipeline(
        feats_src: pd.DataFrame, 
        targets_src: pd.DataFrame, 
        mode: str = "classification", # or "regression"
        n_folds: int = 5, 
        n_imputations: int = 5, 
        n_permutations: int = 5, 
        categorical_vars: list = None, 
        config_params: dict = None, 
        path_output: Path = None,
        random_state: int = 42, 
        overwrite = False # not implemented -> should prevent overwrite when re-running
    ) -> tuple[dict, dict, dict]:
    
    '''
    Performs a complete Random-Forest Training pipeline, including:
        1) k-fold Crossvalidation
        2) Missing Data imputation using MICE 
           (https://scikit-learn.org/stable/modules/impute.html)
        3) Model Configuration via config_params
        4) Train/Validation
        5) Organized Output Generation (Predictions, Performance-Metrics, 
           Models, Feature-Importance)
    
    Currently, testing can be only performed outside this function. Make sure 
    you pass only data allocated to training and/or validation. 
    
    If you want to do a full cross-validation, pass the desired number of folds 
    (n_folds > 1). This should be used for hyperparameter tuning or exploratory 
    purposed only! 
    
    If you want to train a model for real testing purposes, pass n_folds=1. This
    will result in training the model on the full feats dataframe (incl. imputation) 
    but skip all other steps of the pipeline. It returns only the model and the 
    imputer. This is optimally performed after a crossvalidation run for hyperparameter 
    tuning. 
    
    Returns:
        - performance_metrics (dict): performance metrics across each fold and imputation
        - models (dict): trained models across each fold and imputation
        - feature importances (dict): importances across each fold and imputation
        
        All dicts are structured as {"fold-x": {"m-x": <item>, "m-y": ...}, "fold-y": ...}
    '''
    
    assert mode in ["classification", "regression"], (
        "Please provide a valid mode: 'classification', 'regression'\n"
        f"Mode provided: {mode}"
    )
    
    if not path_output is None:
        path_output.mkdir(exist_ok=True, parents=True)
        path_out = path_output / mode
        path_out.mkdir(exist_ok=True, parents=True)
    
    # assign source dataframes to new vars to prevent any unintentional modifications
    feats = feats_src.copy()
    targets = targets_src.copy()
    
    # if n_folds > 1 -> user wants k-fold crossval
    if n_folds > 1:
        if mode == "classification":
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        elif mode == "regression":
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            
        splits = [x for x in kf.split(feats, targets)]
    
    # else: assign full feats as a single training-split
    # this assumes the test-set has been split off outside the function and 
    # will be used accordingly. #TODO: This usecase will be handled more explicitly (see docstring)
    else:
        idx = np.arange(len(feats_src))
        splits = [(idx, idx)]
    
    # overall containers
    performance_metrics = {}
    feature_importances = {}
    models = {}
    
    print(f"Starting {n_folds}-fold crossvalidation | Mode: {mode} | Imputations: {n_imputations}")
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        
        # fold containers
        fold_metrics = {}
        fold_importances = {}
        fold_models = {}
        
        # subset feature dataframes
        x_train_raw = feats.iloc[train_idx]
        x_test_raw = feats.iloc[test_idx]
        
        # subset target (prediction label) dataframes
        y_train = targets.iloc[train_idx]
        y_test = targets.iloc[test_idx]
        
        # save raw (before imputation) features of fold to path_output/fold-x
        if not path_output is None:
            fold_dir = path_out / f"fold-{fold}"
            fold_dir.mkdir(exist_ok=True, parents=True)
            
            x_train_raw.to_excel(fold_dir / "train_features_raw.xlsx")
            x_test_raw.to_excel(fold_dir / "test_features_raw.xlsx")
            y_train.to_excel(fold_dir / "train_targets.xlsx")
            y_test.to_excel(fold_dir / "test_targets.xlsx")
        
        # impute missing data and get imputer
        x_train_imputed_list, x_test_imputed_list, imputers = _impute_missing(
            x_train_raw, 
            x_test_raw, 
            categorical_vars, 
            M = n_imputations, 
            random_state = random_state
        )
        
        # perform m imputations
        for m in tqdm(range(n_imputations), desc=f"  Training models (fold {fold})"):
            
            # get x_train, x_test
            x_train = x_train_imputed_list[m]
            x_test = x_test_imputed_list[m]
            
            # save imputed data and imputer to path_output/fold-x/m-x/
            if not path_output is None:
                m_dir = fold_dir / f"m-{m}"
                m_dir.mkdir(exist_ok=True, parents=True)
                
                x_train.to_excel(m_dir / "train_features_imputed.xlsx")
                x_test.to_excel(m_dir / "test_features_imputed.xlsx")
                
                with open(m_dir / "imputer.pkl", "wb") as f:
                    pkl.dump(imputers[m], f)
            
            # initialize model
            model = _init_model(mode, config_params=config_params)
            
            # fit model
            model.fit(x_train, y_train)
            
            # if n_folds == 1 -> assume user wants to train on full feats and 
            # test outside function (see docstring)
            if n_folds == 1:
                return model, imputers[m]
            
            fold_models[f"m-{m}"] = model
            
            # get performance metrics
            metrics = _performance_metrics(model, x_test, y_test, mode)
            fold_metrics[f"m-{m}"] = metrics.copy()
            
            # get feature importance
            fi = permutation_importance(
                model, x_test, y_test, n_repeats=n_permutations
            )
            
            # format feature importances
            feature_importance = pd.DataFrame(
                fi["importances_mean"], 
                index=x_train.columns, 
                columns=["FI-Score"]
            )
            
            feature_importance.index.name = "Feature"
            
            fold_importances[f"m-{m}"] = feature_importance.copy()
        
        # store fold-specific outputs to overall containers
        performance_metrics[f"fold-{fold}"] = fold_metrics.copy()
        models[f"fold-{fold}"] = fold_models.copy()
        feature_importances[f"fold-{fold}"] = fold_importances.copy()
    
    # save all results
    if not path_output is None:
        #TODO: summary dataframes are returned by saveOutputs but not assigned. 
        #      any usecase for them? 
        _save_outputs(
            path_out, 
            performance_metrics, 
            models, 
            feature_importances,
            n_folds, 
            n_imputations
        )
    
    # Generate Plots
    #TODO
    
    return performance_metrics, models, feature_importances



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




