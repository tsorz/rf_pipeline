
from sklearn.experimental import enable_iterative_imputer #noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve
from sklearn.inspection import permutation_importance

from scipy.stats import pearsonr

import pandas as pd
import numpy as np

from tqdm import tqdm

from pathlib import Path
import pickle as pkl



def _impute_missing(
        train_df, 
        test_df, 
        cat_vars, 
        M=5, 
        random_state=42
    ):
    
    """
    Perform multiple imputation (MICE) on numeric and one-hot encoded categorical variables.
    Returns M imputed datasets (train/test) with categorical columns restored and label-encoded.
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

def initModel(mode, config_params=None):

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

    config = {**base_config, **(config_params or {})}

    model_class = (
        RandomForestClassifier
        if mode == "classification"
        else RandomForestRegressor
    )

    return model_class(**config)

def _performance_metrics(model, x_test, y_test, mode):
    
    if mode == "classification":
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, -1]
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
        def safe_div(n, d): return n / d if d else 0.0
    
        auc = roc_auc_score(y_test, y_prob)
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        ppv = safe_div(tp, tp + fp)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        acc = safe_div(tp + tn, tp + tn + fp + fn)
    
        return pd.Series(
            {"AUC": auc, "Sensitivity": sens, "Specificity": spec, 
             "F1-Score": f1, "PPV": ppv, "Accuracy": acc}
        )
    
    elif mode == "regression":
        y_pred = model.predict(x_test)
        r = pearsonr(y_pred, y_test)
        
        return pd.Series(
            {"Pearson": r.statistic, "p-value": r.pvalue}
        )

def saveOutputs(
        path_output, 
        performance_metrics,
        models, 
        feature_importances, 
        n_folds, 
        n_imputations
    ): 
    
    for fold in range(n_folds):
        fold_key = f"fold-{fold}"
        
        for m in range(n_imputations):
            m_key = f"m-{m}"
            
            output_dir = path_output / fold_key / m_key
            output_dir.mkdir(exist_ok=True, parents=True)
            
            performance_metrics[fold_key][m_key].to_excel(output_dir / "performance-metrics.xlsx")
            feature_importances[fold_key][m_key].to_excel(output_dir / "feature-importances.xlsx")
            
            with open(output_dir / "model.pkl", "wb") as f:
                pkl.dump(models[fold_key][m_key], f)
    
    rows = []
    for fold, fold_dict in performance_metrics.items():
        for imputation, series in fold_dict.items():
            row = {
                "fold": fold,
                "imputation": imputation,
                **series.to_dict()
            }
            rows.append(row)

    summary_metrics = pd.DataFrame(rows)
    summary_metrics.to_excel(path_output / "summary_performance-metrics.xlsx")
    
    cols = []
    for fold, fold_dict in feature_importances.items():
        for imputation, col in fold_dict.items():
            cols.append(col)
    
    summary_importances = pd.concat(
        cols, axis=1
    ).mean(axis=1).sort_values(ascending=False)
    
    summary_importances.to_excel(path_output / "summary_feature-importances.xlsx")
    
    return summary_metrics, summary_importances
    
def RFPipeline(
        feats_src, 
        targets_src, 
        mode = "classification", 
        n_folds = 5, 
        n_imputations = 5, 
        n_permutations = 5, 
        categorical_vars = None, 
        config_params = None, 
        path_output = None,
        random_state = 42, 
        overwrite = False
    ):
    
    assert mode in ["classification", "regression"], (
        "Please provide a valid mode: 'classification', 'regression'\n"
        f"Mode provided: {mode}"
    )
    
    if not path_output is None:
        path_output.mkdir(exist_ok=True, parents=True)
        path_out = path_output / mode
        path_out.mkdir(exist_ok=True, parents=True)
    
    feats = feats_src.copy()
    targets = targets_src.copy()
    
    if n_folds > 1:
        if mode == "classification":
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        elif mode == "regression":
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            
        splits = [x for x in kf.split(feats, targets)]
        
    else:
        idx = np.arange(len(feats_src))
        splits = [(idx, idx)]
    
    performance_metrics = {}
    feature_importances = {}
    models = {}
    
    print(f"Starting {n_folds}-fold crossvalidation | Mode: {mode} | Imputations: {n_imputations}")
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        
        fold_metrics = {}
        fold_importances = {}
        fold_models = {}
        
        x_train_raw = feats.iloc[train_idx]
        x_test_raw = feats.iloc[test_idx]
        
        y_train = targets.iloc[train_idx]
        y_test = targets.iloc[test_idx]
        
        if not path_output is None:
            fold_dir = path_out / f"fold-{fold}"
            fold_dir.mkdir(exist_ok=True, parents=True)
            
            x_train_raw.to_excel(fold_dir / "train_features_raw.xlsx")
            x_test_raw.to_excel(fold_dir / "test_features_raw.xlsx")
            y_train.to_excel(fold_dir / "train_targets.xlsx")
            y_test.to_excel(fold_dir / "test_targets.xlsx")
        
        x_train_imputed_list, x_test_imputed_list, imputers = _impute_missing(
            x_train_raw, 
            x_test_raw, 
            categorical_vars, 
            M = n_imputations, 
            random_state = random_state
        )
        
        for m in tqdm(range(n_imputations), desc=f"  Training models (fold {fold})"):
            
            # init x_train, x_test
            x_train = x_train_imputed_list[m]
            x_test = x_test_imputed_list[m]
            
            if not path_output is None:
                m_dir = fold_dir / f"m-{m}"
                m_dir.mkdir(exist_ok=True, parents=True)
                
                x_train.to_excel(m_dir / "train_features_imputed.xlsx")
                x_test.to_excel(m_dir / "test_features_imputed.xlsx")
                with open(m_dir / "imputer.pkl", "wb") as f:
                    pkl.dump(imputers[m], f)
            
            # init model: func initModel(config, mode)
            model = initModel(mode, config_params=config_params)
            
            # fit model
            model.fit(x_train, y_train)
            
            if n_folds == 1:
                return model, imputers[m]
            
            fold_models[f"m-{m}"] = model
            
            # get performance metrics
            metrics = _performance_metrics(model, x_test, y_test, mode)
            fold_metrics[f"m-{m}"] = metrics.copy()
            
            fi = permutation_importance(
                model, x_test, y_test, n_repeats=n_permutations
            )
            
            # get feature importances
            feature_importance = pd.DataFrame(
                fi["importances_mean"], 
                index=x_train.columns, 
                columns=["FI-Score"]
            )
            feature_importance.index.name = "Feature"
            
            fold_importances[f"m-{m}"] = feature_importance.copy()
        
        performance_metrics[f"fold-{fold}"] = fold_metrics.copy()
        models[f"fold-{fold}"] = fold_models.copy()
        feature_importances[f"fold-{fold}"] = fold_importances.copy()
        
    if not path_output is None:
        saveOutputs(
            path_out, performance_metrics, models, feature_importances,
            n_folds, n_imputations
        )
    
    # Generate Plots
    
    return performance_metrics, models, feature_importances



# Testing
if __name__ == "__main__":
    
    n_rows = 100
    n_cols = 10
    mode = "classification"
    
    path_output = None # path to output directory
    
    columns = [f"num{i}" for i in range (n_cols)]
    
    # generate random metric features
    feats = pd.DataFrame(np.random.rand(n_rows, n_cols), columns=columns)
    
    if mode == "classification":
        labels = pd.Series(np.random.randint(0, 2, size=(n_rows, )))
    else:
        labels = pd.Series(np.random.rand(n_rows))
    
    # generate random categorical features
    feats["cat1"] = np.random.randint(0,2, size=(n_rows, )).astype(str)
    feats["cat2"] = np.random.randint(0,4, size=(n_rows, )).astype(str)
    feats["cat3"] = np.random.randint(0,4, size=(n_rows, )).astype(str)
    feats["cat4"] = np.random.randint(0,3, size=(n_rows, )).astype(str)
    
    # generate some missingness
    mask = np.random.randint(0, 10, size=feats.shape) > 8
    feats[mask] = np.nan
    
    # run pipeline
    model = RFPipeline(
        feats, 
        labels, 
        n_folds=1, 
        n_imputations=1, 
        path_output=path_output, 
        categorical_vars = ["cat1", "cat2", "cat3", "cat4"], 
        mode=mode
    )




