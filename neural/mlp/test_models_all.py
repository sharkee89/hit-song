import os
import copy
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Vraćeno na dataset bez CLAP kolona
DATA_PATH = (
        PROJECT_ROOT / "data" / "dataset" / "spotify_tracks_audiobox_librosa.parquet"
)

SEED = 42

# Number of Optuna trials
RF_TRIALS = 15
XGB_TRIALS = 20
MLP_TRIALS = 15
TRANSFORMER_TRIALS = 15

# Early stopping
EARLY_STOPPING_PATIENCE = 20
XGB_EARLY_STOPPING_ROUNDS = 50

# Maximum epochs for neural networks
MAX_EPOCHS = 200

# Statistical bootstrap
BOOTSTRAP_RESAMPLES = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # More reproducible CUDA behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(DATA_PATH)

print("Original dataset:", df.shape)


# ============================================================
# FEATURES
# ============================================================

base_features = [
    "PQ",
    "PC",
    "CE",
    "CU",
    "artist_popularity",
    "followers",
]

librosa_features = [
    c for c in df.columns
    if c.startswith("librosa_")
]

features = base_features + librosa_features

target = "popularity"


missing_features = [
    c for c in features + [target]
    if c not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing columns: {missing_features}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

data = df[features + [target]].copy()

data = data.dropna()

# Compress heavy-tailed follower distribution
data["followers"] = np.log1p(
    data["followers"].clip(lower=0)
)

X = data[features].astype(np.float32)
y = data[target].astype(np.float32)


print("Valid dataset:", X.shape)
print("Number of features:", X.shape[1])


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

# First: hold out completely untouched test set
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
)

# From remaining 80%, create 25% validation
# => 60% train / 20% validation / 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=SEED,
)


print()
print("TRAIN:", X_train.shape)
print("VALID:", X_val.shape)
print("TEST :", X_test.shape)


# ============================================================
# SCALING
# ============================================================

# IMPORTANT:
# scaler is fitted ONLY on training data.

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Tree models use original numeric features.
X_train_tree = X_train.values
X_val_tree = X_val.values
X_test_tree = X_test.values

y_train_np = y_train.values
y_val_np = y_val.values
y_test_np = y_test.values


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, prediction):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            prediction
        )
    )

    r2 = r2_score(
        y_true,
        prediction
    )

    return rmse, r2


# ============================================================
# 1. RANDOM FOREST
# ============================================================

def rf_objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            700,
            step=100
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            6,
            30
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10
        ),

        "max_features": trial.suggest_float(
            "max_features",
            0.5,
            1.0
        ),

        "random_state": SEED,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)

    model.fit(
        X_train_tree,
        y_train_np
    )

    pred = model.predict(
        X_val_tree
    )

    rmse, _ = calculate_metrics(
        y_val_np,
        pred
    )

    return rmse


print("\n==============================")
print("RANDOM FOREST SEARCH")
print("==============================")

rf_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=SEED
    )
)

rf_study.optimize(
    rf_objective,
    n_trials=RF_TRIALS
)

best_rf_params = rf_study.best_params

best_rf_params.update({
    "random_state": SEED,
    "n_jobs": -1,
})

print("Best RF parameters:")
print(best_rf_params)


rf_model = RandomForestRegressor(
    **best_rf_params
)

rf_model.fit(
    X_train_tree,
    y_train_np
)

rf_pred = rf_model.predict(
    X_test_tree
)


# ============================================================
# 2. XGBOOST
# ============================================================

def xgb_objective(trial):

    params = {
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            15
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            20.0,
            log=True
        ),

        # Large upper bound.
        # Early stopping determines the effective number
        # of boosting rounds.
        "n_estimators": 2000,

        "objective": "reg:squarederror",
        "eval_metric": "rmse",

        "random_state": SEED,
        "n_jobs": -1,

        "device": "cuda"
        if torch.cuda.is_available()
        else "cpu",
    }

    model = xgb.XGBRegressor(
        **params,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS
    )

    model.fit(
        X_train_tree,
        y_train_np,

        eval_set=[
            (
                X_val_tree,
                y_val_np
            )
        ],

        verbose=False
    )

    pred = model.predict(
        X_val_tree
    )

    rmse, _ = calculate_metrics(
        y_val_np,
        pred
    )

    return rmse


print("\n==============================")
print("XGBOOST SEARCH")
print("==============================")

xgb_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=SEED
    )
)

xgb_study.optimize(
    xgb_objective,
    n_trials=XGB_TRIALS
)

best_xgb_params = xgb_study.best_params

best_xgb_params.update({
    "n_estimators": 2000,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": SEED,
    "n_jobs": -1,
    "device": (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
})

print("Best XGBoost parameters:")
print(best_xgb_params)


xgb_model = xgb.XGBRegressor(
    **best_xgb_params,
    early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS
)

xgb_model.fit(
    X_train_tree,
    y_train_np,

    eval_set=[
        (
            X_val_tree,
            y_val_np
        )
    ],

    verbose=False
)

xgb_pred = xgb_model.predict(
    X_test_tree
)

print(
    "XGBoost best iteration:",
    xgb_model.best_iteration
)


# ============================================================
# PYTORCH DATA
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_np,
    dtype=torch.float32
).view(-1, 1)

y_val_tensor = torch.tensor(
    y_val_np,
    dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test_np,
    dtype=torch.float32
).view(-1, 1)


# ============================================================
# NEURAL NETWORK TRAINING WITH EARLY STOPPING
# ============================================================

def train_torch_model(
    model,
    learning_rate,
    weight_decay,
    batch_size,
    trial=None
):

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(xb)

            loss = criterion(
                prediction,
                yb
            )

            loss.backward()

            optimizer.step()


        # ----------------------------
        # validation
        # ----------------------------

        model.eval()

        with torch.no_grad():

            val_pred = model(
                X_val_tensor.to(DEVICE)
            ).cpu().numpy().reshape(-1)

        val_rmse, _ = calculate_metrics(
            y_val_np,
            val_pred
        )


        if trial is not None:

            trial.report(
                val_rmse,
                epoch
            )

            if (
                epoch >= 10
                and trial.should_prune()
            ):
                raise optuna.TrialPruned()


        # ----------------------------
        # early stopping
        # ----------------------------

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            patience_counter = 0

        else:

            patience_counter += 1

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):
                break


    model.load_state_dict(
        best_state
    )

    return model, best_val_rmse, best_epoch


# ============================================================
# 3. MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1,
        hidden2,
        hidden3,
        dropout1,
        dropout2
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden1
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout1
            ),

            nn.Linear(
                hidden1,
                hidden2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout2
            ),

            nn.Linear(
                hidden2,
                hidden3
            ),

            nn.ReLU(),

            nn.Linear(
                hidden3,
                1
            )
        )


    def forward(self, x):

        return self.network(x)


def mlp_objective(trial):

    params = {

        "hidden1": trial.suggest_categorical(
            "hidden1",
            [64, 128, 256]
        ),

        "hidden2": trial.suggest_categorical(
            "hidden2",
            [32, 64, 128]
        ),

        "hidden3": trial.suggest_categorical(
            "hidden3",
            [16, 32, 64]
        ),

        "dropout1": trial.suggest_float(
            "dropout1",
            0.0,
            0.4
        ),

        "dropout2": trial.suggest_float(
            "dropout2",
            0.0,
            0.3
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            3e-3,
            log=True
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-7,
            1e-2,
            log=True
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            [64, 128, 256]
        )
    }


    model = MLP(
        input_dim=X_train.shape[1],
        hidden1=params["hidden1"],
        hidden2=params["hidden2"],
        hidden3=params["hidden3"],
        dropout1=params["dropout1"],
        dropout2=params["dropout2"]
    )


    _, best_rmse, _ = train_torch_model(
        model=model,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        trial=trial
    )

    return best_rmse


print("\n==============================")
print("MLP SEARCH")
print("==============================")


mlp_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=SEED
    ),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5
    )
)

mlp_study.optimize(
    mlp_objective,
    n_trials=MLP_TRIALS
)

best_mlp_params = mlp_study.best_params

print("Best MLP parameters:")
print(best_mlp_params)


mlp_model = MLP(
    input_dim=X_train.shape[1],
    hidden1=best_mlp_params["hidden1"],
    hidden2=best_mlp_params["hidden2"],
    hidden3=best_mlp_params["hidden3"],
    dropout1=best_mlp_params["dropout1"],
    dropout2=best_mlp_params["dropout2"]
)


mlp_model, _, mlp_best_epoch = train_torch_model(
    model=mlp_model,
    learning_rate=best_mlp_params["learning_rate"],
    weight_decay=best_mlp_params["weight_decay"],
    batch_size=best_mlp_params["batch_size"]
)


mlp_model.eval()

with torch.no_grad():

    mlp_pred = (
        mlp_model(
            X_test_tensor.to(DEVICE)
        )
        .cpu()
        .numpy()
        .reshape(-1)
    )


print(
    "MLP best epoch:",
    mlp_best_epoch
)


# ============================================================
# 4. REAL TABULAR TRANSFORMER
# ============================================================

class TabularTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model,
        n_heads,
        n_layers,
        dropout
    ):

        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model


        # Each scalar feature becomes a separate token.
        self.feature_embedding = nn.Linear(
            1,
            d_model
        )


        # Learnable feature-position embeddings
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                input_dim,
                d_model
            )
        )


        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )


        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )
        )


        self.head = nn.Sequential(

            nn.LayerNorm(
                d_model
            ),

            nn.Linear(
                d_model,
                64
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(self, x):

        # x:
        # (batch, features)

        tokens = x.unsqueeze(-1)

        # (batch, features, 1)
        tokens = self.feature_embedding(
            tokens
        )

        # (batch, features, d_model)
        tokens = (
            tokens
            + self.position_embedding
        )

        tokens = self.encoder(
            tokens
        )

        # Pool across feature tokens
        pooled = tokens.mean(
            dim=1
        )

        return self.head(
            pooled
        )


def transformer_objective(trial):

    d_model = trial.suggest_categorical(
        "d_model",
        [32, 64, 128]
    )


    possible_heads = [
        h for h in [2, 4, 8]
        if d_model % h == 0
    ]


    n_heads = trial.suggest_categorical(
        "n_heads",
        possible_heads
    )


    params = {

        "d_model": d_model,

        "n_heads": n_heads,

        "n_layers": trial.suggest_int(
            "n_layers",
            1,
            4
        ),

        "dropout": trial.suggest_float(
            "dropout",
            0.0,
            0.3
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            3e-3,
            log=True
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-7,
            1e-2,
            log=True
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            [64, 128, 256]
        )
    }


    model = TabularTransformer(
        input_dim=X_train.shape[1],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        dropout=params["dropout"]
    )


    _, best_rmse, _ = train_torch_model(
        model=model,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        trial=trial
    )

    return best_rmse


print("\n==============================")
print("TABULAR TRANSFORMER SEARCH")
print("==============================")


transformer_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=SEED
    ),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5
    )
)


transformer_study.optimize(
    transformer_objective,
    n_trials=TRANSFORMER_TRIALS
)


best_transformer_params = (
    transformer_study.best_params
)

print(
    "Best Transformer parameters:"
)

print(
    best_transformer_params
)


transformer_model = TabularTransformer(
    input_dim=X_train.shape[1],

    d_model=best_transformer_params[
        "d_model"
    ],

    n_heads=best_transformer_params[
        "n_heads"
    ],

    n_layers=best_transformer_params[
        "n_layers"
    ],

    dropout=best_transformer_params[
        "dropout"
    ]
)


transformer_model, _, transformer_best_epoch = (
    train_torch_model(
        model=transformer_model,

        learning_rate=best_transformer_params[
            "learning_rate"
        ],

        weight_decay=best_transformer_params[
            "weight_decay"
        ],

        batch_size=best_transformer_params[
            "batch_size"
        ]
    )
)


transformer_model.eval()

with torch.no_grad():

    transformer_pred = (
        transformer_model(
            X_test_tensor.to(DEVICE)
        )
        .cpu()
        .numpy()
        .reshape(-1)
    )


print(
    "Transformer best epoch:",
    transformer_best_epoch
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

predictions = {

    "Random Forest": rf_pred,

    "XGBoost": xgb_pred,

    "MLP": mlp_pred,

    "Tabular Transformer": transformer_pred
}


results = []


for name, pred in predictions.items():

    rmse, r2 = calculate_metrics(
        y_test_np,
        pred
    )

    results.append({
        "Model": name,
        "RMSE": rmse,
        "R2": r2
    })


results_df = (
    pd.DataFrame(results)
    .sort_values("RMSE")
    .reset_index(drop=True)
)


print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# STATISTICAL SIGNIFICANCE
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty_like(
        p_values
    )

    running_max = 0.0

    m = len(p_values)

    for rank, idx in enumerate(order):

        value = (
            m - rank
        ) * p_values[idx]

        running_max = max(
            running_max,
            value
        )

        adjusted[idx] = min(
            running_max,
            1.0
        )

    return adjusted


# ------------------------------------------------------------
# Pairwise Wilcoxon test
#
# We compare absolute errors for the same songs.
# This is a paired test because every model predicts
# exactly the same test observations.
# ------------------------------------------------------------

pairwise_results = []

model_names = list(
    predictions.keys()
)

raw_p_values = []


for i in range(
    len(model_names)
):

    for j in range(
        i + 1,
        len(model_names)
    ):

        model_a = model_names[i]
        model_b = model_names[j]

        errors_a = np.abs(
            y_test_np
            - predictions[model_a]
        )

        errors_b = np.abs(
            y_test_np
            - predictions[model_b]
        )


        test_result = wilcoxon(
            errors_a,
            errors_b,
            zero_method="pratt",
            method="asymptotic",
            alternative="two-sided"
        )


        rmse_a = np.sqrt(
            np.mean(
                errors_a ** 2
            )
        )

        rmse_b = np.sqrt(
            np.mean(
                errors_b ** 2
            )
        )


        pairwise_results.append({
            "Model A": model_a,
            "Model B": model_b,
            "RMSE A": rmse_a,
            "RMSE B": rmse_b,
            "RMSE difference A-B":
                rmse_a - rmse_b,
            "Wilcoxon statistic":
                test_result.statistic,
            "p-value":
                test_result.pvalue
        })

        raw_p_values.append(
            test_result.pvalue
        )


# Holm correction for multiple comparisons
adjusted_p = holm_correction(
    raw_p_values
)


for row, p_adj in zip(
    pairwise_results,
    adjusted_p
):

    row[
        "Holm-adjusted p-value"
    ] = p_adj

    row[
        "Significant (alpha=0.05)"
    ] = p_adj < 0.05


significance_df = pd.DataFrame(
    pairwise_results
)


print("\n==============================")
print("STATISTICAL SIGNIFICANCE")
print("==============================")

print(
    significance_df.to_string(
        index=False
    )
)


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# FOR RMSE DIFFERENCE
# ============================================================

def bootstrap_rmse_difference(
    y_true,
    pred_a,
    pred_b,
    n_boot=1000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    n = len(y_true)

    differences = np.empty(
        n_boot
    )

    for i in range(
        n_boot
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_sample = y_true[
            indices
        ]

        a_sample = pred_a[
            indices
        ]

        b_sample = pred_b[
            indices
        ]


        rmse_a = np.sqrt(
            np.mean(
                (
                    y_sample
                    - a_sample
                ) ** 2
            )
        )

        rmse_b = np.sqrt(
            np.mean(
                (
                    y_sample
                    - b_sample
                ) ** 2
            )
        )


        differences[i] = (
            rmse_a - rmse_b
        )


    lower = np.percentile(
        differences,
        2.5
    )

    upper = np.percentile(
        differences,
        97.5
    )

    return lower, upper


print("\n==============================")
print("RMSE BOOTSTRAP CONFIDENCE INTERVALS")
print("==============================")


best_model_name = results_df.iloc[0][
    "Model"
]

best_prediction = predictions[
    best_model_name
]


for model_name in model_names:

    if model_name == best_model_name:
        continue


    lower, upper = (
        bootstrap_rmse_difference(
            y_test_np,
            predictions[model_name],
            best_prediction,
            n_boot=BOOTSTRAP_RESAMPLES,
            seed=SEED
        )
    )


    difference = (
        np.sqrt(
            np.mean(
                (
                    y_test_np
                    - predictions[model_name]
                ) ** 2
            )
        )
        -
        np.sqrt(
            np.mean(
                (
                    y_test_np
                    - best_prediction
                ) ** 2
            )
        )
    )


    print(
        f"{model_name} vs {best_model_name}: "
        f"RMSE diff={difference:.4f}, "
        f"95% CI=[{lower:.4f}, {upper:.4f}]"
    )


print("\nBest model on untouched test set:")
print(best_model_name)