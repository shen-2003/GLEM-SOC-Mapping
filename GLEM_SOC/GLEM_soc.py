# ============================================================
# GLEM: KAN-Transformer-based spatial ensemble weighting
# Purpose: generate adaptive weights for MGWR, MixedLM, and SVM
# ============================================================
import os
import optuna
import numpy as np
import pandas as pd
from torch.utils import data
import torch
import torch.nn as nn
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
# ============================================================
# Model overview and implementation notes
# ============================================================


# ------------------------------------------------------------
# Reproducibility configuration
# ------------------------------------------------------------
# ========= seed =========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# ------------------------------------------------------------
# Neural-network model definitions
# ------------------------------------------------------------
# ========= model struct =========
# KAN layer: learn nonlinear basis-function responses for each input.
class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        grid = torch.linspace(0, 1, grid_size).repeat(input_dim, output_dim, 1)
        self.grid = nn.Parameter(grid)

        coeff = torch.empty(input_dim, output_dim, grid_size)
        nn.init.xavier_uniform_(coeff, gain=nn.init.calculate_gain("relu"))
        self.coeff = nn.Parameter(coeff * 0.1)

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = x.unsqueeze(-1).unsqueeze(-1)
        distances = torch.abs(x - self.grid)
        activations = torch.relu(1 - distances * self.grid_size)
        output = torch.sum(self.coeff * activations, dim=-1)
        output = torch.sum(output, dim=1)
        return output


# Token-based KAN-Transformer weight generator.
class GenerateW_KAN_Transformer_Token(nn.Module):
    """
    Tokenized KAN-Transformer weight generator.

    Structure of input Weight:
        [D, Res_bias, Res_abs]

    D:
        Spatial distance vector, with dimension n_train.
    Res_bias:
        3-D neighborhood signed residuals, corresponding to MGWR / MixedLM / SVM, respectively.
    Res_abs:
        3-D neighborhood absolute residuals, corresponding to MGWR / MixedLM / SVM, respectively.

    Modeling logic:
        1. KAN_spatial learns the nonlinear response of weights to spatial distance;
        2. KAN_res_bias learns the nonlinear response of weights to local systematic bias;
        3. KAN_res_abs learns the nonlinear response of weights to local error magnitude;
        4. Transformer models the dependencies among spatial distance tokens, residual bias tokens, and residual error tokens;
        5. Output adaptive ensemble weights for the three base models MGWR / MixedLM / SVM.
    """

        # Store model dimensions and token configuration.
    def __init__(
        self,
        spatial_dim,
        out_features,
        residual_dim=3,
        d_model=64,
        nhead=4,
        num_transformer_layers=1,
        kan2_output_dim=32,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.residual_dim = residual_dim
        self.num_tokens = 3

        # Encode each feature group into a common Transformer embedding space.
        self.kan_spatial = KANLayer(spatial_dim, d_model)
        self.kan_res_bias = KANLayer(residual_dim, d_model)
        self.kan_res_abs = KANLayer(residual_dim, d_model)

        self.token_type_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, d_model)
        )
        nn.init.normal_(self.token_type_embedding, mean=0.0, std=0.02)

        # Transformer encoder: model dependencies among the three token types.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout_rate)

        # Decode the fused representation into ensemble weights.
        self.kan2 = KANLayer(d_model, kan2_output_dim)
        self.bn2 = nn.BatchNorm1d(kan2_output_dim)
        self.kan3 = KANLayer(kan2_output_dim, out_features)

        # Split the input into spatial, signed-residual, and absolute-residual groups.
    def forward(self, weight):
        # weight: [batch_size, spatial_dim + residual_dim * 2]
        d_spatial = weight[:, :self.spatial_dim]
        res_bias = weight[
            :, self.spatial_dim:self.spatial_dim + self.residual_dim
        ]
        res_abs = weight[
            :, self.spatial_dim + self.residual_dim:
            self.spatial_dim + 2 * self.residual_dim
        ]

        # Generate one token for each feature group.
        spatial_token = self.kan_spatial(d_spatial)
        bias_token = self.kan_res_bias(res_bias)
        abs_token = self.kan_res_abs(res_abs)

        # tokens: [batch_size, 3, d_model]
        tokens = torch.stack([spatial_token, bias_token, abs_token], dim=1)
        tokens = tokens + self.token_type_embedding

        # Contextualize the tokens through self-attention.
        tokens = self.transformer(tokens)

        # Sample-level representation: fusing spatial distance, residual bias, and residual error information.
        x = tokens.mean(dim=1)
        x = self.norm(x)
        x = self.drop(x)

        # Nonlinear projection from the fused token representation to final weights.
        x = self.kan2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.drop(x)

        weights = self.kan3(x)

        # If the three ensemble weights to be nonnegative and sum to 1, uncomment the next line.
        # weights = torch.softmax(weights, dim=1)
        return weights


    # Ensemble predictor: combine base-model predictions using learned weights.
class EnsembleModel(nn.Module):
    def __init__(self, spatial_dim, out_features, kan_transformer_kwargs):
        super().__init__()
        self.kan_transformer = GenerateW_KAN_Transformer_Token(
            spatial_dim=spatial_dim,
            out_features=out_features,
            **kan_transformer_kwargs,
        ).double()
        self.SAweights = None

        # Generate sample-specific ensemble weights and apply them to base predictions.
    def forward(self, weight, f_pred, betas):
        sa_weights = self.kan_transformer(weight)
        self.SAweights = sa_weights
        pred_y = torch.mm((f_pred * sa_weights), betas)
        return pred_y

        # Return the generated weights for later analysis or export.
    def output(self, weight):
        sa_weights = self.kan_transformer(weight)
        return weight, sa_weights


# ------------------------------------------------------------
# Data, normalization, and feature-construction utilities
# ------------------------------------------------------------
# ========= Tools =========
    # Wrap numpy arrays/tensors into a reproducible PyTorch DataLoader.
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(
        dataset,
        batch_size,
        shuffle=is_train,
        worker_init_fn=lambda worker_id: set_seed(42 + worker_id),
        generator=torch.Generator().manual_seed(42),
    )


    # Estimate regression coefficients using the Moore-Penrose pseudoinverse.
def generalized_least_squares(x, y):
    xt = x.transpose()
    xs = np.matmul(xt, x)
    xsi = np.linalg.pinv(xs)
    betas = np.matmul(np.matmul(xsi, xt), y)
    return betas


    # Row-wise min-max normalization to the [0, 1] interval.
def norm_01(matrix):
    matrix = matrix.astype(float)
    for i in range(len(matrix)):
        minx = np.min(matrix[i])
        maxx = np.max(matrix[i])
        if maxx - minx != 0:
            matrix[i] = (matrix[i] - minx) / (maxx - minx)
        else:
            matrix[i] = 0
    return matrix


    # Robust bandwidth based on the median of positive pairwise distances.
def median_bandwidth(distance_matrix, eps=1e-12):
    vals = distance_matrix[distance_matrix > 0]
    if len(vals) == 0:
        return 1.0
    return float(np.median(vals) + eps)


def build_spatial_kernel(distance_matrix, bandwidth=None, exclude_self=False, eps=1e-12):
    """
    Construct a spatial kernel based on the distance matrix to aggregate training residuals.

    Training set:
    distance_matrix shape = [n_train, n_train], exclude_self=True.
    Validation/test set:
    distance_matrix shape = [n_query, n_train], exclude_self=False.
    """
    if bandwidth is None:
        bandwidth = median_bandwidth(distance_matrix)

    kernel = np.exp(-distance_matrix / bandwidth)

    if exclude_self and kernel.shape[0] == kernel.shape[1]:
        np.fill_diagonal(kernel, 0)

    kernel = kernel / (kernel.sum(axis=1, keepdims=True) + eps)
    return kernel


    # Restrict numerical features to the range expected by the KAN grid.
def clip_01(array):
    return np.clip(array, 0.0, 1.0)


    # Construct neighborhood residual descriptors from training samples.
def build_residual_features(
    y_train,
    f_train,
    d_train,
    d_query,
    scaler=None,
    fit_scaler=False,
):
    """
    Construct neighborhood residual features for query samples using training set residuals.

    y_train:
        True values of the training subset, shape = [n_train, 1]
    f_train:
        Predictions of the three base models on the training subset, shape = [n_train, 3]
    d_train:
        Distance matrix from the training subset to the training subset, shape = [n_train, n_train]
    d_query:
        Distance matrix from query samples to the training subset.
        If it is the training set itself, then d_query=d_train, and exclude self-residuals.
        If it is a validation set or test set, do not exclude self-residuals.
    """
    # Residual sign indicates systematic over-/under-prediction; magnitude measures error size.
    residual_signed_train = y_train - f_train
    residual_abs_train = np.abs(residual_signed_train)

    # Use the same bandwidth for train, validation, and test queries.
    bandwidth = median_bandwidth(d_train)

    # Detect whether the query set is the training set itself.
    is_train_query = (
        d_query.shape[0] == d_train.shape[0]
        and d_query.shape[1] == d_train.shape[1]
        and np.allclose(d_query, d_train)
    )

    kernel_query = build_spatial_kernel(
        d_query,
        bandwidth=bandwidth,
        exclude_self=is_train_query,
    )

    # Aggregate training residuals with the spatial kernel.
    res_bias_feature = kernel_query @ residual_signed_train
    res_abs_feature = kernel_query @ residual_abs_train

    # Concatenate signed and absolute neighborhood residual features.
    residual_feature = np.hstack([res_bias_feature, res_abs_feature])

    if fit_scaler:
        scaler = MinMaxScaler()
        residual_feature = scaler.fit_transform(residual_feature)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False.")
        residual_feature = scaler.transform(residual_feature)

    # The KANLayer grid is defined on [0, 1], so residual features are constrained to [0, 1].
    residual_feature = clip_01(residual_feature)
    return residual_feature, scaler


    # Read the first non-empty column from an Excel feature list.
def read_features_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df.iloc[:, 0].dropna().tolist()


# ------------------------------------------------------------
# KAN-Transformer training and evaluation
# ------------------------------------------------------------
def train_model(
    model,
    train_iter,
    optimizer,
    loss_fn,
    w_val,
    f_val,
    y_val,
    f_betas,
    model_path,
    epochs=200,
    patience=10,
):
    # Track the best validation loss and early-stopping status.
    best_epoch = -1
    no_optim = 0
    best_val_mse = np.inf

        # One optimization epoch over the training batches.
    for epoch in range(1, epochs + 1):
        model.train()
        train_epoch_loss = 0.0
        train_num = 0

            # Apply scheduled learning-rate decay every 20 epochs.
        if epoch % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.9

            # Forward pass, loss computation, and gradient update.
        for weight, f_pred, y_true in train_iter:
            optimizer.zero_grad()
            train_preds = model(weight, f_pred, f_betas)
            mse_loss = loss_fn(train_preds, y_true)
            mse_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += mse_loss.item() * f_pred.size(0)
            train_num += f_pred.size(0)

        train_epoch_loss = train_epoch_loss / max(train_num, 1)

        # Evaluate on the fixed validation set without gradient tracking.
        model.eval()
        with torch.no_grad():
            val_preds = model(w_val, f_val, f_betas).detach().cpu().numpy()
            val_mse = mean_squared_error(y_val.detach().cpu().numpy(), val_preds)

        # Save the best model according to validation MSE.
        if val_mse >= best_val_mse:
            no_optim += 1
        else:
            no_optim = 0
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(model, model_path)

        if no_optim > patience:
            break

    return best_val_mse, best_epoch


    # Load the selected model and calculate independent test-set metrics.
def model_test(model_path, w_test, f_test, y_test, f_betas):
    model = torch.load(model_path, weights_only=False)
    model.eval()

    with torch.no_grad():
        test_preds = model(w_test, f_test, f_betas).detach().cpu().numpy()

    y_test_np = y_test.detach().cpu().numpy()
    r2_test = r2_score(y_test_np, test_preds)
    mae_test = mean_absolute_error(y_test_np, test_preds)
    rmse_test = np.sqrt(mean_squared_error(y_test_np, test_preds))

    # AIC uses the predefined effective parameter count for the ensemble.
    p = 7
    n = len(y_test_np)
    aic_test = n * np.log(rmse_test**2) + 2 * p
    adj_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)

    return r2_test, adj_r2_test, mae_test, rmse_test, aic_test, test_preds


# ============================================================
# Main experiment loop
# Each iteration uses one predefined train/validation/test split.
# ============================================================
# ========= 主循环 =========
for j in range(1, 11):
    os.makedirs("E:/GLEM_results/weights", exist_ok=True)
    os.makedirs("E:/GLEM_results", exist_ok=True)

    mgwr_traindata = pd.read_excel(f"E:/results/MGWR_train_{j}.xlsx")
    mgwr_testdata = pd.read_excel(f"E:/results/MGWR_test_{j}.xlsx")
    llm_traindata = pd.read_excel(f"E:/results/Mixedlm_train_{j}.xlsx")
    llm_testdata = pd.read_excel(f"E:/results/Mixedlm_test_{j}.xlsx")
    svm_traindata = pd.read_excel(f"E:/results/SVM_train_{j}.xlsx")
    svm_testdata = pd.read_excel(f"E:/results/SVM_test_{j}.xlsx")


    # Predictor variables used by the ensemble base models.
    feature_columns = ['precipitation', 'temperature', 'dem', 'slope', 'relief', 'NDVI', 'COHESION_C']

    # Load environmental predictors for the training and test sets.
    x_all = mgwr_traindata[feature_columns].values
    x_test = mgwr_testdata[feature_columns].values

    # Load the response variable and preserve the column-vector shape.
    y_all = llm_traindata["SOC"].values.reshape((-1, 1))
    y_test = llm_testdata["SOC"].values.reshape((-1, 1))

    # Load spatial coordinates used to construct distance matrices.
    train_position_all = svm_traindata[["x", "y"]].values
    test_position = svm_testdata[["x", "y"]].values

    # Collect predictions from the three base learners.
    mgwr_pred_train_all = mgwr_traindata["pred"].values.reshape((-1, 1))
    mgwr_pred_test = mgwr_testdata["pred"].values.reshape((-1, 1))
    llm_pred_train_all = llm_traindata["pred"].values.reshape((-1, 1))
    llm_pred_test = llm_testdata["pred"].values.reshape((-1, 1))
    svm_pred_train_all = svm_traindata["pred"].values.reshape((-1, 1))
    svm_pred_test = svm_testdata["pred"].values.reshape((-1, 1))

    # Stack base-model predictions as columns: MGWR, MixedLM, SVM.
    f_all = np.hstack([mgwr_pred_train_all, llm_pred_train_all, svm_pred_train_all])
    f_test = np.hstack([mgwr_pred_test, llm_pred_test, svm_pred_test])

    # Recover the predefined validation indices and derive the training subset.
    index_all = np.arange(len(x_all))
    index_validation = np.loadtxt(
        f"E:/data/valindices_{j}.csv"
    ).tolist()
    index_validation = np.array([int(i) for i in index_validation], dtype=int)
    index_train = np.array(
        sorted(list(set(index_all.tolist()) - set(index_validation.tolist()))),
        dtype=int,
    )

    # Partition predictors and response according to the fixed split.
    x_train = x_all[index_train]
    x_val = x_all[index_validation]

    y_train = y_all[index_train]
    y_val = y_all[index_validation]

    # Partition base-model predictions using the same indices.
    f_train = f_all[index_train]
    f_val = f_all[index_validation]

    # Partition spatial coordinates for neighborhood construction.
    train_position = train_position_all[index_train]
    val_position = train_position_all[index_validation]

    # Estimate global ensemble coefficients from training data only.
    f_betas_np = generalized_least_squares(f_train, y_train)
    x_betas_np = generalized_least_squares(x_train, y_train)

    f_betas = torch.from_numpy(f_betas_np).double()
    x_betas = torch.from_numpy(x_betas_np).double()

    # Build distances from each query sample to the training samples.
    # ===== Distance matrix =====
    d_train = np.zeros([len(x_train), len(x_train)])
    d_val = np.zeros([len(x_val), len(x_train)])
    d_test = np.zeros([len(x_test), len(x_train)])

    for i in range(len(x_train)):
        d_train[i] = cdist([train_position[i]], train_position)

    for i in range(len(x_val)):
        d_val[i] = cdist([val_position[i]], train_position)

    for i in range(len(x_test)):
        d_test[i] = cdist([test_position[i]], train_position)

    # Normalize each query-to-training distance vector to [0, 1].
    d_train = norm_01(d_train)
    d_val = norm_01(d_val)
    d_test = norm_01(d_test)

    # Build neighborhood residual features without using validation/test responses.
    # ===== Neighborhood residual features =====
    residual_train, residual_scaler = build_residual_features(
        y_train=y_train,
        f_train=f_train,
        d_train=d_train,
        d_query=d_train,
        scaler=None,
        fit_scaler=True,
    )

    residual_val, _ = build_residual_features(
        y_train=y_train,
        f_train=f_train,
        d_train=d_train,
        d_query=d_val,
        scaler=residual_scaler,
        fit_scaler=False,
    )

    residual_test, _ = build_residual_features(
        y_train=y_train,
        f_train=f_train,
        d_train=d_train,
        d_query=d_test,
        scaler=residual_scaler,
        fit_scaler=False,
    )

    # Number of residual channels equals the three base models.
    residual_dim = 3
    spatial_dim = d_train.shape[1]

    # Concatenate spatial distances and the two residual-feature groups.
    # Weight = [spatial distance D, neighborhood signed residual Res_bias, neighborhood absolute residual Res_abs]
    w_train = np.hstack([
        d_train,
        residual_train[:, :residual_dim],
        residual_train[:, residual_dim:],
    ])
    w_val = np.hstack([
        d_val,
        residual_val[:, :residual_dim],
        residual_val[:, residual_dim:],
    ])
    w_test = np.hstack([
        d_test,
        residual_test[:, :residual_dim],
        residual_test[:, residual_dim:],
    ])

    # Convert feature matrices to double-precision tensors for PyTorch.
    w_train_t = torch.from_numpy(w_train).double()
    w_val_t = torch.from_numpy(w_val).double()
    w_test_t = torch.from_numpy(w_test).double()

    f_train_t = torch.from_numpy(f_train).double()
    f_val_t = torch.from_numpy(f_val).double()
    f_test_t = torch.from_numpy(f_test).double()

    y_train_t = torch.from_numpy(y_train).double()
    y_val_t = torch.from_numpy(y_val).double()
    y_test_t = torch.from_numpy(y_test).double()

    # Define the model identifier and checkpoint path.
    name = f"model_SOC_{j}"
    final_model_path = f"E:/GLEM_results/weights/{name}.th"

    # --------------------------------------------------------
    # Hyperparameter optimization with Optuna
    # --------------------------------------------------------
    # ========= Optuna Bayesian optimization objective function =========
    # Objective: minimize validation set MSE
        # Search the Transformer embedding dimension.
    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])

        # Restrict attention-head choices to divisors of d_model.
        nhead_candidates = [1, 2, 4, 8]
        transformer_nhead = trial.suggest_categorical(
            "transformer_nhead",
            [h for h in nhead_candidates if d_model % h == 0],
        )

        # Search the number of Transformer encoder layers.
        num_transformer_layers = trial.suggest_categorical(
            "num_transformer_layers",
            [1, 2],
        )

        # Search the hidden dimension of the second KAN layer.
        kan2_output_dim = trial.suggest_categorical(
            "kan2_output_dim",
            [16, 32, 64, 128],
        )

        # Search dropout regularization strength.
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.2)

        kan_transformer_kwargs = {
            "residual_dim": residual_dim,
            "d_model": d_model,
            "nhead": transformer_nhead,
            "num_transformer_layers": num_transformer_layers,
            "kan2_output_dim": kan2_output_dim,
            "dropout_rate": dropout_rate,
        }

        # Instantiate a trial-specific ensemble model.
        model = EnsembleModel(
            spatial_dim=spatial_dim,
            out_features=f_train_t.shape[1],
            kan_transformer_kwargs=kan_transformer_kwargs,
        ).double()

        # Adam optimizer and mean-squared-error objective.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,
            weight_decay=0,
        )
        loss_fn = nn.MSELoss()
        train_iter = load_array((w_train_t, f_train_t, y_train_t), 64)

        trial_model_path = (
            f"E:/GLEM_results/weights/{name}_trial_{trial.number}.th"
        )

        # Train the trial model using validation MSE for model selection.
        best_val_mse, _ = train_model(
            model=model,
            train_iter=train_iter,
            optimizer=optimizer,
            loss_fn=loss_fn,
            w_val=w_val_t,
            f_val=f_val_t,
            y_val=y_val_t,
            f_betas=f_betas,
            model_path=trial_model_path,
            epochs=200,
            patience=10,
        )

        return best_val_mse

    # Minimize validation MSE; the test set is not used during optimization.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    # Execute the Bayesian optimization study.
    study.optimize(objective, n_trials=30)

    # Report the best hyperparameter configuration.
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (best validation MSE): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --------------------------------------------------------
    # Train the final model with the selected hyperparameters
    # --------------------------------------------------------
    # ========= Train the final model using the best hyperparameters =========
    best_params = study.best_params
    best_kwargs = {
        "residual_dim": residual_dim,
        "d_model": best_params["d_model"],
        "nhead": best_params["transformer_nhead"],
        "num_transformer_layers": best_params["num_transformer_layers"],
        "kan2_output_dim": best_params["kan2_output_dim"],
        "dropout_rate": best_params["dropout_rate"],
    }

    # Instantiate the final model using Optuna's best configuration.
    best_model = EnsembleModel(
        spatial_dim=spatial_dim,
        out_features=f_train_t.shape[1],
        kan_transformer_kwargs=best_kwargs,
    ).double()

    optimizer = torch.optim.Adam(
        best_model.parameters(),
        lr=0.0001,
        weight_decay=0,
    )
    loss_fn = nn.MSELoss()
    train_iter = load_array((w_train_t, f_train_t, y_train_t), 64)

    # Train and checkpoint the final model based on validation performance.
    best_val_mse, best_epoch = train_model(
        model=best_model,
        train_iter=train_iter,
        optimizer=optimizer,
        loss_fn=loss_fn,
        w_val=w_val_t,
        f_val=f_val_t,
        y_val=y_val_t,
        f_betas=f_betas,
        model_path=final_model_path,
        epochs=200,
        patience=10,
    )

    print(f"Final training best validation MSE: {best_val_mse:.6f}")
    print(f"Final training best epoch: {best_epoch}")

    # Independent test-set evaluation after model selection is complete.
    # ========= Test set evaluation =========
    r2_test, adj_r2_test, mae_test, rmse_test, aic_test, test_preds = model_test(
        model_path=final_model_path,
        w_test=w_test_t,
        f_test=f_test_t,
        y_test=y_test_t,
        f_betas=f_betas,
    )

    print(f"test_results_{j}：")
    print(f"R²: {r2_test:.3f}")
    print(f"adj_R²: {adj_r2_test:.3f}")
    print(f"MAE: {mae_test:.3f}")
    print(f"RMSE: {rmse_test:.3f}")
    print(f"AIC: {aic_test:.3f}")

    # --------------------------------------------------------
    # Export model coefficients, predictions, weights, and residual features
    # --------------------------------------------------------
    # ========= Save results =========
    pd.DataFrame(f_betas_np).to_excel(
        f"E:/GLEM_results/Fbetas_final_{j}.xlsx",
        index=False,
    )
    pd.DataFrame(x_betas_np).to_excel(
        f"E:/GLEM_results/Xbetas_final_{j}.xlsx",
        index=False,
    )
    pd.DataFrame(test_preds).to_excel(
        f"E:/GLEM_results/test_predictions_best_{j}.xlsx",
        index=False,
    )

    # Reload the best checkpoint to obtain final sample-level weights.
    loaded_best_model = torch.load(final_model_path, weights_only=False)
    loaded_best_model.eval()

    with torch.no_grad():
        _, sa_weights_test = loaded_best_model.output(w_test_t)
        _, sa_weights_train = loaded_best_model.output(w_train_t)
        _, sa_weights_val = loaded_best_model.output(w_val_t)

    # Export learned ensemble weights for downstream spatial interpretation.
    pd.DataFrame(sa_weights_test.detach().cpu().numpy()).to_excel(
        f"E:/GLEM_results/SAweights_test_{j}.xlsx",
        index=False,
    )
    pd.DataFrame(sa_weights_train.detach().cpu().numpy()).to_excel(
        f"E:/GLEM_results/SAweights_train_{j}.xlsx",
        index=False,
    )
    pd.DataFrame(sa_weights_val.detach().cpu().numpy()).to_excel(
        f"E:/GLEM_results/SAweights_val_{j}.xlsx",
        index=False,
    )

    # Column names correspond to the three base-model residual channels.
    residual_columns = [
        "MGWR_res_bias_neigh",
        "MixedLM_res_bias_neigh",
        "SVM_res_bias_neigh",
        "MGWR_abs_res_neigh",
        "MixedLM_abs_res_neigh",
        "SVM_abs_res_neigh",
    ]

    pd.DataFrame(residual_train, columns=residual_columns).to_excel(
        f"E:/GLEM_results/Residual_features_train_{j}.xlsx",
        index=False,
    )
    pd.DataFrame(residual_val, columns=residual_columns).to_excel(
        f"E:/GLEM_results/Residual_features_val_{j}.xlsx",
        index=False,
    )
    pd.DataFrame(residual_test, columns=residual_columns).to_excel(
        f"E:/GLEM_results/Residual_features_test_{j}.xlsx",
        index=False,
    )
    # Export the best hyperparameter configuration for reproducibility.
    pd.DataFrame([study.best_params]).to_excel(
        f"E:/GLEM_results/best_params_{j}.xlsx",
        index=False,
    )
