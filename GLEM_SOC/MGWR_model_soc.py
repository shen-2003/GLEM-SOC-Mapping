# ============================================================
# MGWR spatial regression model
# Purpose: fit MGWR and generate training/test predictions
# ============================================================
import pandas as pd
import numpy as np
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------
feature_columns = ['precipitation', 'temperature','dem', 'slope', 'relief']
target_column = 'SOC'

# ------------------------------------------------------------
# Repeated train/test experiments
# ------------------------------------------------------------
for j in range(1, 11):
    # Load the predefined train/test split for the current repetition.
    train_data = pd.read_excel(f'E:/data/train_data{j}.xlsx')
    test_data = pd.read_excel(f'E:/data/test_data{j}.xlsx')

    # Extract response, predictors, and spatial coordinates.
    y_train = train_data[target_column].values.reshape((-1,1))
    y_test = test_data[target_column].values.reshape((-1,1))

    X_train = train_data[feature_columns].values
    X_test = test_data[feature_columns].values

    coords_train = list(zip(train_data['x'], train_data['y']))
    coords_test = list(zip(test_data['x'], test_data['y']))

    # Standardize predictors using training-set statistics only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit MGWR bandwidths and the final model on the complete training set.
    selector = Sel_BW(coords_train, y_train, X_train_scaled, multi=True, constant=True)
    bw = selector.search()
    mgwr_model = MGWR(coords_train, y_train, X_train_scaled, selector).fit()

    # Extract location-specific intercepts and coefficients from MGWR.
    # Use the coefficients and intercept of the nearest neighbor point for prediction
    mgwr_params = mgwr_model.params
    intercepts = mgwr_params[:, 0]
    slopes = mgwr_params[:, 1:]

    # Predict each test sample using the coefficients of its nearest training location.
    # Test set prediction
    mgwr_prediction_test = []
    for i, coord in enumerate(coords_test):
        distances = np.linalg.norm(coord - np.array(coords_train), axis=1)
        nearest_index = np.argmin(distances)
        intercept = mgwr_model.params[nearest_index, 0]
        slope = mgwr_model.params[nearest_index, 1:]
        prediction = intercept + np.dot(X_test_scaled[i], slope)
        mgwr_prediction_test.append(prediction)
    mgwr_predictions_test = np.array(mgwr_prediction_test)

    # Generate fitted values for the training samples using local MGWR coefficients.
    # Training set prediction
    mgwr_predictions_train = []
    for i in range(len(coords_train)):
        intercept = mgwr_params[i, 0]
        slopes = mgwr_params[i, 1:]
        prediction = intercept + np.dot(X_train_scaled[i], slopes)
        mgwr_predictions_train.append(prediction)

    # Convert standardized predictions back to the original SOC scale.
    mgwr_predictions_train = np.array(mgwr_predictions_train) * y_train.std(axis=0) + y_train.mean(axis=0)

    # Store predictions in the corresponding data tables.
    train_data['pred'] = mgwr_predictions_train
    test_data['pred'] = mgwr_predictions_test

    # Export model predictions for subsequent ensemble modeling.
    train_data.to_excel(f'E:/results/MGWR_train_{j}.xlsx', index=False)
    test_data.to_excel(f'E:/results/MGWR_test_{j}.xlsx', index=False)

    # Evaluate predictive performance on the independent test set.
    y_test = test_data['SOC'].values.reshape((-1,1))
    r2_test = r2_score(y_test, test_data['pred'])
    mae_test = mean_absolute_error(y_test, test_data['pred'])
    rmse_test = np.sqrt(mean_squared_error(y_test, test_data['pred']))
    # Calculate sample size and the effective parameter count used by the AIC formula.
    # Calculate the number of samples
    n = len(test_data['SOC'])
    # Get the number of parameters
    p = 6
    # Compute the AIC-style criterion used for model comparison.
    aic_test = n * np.log(rmse_test**2) + 2 * p
    # Output results
    # Report test-set performance for the current repetition.
    print(f"test_results_{j}：")
    print(f"RMSE: {rmse_test:.3f}")
    print(f"MAE: {mae_test:.3f}")
    print(f"R²: {r2_test:.3f}")
    print(f"AIC: {aic_test:.3f}")
    print("-" * 50)
