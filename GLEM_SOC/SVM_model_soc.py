# ============================================================
# Support Vector Machine (SVM) regression model
# Purpose: feature selection, hyperparameter tuning, and SOC prediction
# ============================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Feature and target configuration
# ------------------------------------------------------------
feature_columns = ['precipitation', 'temperature','dem', 'slope', 'aspect', 'relief','rough', 'NDVI', 'CONTAG','LPI_L','PRD', 'COHESION_C']
target_column = 'SOC'

# ------------------------------------------------------------
# Repeated train/test experiments
# ------------------------------------------------------------
for j in range(1, 21):
    # Load the predefined train/test split for the current repetition.
    train_data=pd.read_excel(f'E:/data/train_data{j}.xlsx')
    test_data=pd.read_excel(f'E:/data/test_data{j}.xlsx')

    # Extract predictor matrices and the target variable.
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_true = test_data[target_column]

    # --------------------------------------------------------
    # Feature selection with Random Forest importance
    # Fit a random-forest regressor as the feature-importance estimator.
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    # Select features whose importance exceeds the estimator's threshold.
    selector = SelectFromModel(rf_selector)
    selector.fit(X_train, y_train)

    # Recover the names of the retained environmental predictors.
    selected_features = selector.get_support()
    selected_feature_columns = [col for col, sel in zip(feature_columns, selected_features) if sel]
    print(selected_feature_columns)

    # Construct reduced predictor matrices using the selected features.
    X_train_selected = X_train[selected_feature_columns]
    X_test_selected = X_test[selected_feature_columns]

    # Standardize predictors using training-set statistics only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # --------------------------------------------------------
    # RBF-SVM model and hyperparameter search
    # --------------------------------------------------------
    svm_model = SVR(kernel='rbf')
    # Candidate values for the penalty parameter C and RBF kernel width.
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1]
    }
    # Select hyperparameters by five-fold cross-validation on the training set.
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']

    # Refit the final SVM using the best cross-validated parameters.
    final_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma)
    final_model.fit(X_train_scaled, y_train)

    # Generate predictions and append them to the original data tables.
    svm_predYtest = final_model.predict(X_test_scaled)
    test_data['pred'] = svm_predYtest
    train_data['pred'] = final_model.predict(X_train_scaled)

    # Export training/test predictions for downstream ensemble modeling.
    train_data.to_excel(f'E:/results/SVM_train_{j}.xlsx', index=False)
    test_data.to_excel(f'E:/results/SVM_test_{j}.xlsx', index=False)

    # --------------------------------------------------------
    # Test-set accuracy assessment
    # --------------------------------------------------------
    rmse_test = np.sqrt(mean_squared_error(y_true, test_data['pred']))
    mae_test = mean_absolute_error(y_true, test_data['pred'])
    r2_test = r2_score(y_true, test_data['pred'])
    # Calculate sample size and the effective parameter count used in the AIC-style criterion.
    # Calculate the number of samples
    n = len(test_data['SOC'])
    # Get the number of parameters
    p = len(selected_feature_columns)
    # Compute adjusted R² to account for model complexity.
    adj_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
    # Compute the AIC-style criterion used for cross-model comparison.
    aic_test = n * np.log(rmse_test ** 2) + 2 * p

    # Report test-set performance and selected-feature results.
    print(f"test_results_{j}：")
    print(f"R²: {r2_test:.3f}")
    print(f"adj_R²: {adj_r2:.3f}")
    print(f"MAE: {mae_test:.3f}")
    print(f"RMSE: {rmse_test:.3f}")
    print(f"AIC: {aic_test:.3f}")
    print("-" * 50)

