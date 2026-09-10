# ============================================================
# Mixed-effects linear model (MixedLM)
# Purpose: model SOC with fixed environmental effects and
#          land-use-specific random intercepts
# ============================================================
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pykrige import OrdinaryKriging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
from scipy import stats   # ⭐ 用于 p 值

# ------------------------------------------------------------
# Warning management
# =========================
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=HessianInversionWarning)

# ------------------------------------------------------------
# Model specification
# ------------------------------------------------------------
# =========================
# define model formula
# =========================
# Alternative full formula retained for reference; not used in the current run.
formula = 'SOC ~ precipitation + dem + temperature + NDVI + COHESION_C'
# ------------------------------------------------------------
# Repeated train/test experiments
# ------------------------------------------------------------
for j in range(1, 11):
        # Load the predefined train/test split for the current repetition.
    try:
        print(f"\n================ number_{j} ================\n")
        train_df = pd.read_excel(f'E:/data/train_data{j}.xlsx')
        test_df  = pd.read_excel(f'E:/data/test_data{j}.xlsx')

        # Treat land-use type as a categorical grouping variable for random effects.
        train_df['landuse_type'] = train_df['landuse_type'].astype('category')
        test_df['landuse_type']  = test_df['landuse_type'].astype('category')

        # --------------------------------------------------------
        # 1. Standardize the response variable
        # --------------------------------------------------------
        scaler = StandardScaler()
        train_df['SOC'] = scaler.fit_transform(train_df[['SOC']])
        test_df['SOC']  = scaler.transform(test_df[['SOC']])

        # --------------------------------------------------------
        # 2. Standardize predictor variables
        # --------------------------------------------------------
        # Environmental predictors used by the MixedLM.
        features = ['precipitation', 'dem', 'temperature', 'NDVI', 'COHESION_C']
        X_scaler = StandardScaler()
        train_df[features] = X_scaler.fit_transform(train_df[features])
        test_df[features]  = X_scaler.transform(test_df[features])

        # --------------------------------------------------------
        # 3. Fit the mixed-effects model
        # --------------------------------------------------------
        # Fixed effects are defined by 'formula'; land-use type supplies the random intercept.
        model = smf.mixedlm(
            formula=formula,
            data=train_df,
            groups=train_df['landuse_type'],
            re_formula='1'
        )

        result = model.fit()

        # --------------------------------------------------------
        # 4. Predict and restore the original SOC scale
        # --------------------------------------------------------
        # Predictions are produced in standardized SOC units.
        y_train_pred = result.predict(train_df)
        y_test_pred  = result.predict(test_df)

        # Recover the training-set mean and standard deviation for inverse transformation.
        y_train_mean = scaler.mean_[0]
        y_train_std  = np.sqrt(scaler.var_[0])

        # Add predictions and convert observed SOC back to the original scale.
        train_df['pred'] = y_train_pred * y_train_std + y_train_mean
        test_df['pred']  = y_test_pred  * y_train_std + y_train_mean

        train_df['SOC'] = scaler.inverse_transform(train_df[['SOC']])
        test_df['SOC']  = scaler.inverse_transform(test_df[['SOC']])

        # Export model predictions for subsequent ensemble modeling.
        train_df.to_excel(f'E:/results/Mixedlm_train_{j}.xlsx', index=False)
        test_df.to_excel(f'E:/results/Mixedlm_test_{j}.xlsx', index=False)
        # --------------------------------------------------------
        # 5. Evaluate test-set predictive performance
        # --------------------------------------------------------
        r2_test  = r2_score(test_df['SOC'], test_df['pred'])
        mae_test = mean_absolute_error(test_df['SOC'], test_df['pred'])
        rmse_test = np.sqrt(mean_squared_error(test_df['SOC'], test_df['pred']))

        # Calculate sample size and the number of fitted model parameters.
        n = len(test_df['SOC'])
        p = len(result.params)

        # Adjust R² accounts for sample size and model complexity.
        adj_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        aic_test = n * np.log(rmse_test**2) + 2 * p

        # --------------------------------------------------------
        # 6. Summarize fixed-effect significance
        # --------------------------------------------------------
        # Extract fixed-effect estimates and their standard errors.
        fe_params = result.fe_params  # pandas Series
        fe_se = result.bse_fe  # pandas Series

        # Compute Wald-type z statistics and two-sided normal-approximation p values.
        fe_z = fe_params / fe_se
        fe_p = 2 * (1 - stats.norm.cdf(np.abs(fe_z.values)))  # numpy array
            # Convert p values into conventional significance symbols.
        def signif_stars(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            elif p < 0.1:
                return '·'
            else:
                return ''

        # Assemble a compact fixed-effect coefficient/significance table.
        fe_table = pd.DataFrame({
            "Coef": fe_params.values,
            "StdErr": fe_se.values,
            "z": fe_z.values,
            "p_value": fe_p,
            "Signif": [signif_stars(p) for p in fe_p]
        }, index=fe_params.index)

        # --------------------------------------------------------
        # 7. Print model results
        # --------------------------------------------------------
        print("fixed-effect coefficient/significance: ")
        print(fe_table.round(4))

        print(f"test_results_{j}：")
        print(f"R²      = {r2_test:.3f}")
        print(f"adj_R² = {adj_r2:.3f}")
        print(f"MAE    = {mae_test:.3f}")
        print(f"RMSE   = {rmse_test:.3f}")
        print(f"AIC    = {aic_test:.3f}")

        # Catch errors for an individual repetition so later repetitions can continue.
    except Exception as e:
        print(f"error：{e}")
        continue
