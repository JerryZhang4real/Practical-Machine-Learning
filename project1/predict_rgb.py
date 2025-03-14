import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV data
df = pd.read_csv('Pang_Dataset_Values.csv')

# Separate features and targets.
# We drop 'original_sample_name' since it is an identifier,
# and use the rest as predictors.
X = df.drop(['original_sample_name', 'augmentation_classification', 'true_paint_red', 'true_paint_green', 'true_paint_blue'], axis=1)
y = df[['true_paint_red', 'true_paint_green', 'true_paint_blue']]

# Build a pipeline that first preprocesses the data and then applies a multi-output regressor.
pipeline_hgbr = Pipeline([
    ('regressor', MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)))
])

pipeline_rf = Pipeline([
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

pipeline_nn = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', MultiOutputRegressor(MLPRegressor(random_state=42, max_iter=500)))
])

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train each regressor.
pipeline_hgbr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_nn.fit(X_train, y_train)

# Predict on the test set.
y_pred_hgbr = pipeline_hgbr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_nn = pipeline_nn.predict(X_test)

# Evaluate the models using Mean Squared Error.
mse_hgbr = mean_squared_error(y_test, y_pred_hgbr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_nn = mean_squared_error(y_test, y_pred_nn)

print("Test MSE for HistGradientBoosting:", mse_hgbr)
print("Test MSE for RandomForest:", mse_rf)
print("Test MSE for NN:", mse_nn)

# Create DataFrames for predictions with appropriate column names.
hgbr_pred_df = pd.DataFrame(y_pred_hgbr, columns=['HGBR_pred_red', 'HGBR_pred_green', 'HGBR_pred_blue'], index=y_test.index)
rf_pred_df   = pd.DataFrame(y_pred_rf,   columns=['RF_pred_red', 'RF_pred_green', 'RF_pred_blue'], index=y_test.index)
nn_pred_df   = pd.DataFrame(y_pred_nn,   columns=['NN_pred_red', 'NN_pred_green', 'NN_pred_blue'], index=y_test.index)

# Concatenate the true values with the prediction DataFrames.
result = pd.concat([y_test, rf_pred_df, hgbr_pred_df, nn_pred_df], axis=1)

# Create new columns where RGB values are combined into a list and rounded to the nearest integer.
result['true_RGB']     = result[['true_paint_red', 'true_paint_green', 'true_paint_blue']].round(0).astype(int).values.tolist()
result['RF_pred_RGB']  = result[['RF_pred_red', 'RF_pred_green', 'RF_pred_blue']].round(0).astype(int).values.tolist()
result['HGBR_pred_RGB'] = result[['HGBR_pred_red', 'HGBR_pred_green', 'HGBR_pred_blue']].round(0).astype(int).values.tolist()
result['NN_pred_RGB']  = result[['NN_pred_red', 'NN_pred_green', 'NN_pred_blue']].round(0).astype(int).values.tolist()

# Display the combined result.
print("Original RGB         RandomForest        HistGradientBoosting         NN")
print(result[['true_RGB', 'RF_pred_RGB', 'HGBR_pred_RGB', 'NN_pred_RGB']])

result[['true_RGB', 'RF_pred_RGB', 'HGBR_pred_RGB', 'NN_pred_RGB']].to_csv('result.csv')

