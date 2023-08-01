import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Read the CSV files
LoR = pd.read_csv("Average_Northern.csv")


# Remove missing values
LoR = LoR.dropna()

# Convert columns to numeric
LoR['Year'] = pd.to_numeric(LoR['Year'])
LoR['Month'] = pd.to_numeric(LoR['Month'])

# Subset the data
LoR1 = LoR.iloc[:, 0:11]
data= LoR1.iloc[0:90, :]


# Prepare the input features and target variable
X= data[['Prec_Average', 'Average_Temperature_Max', 'Average_RH_Max']]
y= data['Malaria_incidence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Fit the SVR model
svr_model = SVR(kernel='rbf',epsilon=0.1)
svr_model.fit(X_train, y_train)

# Make predictions on the test data
pred = svr_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

# Print the results
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
print(svr_model.n_support_)