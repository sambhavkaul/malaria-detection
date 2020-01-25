import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

#Loading dataset
dataframe = pd.read_csv("csv/dataset.csv")

#Spliting into train and test
x = dataframe.drop(["Label"], axis = 1)
y = dataframe["Label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Bilding Model
model = RandomForestClassifier(n_estimators = 100, max_depth = 5)
model.fit(x_train, y_train)

#Saving it
joblib.dump(model, "rf_malaria_100_5")

#Making predictions
predictions = model.predict(x_test)

print(metrics.classification_report(predictions, y_test))
