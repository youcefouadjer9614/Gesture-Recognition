import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, auc
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

# Import the data

data = pd.read_excel('dataset.xlsx')
new_data = data.dropna()
y = new_data['user id']
x = new_data.drop(["user id"],axis = 1)

# Splitting the data into a training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

xgboost_classifier = xgb.XGBClassifier(
      objective = 'binary:logistic'
)

parameters = {
    'max_depth':range(2,10,1),
    'n_estimators':range(60,220,40),
    'learning_rate':[0.1,0.01,0.05]
}

grid_search = GridSearchCV(
estimator = xgboost_classifier,
param_grid = parameters,
scoring = 'roc_auc',
n_jobs = 10,
cv = 10,
verbose = True
)

grid_search.fit(x_train,y_train)
grid_search.best_estimator_

xgboost_classifier = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

eval_set = [(x_train, y_train), (x_test, y_test)]
%time xgboost_classifier.fit(x_train, y_train.values.ravel(), early_stopping_rounds=15, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

# making predictions on the test set
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(xgboost_classifier,x_test,y_test,cv = 5)
predictions = [round(value) for value in y_pred]




# Extract metrics information from xgboost package
results = xgboost_classifier.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot the log loss and the classification error
# First Log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, results['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost log Loss')
plt.show()

#Second Classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label = 'Train')
ax.plot(x_axis, results['validation_1']['error'], label = 'Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

# Probability predictions to compute area under the curve and ROC curve

y_pred_train = xgboost_classifier.predict(x_train)

y_pred_test = xgboost_classifier.predict(x_test)

y_predict_prob = xgboost_classifier.predict_proba(x_test)[:,1]

[fpr, tpr, thr] = roc_curve(y_test, y_predict_prob)

AUC = auc(fpr,tpr)
Accuracy = accuracy_score(y_test, y_pred_test)
F_score = f1_score(y_test,y_pred_test,'binary')
Kappa_score = cohen_kappa_score(y_test, y_pred_test)
print("AUC: %.2f%%"% (AUC*100.0))
print("Accuracy: %.2f%%" % (Accuracy *100.0))
print("F score: %.2f%% "% (F_score*100))
print("Kappa score: %.2f%%"% (Kappa_score*100.0))

idx = np.min(np.where(tpr>0.95))

plt.figure()
plt.plot(fpr, tpr, color = 'coral', label = "ROC curve area: " + str(AUC*100.0))
plt.plot([0,1], [0,1], 'k--')
plt.plot([0, fpr[idx]],[tpr[idx], tpr[idx]], 'k--', color = 'blue')
plt.plot([fpr[idx], fpr[idx]], [0,tpr[idx]], 'k--', color='blue')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.legend(loc="lower right")
plt.savefig('myfig.png')
plt.show()

# Plotting the feature importance score

from xgboost import plot_importance
plot_importance(xgboost_classifier, max_num_features = 15)

