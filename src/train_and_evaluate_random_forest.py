# %%
import pandas as pd # for read files
import seaborn as sns 
import matplotlib.pyplot as plt # for draw figure

from sklearn import metrics # calculate some performance index ex. accuracy/mcc
from sklearn.ensemble import RandomForestClassifier # random forest model
from sklearn.model_selection import train_test_split # splitting test/train set

from IPython.display import display, HTML # better for export format

import sklearn.tree as tree

import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"

training_set = pd.read_csv(data_dir / "train.csv")
testing_set = pd.read_csv(data_dir / "test.csv")

X_train = training_set.drop(['smiles', 'kinase', 'label'], axis=1)
X_test = testing_set.drop(['smiles', 'kinase', 'label'], axis=1)

y_train = training_set['label']
y_test = testing_set['label']

features_id = list(training_set.drop(['smiles', 'kinase', 'label'], axis=1).columns)

Accuracy = []
Recall_Inhibitor = []
Recall_Not_inhibitor = []
Precision_Inhibitor = []
Precision_Not_inhibitor = []
F1_score_Inhibitor = []
F1_score_Not_inhibitor = []
MCC = []
Roc_auc = []

rf_model = RandomForestClassifier(bootstrap = True , class_weight= None ,max_depth = None, max_features = 'sqrt'
, min_samples_leaf = 1, min_samples_split = 2, n_estimators= 100)


rf_model.fit(X_train, y_train) # train model by X_train and y_train data
y_pred = rf_model.predict(X_test) # predict result by input X_test and trained model
feature_imp = pd.Series(rf_model.feature_importances_, index=features_id).sort_values(ascending=False)


Accuracy.append(metrics.accuracy_score(y_test, y_pred))  
Recall_Inhibitor.append(metrics.recall_score(y_test, y_pred))
Recall_Not_inhibitor.append(metrics.recall_score(y_test, y_pred, pos_label=0))
Precision_Inhibitor.append(metrics.precision_score(y_test, y_pred))
Precision_Not_inhibitor.append(metrics.precision_score(y_test, y_pred, pos_label=0))
F1_score_Inhibitor.append(metrics.f1_score(y_test, y_pred))
F1_score_Not_inhibitor.append(metrics.f1_score(y_test, y_pred, pos_label=0))
MCC.append(metrics.matthews_corrcoef(y_test, y_pred))
Roc_auc.append(metrics.roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))

temp = {
'Accuracy':Accuracy,
'Recall_Inhibitor': Recall_Inhibitor,
'Recall_Not_inhibitor':Recall_Not_inhibitor,
'Precision_Inhibitor':Precision_Inhibitor,
'Precision_Not_inhibitor':Precision_Not_inhibitor,
'F1_score_Inhibitor':F1_score_Inhibitor,
'F1_score_Not_inhibitor':F1_score_Not_inhibitor,
'MCC':MCC,
'Roc_auc':Roc_auc
}

df_Evaluate = pd.DataFrame(temp)
df_Evaluate

# %%
with open('./Test_each_kinase_confusion_matrix.txt','w') as wf:   #Accuracy of each kinase
  pred_df = pd.DataFrame(y_pred, columns=['Predict'])
  pred_df = pd.concat([pred_df,testing_set[['kinase','label']].reset_index(drop=True)],axis=1)

  import math
  num = 0
  count = 0
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  kinase_temp = testing_set['kinase'].unique()
  print('Kinase','TP','FP','TN','FN','Accuracy',file=wf)
  for k in kinase_temp:
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for x in pred_df.loc[pred_df['kinase'] == k].to_numpy():
      if(x[2] != x[0]):
        if int(x[2]) == 1:
          FN +=1
        else:
          FP +=1
      else:
        if int(x[2]) == 1:
          TP +=1
        else:
          TN +=1
    print(k,TP,FP,TN,FN, '{:.3f}'.format((TP+TN)/(TP+FP+TN+FN)),file=wf)

# %%
with open('Test_error.txt','w') as wf:#Print prediction error
    count = 0
    print('Kinase\tSmiles\tPredict\tTrue',file=wf)
    for index, label in y_test.items():
        if(label != y_pred[count]):
            print(testing_set.loc[index]['kinase']+'\t'+testing_set.loc[index]['smiles']+'\t'+str(y_pred[count])+'\t'+str(label),file=wf)
        count += 1

# %%
#Feature importance score
feature_imp = pd.Series(rf_model.feature_importances_, index=features_id).sort_values(ascending=False)

sns_plot = sns.barplot(x=feature_imp[:20], y=feature_imp.index[:20]) 
fig = sns_plot.get_figure()
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# %%
independent_set = pd.read_csv(data_dir / "independent.csv")
X_val = independent_set.drop(['smiles', 'kinase', 'label'], axis=1)
y_val = independent_set['label']

Accuracy_i = []
Recall_Inhibitor_i = []
Recall_Not_inhibitor_i = []
Precision_Inhibitor_i = []
Precision_Not_inhibitor_i = []
F1_score_Inhibitor_i = []
F1_score_Not_inhibitor_i = []
MCC_i = []
Roc_auc_i = []
v_pred = rf_model.predict(X_val)

Accuracy_i.append(metrics.accuracy_score(y_val, v_pred))  
Recall_Inhibitor_i.append(metrics.recall_score(y_val, v_pred))
Recall_Not_inhibitor_i.append(metrics.recall_score(y_val, v_pred, pos_label=0))
Precision_Inhibitor_i.append(metrics.precision_score(y_val, v_pred))
Precision_Not_inhibitor_i.append(metrics.precision_score(y_val, v_pred, pos_label=0))
F1_score_Inhibitor_i.append(metrics.f1_score(y_val, v_pred))
F1_score_Not_inhibitor_i.append(metrics.f1_score(y_val, v_pred, pos_label=0))
MCC_i.append(metrics.matthews_corrcoef(y_val, v_pred))
Roc_auc_i.append(metrics.roc_auc_score(y_val, rf_model.predict_proba(X_val)[:,1]))

temp_i = {
      'Accuracy':Accuracy_i,
      'Recall_Inhibitor': Recall_Inhibitor_i,
      'Recall_Not_inhibitor':Recall_Not_inhibitor_i,
      'Precision_Inhibitor':Precision_Inhibitor_i,
      'Precision_Not_inhibitor':Precision_Not_inhibitor_i,
      'F1_score_Inhibitor':F1_score_Inhibitor_i,
      'F1_score_Not_inhibitor':F1_score_Not_inhibitor_i,
      'MCC':MCC_i,
      'Roc_auc':Roc_auc_i
      }

df_Evaluate_i = pd.DataFrame(temp_i)
df_Evaluate_i

# %%
with open('./Independent_each_kinase_confusion_matrix.txt','w') as wf: #Accuracy of each kinase
  pred_df = pd.DataFrame(v_pred, columns=['Predict'])
  pred_df = pd.concat([pred_df,independent_set[['kinase','label']].reset_index(drop=True)],axis=1)

  import math
  num = 0
  count = 0
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  kinase_temp = independent_set['kinase'].unique()
  print('Kinase','TP','FP','TN','FN','MCC','Accuracy',file=wf)
  for k in kinase_temp:
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for x in pred_df.loc[pred_df['kinase'] == k].to_numpy():
      if(x[2] != x[0]):
        if int(x[2]) == 1:
          FN +=1
        else:
          FP +=1
      else:
        if int(x[2]) == 1:
          TP +=1
        else:
          TN +=1
    print(k,TP,FP,TN,FN, '{:.3f}'.format((TP+TN)/(TP+FP+TN+FN)),file=wf)

# %%
with open('Independent_error.txt','w') as wf:#Print prediction error
    count = 0
    print('Kinase\tSmiles\tPredict\tTrue',file=wf)
    for index, label in y_val.items():
        if(label != v_pred[count]):
            print(independent_set.loc[index]['kinase']+'\t'+independent_set.loc[index]['smiles']+'\t'+str(v_pred[count])+'\t'+str(label),file=wf)
        count += 1


