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
with open('training_log_1.txt','w') as wf1:
   with open('training_log_2.txt','w') as wf2:
      for run in range(1):
        rf_model_4 = RandomForestClassifier(bootstrap = True , class_weight= None ,max_depth = None, max_features = 'sqrt'
        , min_samples_leaf = 1, min_samples_split = 2, n_estimators= 1000)
        # set RF classifier model and parameters class_weight=[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]
        rf_model_4.fit(X_train, y_train) # train model by X_train and y_train data
        y_pred = rf_model_4.predict(X_test) # predict result by input X_test and trained model
        feature_imp = pd.Series(rf_model_4.feature_importances_, index=features_id).sort_values(ascending=False)
        # print(feature_imp[0:20],feature_imp.index[0:20])
        Accuracy.append(metrics.accuracy_score(y_test, y_pred))  
        Recall_Inhibitor.append(metrics.recall_score(y_test, y_pred))
        Recall_Not_inhibitor.append(metrics.recall_score(y_test, y_pred, pos_label=0))
        Precision_Inhibitor.append(metrics.precision_score(y_test, y_pred))
        Precision_Not_inhibitor.append(metrics.precision_score(y_test, y_pred, pos_label=0))
        F1_score_Inhibitor.append(metrics.f1_score(y_test, y_pred))
        F1_score_Not_inhibitor.append(metrics.f1_score(y_test, y_pred, pos_label=0))
        MCC.append(metrics.matthews_corrcoef(y_test, y_pred))
        Roc_auc.append(metrics.roc_auc_score(y_test, rf_model_4.predict_proba(X_test)[:,1]))

        rank = 0
        P_rank = 0
        C_rank = 0
        for name,imp in zip(feature_imp.index, feature_imp):
            rank += 1
            if '#AA' in name:
              P_rank += 1
              wf1.write('Run_'+str(run+1)+'\t'+name+'\t'+str(imp)+'\t'+str(rank)+'\t'+str(P_rank)+'\tx\n')
            else:
               C_rank += 1
               wf1.write('Run_'+str(run+1)+'\t'+name+'\t'+str(imp)+'\t'+str(rank)+'\tx\t'+str(C_rank)+'\n')

        count = 0
        for smiles, label in y_test.items():
          if(label != y_pred[count]):
            wf2.write('Run_'+str(run+1)+'\t'+testing_set.iloc[smiles]['kinase']+'\t'+testing_set.iloc[smiles]['smiles']+'\t'+str(y_pred[count])+'\t'+str(label)+'\n')
          count += 1

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
pred_df = pd.DataFrame(y_pred, columns=['Predict'])
pred_df = pd.concat([pred_df,testing_set[['kinase','label']].reset_index(drop=True)],axis=1)

import math
num = 0
count = 0
TP = 0
FP = 0
TN = 0
FN = 0
TTP = 0
TFP = 0
TTN = 0
TFN = 0
kinase_temp = testing_set['kinase'].unique()
print('Kinase','TP','FP','TN','FN','MCC','Accuracy')
for k in kinase_temp:
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for x in pred_df.loc[pred_df['kinase'] == k].to_numpy():
    if(x[2] != x[0]):
      if int(x[2]) == 1:
        FN +=1
        TFN +=1
      else:
        FP +=1
        TFP +=1
    else:
      if int(x[2]) == 1:
        TP +=1
        TTP +=1
      else:
        TN +=1
        TTN +=1
  print(k,TP,FP,TN,FN,  '{:.3f}'.format((TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        ,  '{:.3f}'.format((TP+TN)/(TP+FP+TN+FN)))
print('Total',TTP,TFP,TTN,TFN,  '{:.3F}'.format((TTP*TTN-TFP*TFN)/math.sqrt((TTP+TFP)*(TTP+TFN)*(TTN+TFP)*(TTN+TFN)))
      ,'{:.3f}'.format((TTP+TTN)/(TTP+TFP+TTN+TFN)))

# %%
Accuracy = []
Recall_Inhibitor = []
Recall_Not_inhibitor = []
Precision_Inhibitor = []
Precision_Not_inhibitor = []
F1_score_Inhibitor = []
F1_score_Not_inhibitor = []
MCC = []
Roc_auc = []
x_pred = rf_model_4.predict(X_train)

Accuracy.append(metrics.accuracy_score(y_train, x_pred))  
Recall_Inhibitor.append(metrics.recall_score(y_train, x_pred))
Recall_Not_inhibitor.append(metrics.recall_score(y_train, x_pred, pos_label=0))
Precision_Inhibitor.append(metrics.precision_score(y_train, x_pred))
Precision_Not_inhibitor.append(metrics.precision_score(y_train, x_pred, pos_label=0))
F1_score_Inhibitor.append(metrics.f1_score(y_train, x_pred))
F1_score_Not_inhibitor.append(metrics.f1_score(y_train, x_pred, pos_label=0))
MCC.append(metrics.matthews_corrcoef(y_train, x_pred))
Roc_auc.append(metrics.roc_auc_score(y_train, rf_model_4.predict_proba(X_train)[:,1]))

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
independent_set = pd.read_csv(data_dir / "independent.csv")
X_val = independent_set.drop(['smiles', 'kinase', 'label'], axis=1)
y_val = independent_set['label']

Accuracy = []
Recall_Inhibitor = []
Recall_Not_inhibitor = []
Precision_Inhibitor = []
Precision_Not_inhibitor = []
F1_score_Inhibitor = []
F1_score_Not_inhibitor = []
MCC = []
Roc_auc = []
v_pred = rf_model_4.predict(X_val)

Accuracy.append(metrics.accuracy_score(y_val, v_pred))  
Recall_Inhibitor.append(metrics.recall_score(y_val, v_pred))
Recall_Not_inhibitor.append(metrics.recall_score(y_val, v_pred, pos_label=0))
Precision_Inhibitor.append(metrics.precision_score(y_val, v_pred))
Precision_Not_inhibitor.append(metrics.precision_score(y_val, v_pred, pos_label=0))
F1_score_Inhibitor.append(metrics.f1_score(y_val, v_pred))
F1_score_Not_inhibitor.append(metrics.f1_score(y_val, v_pred, pos_label=0))
MCC.append(metrics.matthews_corrcoef(y_val, v_pred))
Roc_auc.append(metrics.roc_auc_score(y_val, rf_model_4.predict_proba(X_val)[:,1]))

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
count = 0
print('kinase\tsmiles\tprdict\ttrue')
for index, label in y_val.items():
    if(label != v_pred[count]):
        print(independent_set.loc[index]['kinase']+'\t'+independent_set.loc[index]['smiles']+'\t'+str(v_pred[count])+'\t'+str(label))
    count += 1

# %%
pred_df = pd.DataFrame(v_pred, columns=['Predict'])
pred_df = pd.concat([pred_df,independent_set[['kinase','label']].reset_index(drop=True)],axis=1)

import math
num = 0
count = 0
TP = 0
FP = 0
TN = 0
FN = 0
TTP = 0
TFP = 0
TTN = 0
TFN = 0
kinase_temp = independent_set['kinase'].unique()
print('Kinase','TP','FP','TN','FN','MCC','Accuracy')
for k in kinase_temp:
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for x in pred_df.loc[pred_df['kinase'] == k].to_numpy():
    if(x[2] != x[0]):
      if int(x[2]) == 1:
        FN +=1
        TFN +=1
      else:
        FP +=1
        TFP +=1
    else:
      if int(x[2]) == 1:
        TP +=1
        TTP +=1
      else:
        TN +=1
        TTN +=1
  a = TP+FP
  b = TP+FN
  c = TN+FP
  d = TN+FN
  if TP+FP == 0:
    a = 1
  if TP+FN == 0:
    b = 1
  if TN+FP == 0:
    c = 1
  if TN+FN == 0:
    d = 1
  print(k,TP,FP,TN,FN,  '{:.3f}'.format((TP*TN-FP*FN)/math.sqrt((a)*(b)*(c)*(d)))
        ,  '{:.3f}'.format((TP+TN)/(TP+FP+TN+FN)))
# print('Total',TTP,TFP,TTN,TFN,  '{:.3F}'.format((TTP*TTN-TFP*TFN)/math.sqrt((TTP+TFP)*(TTP+TFN)*(TTN+TFP)*(TTN+TFN)))
#       ,'{:.3f}'.format((TTP+TTN)/(TTP+TFP+TTN+TFN)))

# %%
JMY_set = pd.read_csv(data_dir / "inhouse_50x22.csv")
X_val = JMY_set.drop(['smiles', 'kinase', 'label'], axis=1)
y_val = JMY_set['label']

Accuracy = []
Recall_Inhibitor = []
Recall_Not_inhibitor = []
Precision_Inhibitor = []
Precision_Not_inhibitor = []
F1_score_Inhibitor = []
F1_score_Not_inhibitor = []
MCC = []
Roc_auc = []
v_pred = rf_model_4.predict(X_val)

Accuracy.append(metrics.accuracy_score(y_val, v_pred))  
Recall_Inhibitor.append(metrics.recall_score(y_val, v_pred))
Recall_Not_inhibitor.append(metrics.recall_score(y_val, v_pred, pos_label=0))
Precision_Inhibitor.append(metrics.precision_score(y_val, v_pred))
Precision_Not_inhibitor.append(metrics.precision_score(y_val, v_pred, pos_label=0))
F1_score_Inhibitor.append(metrics.f1_score(y_val, v_pred))
F1_score_Not_inhibitor.append(metrics.f1_score(y_val, v_pred, pos_label=0))
MCC.append(metrics.matthews_corrcoef(y_val, v_pred))
Roc_auc.append(metrics.roc_auc_score(y_val, rf_model_4.predict_proba(X_val)[:,1]))

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
count = 0
print('kinase\tsmiles\tprdict\ttrue')
for index, label in y_val.items():
    if(label != v_pred[count]):
        print(JMY_set.loc[index]['kinase']+'\t'+JMY_set.loc[index]['smiles']+'\t'+str(v_pred[count])+'\t'+str(label))
    count += 1

# %%
pred_df = pd.DataFrame(v_pred, columns=['Predict'])
pred_df = pd.concat([pred_df,JMY_set[['kinase','label']].reset_index(drop=True)],axis=1)

import math
num = 0
count = 0
TP = 0
FP = 0
TN = 0
FN = 0
TTP = 0
TFP = 0
TTN = 0
TFN = 0
kinase_temp = JMY_set['kinase'].unique()
print('Kinase','TP','FP','TN','FN','MCC','Accuracy')
for k in kinase_temp:
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for x in pred_df.loc[pred_df['kinase'] == k].to_numpy():
    if(x[2] != x[0]):
      if int(x[2]) == 1:
        FN +=1
        TFN +=1
      else:
        FP +=1
        TFP +=1
    else:
      if int(x[2]) == 1:
        TP +=1
        TTP +=1
      else:
        TN +=1
        TTN +=1
  a = TP+FP
  b = TP+FN
  c = TN+FP
  d = TN+FN
  if TP+FP == 0:
    a = 1
  if TP+FN == 0:
    b = 1
  if TN+FP == 0:
    c = 1
  if TN+FN == 0:
    d = 1
  print(k,TP,FP,TN,FN,  '{:.3f}'.format((TP*TN-FP*FN)/math.sqrt((a)*(b)*(c)*(d)))
        ,  '{:.3f}'.format((TP+TN)/(TP+FP+TN+FN)))
# print('Total',TTP,TFP,TTN,TFN,  '{:.3F}'.format((TTP*TTN-TFP*TFN)/math.sqrt((TTP+TFP)*(TTP+TFN)*(TTN+TFP)*(TTN+TFN)))
#       ,'{:.3f}'.format((TTP+TTN)/(TTP+TFP+TTN+TFN)))

# %%
JMY_set = pd.read_csv(data_dir / "inhouse_50x22.csv")
X_val = JMY_set.drop(['smiles', 'kinase', 'label'], axis=1)
y_val = JMY_set['label']

Accuracy = []
Recall_Inhibitor = []
Recall_Not_inhibitor = []
Precision_Inhibitor = []
Precision_Not_inhibitor = []
F1_score_Inhibitor = []
F1_score_Not_inhibitor = []
MCC = []
Roc_auc = []
v_pred = rf_model_4.predict(X_val)

Accuracy.append(metrics.accuracy_score(y_val, v_pred))  
Recall_Inhibitor.append(metrics.recall_score(y_val, v_pred))
Recall_Not_inhibitor.append(metrics.recall_score(y_val, v_pred, pos_label=0))
Precision_Inhibitor.append(metrics.precision_score(y_val, v_pred))
Precision_Not_inhibitor.append(metrics.precision_score(y_val, v_pred, pos_label=0))
F1_score_Inhibitor.append(metrics.f1_score(y_val, v_pred))
F1_score_Not_inhibitor.append(metrics.f1_score(y_val, v_pred, pos_label=0))
MCC.append(metrics.matthews_corrcoef(y_val, v_pred))
Roc_auc.append(metrics.roc_auc_score(y_val, rf_model_4.predict_proba(X_val)[:,1]))

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
count = 0
print('kinase\tsmiles\tprdict\ttrue')
for index, label in y_val.items():
    if(label != v_pred[count]):
        print(JMY_set.loc[index]['kinase']+'\t'+JMY_set.loc[index]['smiles']+'\t'+str(v_pred[count])+'\t'+str(label))
    count += 1

# %%
pred_df = pd.DataFrame(v_pred, columns=['Predict'])
pred_df = pd.concat([pred_df,JMY_set[['kinase','label']].reset_index(drop=True)],axis=1)

import math
num = 0
count = 0
TP = 0
FP = 0
TN = 0
FN = 0
TTP = 0
TFP = 0
TTN = 0
TFN = 0
kinase_temp = JMY_set['kinase'].unique()
print('Kinase','TP','FP','TN','FN','MCC','Accuracy')
for k in kinase_temp:
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for x in pred_df.loc[pred_df['kinase'] == k].to_numpy():
    if(x[2] != x[0]):
      if int(x[2]) == 1:
        FN +=1
        TFN +=1
      else:
        FP +=1
        TFP +=1
    else:
      if int(x[2]) == 1:
        TP +=1
        TTP +=1
      else:
        TN +=1
        TTN +=1
  a = TP+FP
  b = TP+FN
  c = TN+FP
  d = TN+FN
  if TP+FP == 0:
    a = 1
  if TP+FN == 0:
    b = 1
  if TN+FP == 0:
    c = 1
  if TN+FN == 0:
    d = 1
  print(k,TP,FP,TN,FN,  '{:.3f}'.format((TP*TN-FP*FN)/math.sqrt((a)*(b)*(c)*(d)))
        ,  '{:.3f}'.format((TP+TN)/(TP+FP+TN+FN)))
# print('Total',TTP,TFP,TTN,TFN,  '{:.3F}'.format((TTP*TTN-TFP*TFN)/math.sqrt((TTP+TFP)*(TTP+TFN)*(TTN+TFP)*(TTN+TFN)))
#       ,'{:.3f}'.format((TTP+TTN)/(TTP+TFP+TTN+TFN)))

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# %%
import shap
shap.initjs()

# %%
# 圖下方的 #feature = 1 or 0 是指 sample中 實際的值， 以我的來說就是 有這個feature 或 沒
#imatinib
def shap_plot(j,data_set):
    explainerModel = shap.Explainer(rf_model_4)
    shap_values_Model = explainerModel.shap_values(data_set.iloc[j][3:], check_additivity=False)
    print(data_set.iloc[j][:3].values[0],data_set.iloc[j][:3].values[1],data_set.iloc[j][:3].values[2],sep='\t')
    base_value = explainerModel.expected_value
    print("Base Value:", base_value[1])
    print('Feature\tshap value\thave feature or not')
    for x,y,z in zip(data_set.columns[3:], shap_values_Model[1],data_set.iloc[j][3:].values):
        print(x,y,z,sep='\t')
    p = shap.force_plot(explainerModel.expected_value[1], shap_values_Model[1], data_set.iloc[[j]].iloc[:,3:])
    return(p)

shap_plot(1520,training_set)


