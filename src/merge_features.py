# %%
import pandas as pd
import re

# %%
df1 = pd.read_csv('./checkmol.csv')
df2 = pd.read_csv('./pubchem.csv')
df3 = pd.read_csv('./ring.csv')
df4 = pd.read_csv('./ac.csv')
df5 = pd.read_csv('./ecfp.csv')


# %%
merge = pd.concat([df1,df2.iloc[:,1:]], axis=1)
merge = pd.concat([merge,df3.iloc[:,1:]], axis=1)
merge = pd.concat([merge,df4.iloc[:,1:]], axis=1)
merge = pd.concat([merge,df5.iloc[:,1:]], axis=1)

# %%
for x in merge.columns:
    print(x)

# %%
count_small_m =[]
pattern = r'(?<!l)@(?![l])'
for x in merge['smiles'][:]:
    matches = re.findall(pattern, x)
    count_small_m.append(len(matches))
merge['count_@'] =count_small_m

# %%
#連續@ or @@ 數量
arr_con_m = []
arr_con_mm = []
arr_fir_m = []
arr_fir_mm = []
for x in merge['smiles'][:]:
    continue_m = 0
    continue_mm = 0
    first_m = 0
    first_mm = 0

    max_continiue_m = []
    max_continiue_mm = []
    for i in range(len(x)):
        if x[i] == '@': 
            if x[i+1] != '@' and x[i-1] != '@': #僅考慮@
                if first_mm == 0:
                    first_m = 1

                continue_m+=1
                if continue_mm !=0:
                    if continue_mm == 1:
                        continue_mm = 0
                        max_continiue_mm.append(continue_mm) 
                    else:    
                        max_continiue_mm.append(continue_mm)
                        continue_mm = 0            
            elif x[i+1] == '@': #僅考慮@@
                if first_m == 0:
                    first_mm = 1

                continue_mm+=1
                if continue_m != 0:
                    if continue_m == 1:
                        continue_m = 0
                        max_continiue_m.append(continue_m) 
                    else:    
                        max_continiue_m.append(continue_m)
                        continue_m = 0
    if continue_m ==1:
        max_continiue_m.append(0)
    else:
        max_continiue_m.append(continue_m)

    if continue_mm ==1:
        max_continiue_mm.append(0)
    else:
        max_continiue_mm.append(continue_mm)
    arr_con_m.append(max(max_continiue_m))
    arr_con_mm.append(max(max_continiue_mm))
    arr_fir_m.append(first_m)
    arr_fir_mm.append(first_mm)

# print(arr_con_m)
# print(arr_con_mm)
# print(arr_fir_m)
# print(arr_fir_mm)
merge['arr_con_m'] =arr_con_m
merge['arr_con_mm'] =arr_con_mm    
merge['arr_fir_m'] =arr_fir_m
merge['arr_fir_mm'] =arr_fir_mm


# %%
count_small_m =[]
pattern = r'(?<!l)\/(?![l])'
for x in merge['smiles'][:]:
    matches = re.findall(pattern, x)
    count_small_m.append(len(matches))
merge['count_/'] =count_small_m

# %%
count_small_m =[]
pattern = r'(?<!l)\\(?![l])'
for x in merge['smiles'][:]:
    matches = re.findall(pattern, x)
    count_small_m.append(len(matches))
merge['count_\\'] =count_small_m

# %%
merge.to_csv('./merge.csv', index=False)


