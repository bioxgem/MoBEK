# %% [markdown]
# more_than_1

# %%
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent
protein_resource_dir = script_dir / "generate_feature" / "protein"
Cell_85 = [8,9,10,11,12,13,14,15,16,17,18,19,20,58,59,60,61,62,63,112,113,114,115,116,117,118,119,120,121,122,123,124,141,142,143,144,145,146,147,148,149,190,191,192,193,194,195,196,197,199,200,201,202,203,204,205,206,207,208,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,664,665,666,667,668,669,670]# 85 位點位置
Science_report_85 = [17,18,19,20,21,22,23,24,25,26,27,28,29,55,56,57,58,59,60,96,97,98,99,100,101,102,103,104,105,106,107,108,125,126,127,128,129,130,131,132,133,173,174,175,176,177,178,179,180,182,183,184,185,186,187,188,189,190,191,312,313,314,315,316,358,359,360,361,362,363,364,365,366,367,368,369,370,371,677,678,679,680,681,682,683]
BBA_85 = [17,18,19,20,21,22,23,24,25,26,27,28,29,55,56,57,58,59,60,96,97,98,99,100,101,102,103,104,105,106,107,108,125,126,127,128,129,130,131,132,133,173,174,175,176,177,178,179,180,182,183,184,185,186,187,188,189,190,191,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,361,362,363,364,365,366,367]
Sci_del = [17,18,19,20,21,22,23,24,25,26,27,28,29,55,56,57,58,59,60,96,97,98,99,100,101,102,103,104,105,106,107,108,125,126,127,128,129,130,131,132,133,173,174,175,176,177,178,179,180,182,183,184,185,186,187,188,189,190,191,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,636,637,638,639,640,641,642]
Row_Cell = [17,	26,	27,	28,	29,	42,	43,	52,	53,	58,	59,	60,	61,	99,	100,	101,	102,	103,	104,	185,	186,	187,	188,	190,	191,	192,	193,	194,	195,	196,	197,	198,	211,	212,	223,	224,	225,	232,	233,	234,	235,	464,	465,	466,	467,	468,	469,	470,	471,	472,	473,	493,	494,	495,	496,	497,	498,	499,	500,	625,	626,	627,	628,	633,	634,	642,	643,	644,	645,	646,	647,	648,	649,	650,	651,	652,	653,	654,	967,	968,	991,	992,	993,	994,	995]
Row_Sci = [21,	22,	23,	24,	25,	26,	27,	44,	45,	46,	47,	48,	49,	97,	98,	99,	100,	101,	102,	142,	143,	144,	145,	146,	147,	148,	149,	150,	151,	152,	153,	180,	182,	183,	184,	185,	186,	187,	188,	189,	190,	422,	423,	424,	425,	426,	427,	428,	429,	430,	445,	446,	447,	448,	449,	450,	451,	452,	453,	958,	959,	960,	961,	962,	1008,	1009,	1010,	1011,	1012,	1013,	1014,	1020,	1021,	1022,	1023,	1024,	1025,	1026,	1336,	1337,	1338,	1339,	1340,	1341,	1342]
Row_BBA = [33,	34,	35,	48,	49,	50,	51,	54,	55,	56,	57,	58,	59,	93,	94,	95,	96,	97,	98,	134,	135,	136,	137,	138,	139,	140,	141,	142,	143,	146,	147,	148,	165,	166,	167,	168,	169,	170,	171,	172,	173,	401,	402,	403,	404,	405,	406,	407,	408,	410,	411,	412,	413,	414,	415,	416,	417,	418,	419,	563,	564,	565,	566,	567,	579,	580,	581,	582,	583,	584,	585,	586,	587,	588,	589,	590,	591,	592,	623,	624,	625,	626,	627,	628,	629]
#------------------#
Array_85 = BBA_85 # our application in model
#------------------------------------------------------------------------------------------------------------#
with open(protein_resource_dir / 'KLIFS_85_feature_list.txt') as rfs:
    title = ['Group,','Kinase,','Uniprot,']
    bits = 1
    position = 1
    rfs = rfs.readlines()
    range_num = 20
    for x,i in enumerate(rfs):
        i = i.strip('\n')
        for j in range((range_num)):
            if x == len(rfs)-1:
                if j == range_num-1:
                    title.append('#AA_'+str(x+1)+'_'+str(j+1)+' '+i+'\n')
                else:
                    title.append('#AA_'+str(x+1)+'_'+str(j+1)+' '+i+',')
            else:
                title.append('#AA_'+str(x+1)+'_'+str(j+1)+' '+i+',')
df = pd.read_csv(protein_resource_dir / 'Filter_BBA.csv')
with open('protein_features.csv','w') as wf:
    for t in title:
        wf.write(t)
    for i in range(df.shape[0]): # kinase 照順序輸入
        print(i+1,'/',df.shape[0])
        Protein_feature = []
        kinase_inf = df.iloc[i,:]
        for j, Klifs_num in enumerate(Array_85): # 85 位置依序開始
            for seq_num in range(len(kinase_inf)): # 原始序列的位置 開始對應 85 位置
                if seq_num < 3 and j == 0:
                    Protein_feature.append(kinase_inf[seq_num]+',')
                    # print(Protein_feature)
                elif seq_num == (Klifs_num+2): #seq_num 的 3 等於 Klifs_num 的 1
                    # if kinase_inf['Kinase'] == 'EGFR': #檢查 對應的85 位置 是否正確
                    #     print(kinase_inf[seq_num])
                    env = 4 # own environment score
                    envB = 0
                    envA = 0
                    envNP = 0
                    envP = 0
                    #A.A. type
                    if(kinase_inf[seq_num] == 'K' or kinase_inf[seq_num] == 'R'): # single kinase conservation weight at this position
                        Protein_feature.append('1,0,0,0,')
                    elif(kinase_inf[seq_num] == 'D' or kinase_inf[seq_num] == 'E'):
                        Protein_feature.append('0,1,0,0,')
                    elif(kinase_inf[seq_num] == 'A' or kinase_inf[seq_num] == 'F' or kinase_inf[seq_num] == 'I' or kinase_inf[seq_num] == 'L' or kinase_inf[seq_num] == 'M' or kinase_inf[seq_num] == 'P' or kinase_inf[seq_num] == 'V' or kinase_inf[seq_num] == 'W'):
                        Protein_feature.append('0,0,1,0,')
                    elif(kinase_inf[seq_num] == 'C' or kinase_inf[seq_num] == 'G' or kinase_inf[seq_num] == 'H' or kinase_inf[seq_num] == 'N' or kinase_inf[seq_num] == 'Q' or kinase_inf[seq_num] == 'S' or kinase_inf[seq_num] == 'T' or kinase_inf[seq_num] == 'Y'):
                        Protein_feature.append('0,0,0,1,')
                    else:
                        Protein_feature.append('0,0,0,0,') # gap
                    # A.A. Evolutionary
                    weight = df.iloc[:,seq_num].tolist().count(kinase_inf[seq_num])/(float(df.shape[0])-1)
                    if(float(weight) < 0.02): # single kinase conservation weight at this position
                        Protein_feature.append('0,0,0,0,')
                    elif(float(weight)  >= 0.02 and float(weight) < 0.1):
                        Protein_feature.append('0,0,0,1,')
                    elif(float(weight)  >= 0.1 and float(weight) < 0.3):
                        Protein_feature.append('0,0,1,1,')
                    elif(float(weight) >= 0.3 and float(weight) < 0.6):
                        Protein_feature.append('0,1,1,1,')
                    elif(float(weight) >= 0.6):
                        Protein_feature.append('1,1,1,1,')
                    # Environment    
                    if(kinase_inf[seq_num] == 'K' or kinase_inf[seq_num] == 'R'):
                        envB+=0
                    elif(kinase_inf[seq_num] == 'D' or kinase_inf[seq_num] == 'E'):
                        envA+=0
                    elif(kinase_inf[seq_num] == 'A' or kinase_inf[seq_num] == 'F' or kinase_inf[seq_num] == 'I' or kinase_inf[seq_num] == 'L' or kinase_inf[seq_num] == 'M' or kinase_inf[seq_num] == 'P' or kinase_inf[seq_num] == 'V' or kinase_inf[seq_num] == 'W'):
                        envNP+=0
                    elif(kinase_inf[seq_num] == 'C' or kinase_inf[seq_num] == 'G' or kinase_inf[seq_num] == 'H' or kinase_inf[seq_num] == 'N' or kinase_inf[seq_num] == 'Q' or kinase_inf[seq_num] == 'S' or kinase_inf[seq_num] == 'T' or kinase_inf[seq_num] == 'Y'):
                        envP+=0

                    if(kinase_inf[seq_num+1] == 'K' or kinase_inf[seq_num+1] == 'R'):
                        envB+= 1
                    elif(kinase_inf[seq_num+1] == 'D' or kinase_inf[seq_num+1] == 'E'):
                        envA+= 1
                    elif(kinase_inf[seq_num+1] == 'A' or kinase_inf[seq_num+1] == 'F' or kinase_inf[seq_num+1] == 'I' or kinase_inf[seq_num+1] == 'L' or kinase_inf[seq_num+1] == 'M' or kinase_inf[seq_num+1] == 'P' or kinase_inf[seq_num+1] == 'V' or kinase_inf[seq_num+1] == 'W'):
                        envNP+= 1
                    elif(kinase_inf[seq_num+1] == 'C' or kinase_inf[seq_num+1] == 'G' or kinase_inf[seq_num+1] == 'H' or kinase_inf[seq_num+1] == 'N' or kinase_inf[seq_num+1] == 'Q' or kinase_inf[seq_num+1] == 'S' or kinase_inf[seq_num+1] == 'T' or kinase_inf[seq_num+1] == 'Y'):
                        envP+= 1

                    if(kinase_inf[seq_num+2] == 'K' or kinase_inf[seq_num+2] == 'R'):
                        envB+= 1
                    if(kinase_inf[seq_num+2] == 'D' or kinase_inf[seq_num+2] == 'E'):
                        envA+= 1
                    if(kinase_inf[seq_num+2] == 'A' or kinase_inf[seq_num+2] == 'F' or kinase_inf[seq_num+2] == 'I' or kinase_inf[seq_num+2] == 'L' or kinase_inf[seq_num+2] == 'M' or kinase_inf[seq_num+2] == 'P' or kinase_inf[seq_num+2] == 'V' or kinase_inf[seq_num+2] == 'W'):
                        envNP+= 1    
                    if(kinase_inf[seq_num+2] == 'C' or kinase_inf[seq_num+2] == 'G' or kinase_inf[seq_num+2] == 'H' or kinase_inf[seq_num+2] == 'N' or kinase_inf[seq_num+2] == 'Q' or kinase_inf[seq_num+2] == 'S' or kinase_inf[seq_num+2] == 'T' or kinase_inf[seq_num+2] == 'Y'):
                        envP+= 1

                    if(kinase_inf[seq_num+3] == 'K' or kinase_inf[seq_num+3] == 'R'):
                        envB+= 1
                    if(kinase_inf[seq_num+3] == 'D' or kinase_inf[seq_num+3] == 'E'):
                        envA+= 1
                    if(kinase_inf[seq_num+3] == 'A' or kinase_inf[seq_num+3] == 'F' or kinase_inf[seq_num+3] == 'I' or kinase_inf[seq_num+3] == 'L' or kinase_inf[seq_num+3] == 'M' or kinase_inf[seq_num+3] == 'P' or kinase_inf[seq_num+3] == 'V' or kinase_inf[seq_num+3] == 'W'):
                        envNP+= 1    
                    if(kinase_inf[seq_num+3] == 'C' or kinase_inf[seq_num+3] == 'G' or kinase_inf[seq_num+3] == 'H' or kinase_inf[seq_num+3] == 'N' or kinase_inf[seq_num+3] == 'Q' or kinase_inf[seq_num+3] == 'S' or kinase_inf[seq_num+3] == 'T' or kinase_inf[seq_num+3] == 'Y'):
                        envP+= 1
                    
                    if(kinase_inf[seq_num-1] == 'K' or kinase_inf[seq_num-1] == 'R'):
                        envB+= 1
                    if(kinase_inf[seq_num-1] == 'D' or kinase_inf[seq_num-1] == 'E'):
                        envA+= 1
                    if(kinase_inf[seq_num-1] == 'A' or kinase_inf[seq_num-1] == 'F' or kinase_inf[seq_num-1] == 'I' or kinase_inf[seq_num-1] == 'L' or kinase_inf[seq_num-1] == 'M' or kinase_inf[seq_num-1] == 'P' or kinase_inf[seq_num-1] == 'V' or kinase_inf[seq_num-1] == 'W'):
                        envNP+= 1    
                    if(kinase_inf[seq_num-1] == 'C' or kinase_inf[seq_num-1] == 'G' or kinase_inf[seq_num-1] == 'H' or kinase_inf[seq_num-1] == 'N' or kinase_inf[seq_num-1] == 'Q' or kinase_inf[seq_num-1] == 'S' or kinase_inf[seq_num-1] == 'T' or kinase_inf[seq_num-1] == 'Y'):
                        envP+= 1

                    if(kinase_inf[seq_num-2] == 'K' or kinase_inf[seq_num-2] == 'R'):
                        envB+= 1
                    if(kinase_inf[seq_num-2] == 'D' or kinase_inf[seq_num-2] == 'E'):
                        envA+= 1
                    if(kinase_inf[seq_num-2] == 'A' or kinase_inf[seq_num-2] == 'F' or kinase_inf[seq_num-2] == 'I' or kinase_inf[seq_num-2] == 'L' or kinase_inf[seq_num-2] == 'M' or kinase_inf[seq_num-2] == 'P' or kinase_inf[seq_num-2] == 'V' or kinase_inf[seq_num-2] == 'W'):
                        envNP+= 1    
                    if(kinase_inf[seq_num-2] == 'C' or kinase_inf[seq_num-2] == 'G' or kinase_inf[seq_num-2] == 'H' or kinase_inf[seq_num-2] == 'N' or kinase_inf[seq_num-2] == 'Q' or kinase_inf[seq_num-2] == 'S' or kinase_inf[seq_num-2] == 'T' or kinase_inf[seq_num-2] == 'Y'):
                        envP+= 1  

                    if(kinase_inf[seq_num-3] == 'K' or kinase_inf[seq_num-3] == 'R'):
                        envB+= 1
                    if(kinase_inf[seq_num-3] == 'D' or kinase_inf[seq_num-3] == 'E'):
                        envA+= 1
                    if(kinase_inf[seq_num-3] == 'A' or kinase_inf[seq_num-3] == 'F' or kinase_inf[seq_num-3] == 'I' or kinase_inf[seq_num-3] == 'L' or kinase_inf[seq_num-3] == 'M' or kinase_inf[seq_num-3] == 'P' or kinase_inf[seq_num-3] == 'V' or kinase_inf[seq_num-3] == 'W'):
                        envNP+= 1    
                    if(kinase_inf[seq_num-3] == 'C' or kinase_inf[seq_num-3] == 'G' or kinase_inf[seq_num-3] == 'H' or kinase_inf[seq_num-3] == 'N' or kinase_inf[seq_num-3] == 'Q' or kinase_inf[seq_num-3] == 'S' or kinase_inf[seq_num-3] == 'T' or kinase_inf[seq_num-3] == 'Y'):
                        envP+= 1

                    if j != (len(Array_85)-1):
                        if(float(envB) == 0 ):
                            Protein_feature.append('0,0,0,')
                        elif(float(envB)  >= 1  and float(envB) <= 2):    
                            Protein_feature.append('0,0,1,')
                        elif(float(envB)  >= 3  and float(envB) <= 4):
                            Protein_feature.append('0,1,1,')
                        elif(float(envB) >= 5):
                            Protein_feature.append('1,1,1,')

                        if(float(envA) == 0 ):
                            Protein_feature.append('0,0,0,')
                        elif(float(envA)  >= 1  and float(envA) <= 2):    
                            Protein_feature.append('0,0,1,')
                        elif(float(envA)  >= 3  and float(envA) <= 4):
                            Protein_feature.append('0,1,1,')
                        elif(float(envA) >= 5):
                            Protein_feature.append('1,1,1,')

                        if(float(envNP) == 0 ):
                            Protein_feature.append('0,0,0,')
                        elif(float(envNP)  >= 1  and float(envNP) <= 2):    
                            Protein_feature.append('0,0,1,')
                        elif(float(envNP)  >= 3  and float(envNP) <= 4):
                            Protein_feature.append('0,1,1,')
                        elif(float(envNP) >= 5):
                            Protein_feature.append('1,1,1,')

                        if(float(envP) == 0 ):
                            Protein_feature.append('0,0,0,')
                        elif(float(envP)  >= 1  and float(envP) <= 2):    
                            Protein_feature.append('0,0,1,')
                        elif(float(envP)  >= 3  and float(envP) <= 4):
                            Protein_feature.append('0,1,1,')
                        elif(float(envP) >= 5):
                            Protein_feature.append('1,1,1,')
                    else:
                        if(float(envB) == 0 ):
                            Protein_feature.append('0,0,0,')
                        elif(float(envB)  >= 1  and float(envB) <= 2):    
                            Protein_feature.append('0,0,1,')
                        elif(float(envB)  >= 3  and float(envB) <= 4):
                            Protein_feature.append('0,1,1,')
                        elif(float(envB) >= 5):
                            Protein_feature.append('1,1,1,')

                        if(float(envA) == 0 ):
                            Protein_feature.append('0,0,0,')
                        elif(float(envA)  >= 1  and float(envA) <= 2):    
                            Protein_feature.append('0,0,1,')
                        elif(float(envA)  >= 3  and float(envA) <= 4):
                            Protein_feature.append('0,1,1,')
                        elif(float(envA) >= 5):
                            Protein_feature.append('1,1,1,')

                        if(float(envNP) == 0 ):
                            Protein_feature.append('0,0,0,')
                        elif(float(envNP)  >= 1  and float(envNP) <= 2):    
                            Protein_feature.append('0,0,1,')
                        elif(float(envNP)  >= 3  and float(envNP) <= 4):
                            Protein_feature.append('0,1,1,')
                        elif(float(envNP) >= 5):
                            Protein_feature.append('1,1,1,')

                        if(float(envP) == 0 ):
                            Protein_feature.append('0,0,0\n')
                        elif(float(envP)  >= 1  and float(envP) <= 2):    
                            Protein_feature.append('0,0,1\n')
                        elif(float(envP)  >= 3  and float(envP) <= 4):
                            Protein_feature.append('0,1,1\n')
                        elif(float(envP) >= 5):
                            Protein_feature.append('1,1,1\n')
        for p in Protein_feature:
            wf.write(p)


