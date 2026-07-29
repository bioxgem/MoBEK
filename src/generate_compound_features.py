import glob
import rdkit
import os
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem #Chem.MolFromSmiles 輸入smile 輸出 mol檔案
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

#2022/10/03/Lin
import csv
import shutil

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('-t', '--fp_type', type=str, action='store', default='all', help='Fingerprint type that user defined.(maccs / ecfp / all/ checkmol)')

parser.add_argument('-checkmol', '--checkmol', type=str, action='store', default='y', help='Generate Checkmol Fingerprint type. ( y / n )')
parser.add_argument('-pubchem' , '--pubchem' , type=str, action='store', default='y', help='Generate PubChem Fingerprint type. ( y / n )')
parser.add_argument('-inhouse' , '--inhouse' , type=str, action='store', default='n', help='Generate PubChem Fingerprint type. ( y / n )')
parser.add_argument('-ring'    , '--ring'    , type=str, action='store', default='y', help='Generate PubChem Fingerprint type. ( y / n )')
parser.add_argument('-maccs'   , '--maccs'   , type=str, action='store', default='n', help='Generate MACCS Fingerprint type. ( y / n )')
parser.add_argument('-ecfp'    , '--ecfp'    , type=str, action='store', default='y', help='Generate ECFP Fingerprint type. ( y / n )')
parser.add_argument('-ac'    , '--ac'    , type=str, action='store', default='y', help='Generate AC Fingerprint type. ( y / n )')

parser.add_argument('-f'       , '--input_format'    , type=str, action='store', default='smiles', help='Format of input files. (smile / sdf)')
parser.add_argument('-i'       , '--input_jobid'  , type=str, action='store', help='Job id of input directory.')
parser.add_argument('-p'       , '--input_path'  , type=str, action='store',default='./', help='input directory.')

#2022/10/03/Lin
parser.add_argument('-Num'       , '--output_number'  , type=str, action='store',default='', help='output file number.')

#parser.add_argument('-o', '--output_dir_path', type=str, action='store', default='./test_output', help='Path of output directory.')
#parser.add_argument('-ce', '--output_csv_ecfp', type=str, action='store', default='all_ECFP4_fea.csv', help='File name of the output ECFP4 CSV file.')
#parser.add_argument('-cm', '--output_csv_maccs', type=str, action='store', default='all_MACCS_fea.csv', help='File name of the output MACCS CSV file.')
args = parser.parse_args()


#FOR WEBSITE by wei4r
#input_dir_path = "/data/HuangYuwei/BioXGEM-drug/Compound/Compound_Feature/data_tmp/"+args.input_jobid
#print("input paht: "+args.input_jobid)

input_dir_path = args.input_jobid

#Check whether the output directory is already exist or not
#if not os.path.exists(args.output_dir_path):
#    os.makedirs(args.output_dir_path)

print("Input file format : " + args.input_format)
#print("Input Job ID : " + args.input_jobid)
print("Checkmol : " + args.checkmol)
print("PubChem : " + args.pubchem)
print("In-house : " + args.inhouse)
print("Ring in drug : " + args.ring)
print("MACCS : " + args.maccs)
print("ECFP : " + args.ecfp)
print("AC : " + args.ac)

#input_dir_path = './data_tmp/' + args.input_jobid
input_dir_path = args.input_path


script_dir = Path(__file__).resolve().parent
resource_dir = script_dir / "generate_feature" / "moieties"
tool_dir = resource_dir
pubchem_moieties = sorted((resource_dir / "pubchem").glob("*.mol"))
inhouse_moieties = sorted((resource_dir / "inhouse").glob("*.mol"))
ring_moieties = sorted((resource_dir / "rings_in_drugs").glob("*.mol"))


#If the user input the argument -F with 'smiles' which means there shall be a txt file contains lines of smiles code in the input file.
if args.input_format == 'smiles':
    #file input and preproccessing
    #os.path.join(input_dir_path, 'compound_list.txt'))
    file = open(os.path.join(input_dir_path, 'compound_list.txt'))
    input_smiles_list = file.read().strip().split("\n") #strip()  去除首尾空格


    #check if compound file directory exist and make directory if it wasn't.
    if not os.path.exists(os.path.join(input_dir_path,'compound_file')):
        print('mkdir ' + os.path.join(input_dir_path,'compound_file'))
        os.mkdir(os.path.join(input_dir_path,'compound_file'))

    output_lines_checkmol = []
    output_lines_pubchem = []
    output_lines_inhouse = []
    output_lines_ring = []
    output_lines_ac = []

    compound_name_list = []
    fea_list = []

    pop_smile_list = []

    count_num = 0
    #get every smiles code's morgan fingerprint
    for input_smile in input_smiles_list:
        count_num += 1
        print(str(count_num) + " /" + str(len(input_smiles_list)), end="\n")
        
        mol = Chem.MolFromSmiles(input_smile)

        # if len(input_smile) > 200:
        #     pop_smile_list.append(input_smile)
        #     print("GG↑\n")
        #     continue

        ##print(input_smile,'\n',len(input_smile))
        #print(count_num)

        molblock = Chem.MolToMolBlock(mol) #讀取Chem.MolFromSmiles 產生的地址，輸出成mol檔
        input_smile_file = input_smile.replace("/", "_") #str.replace(old, new[, max])
        mol_path = os.path.join(input_dir_path, 'compound_file/' + 'test4' + '.mol')

        #img = Chem.Draw.MolToImageFile(mol , os.path.join(input_dir_path, 'compound_img/' + input_smile_file + '.png'), size=(500, 500))

        compound_name_list.append(input_smile)

        with open(mol_path, "w") as smiles2mol:
            smiles2mol.write(molblock)

        if args.ecfp == 'y':
            #mol = Chem.MolFromSmiles(input_smile)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=512)
            
            #convert into numpy array
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, fp_arr)

            if count_num == 1: # input_smiles_list.index(input_smile) == 0: 重複compound會有重複index!!!
                fp_matrix = fp_arr
            else:
                fp_matrix = np.vstack((fp_matrix, fp_arr))


        if args.maccs == 'y':
            fps = MACCSkeys.GenMACCSKeys(mol)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fps,fp_arr)
            #20221017 Lin
            fp_arr = np.delete(fp_arr, 0) #首項index = 0 無feature   
            if(fp_arr.sum != 0):
                fea_list.append(fp_arr)


        if args.checkmol == 'y':
            
            checkmol_tmp_array = np.zeros(204)

            result = subprocess.run([str(tool_dir / 'checkmol'), '-p', mol_path], stdout=subprocess.PIPE)
            output_cm_lines = result.stdout.splitlines()
            check = 0
            for output_cm_line in output_cm_lines:
                
                num_exp = output_cm_line.decode().split(':')    # num_exp[1] >= 1 -> have this feature
                fea_exp = num_exp[0].split('#')                 # fea_exp[1] -> feature num
                #print("check"+num_exp[0])
                #print(fea_exp)
                if(len(fea_exp) == 1):
                    check = 1
                    #a = np.empty(204,)
                    #a[:] = np.nan
                    #output_lines_checkmol.append(input_smile + ',' + ','.join([str(bit) for bit in a]) + '\n')
                    # print("1")
                else:
                    fea_num = int(fea_exp[1])
                    #print(fea_num)
                    if(int(num_exp[1]) >= 1):
                        checkmol_tmp_array[fea_num-1] = 1
                    # print("2")
            if check == 0:
                output_lines_checkmol.append(input_smile + ',' + ','.join([str(bit) for bit in checkmol_tmp_array]) + '\n')
                # print("3")
            else:

                # a = np.empty(204,)
                # a[:] = np.
                # output_lines_checkmol.append(input_smile + ',' + ','.join([str(bit) for bit in a]) + '\n')
                output_lines_checkmol.append(input_smile+',')
                for x in range(204):
                    if x != 203:
                        output_lines_checkmol.append('0,')
                    else:
                        output_lines_checkmol.append('0\n')
                # print("4")

        if args.ac == 'y':
            
            ac_tmp_array = np.zeros(10)

            result = subprocess.run([str(tool_dir / 'mod_ac'), mol_path], stdout=subprocess.PIPE)
            output_cm_lines = result.stdout.splitlines()
            # print(output_cm_lines)
            try: # AC 抓不到數字 不知為何有這個bug # 當compound 包含 離子的smiles時，會有問題 Ex: CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]
                num_exp = output_cm_lines[1].decode().split('\t')
            except:
                num_exp = np.zeros(1)

            if len(num_exp) == 11:
                output_lines_ac.append(input_smile + ',' + ','.join([str(bit) for bit in num_exp[1:]]) + '\n')
            else: # 例如 [Zn] 這種離子類型 AC 也會抓不到
                output_lines_ac.append(input_smile + ',' + ','.join([str(bit) for bit in ac_tmp_array]) + '\n')
            # print(len(num_exp))

        if args.pubchem == 'y':
            
            pubchem_tmp_array = np.zeros(168, int)

            for index, moiety in enumerate(pubchem_moieties):

                result = subprocess.run([str(tool_dir / 'matchmol'), str(moiety), mol_path], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    pubchem_tmp_array[index] = 1

            output_lines_pubchem.append(input_smile + ',' + ','.join([str(bit) for bit in pubchem_tmp_array]) + '\n')
            # print(output_lines_pubchem)

        if args.inhouse == 'y':
            
            inhouse_tmp_array = np.zeros(34, int)

            for index, moiety in enumerate(inhouse_moieties):

                result = subprocess.run([str(tool_dir / 'matchmol'), str(moiety), mol_path], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    inhouse_tmp_array[index] = 1

            output_lines_inhouse.append(input_smile + ',' + ','.join([str(bit) for bit in inhouse_tmp_array]) + '\n')

        
        if args.ring == 'y':
            
            ring_tmp_array = np.zeros(147, int)

            for index, moiety in enumerate(ring_moieties):

                result = subprocess.run([str(tool_dir / 'matchmol'), str(moiety), mol_path], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    ring_tmp_array[index] = 1

            output_lines_ring.append(input_smile + ',' + ','.join([str(bit) for bit in ring_tmp_array]) + '\n')


    for i in pop_smile_list:
        input_smiles_list.remove(i)


    if args.maccs == 'y':
        output_lines_maccs = []        
        for index, compound_fea in enumerate(fea_list):
            compound_fea_int = compound_fea.astype(int)
            output_lines_maccs.append(input_smiles_list[index]+','+','.join(map(str, compound_fea_int))+'\n')

        output_lines_file = open(os.path.join(input_dir_path, 'maccs.csv'), 'w')
        output_lines_file.writelines(output_lines_maccs)
        output_lines_file.close()

    if args.ecfp == 'y':
        #save all features and smiles code name into a dataframe
        #print(fp_matrix)
        if count_num == 1:
            fp_matrix = [fp_matrix]
        df_fp_all = pd.DataFrame(fp_matrix)
        df_isl = pd.DataFrame(input_smiles_list)
        #print(input_smiles_list)
        #print(df_fp_all)
        df_fp_all = pd.concat([df_isl, df_fp_all], axis=1)
        num_ecfp = 512
        header = ["smiles"] + [f"#ECFP_{i}" for i in range(1, num_ecfp + 1)]
        df_fp_all.to_csv(os.path.join(input_dir_path, 'ecfp.csv'), sep=',', index=False, header=header)
    if args.checkmol == 'y':
        #print(output_lines_checkmol)
        output_lines_file = open(os.path.join(input_dir_path, 'checkmol.csv'), 'w')
        output_line = ["smiles,"] + ["#Checkmol_" + str(i) + "," for i in range(1, 204+1)]
        output_line[-1] = "#Checkmol_204\n"
        output_lines_file.write(''.join(output_line))
        output_lines_file.writelines(output_lines_checkmol)
        output_lines_file.close()

    if args.pubchem == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'pubchem.csv'), 'w')
        output_line = ["smiles,"] + ["#Pubchem_" + str(i) + "," for i in range(1, 168+1)]
        output_line[-1] = "#Pubchem_168\n"
        output_lines_file.write(''.join(output_line))
        output_lines_file.writelines(output_lines_pubchem)
        output_lines_file.close()

    if args.inhouse == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'inhouse.csv'), 'w')
        output_lines_file.writelines(output_lines_inhouse)
        output_lines_file.close()

    if args.ring == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'ring.csv'), 'w')
        output_line = ["smiles,"] + ["#Ring_" + str(i) + "," for i in range(1, 147+1)]
        output_line[-1] = "#Ring_147\n"
        output_lines_file.write(''.join(output_line))
        output_lines_file.writelines(output_lines_ring)
        output_lines_file.close()
    
    if args.ac == 'y':
        #print(output_lines_checkmol)
        output_lines_file = open(os.path.join(input_dir_path, 'ac.csv'), 'w')
        output_lines_file.writelines(['smiles,', 'C.ring,', 'C.other,', 'N.ring,', 'N.other,', 'O.ring,', 'O.other,', 'P,', 'S,', 'X,', '#.of.Ring\n'])
        output_lines_file.writelines(output_lines_ac)
        output_lines_file.close()
