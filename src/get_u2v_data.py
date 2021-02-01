import argparse
import pandas as pd
from ipdb import set_trace
import os
import sys

def get_args():
    par = argparse.ArgumentParser(description="extract data to train user embeddings")    
    par.add_argument('-input_file', type=str, required=True, help='input file path')       
    par.add_argument('-output_folder', type=str, required=True, help='output folder path')       
    return par.parse_args()  	

def main(input_file, output_folder):
    if not os.path.exists(os.path.dirname(output_folder+"/u2v/")):
	    os.makedirs(os.path.dirname(output_folder+"/u2v/"))

    df = pd.read_csv(input_file, sep="\t", header=0)
    with open(output_folder+"/u2v/users.txt", "w") as f:
        for i in range(len(df)):
            sys.stdout.write("\ri: {}".format(i))
            sys.stdout.flush()            
            patient = df.iloc[i]["SUBJECT_ID"]
            #sentences are separated by \t 
            try:
                note = df.iloc[i]["TEXT"].split("\t")
            except: 
                print("\nIgnored line: ")
                print(df.iloc[i])
                print()
                continue            
            for sent in note:                
                s = sent.strip()
                if len(s) > 2:
                    f.write("{}\t{}\n".format(patient, s))
            

if __name__ == "__main__":
    args = get_args()
    main(args.input_file, args.output_folder)