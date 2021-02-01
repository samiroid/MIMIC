import argparse
import pandas as pd
from ipdb import set_trace
import os
# import user2vec

MAX_SENT = 20

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
            user_id = df.iloc[i]["SUBJECT_ID"]
            #sentences are separated by \t 
            user_txt = df.iloc[i]["TEXT"].split()
            N = int(len(user_txt)/MAX_SENT)
            # print(user_id)
            # set_trace()
            for n in range(N):
                user_sent = " ".join(user_txt[n*MAX_SENT:(n+1)*MAX_SENT])
                f.write("{}\t{}\n".format(user_id, user_sent))
            # if i >=10: break
    # set_trace()

if __name__ == "__main__":
    args = get_args()
    main(args.input_file, args.output_folder)