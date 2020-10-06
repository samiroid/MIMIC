
import sys
# BASE_PATH = "/content/drive/My Drive/collab/MIMIC/"
BASE_PATH = "/Users/samir/Dev/projects/MIMIC/MIMIC/"
INPUT_PATH = BASE_PATH+"/DATA/input/"
OUTPUT_PATH = BASE_PATH+"/DATA/results/"
TMP_PATH = BASE_PATH+"/DATA/processed/"
sys.path.append(BASE_PATH+"TADAT/") 
N_SEEDS=50
N_VAL_SEEDS = 10
N_VAL_RUNS = 10
PLOT_VARS=["auroc","auprc","sensitivity","specificity"]
model="BERT-POOL"


# %%
from datetime import datetime
import fnmatch
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from pdb import set_trace
import pandas as pd
import pickle
import pprint
import random
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sns
import warnings


#local
from tadat.pipeline import plots
from tadat.core import data, vectorizer, features, helpers, embeddings, berter, transformer_lms

warnings.filterwarnings("ignore")
sns.set(style="darkgrid")


# %%
def read_cache(path):
    X = None
    try:
        with open(path, "rb") as fi:            
            X = pickle.load(fi)
    except FileNotFoundError:
        pass
    return X

def write_cache(path, o):
    with open(path, "wb") as fo:
        pickle.dump(o, fo)

def clear_cache(cache_path, model="*", dataset="*", group="*", ctype="*"):
    assert ctype in ["*","res*","feats"]
    file_paths = os.listdir(cache_path)
    pattern = "{}_{}_{}_*_{}.pkl".format(dataset, model, group, ctype).lower()
    for fname in file_paths:
        if fnmatch.fnmatch(fname, pattern):
            os.remove(cache_path+"/"+fname)
            print("cleared file: {}".format(fname))      

def plot_densities(df, ax, title):
    ax.set_title(title)
    for y in PLOT_VARS:        
        try:
            df.plot.kde(ax=ax, x="seed", y=y)
        except:
            pass
    
def plot_performance(df, title):
    #plots
    fig, ax = plt.subplots(1,1, figsize=(18,5))
#     plots.plot_df(df=df,ax=ax,x="seed",ys=["auroc","auprc","sensitivity","specificity"], annotation_size=10)    
    fig.suptitle(title ,y=1.02)
    plot_densities(df, ax, "") 
#     ax[0].legend(loc='best')
    ax.legend(loc='best')
    plt.tight_layout()

def plot_scatter_metrics(df, title):
    n_rows=2
    n_cols = 3
    mets = list(itertools.combinations(PLOT_VARS,2))
    fig, ax = plt.subplots(n_rows, n_cols,  figsize=(16,9))
    #current coloramap
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    coords = list(itertools.product(range(n_rows),range(n_cols)))
    for m,col,coord in zip(mets, cmap, coords ):
        df.plot.scatter(x=m[0],y=m[1],c=col, ax=ax[coord[0]][coord[1]])
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()  

def plot_scatter_performance(df, title):
    n_rows=2
    n_cols = 2
    mets = [[x+"_delta", x] for x in PLOT_VARS] 
    fig, ax = plt.subplots(n_rows, n_cols,  figsize=(16,9))
    #current coloramap
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    coords = list(itertools.product(range(n_rows),range(n_cols)))
    #get absolute values for the deltas
    for m in PLOT_VARS:
        df[m+"_delta"] = df[m+"_delta"].abs()
    for m,col,coord in zip(mets, cmap, coords ):
        df.plot.scatter(x=m[0],y=m[1],c=col,
                        ax=ax[coord[0]][coord[1]])
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()  

    
def read_dataset(path, dataset_name):
    df_patients = pd.read_csv(path+"patients.csv", 
                              sep="\t", header=0).drop(columns=["TEXT"])
    df_train = pd.read_csv("{}/{}_train.csv".format(path, dataset_name), 
                           sep="\t", header=0)
    df_test  = pd.read_csv("{}/{}_test.csv".format(path, dataset_name),
                           sep="\t", header=0)
    df_val   = pd.read_csv("{}/{}_val.csv".format(path, dataset_name),
                           sep="\t", header=0)
    #set indices
    df_patients.set_index("SUBJECT_ID", inplace=True)
    df_train.set_index("SUBJECT_ID", inplace=True)
    df_test.set_index("SUBJECT_ID", inplace=True)
    df_val.set_index("SUBJECT_ID", inplace=True)

    df_train = df_train.join(df_patients, on="SUBJECT_ID", 
                             how="inner", lsuffix="N_").reset_index()
    df_test = df_test.join(df_patients, on="SUBJECT_ID", 
                           how="inner", lsuffix="N_").reset_index()
    df_val = df_val.join(df_patients, on="SUBJECT_ID", 
                         how="inner", lsuffix="N_").reset_index()

    return df_train, df_test, df_val    


# %%
def get_features(data, vocab_size, feature_type, word_vectors=None):
    if feature_type == "BOW-BIN":
        X = features.BOW(data, vocab_size,sparse=True)
    elif feature_type == "BOW-FREQ":
        X = features.BOW_freq(data, vocab_size,sparse=True)
    elif feature_type == "BOE-BIN":
        X = features.BOE(data, word_vectors,"bin")
    elif feature_type == "BOE-SUM": 
        X = features.BOE(data, word_vectors,"sum")
    elif feature_type == "BERT-POOL":
        X_cls, X_pool =  transformer_lms.transformer_encode_batches(data, 
                                                                    batchsize=64, 
                                                                    device="cuda")
        X=X_pool
    elif feature_type == "BERT-CLS":
        X_cls, X_pool =  transformer_lms.transformer_encode_batches(data, 
                                                                    batchsize=64, 
                                                                    device="cuda")
        X=X_cls
    else:
        raise NotImplementedError
    return X

def extract_features(feature_type, data_path, cache_path):
    X = read_cache(cache_path+"feats_{}".format(feature_type))
    if X:
        print("[reading cached features]")
        subject_ids, X_feats = X
    else:
        df = pd.read_csv(data_path+"patients.csv", sep="\t", header=0)
        subject_ids = list(df["SUBJECT_ID"])
        docs = list(df["TEXT"])
        if "BERT" in feature_type:
            X_feats = get_features(docs, None, feature_type)
        else:
            X, word_vocab = vectorizer.docs2idx(docs)
            X_feats = get_features(X,len(word_vocab),feature_type)
        write_cache(cache_path+"feats_{}".format(feature_type), 
                    [subject_ids, X_feats])
    return subject_ids, X_feats

def vectorize(df_train, df_val, df_test, subject_ids, features, group_label, subgroup):
    #target subgroup vs others
    df_test_G = df_test[df_test[group_label] == subgroup]
    df_test_O = df_test[df_test[group_label] != subgroup]    
    print("{}: {} | others: {}".format(subgroup,
                                       len(df_test_G),len(df_test_O)))        
    #vectorize labels
    train_Y = df_train["Y"]
    val_Y = df_val["Y"]           
    test_Y = df_test["Y"]
    test_Y_G = df_test_G["Y"]
    test_Y_O = df_test_O["Y"]           
    label_vocab = vectorizer.get_labels_vocab(train_Y+test_Y+val_Y)
    train_Y,_ = vectorizer.label2idx(train_Y, label_vocab)
    val_Y,_ = vectorizer.label2idx(val_Y, label_vocab)
    test_Y,_ = vectorizer.label2idx(test_Y, label_vocab)
    test_Y_G,_ = vectorizer.label2idx(test_Y_G, label_vocab)
    test_Y_O,_ = vectorizer.label2idx(test_Y_O, label_vocab)
    #get the subject id indices
    train_idxs = [subject_ids.index(i) for i in list(df_train["SUBJECT_ID"])] 
    val_idxs = [subject_ids.index(i) for i in list(df_val["SUBJECT_ID"])] 
    test_idxs = [subject_ids.index(i) for i in list(df_test["SUBJECT_ID"])] 
    test_idxs_G = [subject_ids.index(i) for i in list(df_test_G["SUBJECT_ID"])] 
    test_idxs_O = [subject_ids.index(i) for i in list(df_test_O["SUBJECT_ID"])] 
    #slice the feature matrix to get the corresponding instances
    train_feats = features[train_idxs, :]
    val_feats = features[val_idxs, :]
    test_feats = features[test_idxs, :]
    test_feats_G = features[test_idxs_G, :]
    test_feats_O = features[test_idxs_O, :]  

    return train_feats, train_Y, val_feats, val_Y, test_feats, test_Y,            test_feats_G, test_Y_G, test_feats_O, test_Y_O, label_vocab

def run(data_path, dataset, feature_type, group_label, subgroup, cache_path=None):
   
    df_train, df_test, df_val = read_dataset(data_path, dataset)
    subject_ids, X_feats = extract_features(feature_type, data_path, cache_path)
    X = vectorize(df_train, df_val, df_test, subject_ids, X_feats, group_label, subgroup)
    train_feats, train_Y, val_feats, val_Y, test_feats, test_Y, test_feats_G, test_Y_G, test_feats_O, test_Y_O, label_vocab = X
    print("train/test set size: {}/{}".format(train_feats.shape[0], test_feats.shape[0]))
    #train/test classifier for each random seed
    random.seed(1) #ensure repeateable runs and leverage cache
    random_seeds = random.sample(range(0, 10000), N_SEEDS)
    results = []
    results_g = []
    results_o = []    
    for seed in random_seeds:        
        res_fname = "{}_{}_{}_{}_res{}.pkl".format(dataset, feature_type, 
                                                   group_label, subgroup, 
                                                   seed).lower()
        R=None
        #look for cached results
        if cache_path: R = read_cache(cache_path+res_fname)      
        
        if not R:
            model = SGDClassifier(loss="log", random_state=seed)
            model.fit(train_feats, train_Y)
            res = evaluate_classifier(model, test_feats, test_Y, 
                                      label_vocab, feature_type, seed)
            res_g = evaluate_classifier(model, test_feats_G, test_Y_G, 
                                        label_vocab, feature_type, seed)
            res_o = evaluate_classifier(model, test_feats_O, test_Y_O, 
                                        label_vocab, feature_type, seed)
            #cache results
            if cache_path: write_cache(cache_path+res_fname, [res, res_g, res_o])                
        else:
            print("loaded cached results | seed: {}".format(seed))
            res, res_g, res_o = R
        results.append(res)
        results_g.append(res_g)
        results_o.append(res_o)
    #dataframes 
    df_res = pd.DataFrame(results)    
    df_res_g = pd.DataFrame(results_g)
    df_res_o = pd.DataFrame(results_o)

    df_res_delta = df_res_g.sub(df_res_o.iloc[:,2:])
    df_res_delta["model"] = df_res_g["model"]
    df_res_delta["seed"] = df_res_g["seed"]   

    return df_res, df_res_g, df_res_o, df_res_delta #  results, results_g, results_o

def tune_SGD(train_X, train_Y, val_X, val_Y, label_vocab, feature_type, seeds, metric):
    best_model = None
    best_perf = 0
    runs = {}
    
    for seed in seeds:
        model = SGDClassifier(loss="log", random_state=seed)
        model.fit(train_X, train_Y)
        res = evaluate_classifier(model, val_X, val_Y, 
                                  label_vocab, feature_type, seed)
        perf = res[metric]
        runs[seed] = perf
        if perf > best_perf:
            best_perf = perf
            best_model = model
    return best_model, best_perf, runs
    
def run_tuning(data_path, dataset, feature_type, group_label, subgroup, cache_path=None):   
    df_train, df_test, df_val = read_dataset(data_path, dataset)
    subject_ids, X_feats = extract_features(feature_type, data_path, cache_path)
    X = vectorize(df_train, df_val, df_test, subject_ids, X_feats, group_label, subgroup)
    train_X, train_Y, val_X, val_Y, test_X, test_Y, test_X_G, test_Y_G, test_X_O, test_Y_O, label_vocab = X
    print("train/test set size: {}/{}".format(train_X.shape[0], test_X.shape[0]))
    #train/test classifier for each random seed

    random.seed(1) #ensure repeateable runs and leverage cache
    random_seeds = random.sample(range(0, 10000), N_VAL_SEEDS*N_VAL_RUNS)
    results = []
    results_g = []
    results_o = []    
    val_runs = []
    
    for i in range(N_VAL_RUNS):
        res_fname = "{}_{}_{}_{}_tuned_res{}.pkl".format(dataset, feature_type, 
                                                         group_label, subgroup, i).lower()
        R=None
        #look for cached results
        if cache_path: R = read_cache(cache_path+res_fname)                      
        if not R:
            seeds = random_seeds[i*N_VAL_SEEDS:(i+1)*N_VAL_SEEDS]
            model, perf, val_run = tune_SGD(train_X, train_Y, val_X, val_Y, label_vocab, 
                                            feature_type, seeds, "sensitivity")
            print("tuning")
            pprint.pprint(val_run)
            res = evaluate_classifier(model, test_X, test_Y, 
                                      label_vocab, feature_type, i)
            res_g = evaluate_classifier(model, test_X_G, test_Y_G, 
                                        label_vocab, feature_type, i)
            res_o = evaluate_classifier(model, test_X_O, test_Y_O, 
                                        label_vocab, feature_type, i)
            #cache results
            if cache_path: write_cache(cache_path+res_fname, [res, res_g, res_o])                
        else:
            print("loaded cached results | run: {}".format(i))
            res, res_g, res_o = R
        results.append(res)
        results_g.append(res_g)
        results_o.append(res_o)
    #dataframes 
    df_res = pd.DataFrame(results)    
    df_res_g = pd.DataFrame(results_g)
    df_res_o = pd.DataFrame(results_o)

    df_res_delta = df_res_g.sub(df_res_o.iloc[:,2:])
    df_res_delta["model"] = df_res_g["model"]
    df_res_delta["seed"] = df_res_g["seed"]   

    return df_res, df_res_g, df_res_o, df_res_delta 


def evaluate_classifier(model, X_test, Y_test,
                   labels, model_name, random_seed, res_path=None):
    Y_hat = model.predict(X_test)
    Y_hat_prob = model.predict_proba(X_test)
    #get probabilities for the positive class
    Y_hat_prob = Y_hat_prob[:,labels[1]]    
    microF1 = f1_score(Y_test, Y_hat, average="micro") 
    macroF1 = f1_score(Y_test, Y_hat, average="macro") 
    try:
        aurocc = roc_auc_score(Y_test, Y_hat_prob)
    except ValueError:
        aurocc = 0
    try:
        prec, rec, thresholds = precision_recall_curve(Y_test, Y_hat)
        auprc = auc(rec, prec)
    except ValueError:
        auprc = 0
    try:
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_hat).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (fn+tp)
    except ValueError:
        specificity, sensitivity = 0, 0
    
    res = {"model":model_name, 
            "seed":random_seed,    
            "microF1":round(microF1,3),
            "macroF1":round(macroF1,3),
            "auroc":round(aurocc,3),
            "auprc":round(auprc,3),
            "specificity":round(specificity,3),
            "sensitivity":round(sensitivity,3)           
            }

    if res_path is not None:    
        helpers.save_results(res, res_path, sep="\t")
    return res

# %% [markdown]
# # Analyses
# %% [markdown]
# ## Ethnicity 

# %%
def ethnicity_plot_deltas(df_delta_W,df_delta_N,df_delta_A,df_delta_H, title):
    df_delta = pd.concat([df_delta_W,df_delta_N,df_delta_A,df_delta_H])    
    #transform results into "long format"
    df_delta_long = df_delta.melt(id_vars=["seed","model","group"], 
                                  value_vars=PLOT_VARS, 
                                  var_name="metric", value_name="delta")
    g = sns.catplot(x="metric", y="delta", data=df_delta_long, 
                    col="group",sharey=True,legend=False)
    ax1, ax2, ax3, ax4 = g.axes[0]
    ax1.axhline(0, ls='--',c="r")
    ax2.axhline(0, ls='--',c="r")
    ax3.axhline(0, ls='--',c="r")
    ax4.axhline(0, ls='--',c="r")
    lim = max(df_delta_long["delta"].abs()) + 0.05
    ax1.set_ylim([-lim,lim])
    ax2.set_ylim([-lim,lim])
    ax3.set_ylim([-lim,lim])
    ax4.set_ylim([-lim,lim])
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()  

def ethnicity_plot_densities(df_W, df_N, df_A, df_H, title):
    #plots
    fig, ax = plt.subplots(1,4, sharey=True, sharex=True, figsize=(18,5))
    plot_densities(df_W, ax[0], "White")
    plot_densities(df_N, ax[1], "Black")
    plot_densities(df_A, ax[2], "Asian")
    plot_densities(df_H, ax[3], "Hispanic")
    fig.suptitle(title,  y=1.02)
    plt.tight_layout()

def ethnicity_plots(df_res, df_res_W, df_res_N, df_res_A, df_res_H, df_res_delta_W, 
                      df_res_delta_N,df_res_delta_A, df_res_delta_H, title):
    plot_performance(df_res, title)
    ethnicity_plot_densities(df_res_W,df_res_N, 
                             df_res_A,df_res_H,title)
    ethnicity_plot_deltas(df_res_delta_W, df_res_delta_N,
                          df_res_delta_A,df_res_delta_H, title)


def ethnicity_analysis(data_path, dataset, feature_type, results_path, cache_path=None, plots=True, tune=False):
    if tune:
        df_res, df_res_W, df_res_W_O, df_res_delta_W = run_tuning(data_path, dataset, feature_type, 
                                                  "ETHNICITY", "WHITE", 
                                                  cache_path=cache_path)
        _, df_res_N, df_res_N_O, df_res_delta_N = run_tuning(data_path, dataset, feature_type, 
                                                  "ETHNICITY", "BLACK", 
                                                  cache_path=cache_path)
        _, df_res_A, df_res_A_O, df_res_delta_A  = run_tuning(data_path, dataset, feature_type, 
                                                  "ETHNICITY", "ASIAN", 
                                                  cache_path=cache_path)
        _, df_res_H, df_res_H_O, df_res_delta_H  = run_tuning(data_path, dataset, feature_type, 
                                                  "ETHNICITY","HISPANIC", 
                                                  cache_path=cache_path)
    else:
        df_res, df_res_W, df_res_W_O, df_res_delta_W = run(data_path, dataset, feature_type, 
                                              "ETHNICITY", "WHITE", 
                                              cache_path=cache_path)
        _, df_res_N, df_res_N_O, df_res_delta_N = run(data_path, dataset, feature_type, 
                                                  "ETHNICITY", "BLACK", 
                                                  cache_path=cache_path)
        _, df_res_A, df_res_A_O, df_res_delta_A  = run(data_path, dataset, feature_type, 
                                                  "ETHNICITY", "ASIAN", 
                                                  cache_path=cache_path)
        _, df_res_H, df_res_H_O, df_res_delta_H = run(data_path, dataset, feature_type, 
                                                  "ETHNICITY","HISPANIC", 
                                                  cache_path=cache_path)
    df_res_delta_W["group"] = ["White v Others"]*len(df_res_delta_W)
    df_res_delta_N["group"] = ["Black v Others"]*len(df_res_delta_N)
    df_res_delta_A["group"] = ["Asian v Others"]*len(df_res_delta_A)
    df_res_delta_H["group"] = ["Hispanic v Others"]*len(df_res_delta_H)
    if tune:
        title="{} x ethnicity x {} (tuned)".format(dataset, feature_type).lower()        
        fname = "{}_{}_ethnicity_all_tuned_res.pkl".format(dataset, 
                                                     feature_type).lower()
    else:
        #save results
        title="{} x ethnicity x {}".format(dataset, feature_type).lower()        
        fname = "{}_{}_ethnicity_all_res.pkl".format(dataset, 
                                                     feature_type).lower()
    with open(results_path+fname, "wb") as fo:
        pickle.dump([df_res, df_res_W, df_res_N, df_res_A, df_res_H, df_res_delta_W, 
                     df_res_delta_N,df_res_delta_A, df_res_delta_H, title], fo)
    if plots:
        ethnicity_plots(df_res, df_res_W, df_res_N, df_res_A, df_res_H, 
                          df_res_delta_W, df_res_delta_N,df_res_delta_A, df_res_delta_H, title)

# %% [markdown]
# ## Ethnicity Binary

# %%
def ethnicity_binary_plot_deltas(df_delta_W,df_delta_N, title):
    df_delta = pd.concat([df_delta_W,df_delta_N])    
    #transform results into "long format"
    df_delta_long = df_delta.melt(id_vars=["seed","model","group"], 
                                  value_vars=PLOT_VARS, var_name="metric", 
                                  value_name="delta")

    g = sns.catplot(x="metric", y="delta", data=df_delta_long, 
                    col="group",sharey=True,legend=False)
    ax1, ax2 = g.axes[0]
    ax1.axhline(0, ls='--',c="r")
    ax2.axhline(0, ls='--',c="r")
    lim = max(df_delta_long["delta"].abs()) + 0.05
    ax1.set_ylim([-lim,lim])
    ax2.set_ylim([-lim,lim])
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()  
    
def ethnicity_binary_plot_densities(df_W, df_N, title):
    #plots
    fig, ax = plt.subplots(1,2, sharey=True, sharex=True, figsize=(18,5))
    plot_densities(df_W, ax[0], "White")
    plot_densities(df_N, ax[1], "Non-White")
    fig.suptitle(title ,y=1.02)
    plt.tight_layout()

def ethnicity_binary_plots(df_res, df_res_W, df_res_N, df_res_delta_W,
                           df_res_delta_N, title):
    plot_performance(df_res, title)
    ethnicity_binary_plot_densities(df_res_W,df_res_N,title)
    ethnicity_binary_plot_deltas(df_res_delta_W, df_res_delta_N, title)
    
def ethnicity_binary_analysis(data_path, dataset, feature_type, results_path,
                              cache_path=None, plots=True, tune=False):
    if tune:        
        df_res, df_res_W, df_res_W_O, df_res_delta_W = run_tuning(data_path, dataset, feature_type, 
                                                  "ETHNICITY_BINARY", 
                                                  "WHITE", cache_path=cache_path)
        df_res, df_res_N, df_res_N_O, df_res_delta_N = run_tuning(data_path, dataset, feature_type, 
                                                  "ETHNICITY_BINARY", 
                                                  "NON-WHITE", cache_path=cache_path)
    else:
        df_res, df_res_W, df_res_W_O, df_res_delta_W = run(data_path, dataset, feature_type, 
                                                  "ETHNICITY_BINARY", 
                                                  "WHITE", cache_path=cache_path)
        df_res, df_res_N, df_res_N_O, df_res_delta_N = run(data_path, dataset, feature_type, 
                                                  "ETHNICITY_BINARY", 
                                                  "NON-WHITE", cache_path=cache_path)       
        
    df_res_delta_W["group"] = ["White v Others"]*len(df_res_delta_W)
    df_res_delta_N["group"] = ["Non-White v Others"]*len(df_res_delta_N)
    
    if tune:
        title="{} x ethnicity-binary x {} (tuned)".format(dataset, feature_type).lower()
        fname = "{}_{}_ethnicity_binary_all_tuned_res.pkl".format(dataset, feature_type).lower()
    else:
        #save results
        title="{} x ethnicity-binary x {}".format(dataset, feature_type).lower()
        fname = "{}_{}_ethnicity_binary_all_res.pkl".format(dataset, feature_type).lower()
    with open(results_path+fname, "wb") as fo:
        pickle.dump([df_res, df_res_W, df_res_N, df_res_delta_W, 
                     df_res_delta_N, title], fo)

    if plots:
        ethnicity_binary_plots(df_res, df_res_W, df_res_N, 
                               df_res_delta_W, df_res_delta_N,title)
    
    

# %% [markdown]
# ## Gender 

# %%
def gender_plot_deltas(df_delta, title):
    #transform results into "long format"
    df_delta_long = df_delta.melt(id_vars=["seed","model"], value_vars=PLOT_VARS, 
                                        var_name="metric", value_name="delta")
    
    lim = max(df_delta_long["delta"].abs()) + 0.05
    g = sns.catplot(x="metric", y="delta",  data=df_delta_long,
                    sharey=True,legend=False)
    ax1 = g.axes[0][0]
    ax1.axhline(0, ls='--',c="r")
    ax1.set_ylim([-lim,lim])
    plt.suptitle(title,y=1.02)
    plt.tight_layout()
    plt.show()  

def gender_plot_densities(df_M, df_F, title):
    #plots
    fig, ax = plt.subplots(1,2, sharey=True, sharex=True, figsize=(18,5))
    plot_densities(df_M, ax[0], "Male") 
    plot_densities(df_F, ax[1], "Female") 
    fig.suptitle(title, y=1.02)
    plt.tight_layout()

def gender_plots(df_res, df_res_M, df_res_F, df_res_delta, title):
    plot_performance(df_res, title)
    gender_plot_densities(df_res_M, df_res_F, title)
    gender_plot_deltas(df_res_delta, title)    

def gender_analysis(data_path, dataset, feature_type, results_path, cache_path=None, plots=True, tune=False):
    if tune:
        df_res, df_res_M, df_res_F, df_res_delta = run_tuning(data_path, dataset, feature_type, 
                                            "GENDER", "M", cache_path=cache_path)
    else:
        df_res, df_res_M, df_res_F, df_res_delta = run(data_path, dataset, feature_type, 
                                            "GENDER", "M", cache_path=cache_path)
    #save results
    if tune:
        title="{} x gender x {} (tuned)".format(dataset, feature_type).lower()
        fname = "{}_{}_gender_all_tuned_res.pkl".format(dataset, feature_type).lower()
    else:
        title="{} x gender x {}".format(dataset, feature_type).lower()
        fname = "{}_{}_gender_all_res.pkl".format(dataset, feature_type).lower()
    with open(results_path+fname, "wb") as fo:
        pickle.dump([df_res, df_res_M, df_res_F, df_res_delta, title], fo)        

    if plots:
        gender_plots(df_res, df_res_M, df_res_F, df_res_delta, title)
    

# %% [markdown]
# # Run

# %%
N_TASKS = 9
def run_analyses(data_path, dataset, feature_type, results_path, 
                 cache_path, clear_results=False, tune=False, plots=False):    

    if clear_results:
        clear_cache(cache_path, model=model, dataset=dataset, ctype="res*")
    
    gender_analysis(data_path, dataset, model, results_path, 
                    cache_path, plots=plots, tune=tune)

    ethnicity_binary_analysis(data_path, dataset, model, results_path, 
                              cache_path, plots=plots, tune=tune)
    
    ethnicity_analysis(data_path, dataset, model, results_path, 
                       cache_path, plots=plots, tune=tune)

#Run All the tasks
def task_done(path,  task):
    with open(path+"completed_tasks.txt", "a") as fod:
        fod.write(task+"\n")

def reset_tasks(path):
    with open(path+"completed_tasks.txt", "w") as fod:
        fod.write("")

def is_task_done(path,  task):
    try:
        with open(path+"completed_tasks.txt", "r") as fid:
            tasks = fid.read().split("\n")            
        return task in set(tasks)
    except FileNotFoundError:
        #create file if not found
        with open(path+"completed_tasks.txt", "w") as fid:
            fid.write("")
        return False

def run_tasks(data_path, tasks_fname, cache_path, results_path, mini_tasks=True, reset=False):
    #if reset delete the completed tasks file
    if reset: reset_tasks(cache_path)
    
    with open(data_path+tasks_fname,"r") as fid:
        for i,l in enumerate(fid):
            if i > N_TASKS: break
            fname, task_name = l.strip("\n").split(",")            
            dataset = "mini-"+fname if mini_tasks else fname
            # dataset = fname
            if is_task_done(cache_path, dataset): 
                print("[dataset: {} already processed]".format(dataset))
                continue                        
            print("******** {} {} ********".format(task_name, dataset))      
            run_analyses(data_path, dataset, model, results_path, cache_path, clear_results=False)
            task_done(cache_path, dataset)

def plot_tasks(data_path, tasks_fname, results_path, mini_tasks=True):
    with open(data_path+tasks_fname,"r") as fid:        
        for i,l in enumerate(fid):
            if i > N_TASKS: break
            fname, task_name = l.strip("\n").split(",")
            dataset = "mini-"+fname if mini_tasks else fname
            plot_analyses(results_path, dataset, model, task_name)


def plot_analyses(cache_path, dataset, model_name, title, tuned=False):
    file_paths = os.listdir(cache_path)
    if tuned:
        pattern = "{}_{}_*_all_tuned_res.pkl".format(dataset, model_name).lower()
    else:        
        pattern = "{}_{}_*_all_res.pkl".format(dataset, model_name).lower()
    print("\n\n{} {} {}\n".format("*"*30, title, "*"*30))
    for fname in file_paths:
        if fnmatch.fnmatch(fname, pattern):
            R = list(read_cache(cache_path+fname))
            if "gender" in fname:
                gender_plots(*R)
                print("-"*100)
            elif "ethnicity_binary" in fname:
                ethnicity_binary_plots(*R)                
                print("-"*100)
            elif "ethnicity" in fname:
                ethnicity_plots(*R)
                print("-"*100)
    print("*"*100)

def metric_scatter_plots(cache_path, dataset, model_name, title):
    file_paths = os.listdir(cache_path)
    pattern = "{}_{}_*_all_res.pkl".format(dataset, model_name).lower()
    print("\n\n{} {} {}\n".format("*"*30, title, "*"*30))
    for fname in file_paths:
        if fnmatch.fnmatch(fname, pattern):
            if "gender" in fname:
                df_res, df_res_M, df_res_F, df_res_delta, title = list(read_cache(cache_path+fname))
                plot_scatter_metrics(df_res_delta, title)
                print("-"*100)
            elif "ethnicity_binary" in fname:
                df_res, df_res_W, df_res_N, df_res_delta_W, df_res_delta_N, title = list(read_cache(cache_path+fname))
                plot_scatter_metrics(df_res_delta_W, title + " (White)")
                plot_scatter_metrics(df_res_delta_N, title + " (Non-White)")
                print("-"*100)
            elif "ethnicity" in fname:
                R = list(read_cache(cache_path+fname))
                df_res, df_res_W, df_res_N, df_res_A, df_res_H, df_res_delta_W, df_res_delta_N,df_res_delta_A, df_res_delta_H, title = R 
                plot_scatter_metrics(df_res_delta_W, title + " (White)")
                plot_scatter_metrics(df_res_delta_N, title + " (Black)")
                plot_scatter_metrics(df_res_delta_H, title + " (Hispanic)")
                plot_scatter_metrics(df_res_delta_A, title + " (Asian)")
                print("-"*100)
            
    print("*"*100)

def performance_scatter_plots(cache_path, dataset, model_name, title):
    file_paths = os.listdir(cache_path)
    pattern = "{}_{}_*_all_res.pkl".format(dataset, model_name).lower()
    print("\n\n{} {} {}\n".format("*"*30, title, "*"*30))
    for fname in file_paths:
        if fnmatch.fnmatch(fname, pattern):
            if "gender" in fname:
                df_res, df_res_M, df_res_F, df_res_delta, title = list(read_cache(cache_path+fname))
                df = df_res.merge(df_res_delta, on=["model","seed"],
                                  suffixes=[None, "_delta"])
                plot_scatter_performance(df, title)
                print("-"*100)
            elif "ethnicity_binary" in fname:
                df_res, df_res_W, df_res_N, df_res_delta_W, df_res_delta_N, title = list(read_cache(cache_path+fname))
                df_W = df_res_W.merge(df_res_delta_W, on=["model","seed"], 
                                      suffixes=[None, "_delta"])
                df_N = df_res_N.merge(df_res_delta_N, on=["model","seed"],
                                      suffixes=[None, "_delta"])
                plot_scatter_performance(df_W, title + " (White)")
                plot_scatter_performance(df_N, title + " (Non-White)")
                print("-"*100)
            elif "ethnicity" in fname:
                R = list(read_cache(cache_path+fname))
                df_res, df_res_W, df_res_N, df_res_A, df_res_H, df_res_delta_W, df_res_delta_N, df_res_delta_A, df_res_delta_H, title = R 
                
                df_W = df_res_W.merge(df_res_delta_W, on=["model","seed"],
                                      suffixes=[None, "_delta"])
                df_N = df_res_N.merge(df_res_delta_N, on=["model","seed"], 
                                      suffixes=[None, "_delta"])
                df_H = df_res_H.merge(df_res_delta_H, on=["model","seed"], 
                                      suffixes=[None, "_delta"])
                df_A = df_res_A.merge(df_res_delta_A, on=["model","seed"],
                                      suffixes=[None, "_delta"])
                
                plot_scatter_performance(df_W, title + " (White)")
                plot_scatter_performance(df_N, title + " (Black)")
                plot_scatter_performance(df_H, title + " (Hispanic)")
                plot_scatter_performance(df_A, title + " (Black)")                
                print("-"*100)
            
    print("*"*100)

    