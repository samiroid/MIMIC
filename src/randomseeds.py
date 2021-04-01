
from collections import defaultdict
from datetime import datetime
import fnmatch
import itertools
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.random import RandomState
import os
from pdb import set_trace
import pandas as pd
import pickle
import pprint
import random
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sns
import warnings
import torch
import uuid
import time
import copy
import ast
import sys
# sys.path.append("/content/drive/My Drive/collab/TADAT/") 
#local
# from tadat.pipeline import plots
# import tadat.core as core
import tadat

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# %% [markdown]
# ## Configs

# %%

# BASE_PATH = "/content/drive/My Drive/collab/MIMIC/"
BASE_PATH = "/Users/samir/Dev/projects/MIMIC/MIMIC_random_seeds/MIMIC/"
INPUT_PATH = BASE_PATH+"/DATA/input/"
FEATURES_PATH = BASE_PATH+"/DATA/features/"
OUTPUT_PATH = BASE_PATH+"/DATA/results/"
# TMP_PATH = BASE_PATH+"/DATA/processed/"

TUNE_OUTPUT_PATH = BASE_PATH+"/DATA/results_fine/"
TUNE_TMP_PATH = BASE_PATH+"/DATA/processed_fine/"

GRID_OUTPUT_PATH = BASE_PATH+"/DATA/results_grid/"
GRID_TMP_PATH = BASE_PATH+"/DATA/processed_grid/"

#configs
N_SEEDS=1000
N_VAL_SEEDS = 5
N_VAL_RUNS = 5
N_TASKS = 3
N_TASKS = 50
# PLOT_VARS=["auroc","auprc","sensitivity","specificity"]
PLOT_VARS=["auroc","sensitivity"]
MODEL="BERT-POOL"
METRIC = "auroc"

GROUPS = { "GENDER": ["M","F"],   
         "ETHNICITY": ["WHITE","BLACK","ASIAN","HISPANIC"]
}

CLASSIFIER = 'sklearn'
# CLASSIFIER = 'torch'
CLASSIFIER = 'mseq'
CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"


# %%

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex = True)

# %% [markdown]
# # Modeling 

# %%
def train_classifier(X_train, Y_train, X_val, Y_val, 
                     init_seed, shuffle_seed=None, input_dimension=None):    
    """ train a classifier
        X_train: training instances 
        Y_yrain: training labels
        X_val: validation instances
        Y_val: validation labels
        init_seed: parameter initialization seed
        shuffle_seed: data shuffling seed
        input_dimension: number of input features
        
        returns: fitted classifier
    """
    #CLASSIFIER is a global variable indicating the type of classifier
    if CLASSIFIER == "torch":        
        x = tadat.core.models.MyLinearModel(in_dim=input_dimension, out_dim=1, 
                    loss_fn=torch.nn.BCEWithLogitsLoss(), 
                    init_seed=init_seed, n_epochs=500, 
                    default_lr=0.01, batch_size=None, 
                    shuffle_seed=shuffle_seed, silent=True,
                    shuffle=True) 
        x.fit(X_train, Y_train, X_val, Y_val)
    elif CLASSIFIER == "mseq":        
        x = tadat.core.models.MultiBERTSeq(in_dim=input_dimension, out_dim=1, 
                    loss_fn=torch.nn.BCELoss(), 
                    init_seed=init_seed, n_epochs=500, 
                    default_lr=0.01, batch_size=None, 
                    shuffle_seed=shuffle_seed, silent=True,
                    shuffle=True) 
        x.fit(X_train, Y_train, X_val, Y_val)
    elif CLASSIFIER == "sklearn":
        x = SGDClassifier(loss="log", random_state=shuffle_seed)
        x.fit(X_train, Y_train)
    else:
        raise NotImplementedError
    return x

def evaluate_classifier(model, X_test, Y_test,
                   labels, model_name, random_seed, subgroup, res_path=None):
    """ evaluate a classifier
        model: classifier to be evaluated        
        X_test: test instances
        Y_test: test labels
        labels: label set
        model_name: model name
        random_seed: random seed that was used to train the classifier
        subgroup: demographic subgroup represented in the data
        res_path: path to save the results
        
        returns: dictionary of evaluation wrt to different metrics
    """
    Y_hat = model.predict(X_test)
    Y_hat_prob = model.predict_proba(X_test)
    #get probabilities for the positive class
    if CLASSIFIER == 'sklearn':
        Y_hat_prob = Y_hat_prob[:,labels[1]]    
    microF1 = f1_score(Y_test, Y_hat, average="micro") 
    macroF1 = f1_score(Y_test, Y_hat, average="macro") 
    try:
        aurocc = roc_auc_score(Y_test, Y_hat_prob)
    except ValueError:
        aurocc = 0
    try:
        prec, rec, thresholds = precision_recall_curve(Y_test, Y_hat_prob)       
        auprc = auc(rec, prec)
    except ValueError:
        auprc = 0
    try:
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_hat).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (fn+tp)
    except ValueError:
        specificity, sensitivity = 0, 0
    try:
        pr, tpr, thresholds = roc_curve(Y_test, Y_hat_prob, drop_intermediate=False)
        pr = pr.tolist()
        tpr = tpr.tolist()
    except ValueError:
        pr, tpr = [],[]

    rocc = f"{pr}::{tpr}"
    
    res = {"model":model_name, 
            "seed":random_seed,  
            "group":subgroup,    
            "microF1":round(microF1,3),
            "macroF1":round(macroF1,3),
            "auroc":round(aurocc,3),
            "auprc":round(auprc,3),
            "specificity":round(specificity,3),
            "sensitivity":round(sensitivity,3),
            "roc_curve": rocc
            }

    if res_path is not None:    
        core.helpers.save_results(res, res_path, sep="\t")
    return res



def vectorize(df_train, df_val, df_test, subject_ids):
    """ vectorize the instances and stratify them by demographic subgroup
        df_train: training data as a DataFrame
        df_test: test data as a DataFrame
        df_val: validation data as a DataFrame
        subject_ids: list of subject ids (the order corresponds to order of the features that were extracted)
        
        returns: vectorized train, validation and test datasets, stratified by demographic subgroup
                 label vocabulary                 
    """

    #vectorize labels
    train_Y = df_train["Y"]
    val_Y = df_val["Y"]           
    test_Y = df_test["Y"]               
    label_vocab = tadat.core.vectorizer.get_labels_vocab(train_Y+val_Y)
    train_Y,_ = tadat.core.vectorizer.label2idx(train_Y, label_vocab)
    val_Y,_ = tadat.core.vectorizer.label2idx(val_Y, label_vocab)
    test_Y,_ = tadat.core.vectorizer.label2idx(test_Y, label_vocab)      
    
    #get indices into the feature matrix
    train_idxs = [subject_ids.index(i) for i in list(df_train["SUBJECT_ID"])] 
    val_idxs = [subject_ids.index(i) for i in list(df_val["SUBJECT_ID"])] 
    test_idxs = [subject_ids.index(i) for i in list(df_test["SUBJECT_ID"])] 
    #construct datasets
    train = {}
    test = {}
    val = {}
    #unstratified 
    train["all"] = [train_idxs, train_Y]
    test["all"] = [test_idxs, test_Y]
    val["all"] = [val_idxs, val_Y]
    #stratified by demographics 
    for group in list(GROUPS.keys()):
        #and subgroups
        for subgroup in GROUPS[group]:                
            df_train_sub = df_train[df_train[group] == subgroup]
            df_test_sub = df_test[df_test[group] == subgroup]
            df_val_sub = df_val[df_val[group] == subgroup]
            # print("[subgroup: {} | tr: {} | ts: {} | val: {}]".format(subgroup, len(df_train_sub), len(df_test_sub), len(df_val_sub)))

            #vectorize labels               
            train_Y_sub,_ = tadat.core.vectorizer.label2idx(df_train_sub["Y"], label_vocab)            
            test_Y_sub,_ = tadat.core.vectorizer.label2idx(df_test_sub["Y"], label_vocab)            
            val_Y_sub,_ = tadat.core.vectorizer.label2idx(df_val_sub["Y"], label_vocab)      
            #get indices into the feature matrix
            train_idxs_sub = [subject_ids.index(i) for i in list(df_train_sub["SUBJECT_ID"])] 
            test_idxs_sub = [subject_ids.index(i) for i in list(df_test_sub["SUBJECT_ID"])] 
            val_idxs_sub = [subject_ids.index(i) for i in list(df_val_sub["SUBJECT_ID"])] 
            if subgroup == "M":
                subgroup = "men"
            elif subgroup == "F":
                subgroup = "women"
            train[subgroup.lower()] = [train_idxs_sub, train_Y_sub]
            test[subgroup.lower()] = [test_idxs_sub, test_Y_sub]
            val[subgroup.lower()] = [val_idxs_sub, val_Y_sub]

    return train, val, test, label_vocab


def get_features(data, vocab_size, feature_type, embeddings=None):
    """ compute features from the data
        data: data instances
        vocab_size: size of the vocabulary
        feature_type: type of feature (e.g bag of words, BERT)
        word_vectors: path to pretrained (static) word vectors
        
        returns: feature matrix
    """
    if feature_type == "BOW-BIN":
        X = core.features.BOW(data, vocab_size,sparse=True)
    elif feature_type == "BOW-FREQ":
        X = core.features.BOW_freq(data, vocab_size,sparse=True)
    elif feature_type == "BOE-BIN":
        X = core.features.BOE(data, embeddings,"bin")
    elif feature_type == "BOE-SUM": 
        X = core.features.BOE(data, embeddings,"sum")
    elif feature_type == "U2V": 
        X = core.features.BOE(data, embeddings,"bin")
    elif feature_type == "BERT-POOL":
        X =  core.transformer_encoders.encode_sequences(data, batchsize=64)        
    elif feature_type == "BERT-CLS":
        X =  core.transformer_encoders.encode_sequences(data, cls_features=True,
                                                        batchsize=64)            
    elif feature_type == "MULTI-BERT-POOL":
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, batchsize=32,
                                                         tmp_path=TMP_PATH)
    elif feature_type == "MULTI-BERT-CLS":
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, 
                                                         cls_features=True,
                                                         batchsize=32,
                                                         tmp_path=TMP_PATH)
    elif feature_type == "CLINICALBERT-POOL":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_sequences(data, batchsize=64, tokenizer=tokenizer,
                                                                    encoder=encoder)        
    elif feature_type == "CLINICALBERT-CLS":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_sequences(data, cls_features=True,batchsize=64,
                                                                    tokenizer=tokenizer, encoder=encoder)        
    elif feature_type == "CLINICALMULTI-BERT-POOL":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, batchsize=32,tmp_path=TMP_PATH,
                                                              tokenizer=tokenizer, encoder=encoder)
    elif feature_type == "CLINICALMULTI-BERT-CLS":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, cls_features=True, 
                                                                batchsize=32,tmp_path=TMP_PATH,
                                                                tokenizer=tokenizer, encoder=encoder)
    else:
        raise NotImplementedError
    return X

def extract_features(feature_type, path):
    """ extract features and save features

        method will first look for computed features on disk and return them if found;
        otherwise it will look for a file with name *patients.csv*;        
        
        feature_type: type of feature (e.g bag of words, BERT)
        path: directory where the data can be found
                
        returns: list of subject ids and feature matrix -- the order of ids corresponds to order of the instances in the feature matrix
    """
    X = read_cache(path+"feats_{}".format(feature_type))
    if X:
        print("[reading cached features]")
        subject_ids, X_feats = X
    else:
        print("[computing {} features]".format(feature_type))
        df = pd.read_csv(path+"patients.csv", sep="\t", header=0)
        subject_ids = list(df["SUBJECT_ID"])
        docs = list(df["TEXT"])
        if "BERT" in feature_type:
            X_feats = get_features(docs, None, feature_type)
        elif "U2V" in feature_type:
            X, user_vocab = core.vectorizer.docs2idx(subject_ids)
            user_embeddings, _ = core.embeddings.read_embeddings(path+"/user_embeddings.txt", user_vocab)
            X_feats = get_features(X, len(user_vocab), feature_type, user_embeddings)
        else:
            embeddings = None
            X, word_vocab = core.vectorizer.docs2idx(docs)
            if "BOE" in feature_type:
                embeddings, _ = core.embeddings.read_embeddings(path+"/word_embeddings.txt", word_vocab)
            X_feats = get_features(X, len(word_vocab), feature_type, embeddings)
        #save features
        print("[saving features]")
        write_cache(path+"feats_{}".format(feature_type), 
                    [subject_ids, X_feats])
    return subject_ids, X_feats

# %% [markdown]
# # Run

# %%
def read_dataset(path, dataset_name, df_patients):    
    
    """ read dataset        
        path: path to the dataset
        dataset_name: name of the dataset
        df_patients: DataFrame of patients
                
        returns: train, test and validation sets as DataFrames
    """
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


def read_cache(path):
    """ read a pickled object
        
        path: path
        returns: object
    """
    X = None
    try:
        with open(path, "rb") as fi:            
            X = pickle.load(fi)
    except FileNotFoundError:
        pass
    return X

def write_cache(path, o):
    """ pickle an object
            
        path: path
        o: object
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, "wb") as fo:
        pickle.dump(o, fo)

def old_run(data_path, dataset, features_path, feature_type, cache_path, metric, n_seeds=N_SEEDS, clear_results=False):
    """ 
        train classifiers with different random seeds and compare the performance over each demographic subgroup

        data_path: path to the data
        dataset: dataset to be evaluted
        features_path: path to the features
        feature_type: type of feature (e.g bag of words, BERT)
        cache_path: cache path 
        metric: evaluation metric
        n_seeds: number of seeds

        returns: results for each subgroup
    """
    #read patients data
    df_patients = pd.read_csv(features_path+"patients.csv", 
                              sep="\t", header=0).drop(columns=["TEXT"])
    #read dataset
    df_train, df_test, df_val = read_dataset(data_path, dataset, df_patients)
    
    print("[train/test set size: {}/{}]".format(len(df_train), len(df_test)))
    print("[running {} classifier]".format(CLASSIFIER))
    #extract features
    subject_ids, feature_matrix = extract_features(feature_type, features_path)      
    train, val, test, label_vocab = vectorize(df_train, df_val, df_test, subject_ids)
    train_idx, train_Y = train["all"]
    val_idx, val_Y = val["all"]
    #slice the feature matrix to get the corresponding instances
    train_X = feature_matrix[train_idx, :]    
    val_X = feature_matrix[val_idx, :]    
    #create the cache directory if it does not exist
    dirname = os.path.dirname(cache_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    #try to open a cached results file or create a new one if it does not exist
    res_fname = cache_path+"/cache_{}_{}_{}.pkl".format(dataset, feature_type, metric).lower()    
    try:
        df_results = pd.read_csv(res_fname)
    except FileNotFoundError:
        df_results = pd.DataFrame(columns = ["seed"] +  list(val.keys()))
        df_results.to_csv(res_fname, index=False, header=True)        
    #we can skip seeds that have already been evaluated
    skip_seeds = set([]) if clear_results else set(df_results["seed"])
    groups = list(val.keys())
    init_randomizer = RandomState(1)
    shuffle_randomizer = RandomState(2)    
    # random.seed(1) #ensure repeateable runs 
    # random_seeds = random.sample(range(0, 10000), n_seeds)        
    ##train/test classifier for each random seed pair
    # for init_seed, shuffle_seed in itertools.product(random_seeds,repeat=2):   
    for j in range(n_seeds):         
        init_seed = init_randomizer.randint(10000)
        shuffle_seed = shuffle_randomizer.randint(10000)        
        seed = "{}x{}".format(init_seed, shuffle_seed)  
        if seed in skip_seeds:
            print("skipped seed: {}".format(seed))
            continue
        curr_results = {"seed":seed}
        print(" > seed: {}".format(seed))                        
        model = train_classifier(train_X, train_Y,val_X, val_Y,  
                                    input_dimension=train_X.shape[-1],
                                    init_seed=init_seed, 
                                    shuffle_seed=shuffle_seed)                                                                                
        #test each subgroup (note thtat *all* is also a subgroup)
        for subgroup in groups:                                
            test_idx_sub, test_Y_sub = test[subgroup]                 
            test_X_sub = feature_matrix[test_idx_sub, :]                
            res_sub = evaluate_classifier(model, test_X_sub, test_Y_sub, 
                                        label_vocab, feature_type, seed, subgroup)                
            curr_results[subgroup]= res_sub[metric]     
        #save results
        df_results = df_results.append(curr_results, ignore_index=True)
        df_results.to_csv(res_fname, index=False, header=True)

    return df_results

def run(data_path, dataset, features_path, feature_type, results_path, metric, n_seeds=N_SEEDS, clear_results=False):
    """ 
        train classifiers with different random seeds and compare the performance over each demographic subgroup

        data_path: path to the data
        dataset: dataset to be evaluted
        features_path: path to the features
        feature_type: type of feature (e.g bag of words, BERT)
        results_path: cache path 
        metric: evaluation metric
        n_seeds: number of seeds

        returns: results for each subgroup
    """
    #read patients data
    df_patients = pd.read_csv(features_path+"patients.csv", 
                              sep="\t", header=0).drop(columns=["TEXT"])
    #read dataset
    df_train, df_test, df_val = read_dataset(data_path, dataset, df_patients)
    
    print("[train/test set size: {}/{}]".format(len(df_train), len(df_test)))
    print("[running {} classifier]".format(CLASSIFIER))
    #extract features
    subject_ids, feature_matrix = extract_features(feature_type, features_path)      
    train, val, test, label_vocab = vectorize(df_train, df_val, df_test, subject_ids)
    train_idx, train_Y = train["all"]
    val_idx, val_Y = val["all"]
    #slice the feature matrix to get the corresponding instances
    train_X = feature_matrix[train_idx, :]    
    val_X = feature_matrix[val_idx, :]    
    #create the cache directory if it does not exist
    dirname = os.path.dirname(results_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    #try to open a cached results file or create a new one if it does not exist
    res_fname = results_path+"/{}_{}_{}.csv".format(dataset, feature_type, metric).lower()    
    try:
        df_results = pd.read_csv(res_fname)
    except FileNotFoundError:
        df_results = pd.DataFrame(columns = ["seed","data"] +  list(val.keys()))
        df_results.to_csv(res_fname, index=False, header=True)        
    #we can skip seeds that have already been evaluated
    skip_seeds = set([]) if clear_results else set(df_results["seed"])
    groups = list(val.keys())
    init_randomizer = RandomState(1)
    shuffle_randomizer = RandomState(2)    
    # random.seed(1) #ensure repeateable runs 
    # random_seeds = random.sample(range(0, 10000), n_seeds)        
    ##train/test classifier for each random seed pair
    # for init_seed, shuffle_seed in itertools.product(random_seeds,repeat=2):   
    for j in range(n_seeds):         
        init_seed = init_randomizer.randint(10000)
        shuffle_seed = shuffle_randomizer.randint(10000)        
        seed = "{}x{}".format(init_seed, shuffle_seed)  
        if seed in skip_seeds:
            print("skipped seed: {}".format(seed))
            continue
        test_results = {"seed":seed, "data":"test"}
        val_results = {"seed":seed, "data":"val"}
        print(" > seed: {}".format(seed))                        
        model = train_classifier(train_X, train_Y,val_X, val_Y,  
                                    input_dimension=train_X.shape[-1],
                                    init_seed=init_seed, 
                                    shuffle_seed=shuffle_seed)                                                                                
        #test each subgroup (note thtat *all* is also a subgroup)
        for subgroup in groups:                                
            test_idx_sub, test_Y_sub = test[subgroup]                 
            test_X_sub = feature_matrix[test_idx_sub, :]                
            test_res_sub = evaluate_classifier(model, test_X_sub, test_Y_sub, 
                                        label_vocab, feature_type, seed, subgroup)                
            test_results[subgroup]= test_res_sub[metric]     
            
            val_idx_sub, val_Y_sub = val[subgroup]                 
            val_X_sub = feature_matrix[val_idx_sub, :]                
            val_res_sub = evaluate_classifier(model, val_X_sub, val_Y_sub, 
                                        label_vocab, feature_type, seed, subgroup)                
            val_results[subgroup]= val_res_sub[metric]                 
            
        #save results
        df_results = df_results.append(test_results, ignore_index=True)
        df_results = df_results.append(val_results, ignore_index=True)
        df_results.to_csv(res_fname, index=False, header=True)


    return df_results


# %% [markdown]
# # Grid Search

# %%
MAX_CRITERIA = [
            "performance", 
            "subgroup avg", 
            "subgroup - std", 
            "subgroup - delta avg", 
#             "subgroup - delta sum", 
            "performance - delta"
            # "a",
            # "b"
            ]

MIN_CRITERIA = [
                "subgroup std", 
                "delta avg", 
                "delta sum"
                ]

CRITERIA =  MAX_CRITERIA

def get_best_seed(df_seeds, groups="both"):    
    """ 
        select the seeds with the best performance at different number of runs
        we compare different seed selection criteria: mean/std performance, mean/std performance averaged over all subgroups
        mean subgroup performance delta
        
        df_seeds: grid search results as DataFrame
        groups: demographic groups
        k: number of runs 

        returns: set of best seeds
    """
    
    gender = ["men","women"]    
    race = ["white","black", "hispanic", "asian"]
    subgroups = []
    if groups == "gender" or groups == "both":
        subgroups += gender
    if groups == "race" or groups == "both":
        subgroups += race

    for g in subgroups:
        df_seeds["val_delta_"+g] = (df_seeds["all"] - df_seeds[g]).abs()    

    df_seeds = df_seeds.reset_index()
    df_seeds["performance"] = df_seeds["all"] 
    df_seeds["subgroup avg"] = df_seeds[[g for g in subgroups]].mean(axis=1)
    df_seeds["subgroup std"] = df_seeds[[g for g in subgroups]].std(axis=1)
    df_seeds["delta avg"] = df_seeds[["val_delta_"+g for g in subgroups]].mean(axis=1)
    df_seeds["delta sum"] = df_seeds[["val_delta_"+g for g in subgroups]].sum(axis=1)

    df_seeds["subgroup - std"] = df_seeds["subgroup avg"] - df_seeds["subgroup std"]
    df_seeds["subgroup - delta avg"] = df_seeds["subgroup avg"] - df_seeds["delta avg"]
    df_seeds["subgroup - delta sum"] = df_seeds["subgroup avg"] - df_seeds["delta sum"]
    df_seeds["performance - delta"] = df_seeds["all"] - df_seeds["delta avg"]
    
    res = []        

    #get seeds with min criteria       
    for crit in MIN_CRITERIA:   
        try:        
            best_df = df_seeds.iloc[df_seeds[crit].idxmin()]
            seed = best_df["seed"]
            perf = best_df[crit]
            res.append({"seed":seed, "criterion":crit, "val":round(perf,3) })        
        except:
            print("Error min: {}".format(crit))
    
    #get seeds with max criteria       
    for crit in MAX_CRITERIA:   
        try:        
            best_df = df_seeds.iloc[df_seeds[crit].idxmax()]
            seed = best_df["seed"]
            perf = best_df[crit]
            res.append({"seed":seed, "criterion":crit, "val":round(perf,3) })        
        except:
            print("Error max: {}".format(crit))
            # set_trace()
            
    df_best_seeds = pd.DataFrame(res)               

    return df_best_seeds

def search_analyses(results_path, dataset, feature_type, metric, groups="both", k=50):
    
    fname = "{}_{}_{}.csv".format(dataset, feature_type, metric).lower()       
    try:
        df_results = pd.read_csv(results_path+fname) 
        df_results_val = df_results[df_results["data"] == "val"].reset_index()
        df_results_test = df_results[df_results["data"] == "test"].reset_index()

        bs = get_best_seed(df_results_val, groups)   
        bs = bs.set_index("seed")
        df_results_test = df_results_test.set_index("seed")
        z = pd.merge(bs, df_results_test, how="left", on="seed")         
        
        subgroups = ["men","women","white","black", "hispanic", "asian"]    
        for g in subgroups:
            z["test_delta_"+g] = (z[g] - z["all"]).abs()                
        z["test delta avg"] = z[["test_delta_"+g for g in subgroups]].mean(axis=1)
        z["test gap"] = z[[g for g in subgroups]].max(axis=1) - z[[g for g in subgroups]].min(axis=1)
        
        # z["test delta sum"] = z[["delta "+g for g in subgroups]].sum(axis=1)
        return z
    except FileNotFoundError:
        print("{} not found...".format(fname))        
        return None

def search_analyse_plot(results_path, dataset, task_name, feature_type, metric, k=50, plot_subgroups=False):
    fig, ax = plt.subplots(1, 3,  figsize=(20,8), sharex=True, sharey=True)    
    y = ["all"] 
    if plot_subgroups:
        y+= ["men", "women", "white", "black","asian", "hispanic"]
    else:
        y+= ["test delta avg"]
    for i,g in enumerate(["both", "gender", "race"]):
        df_res = search_analyses(results_path, dataset, feature_type, metric=metric, groups=g, k=k)
        df_res = df_res.round(3)
        df_res = df_res[df_res["criterion"].isin(CRITERIA)]
        df_res.plot(x="criterion", y=y, kind="bar", rot=45, ax=ax[i], ylim=[0.8, 1.05])
#         plot_search(df_res,  f"{g}" , subgroups=plot_subgroups, ax=ax[i])
        all_scores = list(df_res["all"])
        delta_scores = list(df_res["test delta avg"])
        gaps = list(df_res["test gap"])
        labels = [str(a) + "\n" + str(d) + "\n" + str(g) for a,d,g in zip(all_scores, delta_scores, gaps)]
        rects = ax[i].patches
        leg = "all\n avg delta\n gap"
        
        if i == 0:
            ax[i].text(-1, 1 + 0.01, leg,ha='center', va='bottom', fontsize="small")
        
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax[i].text(rect.get_x() + rect.get_width() / 2, 1 + 0.01, label,
                    ha='center', va='bottom', fontsize="small")
        if i == 2:
            ax[i].legend(loc='lower left', bbox_to_anchor=(1.05, 0))
        else:
            ax[i].get_legend().remove()
        ax[i].set_title(f"{g}", y=1.1)
    plt.tight_layout()    
    plt.suptitle(f"{task_name}" , y=1.15)
    plt.show()
        
        
def all_search_analyse_plot(data_path, tasks_fname, results_path, feature_type, metric, k=50, plot_subgroups=False):
    with open(data_path+"/"+tasks_fname,"r") as fid:        
        for i,l in enumerate(fid):            
            task_abv, task_name = l.strip("\n").split(",")            
            search_analyse_plot(results_path, task_abv, task_name, feature_type, 
                                metric, k=k, plot_subgroups=plot_subgroups)


# %%
def find_similar_seeds(df_results, epsilon=0.01):    
    max_val = df_results["all"].max()    
    df_similar = df_results[df_results["all"] >= (max_val-epsilon)] 
#     df_similar = df_results[df_results[metric] >= (perf-epsilon)] 
    return df_similar

def underspecification(results_path, dataset, feature_type, metric):
    
    fname = "{}_{}_{}.csv".format(dataset, feature_type, metric).lower()       
    try:
        df_results = pd.read_csv(results_path+fname) 
        df_val = df_results[df_results["data"] == "val"]
        df_test = df_results[df_results["data"] == "test"]
        #seeds with similar validation performance
        df_similar = find_similar_seeds(df_val)
        #correspoding test performances
        df_test = df_test[df_test["seed"].isin(df_similar["seed"].tolist())]

        plot_deltas(df_test, dataset)
#         return df_test
    except FileNotFoundError:
        print("{} not found...".format(fname))        
        return None

def all_underspecification(data_path, tasks_fname, results_path, feature_type, metric):
    with open(data_path+"/"+tasks_fname,"r") as fid:        
        for i,l in enumerate(fid):            
            task_abv, task_name = l.strip("\n").split(",")            
            under(results_path, task_abv, feature_type, metric)
#             plt.savefig("underspec_{}.pdf".format(task_abv.lower()),dpi=300, bbox_inches='tight')
            
            
            

# %% [markdown]
# # Analyses

# %%
def run_analyses(data_path, dataset, features_path, feature_type, results_path, 
                 metric, clear_results=False):    

    if not os.path.exists(results_path): os.makedirs(results_path)     

    df_results = run(data_path, dataset, features_path, feature_type, results_path, metric, clear_results=clear_results)         
    fname = "{}_{}_{}.csv".format(dataset, feature_type, metric).lower()
    
    df_results.to_csv(results_path+fname, index=False, header=True)
    return df_results               

#Run All the tasks
def run_tasks(data_path, tasks_fname, features_path, feature_type, results_path,   
             metric, reset=False, mini_tasks=True):
    #if reset delete the completed tasks file
    if reset: reset_tasks(results_path)    
    with open(data_path+tasks_fname,"r") as fid:
        for i,l in enumerate(fid):
            if i > N_TASKS: break
            fname, task_name = l.strip("\n").split(",")            
            dataset = "mini-"+fname if mini_tasks else fname
            # dataset = fname
            if is_task_done(results_path, dataset): 
                print("[dataset: {} already processed]".format(dataset))
                continue                        
            print("******** {} {} ********".format(task_name, dataset))      
            run_analyses(data_path, dataset, features_path, feature_type,  
                         results_path, metric=metric, clear_results=False)
            task_done(results_path, dataset)

def task_done(path,  task):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path+"completed_tasks.txt", "a") as fod:
        fod.write(task+"\n")

def reset_tasks(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path+"completed_tasks.txt", "w") as fod:
        fod.write("")

def is_task_done(path,  task):
    try:
        with open(path+"completed_tasks.txt", "r") as fid:
            tasks = fid.read().split("\n")            
        return task in set(tasks)
    except FileNotFoundError:
        #create file if not found
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path+"completed_tasks.txt", "w") as fid:
            fid.write("")
        return False

# %% [markdown]
# # Plots

# %%
# Generate plots 

def plot_densities(df, ax, title):
    ax.set_title(title)
    for y in PLOT_VARS:        
        try:
            df.plot.kde(ax=ax, x="seed", y=y)
        except:
            pass
        
def plot_analyses(results_path, dataset, task_name, feature_type, metric, minorities=False, data_partition="test"):

    fname = "{}_{}_{}.csv".format(dataset, feature_type, metric).lower()       
    try:
        df_results = pd.read_csv(results_path+fname) 
        try:
            df_results_part = df_results[df_results["data"] == data_partition]
        except KeyError:
            df_results_part = df_results
#         print(df_results)        
        plot_deltas(df_results_part, task_name, minorities=minorities)
        plt.savefig("plots/deltas_{}.pdf".format(dataset.lower()),dpi=300, bbox_inches='tight')
        plt.show()      
        plot_scatters(df_results_part, task_name, minorities=minorities)        
        plt.savefig("plots/scatters_{}.pdf".format(dataset.lower()),dpi=300, bbox_inches='tight')
        plt.show()      
        plot_val_vs_test(df_results, task_name)
        plt.show()      
        plot_val_vs_test(df_results, task_name, var="perf")
        plt.show()      

    except FileNotFoundError:
        print("{} not found...".format(fname))        


def plot_tasks(tasks_fname, feature_type, results_path, metric, mini_tasks=True):
    task_abvs = []
    with open(tasks_fname,"r") as fid:        
        for i,l in enumerate(fid):            
            task_abv, task_name = l.strip("\n").split(",")
            dataset = "mini-"+task_abv if mini_tasks else task_abv
            task_abvs.append(task_abv.lower())
            plot_analyses(results_path, dataset, task_name, feature_type, metric)

# 

def plot_deltas(results, title, minorities=False):      
    if minorities:
        subgroups = ["women","black","asian","hispanic"]
        df_deltas = get_minority_deltas(results)
    else:
        subgroups = ["men","women","white","black","asian","hispanic"]
        df_deltas = get_deltas(results)
    df_delta_long = pd.melt(df_deltas, id_vars=["seed"], value_vars=subgroups, 
                                                      value_name="delta", var_name="group")
    g = sns.catplot(x="group", y="delta", data=df_delta_long, sharey=True,legend=False)       
    
    for ax in g.axes[0]:
        ax.axhline(0, ls='--',c="r")
#         ax.set_ylabel("delta")
        ax.set_ylabel(r"$\Delta$ AUC")

    plt.gcf().set_size_inches(6, 5)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
#     plt.show()  
    

def plot_scatters(results, title, minorities=False):
    if minorities:
        n_rows=2
        n_cols = 2    
        figsize=(12,5)
        results = get_minority_deltas(results)
        subgroups = ["women","black","asian","hispanic"]        
    else:
        n_rows=2
        n_cols = 3    
        figsize=(12,8)
        results = get_deltas(results)
        subgroups = ["men","women","white","black","asian","hispanic"]        
    fig, ax = plt.subplots(n_rows, n_cols,  figsize=figsize, sharex=True, sharey=True)
    #current coloramap
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    coords = list(itertools.product(range(n_rows),range(n_cols)))   
    
    for subgroup, col, coord in zip(subgroups, cmap, coords ):        
        results[subgroup] = results[subgroup].abs()         
        results.plot.scatter(x="all",y=subgroup,
                            color=col, ax=ax[coord[0]][coord[1]])
        x = results["all"]
        y = results[subgroup]
        z = np.polyfit(x, y, 1)
        y_hat = np.poly1d(z)(x)
        ax[coord[0]][coord[1]].plot(x, y_hat, c=col, lw=1)
        ax[coord[0]][coord[1]].set_title(subgroup)
        ax[coord[0]][coord[1]].set_ylabel(r"$\Delta$ AUC")
        ax[coord[0]][coord[1]].set_xlabel(r"AUC")
#         ax[coord[0]][coord[1]].set_ylabel("delta")
    fig.suptitle(title, y=1.02)
    plt.tight_layout(pad=0.1) 

def plot_val_vs_test(results, title, var="delta"):
    
    n_rows=2
    n_cols = 3    
    figsize=(12,8)
    fig, ax = plt.subplots(n_rows, n_cols,  figsize=figsize, sharex=True, sharey=True)

    #current coloramap
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    coords = list(itertools.product(range(n_rows),range(n_cols)))   
    
    
    results_val = results[results["data"] == "val"].reset_index()
    results_test = results[results["data"] == "test"].reset_index()

    if var == "delta":
        results_val = get_deltas(results_val)
        results_test = get_deltas(results_test)
        xlabel = r"Val $\Delta$ AUC"
        ylabel = r"Test $\Delta$ AUC"
    elif var == "perf":
        xlabel = r"Val AUC"
        ylabel = r"Test AUC"
    else:
        raise NotImplementedError
        
    subgroups = ["men","women","white","black","asian","hispanic"]        
    max_delta = results_val[subgroups].abs().max().max()
    
    for subgroup, col, coord in zip(subgroups, cmap, coords ):        
        results_val[subgroup] = results_val[subgroup].abs()  
        results_test[subgroup] = results_test[subgroup].abs()          
        ax[coord[0]][coord[1]].scatter(x=results_val[subgroup], y=results_test[subgroup], c=col)
        x = results_val[subgroup].tolist()
        y = results_test[subgroup].tolist()
#         from pdb import set_trace; set_trace()
        z = np.polyfit(x, y, 1)
        x+=[max_delta]
        y_hat = np.poly1d(z)(x)
        ax[coord[0]][coord[1]].plot(x, y_hat, c=col, lw=1)                
        ax[coord[0]][coord[1]].set_xlabel(xlabel)
        ax[coord[0]][coord[1]].set_ylabel(ylabel)
        ax[coord[0]][coord[1]].set_title(subgroup)
        
#         ax[coord[0]][coord[1]].set_ylabel("delta")

#     for subgroup, col, coord in zip(subgroups, cmap, coords ):        
#         results_val[subgroup] = results_val[subgroup].abs()  
#         results_test[subgroup] = results_test[subgroup].abs()  
# #         results.plot.scatter(x="all",y=subgroup,
# #                             color=col, ax=ax[coord[0]][coord[1]])
        
# #         ax[coord[0]][coord[1]].scatter(x=results_val[subgroup], y=results_test[subgroup], c=col)
#         x = results_val[subgroup].tolist()
#         y = results_test[subgroup].tolist()
# #         from pdb import set_trace; set_trace()
#         z = np.polyfit(x, y, 1)
#         x+=[max_delta]
#         y_hat = np.poly1d(z)(x)
#         ax2[coord[0]][coord[1]].plot(x, y_hat, c=col, lw=1)
        
#         ax2[coord[0]][coord[1]].set_title(subgroup)
#         ax2[coord[0]][coord[1]].set_ylabel(r"Test $\Delta$ AUC")
#         ax2[coord[0]][coord[1]].set_xlabel(r"Val $\Delta$ AUC")

    fig.suptitle(title, y=1.02)
    plt.tight_layout(pad=0.1) 
    
def get_minority_deltas(results):
    df = pd.DataFrame()
    df["seed"] = results["seed"]
    df["all"] = results["all"]
    df["women"] = results["women"] - results["men"]
    #race
    df["black"] = results["black"] - results["white"]
    df["hispanic"] = results["hispanic"] - results["white"]
    df["asian"] = results["asian"] - results["white"]

    return df

def get_deltas(results):
    df = pd.DataFrame()
    df["seed"] = results["seed"]
    df["all"] = results["all"]
    #gender
    df["men"] = results["men"] - results["all"]
    df["women"] = results["women"] - results["all"]
    #race
    df["white"] = results["white"] - results["all"]
    df["black"] = results["black"] - results["all"]
    df["hispanic"] = results["hispanic"] - results["all"]
    df["asian"] = results["asian"] - results["all"]

    return df

def plot_summary(tasks_fname, feature_type, results_path, metric, mini_tasks=True, data_partition="test"):
    dfs = []
    with open(tasks_fname,"r") as fid:                
        for i,l in enumerate(fid):            
            task_abv, task_name = l.strip("\n").split(",")
            dataset = "mini-"+task_abv if mini_tasks else task_abv
            fname = "{}_{}_{}.csv".format(dataset, feature_type, metric).lower()  
            try:
                df_results = pd.read_csv(results_path+fname)     
                try:
                    df_results = df_results[df_results["data"] == data_partition]
                except KeyError:
                    pass        
                df_max = df_results.iloc[:,2:].max(axis=1)
                df_min = df_results.iloc[:,2:].min(axis=1)
                df_results["range"] = df_max - df_min
                df_results["dataset"] = [task_abv]*len(df_results)         
                dfs.append(df_results)
            except FileNotFoundError:
                print("{} not found...".format(fname))        

    dfs = pd.concat(dfs)    
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    aucs = sns.catplot(x="all",y="dataset", data=dfs, sharey=True, legend=True, 
                    legend_out=True, height=6.5, aspect=0.85,palette=cmap)    
    aucs.axes[0][0].set_ylabel(r"Task")    
    aucs.axes[0][0].set_xlabel(r"AUC") 
    plt.tight_layout()
    plt.savefig("plots/tasks.pdf",dpi=300, bbox_inches='tight')
    plt.show()          
    gaps = sns.catplot(x="range",y="dataset", data=dfs, sharey=True,legend=True, 
                    legend_out=True, height=6.5, aspect=0.85,palette=cmap)        
    gaps.axes[0][0].set_ylabel("")    
    gaps.axes[0][0].set_xlabel(r"AUC gap")    
    plt.tight_layout()
    plt.savefig("plots/gaps.pdf",dpi=300, bbox_inches='tight')
    plt.show()      

    plt.tight_layout()
    plt.show()  
    

def plot_rocs(df, seed):    
    n_rows=2
    n_cols = 3    
    figsize=(12,8)
    fig1, ax1 = plt.subplots(n_rows, n_cols,  figsize=figsize, sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(1, 1)

    #current coloramap
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    coords = list(itertools.product(range(n_rows),range(n_cols)))   
    x=[0,1]
    y=[0,1]
    ax2.plot(x, y)
    subgroups = ["men","women","white","black","asian","hispanic"]        
    results = df[df["seed"] == seed]
    
    for subgroup, col, coord in zip(subgroups, cmap, coords ):                
        z = results[subgroup].item()

        tpr, fpr, tr = z.split("::")
        tpr = np.array(ast.literal_eval(tpr))
        fpr = np.array(ast.literal_eval(fpr))
        ax2.plot(tpr,fpr, c=col)
#         set_trace()        
        ax1[coord[0]][coord[1]].plot(tpr,fpr, c=col)
        ax1[coord[0]][coord[1]].plot(x,y,"k:")

#         ax[coord[0]][coord[1]].set_xlabel(xlabel)
#         ax[coord[0]][coord[1]].set_ylabel(ylabel)
        ax1[coord[0]][coord[1]].set_title(subgroup)
        
#     fig.suptitle(title, y=1.02)
#     plt.tight_layout(pad=0.1) 
    plt.show()
    
