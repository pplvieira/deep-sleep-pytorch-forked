

## TURN THE PREDICTIONS PICKLE INTO A TXT OF EACH SLEEP STAGE IN A LINE (PATRICIA'S FORMAT)


### LOAD PICKLE AND DISPLAY HYPNOGRAM

import argparse
import math
import os
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')

import yasa
from yasa import simulate_hypnogram

from yasa import staging

# alexanders
from src.utils.config import process_config
from src.utils.factory import create_instance


# UNUSED
def sleep_stage_map(x):
    r = "W" if x==0 else "N1" if x==1 else "N2" if x==2 else "N3" if x==3 else "REM"
    return r


def predictions_to_txt(eval_window, prediction_dir, fid, begining_index = 18000):
    sleep_stages_list = ["W", "N1", "N2", "N3", "REM"]


    ### MOST OF THESE ARE NOW INPUTS
    # prediction_dir = "C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/deep-sleep-pytorch/experiments/my_experiment1/predictions-best_weights"
    # fid = "a"

    # begining_index = 18000 # cut index from here
    # eval_window = 1
    # eval_window = 60
    # eval_window = 30
    # eval_window = 120

    print("| INFO | LOADING PREDICTIONS |", fid)
    with open(os.path.join(prediction_dir, fid + '.pkl'), 'rb') as handle:
        labels = pickle.load(handle)
    print("| INFO | DONE LOADING PREDICTIONS")

    t = np.concatenate(labels['targets'], axis=0)
    p = np.concatenate(labels['predictions'], axis=1)
    print("[set targets]", t.shape, np.unique(t)) #(9000,)
    print("[set predict]", p.shape, np.unique(p)) #(5, 9000)
    # subset = row.Subset
    # t = np.concatenate(subjects[subset][fid]['true'], axis=0)
    # p = np.concatenate(subjects[subset][fid]['pred'], axis=1)

    # targets = t[::eval_window]
    # preds   = p[:, ::eval_window]
    targets = t[begining_index::eval_window]
    preds   = p[:, begining_index::eval_window]
    preds   = np.mean(p[:, begining_index:].reshape(5, -1, eval_window), axis=2)

    predictions = np.mean(p[:, begining_index:].reshape(5, -1, eval_window), axis=2).argmax(axis=0)

    #predictions = np.array(list(map(sleep_stage_map, predictions))) #sleep_stage_map(predictions)

    # if eval_window > 30:
    #     predictions = np.repeat(predictions, eval_window // 30, axis=0)

    predictions_df = pd.DataFrame(preds.T, columns=sleep_stages_list)
    predictions_df.index.name = "epoch"

    print("[TARGETS    ]", targets.shape, np.unique(targets))
    print("[PREDICTIONS]", predictions.shape, np.unique(predictions))



    f = open(f"{prediction_dir}/predictions_txts/fid-{fid}_predictions_win-{eval_window}.txt", "w")
    for pred_ in predictions:
        f.write(f"{pred_}\n")
    f.close()





if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-window', type=int, default=30, help='eval window size (default: %(30)s)')
    parser.add_argument('-c', '--config', default="config.yaml",
                    type=str, help='Path to configuration file (default: my own :D)')
    parser.add_argument('--prediction-dir', type=str, help='directory of the id.pkl files',
                    default="C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/deep-sleep-pytorch/experiments/my_experiment1/predictions-best_weights")
    args = parser.parse_args()

    # load config file we been using (like with predict_no_hypnogram)
    config = process_config(args.config)
    
    #prediction_dir = "C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/deep-sleep-pytorch/experiments/my_experiment1/predictions-best_weights" #CHANGE
    prediction_dir = args.prediction_dir
    print("[PREDICTIONDIR]:", prediction_dir)

    subsets = ['test']

    subjects = {subset: None for subset in subsets}
    df_total = []

    for subset in subsets:
        # Setup data_loader instances
        dataset = create_instance(config.data_loader)(config, subset=subset)
        #print("\n[dataset]", len(dataset), type(dataset))
        df_subset = dataset.df
        print("»» DF subset", df_subset.shape, "\n", df_subset)
        subjects_in_subset = {r.FileID: {'true': [], 'pred': []} for _, r in df_subset.iterrows()}
        prediction_file_names = [os.path.join(r.FileID + ".pkl") for _, r in df_subset.iterrows()]
        file_ids = [r.FileID for _, r in df_subset.iterrows()]
        print("Subjeitos:", subjects_in_subset)
        print("prediction_file_names:", prediction_file_names)
        print("file_ids:", file_ids)


        # READ PREDICTION FILES
        eval_window = args.eval_window
        for fid in file_ids:
            predictions_to_txt(eval_window, prediction_dir, fid)








    # file_names = [fn for fn in os.listdir(predictions_dir) if fn.endswith("pkl")]
    # print("ATTEMPT FILENAMES:", file_names)





