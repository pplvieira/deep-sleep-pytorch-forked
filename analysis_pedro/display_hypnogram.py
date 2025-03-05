


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





def plot_predict_proba(
    #proba, ### THIS USED TO BE SELF
    proba=None,
    majority_only=False,
    palette=["#99d7f1", "#009DDC", "xkcd:twilight blue", "xkcd:rich purple", "xkcd:sunflower"],
):
    """
    Plot the predicted probability for each sleep stage for each 30-sec epoch of data.

    Parameters
    ----------
    proba : self or DataFrame
        A dataframe with the probability of each sleep stage for each 30-sec epoch of data.
    majority_only : boolean
        If True, probabilities of the non-majority classes will be set to 0.
    """
    # if proba is None and not hasattr(self, "_features"):
    #     raise ValueError("Must call .predict_proba before this function")
    # if proba is None:
    #     proba = self._proba.copy()
    # else:
    assert isinstance(proba, pd.DataFrame), "proba must be a dataframe"
    if majority_only:
        cond = proba.apply(lambda x: x == x.max(), axis=1)
        proba = proba.where(cond, other=0)
    ax = proba.plot(kind="area", color=palette, figsize=(10, 5), alpha=0.8, stacked=True, lw=0)
    # Add confidence
    # confidence = proba.max(1)
    # ax.plot(confidence, lw=1, color='k', ls='-', alpha=0.5,
    #         label='Confidence')
    print("probashape", proba.shape)
    ax.set_xlim(0, proba.shape[0])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time (30-sec epoch)")
    plt.legend(frameon=False, bbox_to_anchor=(1, 1))
    return ax




sleep_stages_list = ["W", "N1", "N2", "N3", "REM"]
def sleep_stage_map(x):
    r = "W" if x==0 else "N1" if x==1 else "N2" if x==2 else "N3" if x==3 else "REM"
    return r

prediction_dir = "C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/deep-sleep-pytorch/experiments/my_experiment1/predictions-best_weights"
fid = "a"

begining_index = 18000 # cut index from here
eval_window = 1
#eval_window = 60
#eval_window = 30
eval_window = 120

print("| INFO | LOADING PREDICTIONS")
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
predictions = np.array(list(map(sleep_stage_map, predictions))) #sleep_stage_map(predictions)

predictions = np.repeat(predictions, eval_window // 30, axis=0)

predictions_df = pd.DataFrame(preds.T, columns=sleep_stages_list)
predictions_df.index.name = "epoch"

print("[TARGETS    ]", targets.shape, np.unique(targets))
print("[PREDICTIONS]", predictions.shape, np.unique(predictions))


# probabilities hypnogram
#plt.figure()
fig1, ax1 = plt.subplots()
ax1 = plot_predict_proba(predictions_df)
#plt.show()


#plt.figure()
print("predictions shape", predictions.shape)
hyp = yasa.Hypnogram(predictions)
fig2, ax2 = plt.subplots()
ax2 = hyp.plot_hypnogram()
plt.savefig(f"hypnogram_{eval_window}.png")
plt.show()











