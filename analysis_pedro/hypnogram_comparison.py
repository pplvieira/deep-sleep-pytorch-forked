

from os import listdir
from os.path import isfile, join

import math
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score


### COMPARES DIFFERENT HYPNOGRAMS FOR THE SAME SLEEP RECORDING



def cross_entropy(y_pred, y_true):

    # computing softmax values for predicted values
    #y_pred = softmax(y_pred)
    loss = 0
        
    # Doing cross entropy Loss
    for i in range(len(y_pred)):

        # Here, the loss is computed using the
        # above mathematical formulation.
        loss += (-1 * y_true[i]*np.log(y_pred[i] + 1e-12))

    return loss


def mutual_information_from_confmat(confmat):
    confmat = np.array(confmat)
    joint_entropy = 0
    mutual_info   = 0
    confmat = confmat / confmat.sum() # TURN INTO PROBABILITIES
    marginal1 = confmat.sum(axis=0) #/ confmat.sum()
    marginal2 = confmat.sum(axis=1) #/ confmat.sum()
    print("SUMS AND MARGINALS", marginal1, marginal2, confmat.sum())
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            joint_entropy += (- confmat[i,j] * np.log(confmat[i,j] + 1e-12))
            mutual_info   += (- confmat[i,j] * np.log(confmat[i,j] / (marginal1[i]*marginal2[j]) + 1e-12))

    #return joint_entropy - marginal1 - marginal2
    return mutual_info



def max_agreeing_predictors(predictions):
    max_agreement = np.array([max(Counter(timestamp_preds).values()) for timestamp_preds in predictions.T])
    return max_agreement


def perform_comparison(prediction_dir: str):
    #prediction_dir   = "C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/deep-sleep-pytorch/experiments/my_experiment1/predictions-best_weights/predictions_txts"
    #files_to_compare = ["fid-a_predictions.txt", "predicted_hypno.txt"]


    txtfiles = [f for f in listdir(prediction_dir) if isfile(join(prediction_dir, f))]
    n_files  = len(txtfiles)
    print("Files in said directory:", txtfiles)

    file_sizes = []
    for i, f in enumerate(txtfiles):
        text_file = open(join(prediction_dir, f), "r")
        ListToSort = text_file.readlines()
        ListToSort = list(map(lambda x: int(x.replace("\n", "")), ListToSort))
        print(f"Size of file {i} ({f}):", len(ListToSort))
        file_sizes.append(len(ListToSort))
        text_file.close()


    max_size = max(file_sizes)
    lcm_size = math.lcm(*file_sizes) #np.lcm(file_sizes)
    print(f"Max size: {max_size} | LCM size: {lcm_size}")
    predictions_lists = np.zeros((len(txtfiles), lcm_size))
    #predictions_lists = np.array([[]]).reshape((0,0))
    for i, f in enumerate(txtfiles):
        text_file = open(join(prediction_dir, f), "r")
        ListToSort = text_file.readlines()
        ListToSort = list(map(lambda x: int(x.replace("\n", "")), ListToSort))
        #predictions_lists[i, :] = np.append(predictions_lists, np.array(ListToSort).reshape(-1,1), axis=0)
        ListToSort = np.repeat(ListToSort, lcm_size // file_sizes[i], axis=0)
        predictions_lists[i, :] = np.array(ListToSort).reshape(1,-1)
        #predictions_lists = np.vstack((predictions_lists, ListToSort))
        text_file.close()
        #print(ListToSort)

    print("shape", predictions_lists.shape)

    num_raters = predictions_lists.shape[0]


    #cm_cat = confusion_matrix(predictions_lists[0], predictions_lists[-1])
    cm_cat = confusion_matrix(predictions_lists[0], predictions_lists[1])
    #                           true/horizontal   -  predicted/vertical
    print("CMATRIX:\n", cm_cat)
    cm_2predictors = ConfusionMatrixDisplay(cm_cat).plot()
    cm_2predictors.ax_.set_title("Confusion matrix of predictions between models 1 and 2, for subject 1")
    cm_2predictors.ax_.set_ylabel(txtfiles[0])
    cm_2predictors.ax_.set_xlabel(txtfiles[1])
    # cm_2predictors.ax_.set_xlabel("model2_subj-1_win-30")
    # cm_2predictors.ax_.set_ylabel("subj-1_win-30")
    cm_2predictors.ax_.set_xticks([i for i in range(5)], ["wake", "N1", "N2", "N3", "REM"])
    cm_2predictors.ax_.set_yticks([i for i in range(5)], ["wake", "N1", "N2", "N3", "REM"])

    

    mutual_information = mutual_information_from_confmat(cm_cat)
    print("mutual information:", mutual_information)



    ## SUPERIMPOSE ALL PREDICTIONS
    dy = 0.02
    ygap = 8 #13
    fig = plt.figure(figsize=(13,6))
    ax = plt.subplot(111)
    x_ = np.arange(0, len(predictions_lists[0]))
    for i, predictions in enumerate(predictions_lists): 
        #plt.plot(predictions, label=txtfiles[i])
        #plt.scatter(x_, predictions*(dy*(len(predictions_lists)+15)) + dy*i, label=txtfiles[i], s=1)
        ax.scatter(x_, predictions*(dy*ygap) + dy*i/2, label=txtfiles[i], s=2)
    ax.set_yticks(ticks=[j*(dy*ygap) for j in range(5)], labels=["wake", "N1", "N2", "N3", "REM"])
    #ax.legend()
    ax.set_xlabel("Epoch # (30 second interval)")
    ax.set_ylabel("Predicted sleep stage")
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, +1.0), ncol=3, fancybox=True, shadow=False)
    ax.set_title("Hypnogram comparison from different predictors", y=1.07)

    ## SUPERIMPOSE HYPNOGRAMS 
    # plt.figure()
    # x_ = np.arange(0, len(predictions_lists[0]))
    # for i, predictions in enumerate(predictions_lists): 
    #     plt.plot(predictions + dy*i, label=txtfiles[i])
    # plt.legend()


    ### NUMBER OF AGREEING 
    # number of max agreeers per each timestamp
    max_agreeing_list = max_agreeing_predictors(predictions_lists)
    print("[ ]", len(max_agreeing_list), np.mean(max_agreeing_list), max_agreeing_list)

    count_number_of_agreeers = [] # len = number of raters
    for i in range(num_raters):
        count_number_of_agreeers += [(max_agreeing_list == i+1).sum()] # / len(max_agreeing_list)
    count_number_of_agreeers = np.array(count_number_of_agreeers)


    fig, ax = plt.subplots()
    rects = ax.bar([i+1 for i in range(num_raters)], count_number_of_agreeers / len(max_agreeing_list) * 100) #, label=bar_labels, color=bar_colors)
    ax.bar_label(rects, padding=3, fmt="{:.1f}%")
    ax.set_title("Proportion of number of agreeers for each timestamp")
    ax.set_xticks([int(i) for i in range(1, num_raters + 1)])


    # weird plot where they agree
    # plt.figure()
    # plt.plot(predictions_lists[0]==predictions_lists[1])

    print("Rater agreement:", np.sum(predictions_lists[0] == predictions_lists[1]) / lcm_size )

    cross_entropy_result = cross_entropy(list(predictions_lists[0]), list(predictions_lists[1]))
    print("Cross entropy loss: ", cross_entropy_result)


    # PAIRWISE METRICS
    cohen_kappa_scores_matrix = np.eye(n_files, n_files)
    rater_agreement_scores_matrix = np.eye(n_files, n_files)
    for rater_i in range(n_files):
        for rater_j in range(rater_i + 1, n_files):
            cohens_kappa_ = cohen_kappa_score(list(predictions_lists[rater_i]), list(predictions_lists[rater_j]))
            rater_agreement_ = np.sum(predictions_lists[rater_i] == predictions_lists[rater_j]) / lcm_size
            print(f"Cohen's kappa score {rater_i},{rater_j}: {cohens_kappa_:.3f}")
            cohen_kappa_scores_matrix[rater_i, rater_j] = cohens_kappa_
            rater_agreement_scores_matrix[rater_i, rater_j] = rater_agreement_

    #print("COHENS MATRIX:\n", cohen_kappa_scores_matrix)
    #plt.figure()
    # # VVVVVV
    cm_kappa = ConfusionMatrixDisplay(cohen_kappa_scores_matrix, display_labels=txtfiles).plot(xticks_rotation=10) #15
    cm_kappa.ax_.set_title("Cohen's kappa scores between raters")
    for i in range(n_files):
        for j in range(n_files):
            cm_kappa.text_[i, j].set_fontsize(13)

    #plt.figure()
    cm_agree = ConfusionMatrixDisplay(rater_agreement_scores_matrix, display_labels=txtfiles).plot(xticks_rotation=10) #15
    cm_agree.ax_.set_title("Agreement between raters")
    #cm_agree.ax_.set_xlabel("")
    #plt.colorbar(cm_agree.figure_, fraction=0.046, pad=0.04)
    #cm_agree.figure_.colorbar(cm_agree.confusion_matrix, fraction=0.046, pad=0.04)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #cm_agree.figure_.set_size_inches(3,3)
    #cm_agree.figure_.tight_layout()
    for i in range(n_files):
        for j in range(n_files):
            cm_agree.text_[i, j].set_fontsize(13) #"x-large"
            # cm_agree.text_[i, j] = ax.text(
            #         j, i, format(cm[i, j], ".2g"), ha="center", va="center", color=color, size='x-large')


    ## TO SEE IF THERE IS ONE STAGE OF SLEEP WITH LESS AGREEMENT
    # stages_with_more_agreement_cm = confusion_matrix(predictions_lists[-1], max_agreeing_list)
    # print("CMATRIX:\n", stages_with_more_agreement_cm)
    # ConfusionMatrixDisplay(stages_with_more_agreement_cm).plot()

    # stages_with_more_agreement_cm = confusion_matrix(predictions_lists[-2], max_agreeing_list)
    # print("CMATRIX:\n", stages_with_more_agreement_cm)
    # ConfusionMatrixDisplay(stages_with_more_agreement_cm).plot()


    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/deep-sleep-pytorch/experiments/my_experiment1/predictions-best_weights/predictions_txts",
        help="folder to load hypnogram prediction txts from (default: %(default)s)",
    )

    
    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)


    perform_comparison(args.folder)



