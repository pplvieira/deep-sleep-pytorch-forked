

from os import listdir
from os.path import isfile, join

import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score



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
    marginal1 = confmat.sum(axis=0) / confmat.sum()
    marginal2 = confmat.sum(axis=1) / confmat.sum()
    print("SUMS AND MARGINALS", marginal1, marginal2, confmat.sum())
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            joint_entropy += (- confmat[i,j] * np.log(confmat[i,j] + 1e-12))
            mutual_info   += (- confmat[i,j] * np.log(confmat[i,j] / (marginal1[i]*marginal2[j]) + 1e-12))

    #return joint_entropy - marginal1 - marginal2
    return mutual_info




prediction_dir   = "C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/deep-sleep-pytorch/experiments/my_experiment1/predictions-best_weights/predictions_txts"
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


cm_cat = confusion_matrix(predictions_lists[0], predictions_lists[1])
print("CMATRIX:\n", cm_cat)
ConfusionMatrixDisplay(cm_cat).plot()

mutual_information = mutual_information_from_confmat(cm_cat)
print("mutual information:", mutual_information)



plt.figure()
for i, predictions in enumerate(predictions_lists): 
    plt.plot(predictions, label=txtfiles[i])
plt.legend()

# weird plot where they agree
# plt.figure()
# plt.plot(predictions_lists[0]==predictions_lists[1])

print("Rater agreement:", np.sum(predictions_lists[0] == predictions_lists[1]) / max_size )

cross_entropy_result = cross_entropy(list(predictions_lists[0]), list(predictions_lists[1]))
print("Cross entropy loss: ", cross_entropy_result)


# PAIRWISE METRICS
cohen_kappa_scores_matrix = np.eye(n_files, n_files)
rater_agreement_scores_matrix = np.eye(n_files, n_files)
for rater_i in range(n_files):
    for rater_j in range(rater_i + 1, n_files):
        cohens_kappa_ = cohen_kappa_score(list(predictions_lists[rater_i]), list(predictions_lists[rater_j]))
        rater_agreement_ = np.sum(predictions_lists[rater_i] == predictions_lists[rater_j]) / max_size
        print(f"Cohen's kappa score {rater_i},{rater_j}: {cohens_kappa_:.3f}")
        cohen_kappa_scores_matrix[rater_i, rater_j] = cohens_kappa_
        rater_agreement_scores_matrix[rater_i, rater_j] = rater_agreement_

#print("COHENS MATRIX:\n", cohen_kappa_scores_matrix)
#plt.figure()
ConfusionMatrixDisplay(cohen_kappa_scores_matrix, display_labels=txtfiles).plot()

#plt.figure()
ConfusionMatrixDisplay(rater_agreement_scores_matrix, display_labels=txtfiles).plot()



plt.show()