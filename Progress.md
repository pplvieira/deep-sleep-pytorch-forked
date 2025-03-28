


# Learning objectives:
- Literature review related to sleep, Alzheimer’s disease, and neurostimulation.
- Learn about EEG automatic sleep stage detection/classification, using deep learning approaches
- Gain further experience with relevant python machine learning and deep learning packages such as Scikit-learn and Pytorch.
- Apply an existing automatic sleep classification algorithm on an existing sleep EEG dataset, and explore improvements to a model.
- Presentation internally of relevant findings for further development. 

--- 

# TODOs

- [ ] More customization at generate cohort files and predict hypnogram from pkl with custom directories for data and results
- [ ] Multi-rater on same psg analysis and same or multiple rater analysis on a cohort of data
- [x] Read Nikolas' friend's data.
- [ ] Use better BIDS notation and folder hierarchy.
- [ ] Make better visualizations
- [ ] Describe pipeline necessary when new eft files are added. For posterity

--- 
--- 


# Progress
### 07/02/25 friday
- Read Alexanders paper [link](https://academic.oup.com/sleep/article/44/1/zsaa161/5897250) and understand
- Install packages (mne, pyedflib, ...) and setup python
- Begin to follow and run the steps from [github](https://github.com/neergaard/deep-sleep-pytorch)

### 14/02/25 friday
- Tracked down the problem being that both the dataset dataloader and precition files expect you to include a labeled hypogram, which we don't have and just want to use the code for obtaining it.
- Run ``` python3 src/utils/channel_label_identifier.py "C:\Users\Pedro\Desktop\Universidade\DTU 2A 1S spring\Specialcourse\data" src/configs/signal_labels/sleepTrial1.json C3 C4 A1 A2 EOGL EOGR LChin RChin EMG``` for first step. Selected the following electrode mapping: C1:C1 | C2:C2 | A1:P3 | A2:P4 | EOGL:FP1??? | EOGR:FP2??? | LChin: | RChin: | EMG:  (0,1,15,16,9,10, , , )
- Run ``` python3 -m src.data.generate_cohort_files_no_hypnogram -c data_sleepTrial1.json ``` instead 
- Run ``` python3 predict_no_hypnogram.py -c config.yaml -r trained_models/best_weights.pth ``` instead
- Mark suggested just making up dummy hypnogram with correct format and run Alexander's pipeline anyway. 
- Modified pipeline uses the 2 cmd lines above. Uses dummy artificial hypnogram data as placeholder to run the rest anyway
- Created simple aditional py script to read the predictions and plot the hypnograms predicted

### 21/02/25 friday
- Rerun the pipeline with proper electrode mapping. Workaround because pipeline expects EOG and chin EMG. Make dummy for those or ignore.
- Presentation from Mark on BIDS brain imaging data structure. Organized folder hierarchy and naming convention. Keep original version of the data and dont overwrite (source; raw (minimaly processed but in compliance to BIDS. processed using a _bidsify_ script (MNE bits VER!!), not manually); derivative). Data and Analysis folders should be on the same level of the project hierarchy.
- Organized a bit the hierarchy and folder structure. 
- Created ```predictions_to_txt.py``` to convert predictions probabilities from pickle format to a txt of final predictions [predictions_to_txt.py](analysis_pedro/predictions_to_txt.py) ***(!! NEED TO MAKE IT USABLE IN THE PIPELINE)***
- Created ```hypnogram_comparison.py``` to read those txt from different (2) sources and plot the joint hypnogram and labels where they coincide [hypnogram_comparison.py](analysis_pedro/hypnogram_comparison.py) ***(!! NEED TO MAKE IT USABLE IN THE PIPELINE)***
- ### Next steps: compare me and Patricia's; use on more data; how the algorithms work, state of the art and how much it improved; OpenNeuro database; look for models that dont need eog and emg; make better visualizations

### 28/02/25
- Make better multi-hypnogram result analysis pipeline (reading from the predictions txt), inter-rater kappa score and agreement + matrix plots
- Use ```python3 predictions_to_txt.py --eval-window 30``` or ```python3 -m analysis_pedro\predictions_to_txt --eval-window 30``` (or a different number), to generate the *predictions.txt* files with different window sizes
- Troubleshooting when trying to run the pipeline with another edf file. Perhaps the file is at fault? Or some other modifications I made? Model was predicting nan allways.   

### (Some other days. Missed one friday)
- Polishing pipeline, customizable and reproducible.
- Forked Alexander's repo and included my modifications on his pipeline plus analysis and processing/comparisson of predicted sleep stages.

### 14/03/25
In ``` data_sleepTrial1.json ``` use "edf": "D:/Universidade/DTU 2A 1S spring/Specialcourse/data"  OR  "edf": "C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data",
- Solved problem from 2 weeks ago, by turning 1 of the 4 channels from nan to zeros at the predictor.
- Development on the hypnogram predictions comparator (comparison between predictors). Now returns number of agreeing raters for each time stamp; barplot of total count of each max number of agreeers at each time step; matrix between number of agreers and single rater prediction class (maybe scrap).
- Use ```python3 analysis_pedro\hypnogram_comparison.py``` after ```python3 analysis_pedro\predictions_to_txt.py``` to compare **different raters ON THE SAME SLEEP DATA**.

### ( Missed one friday )

### 28/03/25
- Better visualization and pipelines. For hypnogram comparison of multiple predictors for the same sleep EEG
- Attempt to run pipeline on sleepEDF database. Light (few channels) polysomnography data.





--- 

```"edf": "C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data",```

OR

```"edf": "D:/Universidade/DTU 2A 1S spring/Specialcourse/data",```



---

## Papers
- Automatic sleep staging of EEG signals: recent development, challenges, and future directions. [link](https://pure.au.dk/ws/portalfiles/portal/333848852/Phan_2022_Physiol._Meas._43_04TR01.pdf)

---
---

# - How to run the pipeline -
1. Place all EDF files in &lt;*DATAFOLDER*&gt; folder (chose whatever name or directory for &lt;*DATAFOLDER*&gt;).
2. (optional) Copy the contents of src.configs.data_base.json to another file data_<COHORT_NAME>.json in the same directory.
3. (optional) Insert the name of your test data COHORT_NAME in line 3 of the file, and change the edf and stage paths to point to the location of your EDFs and hypnograms (this can be the same directory).
4. (optional) Change the output directory to a custom location
5. Change the ``COHORT>sleepTrial1>"edf"`` line in ```src/configs/signal_labels/<sleepTrial1>.json``` to &lt;*DATAFOLDER*&gt;. 
6. In command line, change directory to this folder (deep-sleep-torch).
7. (Optional) The electrode mapping can be changed with by running ``` python3 src/utils/channel_label_identifier.py "<DATAFOLDER>" src/configs/signal_labels/<sleepTrial1>.json C3 C4 A1 A2 EOGL EOGR LChin RChin EMG```. When prompted, select the following electrode mapping: C1:C1 | C2:C2 | A1:P3 | A2:P4 | EOGL:FP1??? | EOGR:FP2??? | LChin: (enter) | RChin: (enter) | EMG: (enter) (0,1,15,16,9,10, , , ). Otherwise skip this step.
8. Run `python3 -m src.data.generate_cohort_files_no_hypnogram -c data_sleepTrial1.json` to preprocess the edf and hypnogram files, generate pkl and h5 files.
9. Edit `config.yaml` appropriately
10. Run `python3 predict_no_hypnogram.py -c config.yaml -r trained_models/best_weights.pth ` to generate the predictions for every timestamp, in a `&lt;UserID&gt;.pkl` file in `\experiments\my_experiment1\predictions-best_weights` and overviews .csv.
11. (Optional) Run `python3 analysis_pedro\display_hypnogram.py <PREDICTION.pkl filepath>` to display the predicted sleep staging (predicted hypnogram) and probabilities for each timestamp. With `--eval-window 30` (size of moving window) and `--begining-index 0` (instant where sleep actually begins) as costumizable arguments.
12. Run `python3 -m analysis_pedro.predictions_to_txt.py --eval-window 30` to apply the moving eval-window on the predictions made before, and write a txt file with the sleep stage predicted for each epoch. These files appear in `\experiments\my_experiment1\predictions-best_weights\predictions_txts`
13. (Optional) Place any number of files in the same format as `predictions.txt`, from multiple predictors/different pipelines, but FOR THE SAME SLEEP EDF DATA (can have different sizes because of different window sizes). This performs multi-rater comparisons for the same sleep data.


---


<!-- 

```
C:\Users\Pedro\Desktop\Universidade\DTU 2A 1S spring\Specialcourse\deep-sleep-pytorch>python3 -m src.data.generate_cohort_files_no_hypnogram -c data_sleepTrialMulti.json
2025-02-28 13:57:58.336 | INFO | Processing cohorts: ['sleepTrial1', 'sleepTrialExternalDrive']
2025-02-28 13:57:58.336 | INFO | Processing cohort: sleepTrial1
hgjad C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data
[paths and names] {'edf': 'C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data', 'stage': ''} sleepTrial1
[listedf] 1 ['C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data\\2024-11-22_14-08-21_8e19cc41_27_electrodes.edf']
[list_ID_union] ['2024-11-22_14-08-21_8e19cc41_27_electrodes']
[*-1*] ['C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data\\2024-11-22_14-08-21_8e19cc41_27_electrodes.edf']
basedir, listfileID ['a'] ['a']
[] lists of coisas ['C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data\\2024-11-22_14-08-21_8e19cc41_27_electrodes.edf']  |  ['a']
2025-02-28 13:57:58.336 | INFO | Current cohort: sleepTrial1 | Total: 1 subjects, 1 EDFs
2025-02-28 13:57:58.336 | INFO | sleepTrial1 | Assigning subjects to subsets: 0/0/1 train/eval/test
2025-02-28 13:57:58.351 | INFO | sleepTrial1 | test  | a     | Loading matching hypnogram
[UNIQUE @ HYPNOGRAM] [0]
2025-02-28 13:57:58.351 | INFO | sleepTrial1 | test  | C:/Users/Pedro/Desktop/Universidade/DTU 2A 1S spring/Specialcourse/data\2024-11-22_14-08-21_8e19cc41_27_electrodes.edf | Loading EDF FILE
=== edf.getNSamples() [13518080 13518080 13518080 13518080 13518080 13518080 13518080 13518080
 13518080 13518080 13518080 13518080 13518080 13518080 13518080 13518080
 13518080 13518080 13518080 13518080 13518080 13518080 13518080 13518080
 13518080 13518080 13518080 13518080 13518080 13518080]
=== signal_label_idx {'C3': 4, 'C4': 5, 'A1': 6, 'A2': 7, 'EOGL': 0, 'EOGR': 1, 'LChin': [], 'RChin': [], 'EMG': []}
=== signal_data (1, 13518080) []
2025-02-28 13:58:21.763 | INFO | sleepTrial1 | test  | a     | Referencing data channels
##» C3
### (1, 13518080)
### (1, 13518080)
##» EOGL
### (1, 13518080)
### (1, 13518080)
=== signal_label_idx {'C3': 4, 'C4': 5, 'A1': 6, 'A2': 7, 'EOGL': 0, 'EOGR': 1, 'LChin': [], 'RChin': [], 'EMG': []}
2025-02-28 13:58:22.033 | INFO | sleepTrial1 | test  | a     | Resampling data
[For chn in signal data key] 128 | C3 : 4 | 256.0
[For chn in signal data key] 128 | C4 : 5 | 256.0
[For chn in signal data key] 128 | EOGL : 0 | 256.0
[For chn in signal data key] 128 | EOGR : 1 | 256.0
2025-02-28 13:58:23.660 | INFO | sleepTrial1 | test  | a     | Selecting C3 as EEG
C:\Users\Pedro\Desktop\Universidade\DTU 2A 1S spring\Specialcourse\deep-sleep-pytorch\src\data\generate_cohort_files_no_hypnogram.py:283: RuntimeWarning: invalid value encountered in divide
  psg[chn][k, :] = (X - m)/s
2025-02-28 13:58:24.776 | INFO | sleepTrial1 | test  | a     | Trim/max length: 1760/1760
[UNIQUE @ HYPNOGRAM] [0]
@ process files 1760 1760
2025-02-28 13:58:24.857 | INFO | sleepTrial1 | a | Writing 1760 epochs
2025-02-28 13:58:25.247 | INFO | Processing cohort: sleepTrialExternalDrive
hgjad D:/Universidade/DTU 2A 1S spring/Specialcourse/data
[paths and names] {'edf': 'D:/Universidade/DTU 2A 1S spring/Specialcourse/data', 'stage': ''} sleepTrialExternalDrive
[listedf] 2 ['D:/Universidade/DTU 2A 1S spring/Specialcourse/data\\2024-11-22_14-08-21_8e19cc41_27_electrodes.edf', 'D:/Universidade/DTU 2A 1S spring/Specialcourse/data\\2025-02-18_23-51-08_1a6055f0_27_electrodes.edf']
2025-02-28 13:58:25.541 | INFO | Processing cohorts finalized.
``` -->

