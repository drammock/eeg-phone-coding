# Exploring speech sound coding using EEG
Experiment to test performance of EEG signal classifiers trained on brain responses to CV-syllables using only English consonants, when tested on brain responses to held-out English and foreign consonant CV-syllables.

## Preparation
- `slide-prompts` folder: Generate textual prompts for recording foreign-language stimuli
- `stimulus-generation` folder: Process audio recordings to generate stimuli

## Data Collection
- `run-experiment.py`: Run experiment and collect EEG data (done on a separate acquisition computer using BrainVision pyCorder software; no additional scripts associated with the recording process)

## Analysis Pipeline
Scripts that are ancillary (not precursors to the next step in the pipeline) are indented.

- `010-merge-eeg-raws.py`: convert BrainVision data format to `mne.io.Raw` objects; deal with subjects who have two separate recordings (due to equipment malfunction / restarting blocks); auto-add annotations to ignore between-block periods.
- `015-reannotate.py`: interactive annotation of `Raw` files: to mark movement artifacts, bad channels, or other undesirable noise in the data.
- `018-add-projectors.py`: detect blinks and add SSP projectors to remove blink artifacts. Also bandpass-filters the data.
- `020-make-epochs.py`: epoching, baseline correction, and downsampling.
    - `022-check-snr.py`: compares baseline power to evoked power, to assess how good a job the preprocessing did at data cleaning.
    - `025-plot-erps.py`: sanity check that the ERPs look reasonable.
- `030-dss.py`: run denoising source separation on epoched data.
    - `031-validate-dss.py`: plot relative signal power per DSS component. Adjust parameter `[dss][n_components]` based on this plot.
    - `032-plot-dss-topomap.py`: plot scalp topography of the DSS components.
    - `035-find-redundant-features.py`: determine which phonological features are equivalent across different feature systems, so we don’t unnecessarily run redundant classifiers. Add parameter `[feature_mappings]` based on results of this script.
- `036-munge-feature-sets.py`: combine all feature systems into a single spreadsheet. This determines the classifiers that will be trained and the labels they will get with their training data.
- `037-time-domain-redux.py`: reduce correlation of time samples via PCA, and unroll channels (or DSS components) to make classifier-friendly unidimensional vector for each trial.
    - `038-check-trial-counts.py`: sanity check that dropping noisy epochs did not cause too great an imbalance across phone types.
- `039-make-parallel-jobfile.py`: generates the Bash lines that call the classification script with command line args for each subject and feature.
- `040-classify.py`: The machine learning workhorse that runs the grid search / cross validation.
    - `042-rank-classifier-performance.py`: generates sorted EER tables and generates plots comparing classifier performance for each feature system.
- Three scripts to convert classifier output into various kinds of confusion matrices:
    - `050a-make-eer-confusion-matrices.py`: Make confusion matrices based on a fixed error rate for each feature, determined by that feature’s classifier’s “equal error rate” (EER).
    - `050b-make-phone-confusion-matrices.py`: Make confusion matrices based on the phone-level error rates from classifying the held-out data.
    - `050c-make-theoretical-confusion-matrices.py`: Make confusion matrices based on a fixed error rate across all features, for simulating robustness of feature systems to varying levels of noise.
- Two scripts to order the rows/columns of the confusion matrices in informative ways:
    - `052a-optimal-matrix-sorting.py`: Re-orders the rows/columns of confusion matrices using hierarchical clustering with “optimal leaf ordering”.
    - `052b-featural-matrix.sorting.py`: Re-orders the rows/columns of confusion matrices using hierarchical clustering with “optimal leaf ordering”.
    - Some scripts for measuring the diagonality of the resulting matrices:
        - `055-measure-diagonality.py`: Compute how much of the weight of each matrix lies near the diagonal.
        - `056-plot-diagonality.py`: Plot diagonality values for each subject.
        - `057-plot-snr-vs-diagonality.py`: Plot SNR vs diagonality.
- `060-plot-confusion-matrices.py`: Plots grids of confusion matrices for comparing performance of the different feature systems.
