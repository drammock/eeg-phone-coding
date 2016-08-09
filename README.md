# Exploring speech sound coding using EEG
Experiment to test performance of EEG signal classifiers trained on responses to CV-syllables using only English consonant contrasts, when tested on foreign consonant contrasts.

## Pipeline:
- `slide-prompts` folder: Generate textual prompts for recording foreign-language stimuli
- `stimulus-generation` folder: Process audio recordings to generate stimuli
- `run-experiment.py`: Run experiment and collect EEG data (done on a separate acquisition computer using BrainVision pyCorder software; no associated scripts)
- `clean-eeg.py`: Preprocess EEG data
- `apply-dss-and-merge-subjects.py`: combine cleaned EEG data across subjects
- `classify-eeg.py`: Train classifiers on EEG data (responses to English syllables) and classify remaining EEG data (responses to foreign syllables)
- analyze: `make-feature-based-confusion-matrices.py`, `make-confmats-from-classifier-output.py`, `apply-weights-and-column-order.py`
- plot: `plot-weighted-confusion-matrices.py`

## TODOs

### clustering
- [ ] determine optimal number of DSS channels to use (penalized logistic regression?)
- [ ] compute “mediod” (cluster center analogue) for spectral clustering approach (point corresponding to row of within-cluster pairwise distance matrix that has lowest sum)

### classification
- [ ] variance analysis across listeners: investigate cross-subject generalizability (train on 1/2/4/8 subjects instead of all 12). If true, justifies using pre-computed EEG-based confusion matrices to weight the G2P for transcriptions of any (native English-speaking) crowd worker.
- [ ] switch from PHOIBLE phoneme sets to the phone sets determined by the PT system's G2P output
- [ ] find coefficients for a linear combination of the two confusion matrices (as was done last summer) by testing their performance in the mismatched transcript system and using cross-validation to optimize (JHU CLSP server)
- [ ] use EEG responses to the 2 held-out English talkers as a cross-validation set to optimize classifier thresholds using F1 score instead of equal error rate (not sure if this will be particularly useful).
- [ ] try other classifiers?

### experimental
- [ ] repeat experiment focusing on vowels instead of consonants
- [ ] explore efficacy of ESR for different types of foreign contrasts (register tone, contour tone, clicks, ejectives, voice quality distinctions...)
- [ ] explore differences in EEG-based confusion matrices from listeners with different native languages (Spanish, Mandarin, others?), esp. w/r/t certain contrasts.
