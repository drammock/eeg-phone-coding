# Exploring speech sound coding using EEG
Experiment to test performance of EEG signal classifiers trained on responses to CV-syllables using only English consonant contrasts, when tested on foreign consonant contrasts.

## Pipeline:
- `slide-prompts` folder: Generate textual prompts for recording foreign-language stimuli
- `stimulus-generation` folder: Process audio recordings to generate stimuli
- `run-experiment.py`: Run experiment and collect EEG data (done on a separate acquisition computer using BrainVision pyCorder software; no associated scripts)
- `clean-eeg.py`: Process EEG data
- `classify_eeg.py`: Train classifiers on EEG data (responses to English syllables) and classify remaining EEG data (responses to foreign syllables)
- analyze, plot, etc (WIP)
