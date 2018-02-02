---
title: Does the brain encode phonological features? Evaluating feature theories against EEG measures of phoneme perception
author:
- name: Daniel R. McCloy
- name: Adrian K. C. Lee
  email: akclee@uw.edu
  affiliation:
  - Institute for Learning and Brain Sciences, University of Washington
documentclass: article
classoption: oneside
fontsize: 12pt
geometry:
- letterpaper
- margin=1in
csl: bib/jasa-numeric.csl
bibliography: bib/eeg-phone-coding.bib
abstract: >
  Foo. Bar. Baz.
---

<!--
# BIG PICTURE QUESTIONS

- SYSTEM LEVEL: which feature system from the literature best captures the contrasts that are recoverable with EEG?

- PHONEME LEVEL: which consonants are well-discernable based on EEG+classifier? Is it consistent with what we'd expect based on temporal properties of the consonants? Is the set different for different feature systems?

- FEATURE LEVEL: which phonological features are more/less learnable from EEG?

# TODO
- [x] figure out why classifier failures
- [x] average maps
- [x] subtraction from simulated maps?
- [ ] stats on diagonality?
- unpack the dendrograms: what do we learn from them about:
    - [ ] features within a system
    - [ ] diffs. between the systems
- discussion points:
    - [ ] sparsity
    - [ ] levels of computation / representation in the brain
-->

# Introduction

Phonemes are the abstract representations of speech sounds that represent all
and only the contrastive relationships between sounds (i.e., the set of sounds
different enough to change the identity of a word if one sound were substituted
for another).  Within any given language, the set of  phonemes is widely held
to be structured in terms of *phonological distinctive features* (hereafter
“phonological features”) — properties common to some subset of the phoneme
inventory.  For example, all phonemes involving complete occlusion of the oral
tract might share the feature “non-continuant,” to the exclusion of all other
phonemes in the language.

Phonological features have been called the most important advance of linguistic
theory of the 20th century, for their descriptive power in capturing sound
patterns, and for their potential to capture the structure of phonological
knowledge as represented in the brain [@MielkeHume2006].  Few linguists would
dispute the first of these claims; as every first-year phonology student
learns, descriptions of how a phoneme is pronounced differently word-initially
versus word-medially, or in stressed versus unstressed syllables, can be
readily generalized to other phonemes undergoing similar pronunciation changes
if the change is expressed in terms of features rather than individual phones.
Phonological features are equally useful for describing sound change over time
[@BrombergerHalle1989], such as Grimm’s law describing parallel changes of
reconstructed proto-Indo-European stops /bʱ dʱ ɡʱ ɡʷʱ/ into fricatives
/ɸ θ x xʷ/ in proto-Germanic.

In contrast, the promise of phonological features as a model of speech sound
representation or processing in the human brain is far from being conclusively
established<!--, in part because the epistemic basis of phonological features
is still unclear-->.  There is growing evidence that neural representation in
the superior temporal gyrus predominantly reflects *acoustic phonetic* features
such as voice onset time, vowel formant frequency, or other spectrotemporal
properties of speech [@MesgaraniEtAl2014; @LeonardEtAl2015], <!--; that the response properties of these populations are often non-linear with respect to acoustic properties of the input, which would support a transformation from continuous to categorical representations;--> and that more abstract representations of phoneme or syllable identity are represented in adjacent temporal areas such as posterior superior temporal sulcus [@VadenEtAl2010] **and Spt XXX**, as well as frontal areas such as left inferior frontal sulcus [@MarkiewiczBohland2016] and left precentral gyrus and sulcus [@EvansDavis2015].

**TODO** Review studies specifically addressing features [@Correia2015; @ArsenaultBuchsbaum2015] various Lahiri papers, and Evans & Davis’s lack of findings.

**TODO** maybe review some neuro speech *production* papers re: phonological features

**TODO** review some relevant ling papers re: phonological features (e.g., word-specific phonetics (Pierrehumbert), maybe the different lexical extents of sound changes like canadian raising)

# Methods

**TODO** add methods figure.

## Stimulus design

Auditory stimuli were recordings of isolated consonant-vowel (CV) syllables
from four American English talkers (2 male, 2 female).  <!--The two females
were natives of the Pacific Northwest, and the two males were natives of the
U.S. Midwest residing in the Pacific Northwest; all four participate in the
low-back vowel merger.-->  Recordings were made in a sound-attenuated booth at
a sampling frequency of 44.1 kHz and 16-bit sample precision.  During
recording, syllables were presented orthographically on a computer screen;
example words with target syllables highlighted were also presented to
disambiguate orthographic representations as necessary.  All consonant phonemes
of English were used except /ŋ/ (which is phonotactically restricted to coda
position in English); in all syllables the vowel spoken was /ɑ/.  Several
repetitions of the syllable list were recorded, to ensure clean recordings of
each syllable type.

The recordings were segmented by a phonetician to mark the onset and offset of
each CV syllable, as well as the consonant-vowel transition.  Recordings were
then highpass filtered using a fourth-order Butterworth filter with a cutoff
frequency of 50 Hz (to remove very low frequency background noise present in
some of the recordings).  The segmented syllables were then excised into
separate WAV files, and root-mean-square normalized to equate loudness across
talkers and stimulus identities.  Final duration of stimuli ranged from 311 ms
(female 2 /tɑ/) to 677 ms (female 1 /θɑ/).

One male and one female talker were arbitrarily specified as “training”
talkers, and the other two as “test” talkers.  For the training talkers, 3
recordings of each syllable were included in the stimulus set; for the test
talkers, only one token of each syllable was included.  Each stimulus token was
presented 20 times during the experiment.  The imbalance between training and
test talkers was used to ensure sufficient training data for the classifiers,
and to avoid an uncomfortably long experiment duration.  Total number of
syllables presented was 23 consonants × (3 training tokens + 1 test token) × 2
talker genders × 20 repetitions = 3680 stimulus presentations.  Additionally,
intermixed with the English syllables were a smaller number of similar
monosyllabic stimuli spoken by native speakers of four other languages; those
data are not analyzed or discussed further here.

## Participants

Twelve listeners (9 female, 19-67 years, median 26) were recruited for this
experiment, and were paid an hourly rate for participation.  All procedures
were approved by the University of Washington Institutional Review Board.

## Procedure

Stimuli were delivered through a TDT RP2 real-time processor (Tucker Davis
Technologies, Alachula, FL) via Etymotic ER-2 insert earphones, at a level of
65 dB SPL.  Inter-stimulus interval was varied uniformly between 300 and 800 ms
to avoid rhythmicity of the stimulus presentation, which could induce
anticipatory listening effects.  Stimuli were presented in a unique random
order for each listener.  The listening task was passive; no verbal,
button-press, or other response was required of the participant.

To forestall boredom, listeners were shown cartoons during presentation of the
auditory stimuli.  The cartoons were episodes of Shaun the Sheep (6-7 minutes
in duration) edited to remove the opening and closing credits, and presented
without audio or subtitles (the cartoon does not normally include any dialog,
so the plots are easy to follow without subtitles or sound).  Cartoon episode
order was randomized for each participant.  Auditory stimulus blocks were timed
to correspond to the duration of the cartoons.  Listeners were given the chance
to take breaks between each block, and encouraged to sit still during each
block to reduce motion artifacts in the EEG recording.

## Data acquisition

During stimulus presentation, EEG signals were continuously recorded using a
BrainVision 32-channel ActiChamp system at 1000 Hz sampling frequency.
Electrode placement followed a modified form of the standard 10-20 montage,
with electrode TP9 moved to A1 (earlobe reference) and electrodes FT9 and FT10
moved to POO9h and POO10h (for compatibility with concurrent studies using
shared equipment).  Prior to each stimulus, a unique binary identifier of
stimulus identity was sent from the presentation computer to the EEG
acquisition computer via TTL (carried on the third and fourth bits) and
recorded alongside the EEG signal.  A second TTL signal (a single
least-significant bit) was sent from the TDT RP2 to the acquisition computer,
synchronized to the stimulus onset.  This allowed reliable confirmation of
stimulus identity and timing during post-processing.

## Data cleaning

EEG signal processing was carried out in python using the `mne-python`
module.[@mnepython]  Raw EEG traces were manually inspected and annotated for
bad channels, and for spans of time with gross movement artifacts.  For a given
subject, 0 to 3 channels were marked “bad” and excluded from further
processing.  Subsequently, a blink detection algorithm was applied to the
signal from forehead electrode FP1 (or in one case FP2, because FP1 was
extremely noisy for that subject).  Annotation to exclude segments of the
recording was repeated until the algorithm labels were in close agreement with
experimenter-judged blinks.  Blinks were then extracted in 1-second epochs
centered around the peak, which were used to generate a signal-space projection
(SSP) operator [@SSP] separating blink-related activity from activity due to
other sources.  Four SSP components were necessary to remove blink artifacts
across all subjects.  Before applying SSP projectors, the data were
mean-subtracted and bandpass-filtered using zero-phase FIR filters, with
cutoffs at 0.1 Hz and 40 Hz and transition bandwidths of 0.1 Hz and 10 Hz.

Next, epochs were created around stimulus onsets, and baseline-corrected to
yield zero mean for the 100 ms immediately preceding stimulus onset.  Prior
annotations of the signal were ignored at this stage, but epochs with absolute
voltage changes exceeding 75 μV in any channel (excluding channels previously
marked as “bad”) were dropped.  Across subjects, between 3.6% and 8.8% of the
3680 English syllable presentations were dropped.

<!-- TODO: discussion / plot of SNR here? -->

Retained epochs were time-shifted to align on the consonant-vowel transition
time instead of the stimulus onset.  This was done because information about
consonant identity is encoded in the first ~100 ms of following vowels, so
temporal alignment of this portion of the EEG response should hopefully improve
the ability of classifiers to learn consonant features.  After this
re-alignment, epochs had a temporal span from −335 ms (the onset of the longest
consonant), to +733 ms (200 ms after the end of the longest vowel), with $t=0$
fixed at the consonant-vowel transition point.

## Dimensionality reduction

Retained, transition-aligned epochs were downsampled from 1000 Hz to 100Hz
sampling frequency to speed further processing.  Spatial dimensionality
reduction was performed with denoising source separation
(DSS).[@SarelaValpola2005; @deCheveigneSimon2008]  DSS creates orthogonal
virtual channels (or “components”) that are linear sums of the physical
electrode channels, constrained by a bias function (typically the average
evoked response across trials of a given type) that orients the components to
maximally approximate the bias function.  In this case a single per-subject
bias function was used, based on the average of all epochs for that subject.
This was a more conservative strategy than is typically seen in the literature,
where separate bias functions are used for each experimental condition.  The
conservative strategy was chosen to eliminate a possible source of variability
in classifier performance: if computing separate bias functions for each
consonant or for each consonant token, there would be potentially different
numbers of trials being averaged for the different consonants / tokens.  This
could lead to different SNRs for the different consonants, in a way that might
bias the classifiers or make certain classifiers’ tasks easier that others.  An
alternative approach would have been to equalize the trial counts for each
consonant prior to DSS analysis, but this was rejected due to the exploratory
nature of this research (i.e., since the overall amount of data needed was not
known, more trials with processing akin to “shrinkage toward the mean” was
deemed preferable to fewer trials with less conservative processing).

Based on scree plots of the relative signal power in each DSS component, five
components were retained for all subjects, replacing the 28-31 retained
electrode channels with 5 orthogonal DSS components for each subject.  Temporal
dimensionality reduction was also applied using PCA, to reduce collinearity of
(adjacent) temporal samples.  Such collinearity can make it difficult for a
classifier to learn a decision boundary from the training data.  For 6 of the
12 subjects, this reduced the 107 time points to 100 or 101 time points; for
the remaining 6 subjects there was no reduction. Finally, the 5 DSS components
for each epoch were unrolled into a one-dimensional vector to be passed to the
classifiers.

## Supervised learning

At this point, data for one subject comprises a numeric matrix of ~3400 rows
(one row = one retained epoch) and ~500 columns (5 unstacked DSS components of
~100 time points each).  Each row could be labelled with the consonant that was
presented during that epoch, posing a 23-way classification problem that could
be approached with various multiclass classification techniques.  Instead, the
approach used here attempts to solve 9 (PSA), 10 (PHOIBLE), or 11 (SPE)
binary classification problems, by re-labeling the epochs using the
phonological feature values associated with each consonant, and training a
separate classifier for each phonological feature.

In initial experiments, the classifier was a support vector machine (SVM) with
a radial basis function kernel, which allows nonlinear decision boundaries to
be learned readily.  SVM solutions were highly unstable across repeated runs
with different random seeds, logistic regression classifiers were used instead.
Classifier fitting used stratified 5-fold cross-validation at each point in a
grid search for hyperparameter $C$ (regularization parameter; small $C$ yields
smoother decision boundary)<!-- and $\gamma$ (kernel variance parameter,
controlling how far from the decision boundary a data point can be and still
count as a support vector; small $\gamma$ yields fewer support vectors,
which often leads to smoother decision boundaries)-->.  The grid search was
quite broad; $C$ ranged logarithmically (base 2) from $2^{-5}$ to $2^{16}$
<!--and $\gamma$ from $2^{-15}$ to $2^{4}$-->.

Many phonological feature values are unevenly distributed (e.g., of the 23
consonants presented, only 6 are “+strident”), posing a classification problem
known as “class imbalance” which can lead to high accuracy scores by simply
guessing the more numerous class in most or all cases.  To address this, a
custom score function was written for the classifiers that moved the decision
boundary threshold from 0.5 to a value that equalized the *ratios* of incorrect
classifications (i.e., equalized false positive and false negative *rates*
instead of equalizing the raw number of such errors), and returned a score of 1
− equal error rate.

Additionally, many phonological feature systems involve some sparsity (i.e.,
some features are undefined for certain phonemes; for example, in the PSA
feature system, the feature “nasal” is unvalued for phonemes /h l ɹ j w/).  In
such cases, trials for phonemes that are undefined on a given phonological
feature were excluded from the training set for that classifier.

After grid search, each classifier was re-fit on the full set of training data,
using the best hyperparameters.  Held-out test data (trials from the other two
English talkers) were then submitted to the classifiers, and the resulting
classifications <!--and classification probabilities-->were recorded for each
trial.

## Aggregation of classifier results

At this point, data for one subject comprises a matrix of ~900 rows (1 per test
trial) and 9-11 columns (1 per phonological feature classifier; number of
columns depends on which phonological feature system is being analyzed); each
cell is a 0 or 1 classification for that combination of trial and phonological
feature.  From these data, the accuracy of each classifier (subject to the same
equal-error-rate constraint used during training) was computed.  Next, a
23×23×N array was constructed (where N is the number of phonological features
in the current system, i.e., between 9 and 11).  The first axis represents
which consonant was presented in the stimulus, and the second axis represents
which consonant was likely perceived by the listener (as estimated from the EEG
signal).  For each plane along the third axis (i.e., for a given phonological
feature classifier), the cells of the plane are populated with either the
accuracy or the error rate (1 minus the accuracy) of that classifier, depending
on whether the consonant indices of the row and column of that cell match
(accuracy) or mismatch (error rate) in that feature.  For example, a cell in
row /p/, column /b/ of the voicing classifier plane would have as its entry the
error rate of the voicing classifier, signifying the probability that a
voiceless /p/ would be mis-classified as a voiced /b/.

Finally, the feature planes are collapsed by taking the product along the last
dimension, yielding the joint probability that the input consonant (the
stimulus) would be classified as any given output consonant (the percept).  For
features involving sparsity, cells corresponding to undefined feature values
were given a chance value of 0.5; results were broadly similar when undefined
values were left as unvalued and excluded from the computation.

## Analysis of confusion matrices

**TODO: resume here** At this point, there is one 23×23 matrix for each feature system; a cell in such a matrix represents the probability that...

## Calculation of diagonality


# Results

## SPE, JFH, PHOIBLE

![Confusion matrices for three baseline conditions (top row) and the three feature systems (bottom row).  TODO: say more](../../figures/publication/fig-avg-confmats.pdf)

## phone-specific error rates

![Error rates for pairwise classifiers. The three highest confusions are labelled.](../../figures/publication/fig-avg-pairwise-error.pdf)

## Specific features

**TODO** add figure.

# Discussion

- Sparsity
- EEG limitations
- Variability across subjects

Of course, neural systems are hierarchical, there's not just one level of representation...  certainly both auditory templates and motor plans both exist...  maybe there is no sense in which our phonological knowledge can be fully distilled, and descriptive adequacy is the best phonologists can do.  (i.e., there is no answer to what is ***the*** most fundamental level of representation)

## Future directions

- not just consonants
- MEG
- ArtPhon / FUL
- Foreign phones
- AG’s suggestion

# References

\setlength{\parindent}{-0.25in}
\setlength{\leftskip}{0.25in}
\noindent
