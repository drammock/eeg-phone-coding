---
title: >
  EEG-derived phoneme confusion matrices show the fit between phonological feature systems and brain responses to speech
author:
- name: Daniel R. McCloy
  orcid: 0000-0002-7572-3241
- name: Adrian K. C. Lee
  orcid: 0000-0002-7611-0500
  email: akclee@uw.edu
  affiliation:
  - University of Washington, Institute for Learning and Brain Sciences, 1715 NE Columbia Rd., Seattle, WA 98195-7988
bibliography: bib/eeg-phone-coding.bib
abstract: >
  This paper describes a technique to assess the correspondence between patterns of similarity in the brain’s response to speech sounds and the patterns of similarity encoded in phonological feature systems, by quantifying the recoverability of phonological features from the neural data using supervised learning. The technique is applied to EEG recordings collected during passive listening to consonant-vowel syllables. Three published phonological feature systems are compared, and are shown to differ in their ability to recover certain speech sound contrasts from the neural data. For the phonological feature system that best reflects patterns of similarity in the neural data, a leave-one-out analysis indicates some consistency across subjects in which features have greatest impact on the fit, but considerable across-subject heterogeneity remains in the rank ordering of features in this regard.
articletype: research article
keywords:
- neurolinguistics
- phonology
- phonological features
- speech perception
- EEG
---

<!-- ORDER: title page; abstract (150 words); keywords (4-8); main text
introduction, materials and methods, results, discussion; acknowledgments;
declaration of interest statement; references; appendices (as appropriate);
table(s) with caption(s) (on individual pages); figures; figure captions (as a
list) -->

<!-- FIGURES: 3.25 in (single column width); 6.75 in (full-width); 9.5 in (page height) -->

# Introduction

Phonemes are the abstract representations of speech sounds that represent all
and only the contrastive relationships between sounds [i.e., the set of sounds
different enough to change the identity of a word if one sound were substituted
for another; @Jones1957].  Within any given language, the set of  phonemes is
widely held to be structured in terms of *phonological distinctive features*
(hereafter “phonological features”) — properties common to some subset of the
phoneme inventory.  For example, in some phonological feature systems, all
phonemes characterised by sounds made with complete occlusion of the oral tract
share the feature “non-continuant.”  There are a number of competing hypotheses
regarding the particular details of what the features are, and whether there is
a single universal feature system or a separate feature system for each
language.

Phonological features have been called the most important advance of linguistic
theory of the 20th century, for their descriptive power in capturing sound
patterns, and for their potential to capture the structure of phonological
knowledge as represented in the brain [@MielkeHume2006].  Few linguists would
dispute the first of these claims; as every first-year phonology student
learns, descriptions of how a phoneme is realised differently in different
contexts can be readily generalised to other phonemes undergoing similar
pronunciation changes if the change is expressed in terms of phonological
features rather than individual sounds. To give a common example, English
voiceless stops \ipa{/p t k/} are aspirated in word-initial position or before
a stressed vowel; a change that in the feature system of Hayes [-@Hayes2009]
can be expressed as:

$$\begin{bmatrix}\textrm{−voiced, −continuant, −delayedRelease}\end{bmatrix}
\rightarrow \begin{bmatrix}\textrm{+spreadGlottis}\end{bmatrix} / \left\{
\begin{matrix}
\textrm{\#}\underline{\hspace{9pt}}\phantom{\begin{bmatrix}\textrm{+stress, +vocalic}\end{bmatrix}} \\
\phantom{\textrm\#}\underline{\hspace{9pt}}\begin{bmatrix}\textrm{+stress, +vocalic}\end{bmatrix}
\end{matrix}
\right.$$

where the term left of the arrow captures the class undergoing contextual
change, the term between the arrow and the slash describes the change that
occurs, and the term right of the slash describes the context(s) in which the
change occurs. Phonological features are equally useful for describing sound
change over time, such as Grimm’s law describing parallel changes of
reconstructed proto-Indo-European stops \ipa{/bʱ dʱ ɡʱ ɡʷʱ/} into the
fricatives \ipa{/ɸ θ x xʷ/} of proto-Germanic [@BrombergerHalle1989].

In contrast, the promise of phonological features as a model of speech sound
representation or processing in the human brain is far from being conclusively
established<!--, in part because the epistemic basis of phonological features
is still unclear-->.  There is growing evidence for particular brain regions
<!-- (esp. superior temporal gyrus) -->
representing *acoustic phonetic* features such as voice onset time, vowel
formant frequency, or other spectrotemporal properties of speech
[@MesgaraniEtAl2014; @LeonardEtAl2015], <!--; that the response properties of
these populations are often non-linear with respect to acoustic properties of
the input, which would support a transformation from continuous to categorical
representations;--> and for more abstract representations of phoneme or
syllable identity in other brain regions [@VadenEtAl2010;
@MarkiewiczBohland2016; @EvansDavis2015].<!--(adjacent temporal areas such as
posterior superior temporal sulcus (Vaden) and the sylvian parieto-temporal
region, as well as frontal areas such as left inferior frontal sulcus
(Markiewicz) and and left precentral gyrus and sulcus (EvansDavis) -->
However, some studies that have looked for evidence of phonological
feature representations in the brain have used stimuli that don’t
distinguish whether the neural representation reflects a truly abstract
phonological category or merely spectrotemporal properties shared among the
stimuli [@ArsenaultBuchsbaum2015], and in one study that took pains to rule out
the possibility of spectrotemporal similarity, no evidence for the
representation of phonological features was found [@EvansDavis2015].

Moreover, in most studies of how speech sounds are represented in the brain,
the choice of which features to investigate (and which sounds to use in
representing them) is often non-standard from the point of view of phonological
theory. For example, one recent study [@ArsenaultBuchsbaum2015] grouped the
English consonants into five place of articulation “features” (labial, dental,
alveolar, palatoalveolar, and velar); in contrast, a typical phonological
analysis of English would treat the dental, alveolar, and palatoalveolar
consonants as members of a single class of “coronal” consonants, with
differences between dental \ipa{/θ ð/}, alveolar \ipa{/s z/}, and
palatoalveolar \ipa{/ʃ ʒ/} fricatives encoded through features such as
“strident” (which groups \ipa{/s z ʃ ʒ/} together) or “distributed” [which
groups \ipa{/θ ð ʃ ʒ/} together; @Hayes2009]. Consequently, encoding the
dental, alveolar, and palatoalveolar sounds as completely disjoint sets ignores
the fact that those sounds tend to pattern together in speech, and fails to
test the relationship between the neural representation of phonemes and
phonological models of structured relationships among phonemes.

<!-- TODO? give examples of rules that apply to strident and distributed
classes -->

An exception to this trend of mismatches between the features used or tested in
neuroscience experiments and the features espoused by phonological theory is
the work of Lahiri and colleagues [see @LahiriReetz2010 for overview]. However,
their work tests hypotheses about specific phonological contrasts in specific
languages, such as the consonant length distinction in Bengali
[@RobertsEtAl2014] or vowel features in German [@ObleserEtAl2004], but has not
thusfar tested an entire system of phonological features against neural
recordings.

<!-- TODO? maybe review some relevant ling papers re: phonological features
(e.g., word-specific phonetics (Pierrehumbert), maybe the different lexical
extents of sound changes like Canadian raising) -->

To address these issues, the present study takes a somewhat different approach.
First, rather than testing a specific phonemic contrast, the experiments
reported here address all consonant phonemes of English (with one exception,
\ipa{/ŋ/}, which never occurs word-initially in natural English speech).
Second, these experiments assess the fit between neural recordings during
speech perception and several different feature systems drawn directly from the
phonological literature.

# Materials and Methods

An overview of the analysis pipeline is given in figure \ref{fig-methods}.
Briefly, syllables were presented auditorily while recording EEG from the
listener. EEG data were epoched around each presented syllable, underwent
dimensionality reduction, and were labelled with the phonological feature
values of each syllable’s consonant. Those labels were used to train a set of
phonological feature classifiers (one for each phonological feature in the
system), and the performance of the system of classifiers was combined to
create a confusion matrix summarising the similarity of neural responses to the
different consonants, as well as the degree to which the phonological features
defined by the given system capture those patterns of similarity.

![Methods overview. **A:** portion of a stimulus waveform presented to one subject, with one stimulus highlighted; layout of EEG sensors is shown at right. **B:** corresponding EEG recording with middle epoch highlighed. Light-colored channels are “bad” channels for this subject, and are excluded from processing (hence they show large deflections due to blinks, which have been regressed out of the “good” channels). **C:** the highlighted epoch from panel B after Denoising Source Separation (DSS) to reduce dimensionality across channels. **D:** phonological feature matrix, with heavy boxes indicating values for the three syllables shonw in panel A for a given feature (“voiced”). **E:** DSS signals after concatenation, labelled with the “voiced” feature values from panel D and ready for classification. The process is repeated for each of the phonological features (rows) in panel D. **F:** confusion matrix made by aggregating classification results across all phonological features in the phonological system.\label{fig-methods}](fig-methods-diagram.eps)

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
of English were used except \ipa{/ŋ/} (which is phonotactically restricted to
coda position in English); in all syllables the vowel spoken was \ipa{/ɑ/}.
Several repetitions of the syllable list were recorded, to ensure clean
recordings of each syllable type.

The recordings were segmented by a phonetician to mark the onset and offset of
each CV syllable, as well as the consonant-vowel transition.  Recordings were
then highpass filtered using a fourth-order Butterworth filter with a cutoff
frequency of 50 Hz (to remove very low frequency background noise present in
some of the recordings).  The segmented syllables were then excised into
separate WAV files, and root-mean-square normalised to equate loudness across
talkers and stimulus identities.  Final duration of stimuli ranged from 311 ms
(female 2 \ipa{/tɑ/}) to 677 ms (female 1 \ipa{/θɑ/}).

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
experiment, and were paid an hourly rate for participation.  All participants
had normal audiometric thresholds (20 dB HL or better at octave frequencies
from 250 Hz to 8 kHz).  All procedures were approved by the University of
Washington Institutional Review Board.

## Procedure

Stimuli were delivered through a TDT RP2 real-time processor (Tucker Davis
Technologies, Alachula, FL) via Etymotic ER-2 insert earphones, at a level of
65 dB SPL.  Inter-stimulus interval was varied uniformly between 300 and 800 ms
to avoid rhythmicity of the stimulus presentation, which could induce
anticipatory listening effects.  Stimuli were presented in a unique random
order for each listener.  The listening task was passive; no verbal,
button-press, or other response was required of the participant.

To forestall boredom, listeners were shown cartoons during presentation of the
auditory stimuli.  The cartoons were episodes of *Shaun the Sheep*
[@ShaunTheSheep; 6-7 minutes in duration] edited to remove the opening and
closing credits, and presented without audio or subtitles (the cartoon does not
normally include any dialog, so the plots are easy to follow without subtitles
or sound).  Cartoon episode order was randomised for each participant.
Auditory stimulus blocks were timed to correspond to the duration of the
cartoons.  Listeners were given the chance to take breaks between each block,
and encouraged to sit still during each block to reduce motion artifacts in the
EEG recording.

## Data acquisition

During stimulus presentation, EEG signals were continuously recorded using a
BrainVision 32-channel ActiChamp system at 1000 Hz sampling frequency.
Electrode placement followed a modified form of the standard 10-20 montage,
with electrode TP9 moved to A1 (earlobe reference) and electrodes FT9 and FT10
moved to POO9h and POO10h from the 10-05 montage (for compatibility with
concurrent studies using shared equipment).  Prior to each stimulus, a unique
binary identifier of stimulus identity was sent from the presentation computer
to the EEG acquisition computer via TTL;<!-- (carried on the third and fourth
bits) and recorded alongside the EEG signal.  A-->
a second TTL signal <!--(a single least-significant bit)-->
was sent from the TDT RP2 to the acquisition computer, synchronised to the
stimulus onset.  This allowed reliable confirmation of stimulus identity and
timing during post-processing.

## Data cleaning

EEG signal processing was carried out in python using the `mne-python`
module [@mnepython].  Raw EEG traces were manually inspected and annotated for
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
[DSS; @SarelaValpola2005; @deCheveigneSimon2008].  DSS creates orthogonal
virtual channels (or “components”) that are linear sums of the physical
electrode channels, constrained by a bias function (typically the average
evoked response across trials of a given type) that orients the components to
maximally recover the bias function.  In this case a single per-subject
bias function was used, based on the average of all epochs for that subject.
This was a more conservative strategy than is typically seen in the literature,
where separate bias functions are used for each experimental condition.  The
conservative strategy was chosen to eliminate a possible source of variability
in classifier performance: if computing separate bias functions for each
consonant or for each consonant token, there would be potentially different
numbers of trials being averaged for the different consonants / tokens.  This
could lead to different SNRs for the different consonants, in a way that might
bias the classifiers or make certain classifiers’ tasks easier that others.  An
alternative approach would have been to equalise the trial counts for each
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
(one row = one retained epoch or “trial”) and ~500 columns (5 unstacked DSS
components of ~100 time points each).  Recall that ¾ of the total trials are
from training talkers, so ~2500 trials are available to train classifiers,
employing various labelling and partitioning strategies; the remaining ~900
trials are then used to assess the information that can be recovered from the
neural signals.

In experiment 1, each trial is labelled with the consonant that was presented
during that epoch, and a classifier was trained to discriminate between each
pair of consonants (for 23 consonants, this yields 276 pairwise comparisions,
with each comparison having ~220 trials of training data). This initial
experiment serves as a “sanity check” that sufficient information about the
neural processing of consonant identity is picked up by EEG, and that the
information remains in the neural signals after all preprocessing steps have
been carried out.

In experiment 2, trials are again labelled using consonant identity, but
instead of pairwise classifiers, 23 one-versus-rest (OVR) classifiers were
trained, each receiving the full set of training data and learning to identify
a different “target” consonant. This serves as a baseline condition against
which the phonological-feature-based classifiers can be compared.
<!-- It is analogous to a model of phoneme perception in which a bank of
phoneme detectors all receive the same input, and a whichever detector is most
confident determines what is perceived. -->

In experiment 3, trials were labeled with phonological feature values of the
consonant heard on that trial. One classifier was trained for each phonological
feature, and three different feature systems were explored: the system
described in Jakobson, Fant, and Halle’s _Preliminaries to Speech Analysis_
[PSA; @JakobsonFantHalle1952], the system in Chomsky and Halle’s _Sound
Pattern of English_ [SPE; @ChomskyHalleSPE], and the system used in the
PHOIBLE database [@phoible2013], which is a corrected and expanded version of
the system described in Hayes’s _Introduction to Phonology_ [-@Hayes2009]. The
three feature systems each require a different number of features to encode all
English consonant contrasts (PSA uses 9, SPE uses 11, and PHOIBLE uses 10), but
all represent a reduction of the 23-way multiclass classification problem into
a smaller number of binary classification problems. Feature value assignments
in the three feature systems are illustrated in figure
\ref{fig-feature-matrices}.

![Feature matrices for the three feature systems used in Experiment 3. Dark gray cells indicate positive feature values, light gray cells indicate negative feature values, and white cells indicate phoneme-feature combinations that are undefined in that feature system. The names of the features reflect the original sources; consequently the same feature name may have different value assignments in different systems.\label{fig-feature-matrices}](fig-featsys-matrices.eps)

In all three experiments, logistic regression classifiers were used, as
implemented in `scikit-learn` [@sklearn]. Stratified 5-fold cross-validation
was employed at each point in a grid search for hyperparameter $C$
(regularisation parameter; small $C$ yields smoother decision boundary)
<!-- and $\gamma$ (kernel variance parameter, controlling how far from the
decision boundary a data point can be and still count as a support vector;
small $\gamma$ yields fewer support vectors, which often leads to smoother
decision boundaries)-->.  The grid search was quite broad; $C$
ranged logarithmically (base 2) from $2^{-5}$ to $2^{16}$ <!--and $\gamma$
from $2^{-15}$ to $2^{4}$-->.  After grid search, each classifier was re-fit on
the full set of training data, using the best hyperparameters.  The trained
classifiers were then used to make predictions about the class of the
consonant on each held-out test data trials; in experiments 1 and 2 those
class predictions were consonant labels; in experiment 3 those class
predictions were phonological feature values.

Because many phonological feature values are unevenly distributed (e.g., of the
23 consonants in the stimuli, only 6 are “+strident”), a custom score function
was written for the classifiers that moved the decision boundary threshold from
0.5 to a value that equalised the *ratios* of incorrect classifications (i.e.,
equalised false positive and false negative *rates*), minimised this value
(called the “equal error rate”), and returned a score of 1 − equal error rate.
This precludes the situation where high accuracy scores are achieved by simply
guessing the more numerous class in most or all cases.  Since such class
imbalance can occur in any of the experiments (and is virtually guaranteed to
happen for the one-versus-rest classifiers in experiment 2), a score function
that minimised equal error rate was used for all experiments.

Additionally, many phonological feature systems involve some sparsity (i.e.,
some features are undefined for certain phonemes; for example, in the PSA
feature system, the feature “nasal” is unvalued for phonemes \ipa{/h l ɹ j
w/}).  In such cases, trials for phonemes that are undefined on a given
phonological feature were excluded from the training set for that classifier.

## Aggregation of classifier results

At this point in experiment 3, data for one subject comprises a matrix of ~900
rows (1 per test trial) and 9-11 columns (1 per phonological feature
classifier; number of columns depends on which phonological feature system is
being analyzed). Each cell of that matrix is a 0 or 1 classification for that
combination of trial and phonological feature.  From these data, the accuracy
of each classifier (subject to the same equal-error-rate constraint used during
training) was computed.  Next, a 23×23×N array was constructed (where 23 is the
number of conosonants in the stimuli, and N is the number of phonological
features in the current system, i.e., between 9 and 11).  The first dimension
represents which consonant was presented in the stimulus, and the second
dimension represents which consonant was likely perceived by the listener (as
estimated from the EEG signal).  For each plane along the third axis (i.e., for
a given phonological feature classifier), the cells of the plane are populated
with either the accuracy or the error rate (1 minus the accuracy) of that
classifier, depending on whether the consonant indices of the row and column of
that cell match (accuracy) or mismatch (error rate) in that feature.  For
example, a cell in row \ipa{/p/}, column \ipa{/b/} of the voicing classifier
plane would have as its entry the error rate of the voicing classifier,
signifying the probability that a voiceless \ipa{/p/} would be mis-classified
as a voiced \ipa{/b/}.

Finally, the feature planes are collapsed by taking the product along the last
dimension, yielding the joint probability that the input consonant (the
stimulus) would be classified as any given output consonant (the percept).  For
features involving sparsity, cells corresponding to undefined feature values
were given a chance value of 0.5 (results were broadly similar when such cells
were coded as `NaN` and excluded from the computation).  The resulting 23×23
matrices can be thought of as confusion matrices: each cell gives the
probability that the bank of phonological feature classifiers in that feature
system would classify a heard consonant (given by the row label) as a
particular consonant percept (given by the column label).

In such matrices, a perfectly performing system would be an identity matrix:
values of 1 along the main diagonal, and values of 0 everywhere else.  To
quantify the extent to which each confusion matrix resembles an identity
matrix, a measure of relative diagonality was computed as in Equation
@eq-diagonality, where $A$ is the matrix, $i$ is an integer vector of
row or column indices (starting at 1 in the upper-left corner), and $w$ is a vector of ones of the same length as $i$.

(@eq-diagonality) $$\text{diagonality} = \frac{wAw^T \times iAi^T - iAw^T \times wAi^T}{\sqrt{wAw^T \times i^2Aw^T - (iAw^T)^2} \times \sqrt{wAw^T \times wA(i^2)^T - (wAi^T)^2}}$$


Briefly, this is an adaptation of the Pearson correlation for samples. It
computes the correlation between the matrix’s rows and columns, which reflects
the fraction of the total mass of the matrix that lies on the main diagonal.
This measure yields a value of 1 for an identity matrix, zero for a uniform
matrix, and −1 for a matrix with all its mass on the minor diagonal.  A
derivation of this measure is provided in the Supplementary Material.

This notion of diagonality requires that adjacent rows be relatively similar
and distant rows be relatively dissimilar (and likewise for columns);
otherwise, the formula’s notion of distance between rows (or columns) cannot
be justifiably applied.  Therefore, before computing diagonality, the matrices
were submitted to a heirarchical clustering of the rows using the optimal leaf
ordering algorithm [@BarJosephEtAl2001].

# Results

## Experiment 1: Pairwise classifiers

Results (aggregated across subjects) for the pairwise classifiers are shown in
figure \ref{fig-pairwise-confmat}.  Because the classifier score function
imposed an equal-error-rate constraint, there is no difference between, e.g.,
proportion of \ipa{/p/} trials mistaken for \ipa{/b/} and proportion of
\ipa{/b/} trials mistaken for \ipa{/p/}, so the upper triangle is omitted.

![Across-subject average accuracy/error for pairwise classifiers. Off-diagonal cells represent the error rates for the pairwise classifier indicated by that cell’s row/column labels; diagonal cells represent the mean accuracy for all pairs in which that consonant is one element.\label{fig-pairwise-confmat}](fig-pairwise.eps)

In general, the mean accuracy across subjects for a given pairwise comparison
was always above 90%; individual accuracy scores for each subject were generally above 80% and are shown in
the supplementary material, figure S4.  These plots indicate that consonant
identity can be recovered fairly well from brain responses to the stimuli.
However, a suite of pairwise classifiers is not a particularly realistic model
of how speech perception is likely to work: during normal comprehension it
isn’t generally the case that listeners are always choosing between 1 of 2
options for “what that consonant was.” Rather, consonant identification is a
closed-set identification task: listeners know the set of possible consonants
they might hear, and must determine which one was actually spoken. Experiment 2
provides a more realistic model of this scenario.

## Experiment 2: OVR classifiers

Results for the OVR classifiers in experiment 2 are shown in figures
\ref{fig-ovr-boxplot} and \ref{fig-ovr-confmat}. As a reminder, these
classifiers learn to discriminate brain responses to one particular consonant
against all other consonants as a group (which is a harder problem than
pairwise comparison, as the non-target class is much more heterogeneous).
Unsurprisingly, the OVR
classifiers were not as good as the pairwise classifiers at identifying
consonants from the EEG data; accuracies ranged from below chance to
near-perfect on individual subject data, with first quartiles between 74% and
82% and third quartiles between 83% and 92% (see figure \ref{fig-ovr-boxplot}).

![Within-subject distributions of accuracy for one-versus-rest classifiers. Boxes show quartiles; dots are individual classifier accuracies.\label{fig-ovr-boxplot}](fig-ovr-boxplot.eps)

Moreover, despite passable accuracy scores, the classifiers don’t actually
learn much, classifying both non-target and target trials as the target
consonant at similar rates (i.e., little within-column differentiation in
figure \ref{fig-ovr-confmat}A). The faint checkerboard pattern in the upper
left quadrant of the matrix does indicate that, e.g., the \ipa{/b/} classifier
in the second column is relatively more likely to mistakenly classify trials
with \ipa{/d/} or \ipa{/ɡ/} stimuli as \ipa{/b/} trials, than it is to make the
same mistake on trials with \ipa{/p/}, \ipa{/t/}, or \ipa{/k/} stimuli. There
is also some indication that classifiers for fricative consonants (especially
\ipa{/f s z ʃ ʒ/}) tended to make more misclassifications to trials with
(non-target) fricative stimuli, compared to trials with non-fricative stimuli
(the broad vertical stripe near the middle of the matrix, which is darker in
its upper and lower thirds). However, the classifiers’ ability to make these
basic discriminations of voiced versus voiceless stop consonants or fricatives
versus non-fricatives is still rather poor (e.g., the \ipa{/b/} classifier
marks 19% of \ipa{/p/} trials as \ipa{/b/}, and only 41% of \ipa{/b/} trials as
\ipa{/b/}). Finally, looking across classifiers for a given stimulus phoneme,
it is rarely the case that the most frequent classification is the correct one
(cf. lack of diagonal elements in figure \ref{fig-ovr-confmat}B), further
underscoring the impression that a bank of OVR classifiers is probably a poor
model of the information extraction carried out by the brain during speech
perception.

![Results for one-versus-rest classifiers, aggregated across subjects. Each column represents a single classifier, with its target class indicated by the column label. Row labels correspond to the test data input to each classifier. **A:** cells on the diagonal represent the ratio of true positive classifications to total targets (also called “hit rate” or “recall”); off-diagonal elements represent the ratio of false positive classifications to total non-targets (“false alarm rate”) for the consonant given by the row label. **B:** most frequent classification of each stimulus consonant, emulating across-classifier voting. Consonants that are correctly identified are indicated by dark gray cells along the main diagonal; consonants that are most frequently incorrectly identified are medium-gray cells.\label{fig-ovr-confmat}](fig-ovr.eps)

## Experiment 3: Phonological feature classifiers

Whereas experiments 1 and 2 test classification of neural signals based on
_identity_ of the consonant in each stimulus, experiment 3 tests classification
of the same signals based on _phonological feature values_ of those consonants,
and classifications of test data are aggregated across systems of phonological
features to yield consonant-level confusion matrices similar to those seen in
figures \ref{fig-pairwise-confmat} and \ref{fig-ovr-confmat}. The results of
this aggregation for the three phonological feature systems tested (PSA, SPE,
and PHOIBLE) are shown in figures \ref{fig-psa-confmat}, \ref{fig-spe-confmat},
and \ref{fig-phoible-confmat}, respectively (plots for individual subject’s
data can be seen in the supplementary material, figures S5-S7). Unlike the
prior confusion matrices, where rows and columns followed a standard order
based on consonant manner of articulation, in figures \ref{fig-psa-confmat},
\ref{fig-spe-confmat}, and \ref{fig-phoible-confmat} the matrices are ordered
based on a heirarchical clustering of the rows (performed separately for each
matrix) using optimal leaf ordering as implemented in `scipy` [@scipy1.0.0].
Therefore, the row and column orders do not necessarily match across the three
figures, so attention to the row and column labels is necessary when visually
comparing the matrices.

![Results for the PSA feature system. **A:** Confusion matrix derived from the PSA phonological feature classifiers. Row labels correspond to the test data input to each classifier. Notable features include fairly reliable identification of \ipa{/ɹ/} trials, and relatively uniform confusability of the “+consonantal, −nasal” consonants (everything below and to the right of \ipa{/s/}). **B:** most frequent classification of each stimulus consonant.\label{fig-psa-confmat}](fig-psa.eps)

Compared to the confusion matrices for pairwise (figure
\ref{fig-pairwise-confmat}A) and OVR (figure \ref{fig-ovr-confmat}A)
classifiers, the magnitude of the values in the phonological-feature-based
confusion matrices is smaller (the highest classification scores tend to be
around 2-4%, this arises because each cell is a product of several probability
values, which naturally makes the resulting numbers smaller).  However, the
smallest values are a few orders of magnitude smaller than the values in the
pairwise and OVR confusion matrices, making the distinction between similar and
dissimilar cells more apparent.  Consequently, much more structure is apparent
in the phonological-feature-based confusion matrices.  For example, figure
\ref{fig-psa-confmat}A shows that when information is combined from the 9
feature classifiers in the PSA system, 2×2 submatrices for voiced-voiceless
consonant pairs \ipa{/ð θ/} and \ipa{/ɡ k/} are visible in the lower-right
corner, suggesting that the classifier trained to discriminate voicing (encoded
by the “tense” feature in the PSA system) was relatively less accurate than
other phonological feature classifiers in that system. In contrast, trials with
\ipa{/ɹ/} stimuli are fairly well discriminated: the top row of the matrix is
mostly darker colors with its diagonal element relatively bright, and other
consonants are rarely mis-classified as \ipa{/ɹ/} (the leftmost column is
mostly dark, with only its diagonal element bright). Nonetheless, combining
information from all the phonological feature classifiers in the PSA system
still results in the correct classification being the dominant one, for all
consonants except \ipa{/n/} (figure \ref{fig-psa-confmat}B).

![Results for the SPE feature system. **A:** Confusion matrix derived from the SPE phonological feature classifiers.  Notable features include the 4×4 block of post-alveolar fricatives and affricates in the upper left quadrant, the reliable identification of \ipa{/h/}, and several 2×2 submatrices indicating confusible pairs of consonants (e.g., \ipa{/ɡ k/}, \ipa{/ɹ l/}, and \ipa{/m n/}). **B:** most frequent classification of each stimulus consonant.\label{fig-spe-confmat}](fig-spe.eps)

Looking across feature systems, similar 2×2 submatrices of poorly discriminated
consonant pairs are seen in figure \ref{fig-spe-confmat}A (e.g., \ipa{/ɹ l/}
and \ipa{/m n/}) and figure \ref{fig-phoible-confmat}A (e.g., \ipa{/s z/} and
\ipa{/j w/}), and both SPE and PHOIBLE show a 4×4 submatrix in the upper left
quadrant corresponding to the post-alveolar consonants \ipa{/ʃ ʒ tʃ dʒ/},
suggesting that, in addition to the voicing distinction, the
fricative-affricate distinction (encoded by the “continuant” feature in both
systems) was not well learned by the classifiers. Interestingly, the pair
\ipa{/w j/} is poorly discriminated by the classifiers in any of the three
systems, although the distinction is encoded by different features in each:
“grave” in PSA, “back” in SPE, and “labial” in PHOIBLE. Additionally, the
PHOIBLE system show a large block in the lower right quadrant corresponding to
\ipa{/t d θ ð s z l n/} (the “+anterior” consonants in that system).

![Confusion matrices for the PHOIBLE feature system. Notable features include the 4×4 block of post-alveolar fricatives and affricates in the upper left,  the 8×8 block of anterior alveoloar consonants in the lower right (with 2×2 voiced-voiceless submatrices \ipa{/s z/} and \ipa{/ð θ/} within it), and the relative distinctiveness of \ipa{/j w/} from all other consonants, but not from each other.\label{fig-phoible-confmat}](fig-phoible.eps)

To quantify the degree to which the neural responses reflect the contrasts
encoded by each feature system, we compute the diagonality of each matrix
(the degree to which the mass of the matrix falls along the main diagonal).
Matrix diagonality values for each subject’s data, along with the
across-subject average matrices, are shown in figure \ref{fig-diag-boxplot}.
The PHOIBLE feature system fares considerably better than the PSA and SPE
feature systems on this measure, suggesting that the contrasts encoded by the
PHOIBLE system more closely reflect the kinds of information extracted by the
brain during speech processing and subsequently detected in the EEG signals.
<!-- Prior to computing those diagonality values, heirarchical clustering was
performed on the rows of each matrix individually, so the diagonality valuet
reflects the maximal diagonality possible for each subject’s matrix. -->
To ensure that the diagonality measure was not merely reflecting differences in
signal quality for different subjects (i.e., due to scalp conductivity, EEG cap
fit, etc.), we regressed subjects’ diagonality scores in each feature system
against the signal-to-noise ratio of their epoched EEG data, and found no
evidence to support a correlation (see supplementary material, figures S2 and
S3).

![Matrix diagonality measures for each of the three feature systems tested. Gray boxes show quartiles; circles represent diagonality measures for individual subject data, and black horizontal lines represent the diagonality measures for the across-subject average matrices shown in figures \ref{fig-psa-confmat}A, \ref{fig-spe-confmat}A, and \ref{fig-phoible-confmat}A. Brackets indicate significant differences between feature systems (paired-samples t-tests, bonferroni-corrected, all corrected p-values < 0.01).\label{fig-diag-boxplot}](fig-diagonality-barplot-individ.eps)

To determine which features contributed most to this measure of diagonality, we
performed a leave-one-out analysis, in which each feature was excluded in turn
from the computation of the confusion matrix, heirarchical clustering with
optimal leaf ordering was performed, and the resulting diagonality was
calculated. This analysis was done on the individual listeners’ confusion
matrices, rather than on the average confusion matrix across listeners, in
order to assess cross-subject agreement in the rank ordering of feature
importance. Results of this analysis for the PHOIBLE feature system are shown
in figure \ref{fig-loo}<!--; similar figures for the other two phonological
feature systems can be found in the supplementary material (figures SXXX and
SXXX)-->. There appears to be considerable heterogeneity in the
rank ordering of feature importance across subjects, evidenced by the lack of
monotonicity in the connecting lines for several of the subjects in figure
\ref{fig-loo}.  Nonetheless, most subjects (7 of 12) showed the greatest
decrease in diagonality when omitting the “strident” feature, which
distinguishes \ipa{/s z ʃ ʒ tʃ dʒ/} from the rest of the coronal consonants
(\ipa{/t d θ ð l ɹ n/}).

![Leave-one-out analysis of feature influence on the diagonality measure, for the PHOIBLE feature system. Abscissa labels indicate which feature was excluded; connected points reflect diagonality values for a given listener; boxes show quartiles across listeners for each left-out feature. For each listener, the lowest diagonality is indicated by a large dot; 7 of the 12 listeners showed the greatest decline in diagonality when omitting the “strident” feature. \label{fig-loo}](fig-loo-PHOIBLE.eps)

# Discussion

This paper describes a technique for comparing theoretical accounts of
phonological features to recordings of brain activity during speech perception,
and illustrates the technique with three phonological feature systems drawn
from the linguistic literature.  The approach uses machine learning classifiers
to model the representation of abstract classes of speech sounds by the brain,
and combines information across classifiers to construct predicted patterns of
similarity or confusion among phonemes.

This work differs from past studies of phonological feature representation in
its emphasis on phonological feature systems drawn from the linguistic
literature, and in its attempt to model the entire consonant inventory of a
language rather than a few select contrasts.  Of course, languages comprise
vowels as well as consonants, and a natural extension of this work would model
the vowels as well.  Moreover, the phonemic contrasts present in English are
only a subset of the attested contrasts in the world’s languages, and another
natural extension would be to apply these techniques to modeling the brain
activity of native listeners of a variety of languages (to explore the
representation of lexical tone, voice quality contrasts, ejective and implosive
consonants, etc., and to characterise any differences in representation that
emerge across different languages). Finally, a true test of *abstract*
phonological representation should at least account for patterns of allophony
(e.g., the differing pronunciations of \ipa{/t/} in different positions in a
word). Whereas these experiments did include multiple tokens of each consonant
from multiple talkers, they did not model allophonic variation, and doing so is
another natural extension of this work (either with polysyllabic stimuli or
continuous speech).

<!--
This work is similar in spirit to prior studies using representational
similarity analysis (RSA)[@KriegeskorteEtAl2008] to examine phonological
feature representation in the brain [e.g., @EvansDavis2015], but differs in a
few respects. First, rather than using RSA to locate specific brain regions
that respond preferentially to (say) continuant versus non-continuant
consonants, this study asks whether that distinction can be recovered from the
neural recordings regardless of which cortical region(s) are sensitive to the
distinction. This leaves open the question of whether the classifiers in this
study are learning a distinction between classes of speech sounds based on
emergent patterns in the brain’s representation of the *acoustic* properties of
sounds in those classes, or instead learning a distinction based on activity in
a (hypothetical) population of phonological-feature-sensitive neurons
representing *abstract* class membership.  However, even with millimeter-scale
spatial resolution of cortical activity offered by other techniques such as
fMRI, there is no guarantee that a patch of cortex that appears to respond
preferentially to continuant sounds over non-continuant ones is in fact
carrying information about “continuancy” that is truly abstract (i.e., devoid
of any information about the acoustic properties of the input) given
acoustically natural speech stimuli (cf. Evans and Davis [@EvansDavis2015] for
an approach that addresses this worry, but finds no evidence of abstract
representation after accounting for stimulus acoustics).  Again, studies
involving phonemes in different phonological contexts (i.e., polysyllabic words
or continuous speech) might yield more definitive answers regarding the
abstractness of the representation, though such stimuli introduce many
complexities such as lexico-semantic processing, frequency effects, and both
word- and sentence-level predictive processing, all of which must be accounted
for in stimulus design and/or analysis.
-->

Perhaps the most promising aspect of this work is its potential to provide
complementary information to behavioral studies of phoneme confusion [e.g., the
classic study of @MillerNicely1955].  In such studies,
confusions are induced by degrading the stimuli (usually by adding background
noise), which has unequal effects on different speech sounds due to variations
in their cue robustness [@Wright2001, @Wright2004].  In contrast, this
technique estimates phoneme confusion based on brain responses to unmodified
natural speech, not speech-in-noise, thereby removing biases related to cue
robustness.  Of course, the nature of the auditory system, the brain, and EEG
also favors some speech sound contrasts over others.  For example, EEG
relatively faithfully reflects the total energy of acoustic stimuli, making it
ideal for tracking things like the amplitude envelope of an utterance
[@PowerEtAl2012], but this also favors phoneme-level contrasts based on
differences in rise time (e.g., fricative-affricate distinctions like \ipa{/ʃ/}
versus \ipa{/tʃ/}) compared to purely spectral contrasts such as \ipa{/f/}
versus \ipa{/θ/}.
<!-- More generally, the nature of sound transduction in the cochlea means
that spectral distinctions are primarily conveyed through a place code, which
may or may not be readily recoverable depending on the spatial scale of the
tonotopy and the resolution of the imaging technique. -->
Therefore, failure of a classifier to recover a given contrast in the neural
recordings should not be taken as definitive evidence that the brain does not
represent that particular abstraction — such failures may arise from the
limitations of our measurement tools.

Nevertheless, even if we can’t rule out any *particular feature* being
represented in the brain based on these data alone, we can still say that the
abstractions implied by a *system* with a better fit to the neural data are
likely a closer approximation of a listener’s phonological knowledge than the
abstractions implied by a system with poor fit to the neural data.  Even such
coarse insights can provide a foundation for follow-up studies that focus on
specific features or contrasts, and drilling down on the patterns of errors
made by each classifier may reveal specific shortcomings of particular feature
systems that could guide their refinement (e.g., whether the “consonantal”
feature includes \ipa{/ɹ/}, as in SPE, or does not, as in PSA and PHOIBLE; or
whether sparsity in the assignment of feature values improves or degrades the
fit between phonological feature system and neural data).

<!--
The discussion thusfar has skirted the fact that there are many stages of
peripheral and central auditory processing, with both feed-forward and feedback
connections between stages, as well as interconnections among units within a
given stage.  Thus there are multiple levels of representation present, all of
which are potentially reflected in the EEG signal.  It may turn out that
several of those levels correspond to what is traditionally considered
“phonological” (as opposed to phonetic) knowledge, no one of which can be
uncontroversially labeled as *the* phonological representation.  For example,
representation of phonological features with more direct parallels to acoustic
form (such as sonorancy) may arise at an earlier stage than representations of
features that have more heterogeneous acoustic correlates (such as place
features like “coronal”).
-->

Finally, it should be noted that because we used only acoustically natural
speech stimuli, the patterns of neural representation seen in this work may not
be fully divorced from patterns of similarity in the acoustic properties of the
stimuli.  As mentioned above, Evans and Davis [-@EvansDavis2015] used a mixture
of acoustically natural and synthetic stimuli that addresses this worry, and
found no evidence of abstract representation after accounting for stimulus
acoustics in this way (though with a fairly limited stimulus set).  For our
purposes, the question of whether the representations are fully abstract is not
crucial, since our goal is not to prove that the brain carries out such
abstractions nor to localise such representations in the cortex. Rather, we set
out to assess whether this technique can tell us which published phonological
feature system best reflects the patterns of similarity seen in the neural
data. Thus even if the information learned by the classifiers is partially
acoustic in nature, knowing which phonological feature system best
recapitulates those patterns still helps to build a bridge between the
neuroscience of language and phonological theory.

<!-- TODO? paragraph about variability across subjects (reference to individ.
subject matrices in supplement?)  Which features consistently high SNR across
subjects, which most noisy / inconsistent (possible reference to supplemental
figure?) -->

<!-- TODO? paragraph about feature weighting, and how EEG may provide evidence
for salience (refs: Miller & Nicely 1955, Blumstein & Cooper 1972, "perceptual
prominence" of voice over place features (grave and flat)) -->

## Future directions

As mentioned above, natural extensions of this work are expanding the stimulus
set to include vowels, other languages, and continuous speech, and testing
listeners of other language backgrounds.  In addition, there are other theories
of phonological representation that could be assessed using this technique,
such as articulatory phonology [@BrowmanGoldstein1989; @BrowmanGoldstein1992],
or the featurally underspecified lexicon model [@LahiriReetz2002;
@LahiriReetz2010].  There is also the potential to derive a system of features
directly from the neural data, without reference to linguistic theory, and
assess its similarities to published accounts *post hoc*.

From a methodological perspective, different preprocessing or classification
strategies might improve the clarity of the confusion matrices by reducing
classification errors that are strictly due to noise, as opposed to those that
reflect a genuine mistake about which sounds ought to pattern together.  Other
imaging modalities (such as MEG or combined EEG/MEG) could also increase signal
and thereby improve our ability to resolve differences between models.

Finally, our earliest application of this technique was actually as a model of
how speech sounds from an unfamiliar language are filtered through native
phonological categories [@HasegawaJohnsonEtAl2016-UnderresourcedASR], as part
of a larger project involving non-traditional approaches to automatic speech
recognition.  Much work remains to realise the potential of that application as
well.


# Acknowledgments {-}

Special thanks to Mark Hasegawa-Johnson and Ed Lalor who were instrumental in
the early development of this technique, and to Nick Foti, Alex Gramfort,
Kathleen Hall, Eric Larson, Majid Mirbagheri, and Doug Pulleyblank for helpful
and stimulating discussions about various aspects of the project along the way.


# Disclosure statement {-}

The authors report no conflict of interest.

# Funding {-}

This work was supported in part by the NIH under grant T32DC005361.

# References {-}

\setlength{\parindent}{-0.25in}
\setlength{\leftskip}{0.25in}
\noindent
