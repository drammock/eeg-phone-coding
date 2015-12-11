# JSALT validation
Experiment to test performance of EEG signal classifiers trained on responses to CV-syllables using only English consonant contrasts, when tested on foreign consonant contrasts.

# Notes
The foreign languages include only 1 token per syllable, and only 1 talker per language. Transcriptions for these syllables have been done on a broad phonetic level, rather than a phonemic level. For example, Swahili voiced stop consonants are supposed to be implosive, but only the bilabial and alveolar stops were actually produced as implosive by the recorded talker; the palatal stop was modal and the velar stop was spirantized. The purpose of including this level of detail is to get a better estimate of the distance in phonetic feature space between the English and foreign sounds.

On the other hand, the English syllables include 3 tokens from each of 4 talkers (2 male, 2 female) or a total of 12 tokens.  For the purpose of training the classifier, all tokens corresponding to a given English phoneme need to have the same label. In the majority of cases, allophonic variation was quite minimal across talkers and tokens, but there were a few cases where tokens differed enough that they arguably deserved different transcriptions if the same “broad phonetic” standards used in the foreign speech were applied to the English speech. In all cases, the transcription chosen was one that was most consistent with the *majority* of English tokens. In particular:

1. The English stop/affricate voicing contrast was always encoded as a “plain voiceless” versus “aspirated voiceless” contrast, and transcribed as, e.g., /p/ vs /pʰ/.  In a minority of tokens, the “plain voiceless” token included prevoicing (suggesting a transcription of /b/ instead of /p/).

2. Allophonic alternation of /z/: there were a few tokens which showed no periodicity during the frication, but were instead distinguished from /s/ by duration, amplitude, and possibly rise time.

3. No attempt was made to distinguish among different possible articulations of /ɹ/.

4. No attempt was made to distinguish between “light” (alveolar) and “dark” (velarized-alveolar) /l/.
