# Data

## Lakh Pianoroll Dataset

We use the *cleansed* version of
[Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
(LPD). LPD contains 174,154 unique
[multi-track piano-rolls](https://salu133445.github.io/musegan/representation)
derived from the MIDI files in the
[Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/) (LMD),
while the cleansed version contains 21,425 piano-rolls that
are in 4/4 time and have been matched to distinct entries in
[Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/) (MSD).

## Training Data

- Use *symbolic timing*, which discards tempo information
  (see [here](https://salu133445.github.io/lakh-pianoroll-dataset/representation) for more
  details)
- Discard velocity information (using binary-valued piano-rolls)
- 84 possibilities for note pitch (from C1 to B7)
- Merge tracks into 8 categories: *Drums*, *Piano*, *Guitar*, *Bass*,
  *Ensemble*, *Reed*, *Synth Lead* and *Synth Pad*
- Consider only songs with an *alternative* tag
- Randomly pick 6 phrases of 4-bar long from each song

Hence, the size of the target output tensor is 4 (bar) &times; 96 (time step)
&times; 84 (pitch) &times; 8 (track).

- [lastfm_alternative_8b_phrase.npy](https://drive.google.com/uc?id=1x3CeSqE6ElWa6V7ueNl8FKPFmMoyu4ED&export=download)
  (3.38 GB) contains 13,746 four-bar phrases from 2,291 songs. The shape is
  (2291, 6, 4, 96, 84, 8).

![train_samples](figs/train_samples.png)
<p class="caption">Six examples of eight-track piano-roll of four-bar long (each block represents a bar) seen in our training data. The tracks are (from top to bottom): <i>Drums</i>, <i>Piano</i>, <i>Guitar</i>, <i>Bass</i>, <i>Ensemble</i>, <i>Reed</i>, <i>Synth Lead</i> and <i>Synth Pad</i>.</p>
