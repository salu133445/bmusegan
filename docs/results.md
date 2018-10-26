# Results

## Qualitative Results

> Note that only the guitar track is shown.

| Strategy                                | Result                                                                                                      |
|:---------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|
| raw prediction of<br>the pretrained _G_ | <img src="figs/closeup_raw.png" alt="closeup_raw" style="max-width:400px;">                                 |
| Bernoulli sampling<br>(at test time)    | <img src="figs/closeup_test_time_bernoulli.png" alt="closeup_test_time_bernoulli" style="max-width:400px;"> |
| hard thresholding<br>(at test time)     | <img src="figs/closeup_test_time_round.png" alt="closeup_test_time_round" style="max-width:400px;">         |
| proposed model<br>with SBNs             | <img src="figs/closeup_bernoulli.png" alt="closeup_bernoulli" style="max-width:400px;">                     |
| proposed model<br>with DBNs             | <img src="figs/closeup_round.png" alt="closeup_round" style="max-width:400px;">                             |

## Quantitative Results

### Evaluation metrics

- __Qualified note rate__ (QN) computes the ratio of the number of the qualified
  notes (notes no shorter than three time steps, i.e., a 32th note) to the total
  number of notes. Low QN implies overly fragmented music.
- __Polyphonicity__ (PP) is defined as the ratio of the number of time steps
  where more than two pitches are played to the total number of time steps.
- __Tonal distance__ (TD) measures the distance between the chroma features (one
  for each beat) of a pair of tracks in the tonal space proposed in [1].

### Comparisons of training and binarization strategies

<div style="text-align:center">
  <img src="figs/two-stage_polyphonicity.png" alt="two-stage_polyphonicity" style="max-width:300px; display:inline-block;">&emsp;
  <img src="figs/two-stage_qualified_note_rate.png" alt="two-stage_qualified_note_rate" style="max-width:300px; display:inline-block;">
</div>

<div style="text-align:center">
  <img src="figs/end2end_qualified_note_rate.png" alt="end2end_qualified_note_rate" style="max-width:300px; display:inline-block;">
</div>

### Effects of the shared/private and multi-stream design of the discriminator

<div style="text-align:center">
  <img src="figs/ablated_qualified_note_rate.png" alt="ablated_qualified_note_rate" style="max-width:300px; display:inline-block;">&emsp;
  <img src="figs/ablated_tonal_distance.png" alt="ablated_tonal_distance" style="max-width:300px; display:inline-block;">
</div>

## Audio Samples

The audio samples presented below are in a high temporal resolution of 24 time
steps per beat. Hence, they are not directly comparable to the results presented
on the MuseGAN [website](https://salu133445.github.io/musegan/results).

> No cherry-picking. Some might sound unpleasant. __Lower the volume first!__

| Model                             | Result                                                             |
|:---------------------------------:|:------------------------------------------------------------------:|
| hard thresholding (at test time)  | {% include audio_player.html filename="test_time_round.mp3" %}     |
| Bernoulli sampling (at test time) | {% include audio_player.html filename="test_time_bernoulli.mp3" %} |
| proposed model (+SBNs)            | {% include audio_player.html filename="proposed_bernoulli.mp3" %}  |
| proposed model (+DBNs)            | {% include audio_player.html filename="proposed_round.mp3" %}      |
| end-to-end model (+SBNs)          | {% include audio_player.html filename="end2end_bernoulli.mp3" %}   |
| end-to-end model (+DBNs)          | {% include audio_player.html filename="end2end_round.mp3" %}       |

## Reference

1. Christopher Harte, Mark Sandler, and Martin Gasser,
   "Detecting Harmonic Change In Musical Audio,"
   in _Proc. ACM MM Workshop on Audio and Music Computing Multimedia_, 2006.
