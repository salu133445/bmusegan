# Results

## Qualitative Results

> Note that only the guitar track is shown.

| Strategy | Result |
|:--------:|:------:|
| raw prediction of <br> the pretrained *G* |<img src="figs/closeup_raw.png" alt="closeup_raw" style="max-height:150px; display:block;">|
| Bernoulli sampling<br>(at test time) | <img src="figs/closeup_test_time_bernoulli.png" alt="closeup_test_time_bernoulli" style="max-height:150px; display:block;"> |
| hard thresholding<br>(at test time) | <img src="figs/closeup_test_time_round.png" alt="closeup_test_time_round" style="max-height:150px; display:block;"> |
| proposed model<br>with SBNs | <img src="figs/closeup_bernoulli.png" alt="closeup_bernoulli" style="max-height:150px; display:block;"> |
| proposed model<br>with DBNs | <img src="figs/closeup_round.png" alt="closeup_round" style="max-height:150px; display:block;"> |

## Evaluation Metrics

- **Qualified note rate** (QN) computes the ratio of the number of the qualified
  notes (notes no shorter than three time steps, i.e., a 32th note) to the total
  number of notes. Low QN implies overly fragmented music.
- **Polyphonicity** (PP) is defined as the ratio of the number of time steps
  where more than two pitches are played to the total number of time steps.
- **Tonal distance** (TD) measures the distance between the chroma features (one
  for each beat) of a pair of tracks in the tonal space proposed in [1].

## Comparisons of Training Strategies and Binarization Strategies

<div style="text-align:center">
<img src="figs/two-stage_polyphonicity.png" alt="two-stage_polyphonicity" style="max-height:200px; display:inline-block;">&emsp;
<img src="figs/two-stage_qualified_note_rate.png" alt="two-stage_qualified_note_rate" style="max-height:200px; display:inline-block;">
</div>

<div style="text-align:center">
<img src="figs/end2end_qualified_note_rate.png" alt="end2end_qualified_note_rate" style="max-height:200px; display:inline-block;">
</div>

## Effects of the Shared/private and Multi-stream Design of the Discriminator

<div style="text-align:center">
<img src="figs/ablated_qualified_note_rate.png" alt="ablated_qualified_note_rate" style="max-height:200px; display:inline-block;">&emsp;
<img src="figs/ablated_tonal_distance.png" alt="ablated_tonal_distance" style="max-height:200px; display:inline-block;">
</div>

## Reference

1. Christopher Harte, Mark Sandler and Martin Gasser,
   "Detecting Harmonic Change In Musical Audio,"
   in *Proc. ACM MM Workshop on Audio and Music Computing Multimedia*, 2006.
