#!/bin/bash
for postfix in proposed ablated baseline
do
  sh download.sh lastfm_alternative_pretrain_d_$postfix
done

