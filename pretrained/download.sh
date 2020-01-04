#!/bin/bash
case $1 in
  *.tar.gz)
    filename=$1
    ;;
  *)
    filename=$1.tar.gz
    ;;
esac

case $filename in
  "lastfm_alternative_pretrain_d_proposed.tar.gz")
    fileid=16LKWjiEjDjgiTjMLFcgnzZdCT-v8fp3T
    ;;
  "lastfm_alternative_pretrain_d_ablated.tar.gz")
    fileid=1YyKAiPV0AuGuQB1K05dQAkPnsRMAqjtJ
    ;;
  "lastfm_alternative_pretrain_d_baseline.tar.gz")
    fileid=1ZVASqhTApVWSvtM0N-952BAEbfUqRTfK
    ;;
  *)
    echo "File not found"
    exit 1
    ;;
esac

wget -O $filename --no-check-certificate \
  "https://drive.google.com/uc?export=download&id="$fileid
