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
    fileid=12tEzs-Qa-qi59hLJB8TlD-vcZgVEQZu6
    ;;
  "lastfm_alternative_pretrain_d_ablated.tar.gz")
    fileid=1GolkoE2ktmHF2Pt7POd8TBBYZARu6ih8
    ;;
  "lastfm_alternative_pretrain_d_baseline.tar.gz")
    fileid=1qWWWU6UTMJvzdK6y4bvh3PRXF5Xbk09v
    ;;
  *)
    echo "File not found"
    exit 1
    ;;
esac

wget -O $filename --no-check-certificate \
  "https://drive.google.com/uc?export=download&id="$fileid

