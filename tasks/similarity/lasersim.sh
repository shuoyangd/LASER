#! /bin/sh
#
# lasersim.sh
# Copyright (C) 2020 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#


if [ -z ${LASER+x} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit 1
fi

if [ $# -ne 3 ] ; then
  echo "usage embed.sh input-file language output-file"
  exit 1
fi

ifile=$1
lang=$2
ofile=$3

# encoder
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

cat $ifile \
  | python3 ${LASER}/source/sent_similarity.py \
    --encoder ${encoder} \
    --token-lang ${lang} \
    --bpe-codes ${bpe_codes} \
    --output ${ofile} \
    --verbose
