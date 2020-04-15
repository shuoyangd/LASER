#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-04-08
#
# Distributed under terms of the MIT license.
#

import argparse
from collections import namedtuple
import logging
import os
import re
import sys
import tempfile
import time
import torch
import torch.nn.functional as F

from embed import SentenceEncoder, EncodeFile, EncodeTime, buffered_read
from text_processing import Token, BPEfastApply

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="LASER: compute sentence similarities")
opt_parser.add_argument('--encoder', type=str, required=True, help='encoder to be used')
opt_parser.add_argument('--token-lang', type=str, default='--', help="Perform tokenization with given language ('--' for no tokenization)")
opt_parser.add_argument('--bpe-codes', type=str, default=None, help='Apply BPE using specified codes')
opt_parser.add_argument('-v', '--verbose', action='store_true', help='Detailed output')

opt_parser.add_argument('-o', '--output', required=True, help='Output sentence embeddings')
opt_parser.add_argument('--buffer-size', type=int, default=10000, help='Buffer size (sentences)')
opt_parser.add_argument('--max-tokens', type=int, default=None, help='Maximum number of tokens to process in a batch')
opt_parser.add_argument('--max-sentences', type=int, default=32, help='Maximum number of sentences to process in a batch')
opt_parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
opt_parser.add_argument('--stable', action='store_true', help='Use stable merge sort instead of quick sort')

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
from text_processing import Token, BPEfastApply

SPACE_NORMALIZER = re.compile("\s+")
Batch = namedtuple('Batch', 'srcs tokens lengths')

EOS_IDX = 2


class SentencePairSimilarity(SentenceEncoder):

  def _process_batch(self, batch):
    tokens = batch.tokens
    lengths = batch.lengths
    if self.use_cuda:
      tokens = tokens.cuda()
      lengths = lengths.cuda()
    self.encoder.eval()
    embeddings = self.encoder(tokens, lengths)['sentemb']
    return embeddings.detach()

  def compute_sentpair_similarity(self, sentence_pairs):
    indices = []
    results = []

    total_num_sents = len(sentence_pairs[0])
    for start_idx in range(0, total_num_sents, self.max_sentences):
      if total_num_sents - start_idx > self.max_sentences:
        sent_batch1 = sentence_pairs[0][start_idx:start_idx+self.max_sentences]
        sent_batch2 = sentence_pairs[1][start_idx:start_idx+self.max_sentences]
        batch1, batch_indices1 = next(self._make_batches(sent_batch1))
        batch2, batch_indices2 = next(self._make_batches(sent_batch2))
      else:
        sent_batch1 = sentence_pairs[0][start_idx:]
        sent_batch2 = sentence_pairs[1][start_idx:]
        batch1, batch_indices1 = next(self._make_batches(sent_batch1))
        batch2, batch_indices2 = next(self._make_batches(sent_batch2))

      batch_indices1 = torch.LongTensor(batch_indices1)
      batch_indices2 = torch.LongTensor(batch_indices2)
      embeds1 = self._process_batch(batch1)[torch.argsort(batch_indices1)]
      embeds2 = self._process_batch(batch2)[torch.argsort(batch_indices2)]
      sims = F.cosine_similarity(embeds1, embeds2)
      indices.extend(sorted(batch_indices1))
      results.append(sims.cpu())

    return torch.cat(results, dim=0).tolist()


def write_list(l, out_file):
  for e in l:
    out_file.write(str(e) + "\n")


def SimFilep(lasersim, inp1_file, inp2_file, out_file, buffer_size=10000, verbose=False):
  n = 0
  t = time.time()
  for sentences in zip(buffered_read(inp1_file, buffer_size), buffered_read(inp2_file, buffer_size)):
    res = lasersim.compute_sentpair_similarity(sentences)
    write_list(res, out_file)
    n += len(sentences[0])
    if verbose and n % 10000 == 0:
      print('\r - Encoder: {:d} sentence pairs'.format(n), end='')
  if verbose:
    print('\r - Encoder: {:d} sentence pairs'.format(n), end='')
    EncodeTime(t)


def SimFile(lasersim, inp1_fname, inp2_fname, out_fname,
            buffer_size=10000, verbose=False, over_write=False,
            inp_encoding='utf-8'):
  if verbose:
    print(' - Encoder: {} and {} to {}'.
          format(os.path.basename(inp1_fname) if len(inp1_fname) > 0 else 'stdin',
            os.path.basename(inp2_fname) if len(inp2_fname) > 0 else 'stdin',
            os.path.basename(out_fname)))
  fin1 = open(inp1_fname, 'r', encoding=inp_encoding, errors='surrogateescape') if len(inp1_fname) > 0 else sys.stdin
  fin2 = open(inp2_fname, 'r', encoding=inp_encoding, errors='surrogateescape') if len(inp2_fname) > 0 else sys.stdin
  fout = open(out_fname, mode='w')
  SimFilep(lasersim, fin1, fin2, fout, buffer_size=buffer_size, verbose=verbose)
  fin1.close()
  fin2.close()
  fout.close()


def split_fields(f_path, out_paths):
  if f_path != "":
    f = open(f_path)
  else:
    f = sys.stdin
  out_files = [ open(out_path, 'w') for out_path in out_paths ]
  for fields_line in f:
    fields = fields_line.strip().split('\t')
    for idx, field in enumerate(fields):
      out_files[idx].write(field + '\n')


def main(options):
  options.buffer_size = max(options.buffer_size, 1)
  assert not options.max_sentences or options.max_sentences <= options.buffer_size, \
    '--max-sentences/--batch-size cannot be larger than --buffer-size'

  if options.verbose:
    print(' - Encoder: loading {}'.format(options.encoder))
    lasersim = SentencePairSimilarity(options.encoder,
                max_sentences=options.max_sentences,
                max_tokens=options.max_tokens,
                sort_kind='mergesort' if options.stable else 'quicksort',
                cpu=options.cpu)

  with tempfile.TemporaryDirectory() as tmpdir:
    ifname = ''  # stdin will be used
    ifname1 = os.path.join(tmpdir, 'ifile1')
    ifname2 = os.path.join(tmpdir, 'ifile2')
    split_fields(ifname, [ifname1, ifname2])
    if options.token_lang != '--':
      tok_fname1 = os.path.join(tmpdir, 'tok1')
      Token(ifname1,
          tok_fname1,
          lang=options.token_lang,
          romanize=True if options.token_lang == 'el' else False,
          lower_case=True, gzip=False,
          verbose=options.verbose, over_write=False)
      tok_fname2 = os.path.join(tmpdir, 'tok2')
      Token(ifname2,
          tok_fname2,
          lang=options.token_lang,
          romanize=True if options.token_lang == 'el' else False,
          lower_case=True, gzip=False,
          verbose=options.verbose, over_write=False)
      ifname1 = tok_fname1
      ifname2 = tok_fname2

    if options.bpe_codes:
      bpe_fname1 = os.path.join(tmpdir, 'bpe1')
      BPEfastApply(ifname1,
             bpe_fname1,
             options.bpe_codes,
             verbose=options.verbose, over_write=False)
      bpe_fname2 = os.path.join(tmpdir, 'bpe2')
      BPEfastApply(ifname2,
             bpe_fname2,
             options.bpe_codes,
             verbose=options.verbose, over_write=False)
      ifname1 = bpe_fname1
      ifname2 = bpe_fname2

    SimFile(lasersim,
           ifname1, ifname2,
           options.output,
           verbose=options.verbose, over_write=False,
           buffer_size=options.buffer_size)


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
