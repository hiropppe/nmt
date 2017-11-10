# -*- coding:utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import MeCab
import os
import re
import tensorflow as tf

from tensorflow.python.platform import gfile


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")
_BLANK_RE = re.compile(ur"[　\s]+")

tagger = MeCab.Tagger("-Owakati")


def sampling_data(src_file, tgt_file, size, prefix='dev'):
    import numpy as np
    n_data = 0
    data_dir = os.path.dirname(src_file)
    src_suffix = src_file[src_file.rindex('.'):]
    tgt_suffix = tgt_file[tgt_file.rindex('.'):]
    src_sample_file = os.path.join(data_dir, "%s%s" % (prefix, src_suffix))
    tgt_sample_file = os.path.join(data_dir, "%s%s" % (prefix, tgt_suffix))
    with codecs.getwriter("utf-8")(tf.gfile.GFile(src_sample_file, "w")) as src_sample:
        with codecs.getreader("utf-8")(tf.gfile.GFile(src_file, "r")) as src:
            for sentence in src:
                n_data += 1
            sample_idxes = np.random.randint(0, n_data, size)
        with codecs.getreader("utf-8")(tf.gfile.GFile(src_file, "r")) as src:
            for i, sentence in enumerate(src):
                if i in sample_idxes:
                    src_sample.write('%s' % sentence)

    with codecs.getwriter("utf-8")(tf.gfile.GFile(tgt_sample_file, "w")) as tgt_sample:
        with codecs.getreader("utf-8")(tf.gfile.GFile(tgt_file, "r")) as tgt:
            for i, sentence in enumerate(tgt):
                if i in sample_idxes:
                    tgt_sample.write('%s' % sentence)


def mecab_tokenizer(sentence):
    assert type(sentence) is str
    result = tagger.parse(sentence)
    return result.split()


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True, overwrite=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if overwrite or not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          word = _BLANK_RE.sub("_", word.decode('utf8')).encode('utf8')
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")
