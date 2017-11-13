# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from nmt import inference
from nmt import attention_model
from nmt import gnmt_model
from nmt import model as nmt_model
from nmt import model_helper
from nmt.utils import misc_utils as utils
from nmt.utils import nmt_utils
from nmt.utils import data_utils


class Seq2SeqBot(object):

    def __init__(self, ckpt, hparams, tokenizer=None, normalize_digits=False):
        if not hparams.attention:
            model_creator = nmt_model.Model
        elif hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
            model_creator = gnmt_model.GNMTModel
        else:
            raise ValueError("Unknown model architecture")

        self.ckpt = ckpt
        self.infer_model = inference.create_infer_model(model_creator, hparams)
        self.hparams = hparams
        self.tokenizer = tokenizer if tokenizer else data_utils.basic_tokenizer
        self.normalize_digits = normalize_digits

    def decode(self, inpt):
        with tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto()) as self.sess:
            self.model = model_helper.load_model(
                self.infer_model.model,
                self.ckpt,
                self.sess,
                "infer")

            words = self.tokenizer(inpt)
            if self.normalize_digits:
                words = [data_utils._DIGIT_RE.sub(b"0", w) for w in words]
            sentence = ' '.join(words)
            sentence = sentence.decode('utf8')
            infer_data = [sentence]
            self.sess.run(
                self.infer_model.iterator.initializer,
                feed_dict={
                    self.infer_model.src_placeholder: infer_data,
                    self.infer_model.batch_size_placeholder: 1
                })

            nmt_outputs, infer_summary = self.model.decode(self.sess)

            assert nmt_outputs.shape[0] == 1
            reply = nmt_utils.get_translation(
                nmt_outputs,
                sent_id=0,
                tgt_eos=self.hparams.eos,
                bpe_delimiter=self.hparams.bpe_delimiter)

            return reply
