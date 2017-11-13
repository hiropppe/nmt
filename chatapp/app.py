# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import sys
import tensorflow as tf
import time

from flask import Flask
from flask import request, render_template, jsonify


app = Flask(__name__)

out_dir = '/tmp/meidai_model'
chatbot = None


@app.route("/chat")
def index():
    return render_template("index.html")


@app.route("/chat/talk")
def talk():
    inpt = request.args.get("inpt", "")
    reply = chatbot.decode(inpt.encode('utf8'))
    return jsonify({'reply': reply})


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
    from nmt import nmt
    from nmt.utils import data_utils
    from chatapp import bot

    nmc_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmc_parser)
    FLAGS, unparsed = nmc_parser.parse_known_args()
    ckpt = tf.train.latest_checkpoint(out_dir)
    default_hparams = nmt.create_hparams(FLAGS)
    hparams = nmt.create_or_load_hparams(out_dir, default_hparams, None)
    chatbot = bot.Seq2SeqBot(ckpt, hparams, tokenizer=data_utils.mecab_tokenizer)
    app.run(host="0.0.0.0", port=8888, processes=3, debug=False)
