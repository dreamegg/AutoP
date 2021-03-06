from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import load_examples
from utils import convert
from model import create_model

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import random
import collections
import time
import math


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",  default = "src_input", help="path to folder containing images")
parser.add_argument("--target_dir",  default = "src_target", help="path to folder containing images")
parser.add_argument("--output_dir", default = "toon_out", help="where to put output files")
parser.add_argument("--seed", type=int)

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default = 2500, help="number of training epochs")
parser.add_argument("--aspect_ratio"
                    , type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=20, help="number of images in batch")

parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=1000, help="save model every save_freq steps, 0 to disable")

parser.set_defaults(flip=True)

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()


if tf.__version__.split('.')[0] != "1":
    raise Exception("Tensorflow version 1 required")

if a.seed is None:
    a.seed = random.randint(0, 2**31 - 1)

tf.set_random_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)

if not os.path.exists(a.output_dir):
    os.makedirs(a.output_dir)

for k, v in a._get_kwargs():
    print(k, "=", v)

with open(os.path.join(a.output_dir, "options.json"), "w") as f:
    f.write(json.dumps(vars(a), sort_keys=True, indent=4))

examples = load_examples(a.input_dir, a.target_dir, a.batch_size)
print("examples count = %d" % examples.count)

# inputs and targets are [batch_size, height, width, channels]
model = create_model(examples.inputs, examples.targets)

inputs = examples.inputs
targets = examples.targets
outputs = model.outputs

# reverse any processing on images so they can be written to disk or displayed to user
with tf.name_scope("convert_inputs"):
    converted_inputs = convert(inputs, a.aspect_ratio)

with tf.name_scope("convert_targets"):
    converted_targets = convert(targets, a.aspect_ratio)

with tf.name_scope("convert_outputs"):
    converted_outputs = convert(outputs, a.aspect_ratio)

with tf.name_scope("encode_images"):
    display_fetches = {
        #"paths": examples.paths,
        "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
        "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
        "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
    }

# summaries
with tf.name_scope("inputs_summary"):
    tf.summary.image("inputs", converted_inputs)

with tf.name_scope("targets_summary"):
    tf.summary.image("targets", converted_targets)

with tf.name_scope("outputs_summary"):
    tf.summary.image("outputs", converted_outputs)

with tf.name_scope("predict_real_summary"):
    tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

with tf.name_scope("predict_fake_summary"):
    tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

tf.summary.scalar("discriminator_loss", model.discrim_loss)
tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
tf.summary.scalar("tv_loss", model.tv_loss)
tf.summary.scalar("content_loss", model.content_loss)
for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

#for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
#    tf.summary.histogram(var.op.name + "/gradients", grad)

with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1)

logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
with sv.managed_session() as sess:
    print("parameter_count =", sess.run(parameter_count))

    checkpoint = tf.train.latest_checkpoint(a.output_dir)
    if ( checkpoint != None ) :
        print("loading model from checkpoint")
        saver.restore(sess, checkpoint)

    max_steps = 2**32
    if a.max_epochs is not None:
        max_steps = examples.steps_per_epoch * a.max_epochs
    if a.max_steps is not None:
        max_steps = a.max_steps

    # training
    start = time.time()

    for step in range(max_steps):
        def should(freq):
            return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

        options = None
        run_metadata = None
        if should(a.trace_freq):
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        fetches = {
            "train": model.train,
            "global_step": sv.global_step,
        }

        if should(a.progress_freq):
            fetches["discrim_loss"] = model.discrim_loss
            fetches["gen_loss_GAN"] = model.gen_loss_GAN
            fetches["gen_loss_L1"] = model.gen_loss_L1
            fetches["tv_loss"] = model.tv_loss
            fetches["content_loss"] = model.content_loss

        if should(a.summary_freq):
            fetches["summary"] = sv.summary_op

        if should(a.display_freq):
            fetches["display"] = display_fetches

        results = sess.run(fetches, options=options, run_metadata=run_metadata)

        if should(a.summary_freq):
            print("recording summary")
            sv.summary_writer.add_summary(results["summary"], results["global_step"])

        if should(a.trace_freq):
            print("recording trace")
            sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

        if should(a.progress_freq):
            # global_step will have the correct step count if we resume from a checkpoint
            train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
            train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
            rate = (step + 1) * a.batch_size / (time.time() - start)
            remaining = (max_steps - step) * a.batch_size / rate
            print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dh" % (train_epoch, train_step, rate, remaining / (60*60)))
            print("discrim_loss", results["discrim_loss"])
            print("gen_loss_GAN", results["gen_loss_GAN"])
            print("gen_loss_L1", results["gen_loss_L1"])
            print("tv_loss", results["tv_loss"])
            print("content_loss", results["content_loss"])


        if should(a.save_freq):
            print("saving model")
            saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

        if sv.should_stop():
            break

