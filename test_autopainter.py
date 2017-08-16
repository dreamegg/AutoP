from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import load_examples
from utils import convert
from utils import deprocess
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
parser.add_argument("--test_input_dir",  default = "test_input", help="path to folder containing images")
parser.add_argument("--target_dir",  default = "src_target", help="path to folder containing images")
parser.add_argument("--output_dir", default = "toon_out", help="where to put output files")
parser.add_argument("--seed", type=int)

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default = 2000, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


if tf.__version__.split('.')[0] != "1":
    raise Exception("Tensorflow version 1 required")

if a.seed is None:
    a.seed = random.randint(0, 2**31 - 1)

tf.set_random_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)

if not os.path.exists(a.output_dir):
    os.makedirs(a.output_dir)

# load some options from the checkpoint
options = {"which_direction", "ngf", "ndf", "lab_colorization"}
with open(os.path.join(a.output_dir, "options.json")) as f:
    for key, val in json.loads(f.read()).items():
        if key in options:
            print("loaded", key, "=", val)
            setattr(a, key, val)

for k, v in a._get_kwargs():
    print(k, "=", v)

with open(os.path.join(a.output_dir, "options.json"), "w") as f:
    f.write(json.dumps(vars(a), sort_keys=True, indent=4))

examples = load_examples(a.test_input_dir, a.target_dir, a.batch_size)
print("examples count = %d" % examples.count)

# inputs and targets are [batch_size, height, width, channels]
model = create_model(examples.inputs, examples.targets)

inputs = deprocess(examples.inputs)
targets = deprocess(examples.targets)
outputs = deprocess(model.outputs)

def convert(image):
    if a.aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

# reverse any processing on images so they can be written to disk or displayed to user
with tf.name_scope("convert_inputs"):
    converted_inputs = convert(inputs)

with tf.name_scope("convert_targets"):
    converted_targets = convert(targets)

with tf.name_scope("convert_outputs"):
    converted_outputs = convert(outputs)

with tf.name_scope("encode_images"):
    display_fetches = {
        "paths": examples.paths,
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

for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
    tf.summary.histogram(var.op.name + "/gradients", grad)

with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1)

logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
with sv.managed_session() as sess:
    print("parameter_count =", sess.run(parameter_count))

    print("loading model from checkpoint")
    checkpoint = tf.train.latest_checkpoint(a.output_dir)
    saver.restore(sess, checkpoint)

    max_steps = 2**32
    if a.max_epochs is not None:
        max_steps = examples.steps_per_epoch * a.max_epochs
    if a.max_steps is not None:
        max_steps = a.max_steps

    # testing
    # at most, process the test data once
    max_steps = min(examples.steps_per_epoch, max_steps)
    for step in range(max_steps):
        results = sess.run(display_fetches)
        filesets = save_images(results)
        for i, f in enumerate(filesets):
            print("evaluated image", f["name"])
        index_path = append_index(filesets)

    print("wrote index at", index_path)
