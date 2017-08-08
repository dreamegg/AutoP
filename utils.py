
import tensorflow as tf
import os
import glob
import random
import math
import collections

CROP_SIZE = 256

#Examples = collections.namedtuple("Examples", "inputs, targets, count, steps_per_epoch")
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def convert(image, aspect_ratio):
    if aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def transform(image,scale_size, seed):
    r = image
    #if a.flip: flip is true
    r = tf.image.random_flip_left_right(r, seed=seed)

    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
    if scale_size > CROP_SIZE:
        r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
    elif scale_size < CROP_SIZE:
        raise Exception("scale size cannot be less than crop size")
    return r

def listup_path(input_dir)  :
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    return input_paths

def load_examples(input_dir, target_dir, batch_size):
    scale_size = CROP_SIZE

    input_paths = listup_path(input_dir)
    target_paths = listup_path(target_dir)

    decode = tf.image.decode_png

    with tf.name_scope("load_images"):
        reader = tf.WholeFileReader()

        path_queue = tf.train.string_input_producer(input_paths, shuffle=False)
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        path_queue_t = tf.train.string_input_producer(target_paths, shuffle=False)
        paths_t, contents_t = reader.read(path_queue_t)
        raw_target = decode(contents_t)
        raw_target = tf.image.convert_image_dtype(raw_target, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        raw_target.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = raw_input
        b_images = raw_target

    inputs, targets = [a_images, b_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)

    with tf.name_scope("input_images"):
        input_images = transform(inputs, scale_size, seed)

    with tf.name_scope("target_images"):
        target_images = transform(targets, scale_size, seed)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
    inputs_batch, targets_batch = tf.train.batch([input_images, target_images],batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )