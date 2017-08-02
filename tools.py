from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import subprocess
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import threading
import time

edge_pool = None

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="toon/src")
parser.add_argument("--output_dir", default="toon/cropped", help="output path")
parser.add_argument("--src_target", default="src_target", help="output path")
parser.add_argument("--src_input", default="src_input", help="output path")
parser.add_argument("--th", default=200, help="output path")
parser.add_argument("--threshold", default=50, help="output path")
a = parser.parse_args()


#global outcount
outcount= 0


def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now


def cropping_w(src, up, down):
    i = 0
    while i < src.width-5:
        out = np.mean(src.crop((i, up, i + 5, down)).getdata())
        if out < a.th:
            left_bound = i
            #print("left", i, ":", out)
            while i < src.width-5:
                out = np.mean(src.crop((i, up, i + 5, down)).getdata())
                if out > a.th:
                    right_bound = i
                    #print("right_bound ", i, ":", out)

                    if (right_bound - left_bound) > 200 : #and (down-up)//(right_bound - left_bound) == 1 :
                        print (up, down, left_bound, right_bound)
                        cropped = src.crop((left_bound, up, right_bound, down))
                        global outcount
                        out_path = os.path.join(a.output_dir, str(outcount) + ".png")
                        cropped.save(out_path)


                        out_src_input = os.path.join(a.src_input, str(outcount) + ".png")
                        out_src_target = os.path.join(a.src_target, str(outcount) + ".png")
                        reszied = cropped.resize((512, 512))
                        reszied.save(out_src_target)

                        gray_image = reszied.convert('L', colors = 5)
                        edge = gray_image.filter(ImageFilter.GaussianBlur(radius=5))
                        edge = edge.filter(ImageFilter.FIND_EDGES)
                        edge = ImageOps.invert(edge)
                        line = gray_image.point(lambda x: 0 if x < a.threshold else 255, '1')

                        bw = Image.blend(edge,line,0.8)
                        bw=bw.convert("RGB")
                        bw.save(out_src_input)
                        outcount=outcount+1
                    break
                i = i + 1
        i = i + 1



def cropping(src_file, des_path):
    src = Image.open(src_file)
    i=0
    while i < src.height :
        out = np.mean(src.crop((0,i,src.width, i+5)).getdata())
        if out < a.th :
            upper_bound = i
            #print ("up", i, ":", out)
            while i < src.height :
                out = np.mean(src.crop((0, i, src.width, i + 5)).getdata())
                if out > a.th:
                    down_bound = i
                    #print("down ", i, ":", out)

                    if (down_bound - upper_bound ) > 100 : cropping_w(src,upper_bound, down_bound)
                    break
                i=i+1
        i=i+1


    #im.save(dst, des_path)

complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

if not os.path.exists(a.output_dir):
    os.makedirs(a.output_dir)

src_paths = []
dst_paths = []

skipped = 0
for src_path in os.listdir(a.input_dir):
    name, _ = os.path.splitext(os.path.basename(src_path))
    src_path = os.path.join(a.input_dir, name + ".png")
    src_paths.append(src_path)

print("skipping %d files that already exist" % skipped)

total = len(src_paths)

print("processing %d files" % total)

start = time.time()

for src_path in src_paths :
    print (src_path)
    cropping(src_path, a.output_dir)
    complete()
