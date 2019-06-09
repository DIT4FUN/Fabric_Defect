#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image
import labelme

path = "./trainData/20181024"
outPath = "./trainData/20181024/out"
labels="./trainData/labels.txt"

def main():
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument('input_dir', help='input annotated directory')
    # parser.add_argument('output_dir', help='output dataset directory')
    # parser.add_argument('--labels', help='labels file', required=True)
    # args = parser.parse_args()

    if osp.exists(outPath):
        print('Output directory already exists:', outPath)
        sys.exit(1)
    os.makedirs(outPath)
    os.makedirs(osp.join(outPath, 'JPEGImages'))
    os.makedirs(osp.join(outPath, 'SegmentationClass'))
    os.makedirs(osp.join(outPath, 'SegmentationClassPNG'))
    os.makedirs(osp.join(outPath, 'SegmentationClassVisualization'))
    print('Creating dataset:', outPath)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(outPath, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    for label_file in glob.glob(osp.join(path, '*.json')):
        try:
            print('Generating dataset from:', label_file)
            with open(label_file) as f:
                base = osp.splitext(osp.basename(label_file))[0]
                out_img_file = osp.join(
                    outPath, 'JPEGImages', base + '.jpg')
                out_lbl_file = osp.join(
                    outPath, 'SegmentationClass', base + '.npy')
                out_png_file = osp.join(
                    outPath, 'SegmentationClassPNG', base + '.png')
                out_viz_file = osp.join(
                    outPath,
                    'SegmentationClassVisualization',
                    base + '.jpg',
                )

                data = json.load(f)

                img_file = osp.join(osp.dirname(label_file), data['imagePath'])
                img = np.asarray(PIL.Image.open(img_file))
                PIL.Image.fromarray(img).save(out_img_file)

                lbl = labelme.utils.shapes_to_label(
                    img_shape=img.shape,
                    shapes=data['shapes'],
                    label_name_to_value=class_name_to_id,
                )
                labelme.utils.lblsave(out_png_file, lbl)

                np.save(out_lbl_file, lbl)

                viz = labelme.utils.draw_label(
                    lbl, img, class_names, colormap=colormap)
                PIL.Image.fromarray(viz).save(out_viz_file)
        except:
            continue


if __name__ == '__main__':
    main()
