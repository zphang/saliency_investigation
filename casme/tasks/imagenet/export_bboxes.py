import argparse
import glob
import json
import os
import tqdm
from bs4 import BeautifulSoup


def get_ground_truth_boxes(anno, category):
    boxes = []
    objs = anno.findAll('object')
    for obj in objs:
        obj_names = obj.findChildren('name')
        for name_tag in obj_names:
            # None of the examples fail this check? At least in val.
            if str(name_tag.contents[0]) == category:
                # fname = anno.findChild('filename').contents[0]
                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])
                ymin = int(bbox.findChildren('ymin')[0].contents[0])
                xmax = int(bbox.findChildren('xmax')[0].contents[0])
                ymax = int(bbox.findChildren('ymax')[0].contents[0])

                boxes.append([xmin, ymin, xmax, ymax])

    assert len(boxes) > 0
    return boxes


def get_resized_pos(gt_boxes, width, height, break_ratio):
    resized_boxes = []
    for box in gt_boxes:
        resized_boxes.append(resize_pos(box, width, height, break_ratio))

    return resized_boxes


def resize_pos(raw_pos, width, height, break_ratio):
    if break_ratio:
        ratio_x = 224/width
        ratio_y = 224/height
        xcut = 0
        ycut = 0
    else:
        if width > height:
            ratio_x = 224/height
            ratio_y = 224/height
            xcut = (width*ratio_x - 224) / 2
            ycut = 0
        else:
            ratio_x = 224/width
            ratio_y = 224/width
            xcut = 0
            ycut = (height*ratio_y - 224) / 2

    semi_cor_pos = [(ratio_x*raw_pos[0] - xcut),
                    (ratio_y*raw_pos[1] - ycut),
                    (ratio_x*raw_pos[2] - xcut),
                    (ratio_y*raw_pos[3] - ycut)]

    # The box coordinates returned here are rescaled to apply to a 224x224 image
    # but NOT clipped, so the coordinates may be negative or >224
    return [int(x) for x in semi_cor_pos]


def get_annotations(data_path, annotation_path, break_ratio):
    paths = glob.glob(os.path.join(data_path, "*", "*"))
    bboxes = {}
    metadata_dict = {}
    for path in tqdm.tqdm(paths):
        ann_path = os.path.join(annotation_path, os.path.basename(path)).split('.')[0] + '.xml'

        if not os.path.isfile(ann_path):
            raise KeyError(f"Annotations aren't found at {ann_path}. Aborting!")

        category = path.split('/')[-2]
        bbox_ls, metadata = get_gt_boxes(
            ann_path=ann_path,
            category=category,
            break_ratio=break_ratio,
        )
        key = os.path.basename(path).split(".")[0]
        bboxes[key] = bbox_ls
        metadata_dict[key] = metadata
    return bboxes, metadata_dict


def get_gt_boxes(ann_path, category, break_ratio, html_lib="html.parser"):
    with open(ann_path) as f:
        xml = f.readlines()
    anno = BeautifulSoup(''.join([line.strip('\t') for line in xml]), html_lib)

    size = anno.findChildren('size')[0]
    width = int(size.findChildren('width')[0].contents[0])
    height = int(size.findChildren('height')[0].contents[0])

    # get ground truth boxes positions in the original resolution
    original_gt_boxes = get_ground_truth_boxes(anno, category)
    # get ground truth boxes positions in the resized resolution
    gt_boxes = get_resized_pos(original_gt_boxes, width, height, break_ratio)
    gt_boxes_dicts = [
        dict(zip(["xmin", "ymin", "xmax", "ymax"], gt_box))
        for gt_box in gt_boxes
    ]
    original_gt_boxes_dicts = [
        dict(zip(["xmin", "ymin", "xmax", "ymax"], gt_box))
        for gt_box in original_gt_boxes
    ]
    metadata = {
        "width": width,
        "height": height,
        "gt_boxes": original_gt_boxes_dicts,
    }
    return gt_boxes_dicts, metadata


def get_annotations_and_write(data_path, annotation_path, break_ratio, output_path, metadata_output_path=None):
    annotations, metadata = get_annotations(
        data_path=data_path,
        annotation_path=annotation_path,
        break_ratio=break_ratio,
    )
    with open(output_path, "w") as f:
        f.write(json.dumps(annotations, indent=2))
    if metadata_output_path:
        with open(metadata_output_path, "w") as f:
            f.write(json.dumps(metadata, indent=2))


def main():
    args = get_args()
    get_annotations_and_write(
        data_path=args.data_path,
        annotation_path=args.annotation_path,
        break_ratio=args.break_ratio,
        output_path=args.output_path,
        metadata_output_path=args.metadata_output_path,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--annotation_path')
    parser.add_argument('--break_ratio', action='store_true')
    parser.add_argument('--output_path')
    parser.add_argument('--metadata_output_path', default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
