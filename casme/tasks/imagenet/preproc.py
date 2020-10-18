from torchvision.datasets.folder import has_file_allowed_extension
import glob
import numpy as np
import pyutils.io as io
import os
import casme.tasks.imagenet.export_bboxes as export_bboxes
import tqdm
import zconf

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    train_path = zconf.attr(type=str)
    val_path = zconf.attr(type=str)
    val_annotation_path = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)

    extended_annot_base_path = zconf.attr(type=str, default=None)
    num_per_class_in_train_val = zconf.attr(type=int, default=50)
    seed = zconf.attr(type=int, default=1234)


def find_classes(base_path):
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(fol, class_to_idx, extensions):
    images = []
    fol = os.path.expanduser(fol)
    for target in sorted(os.listdir(fol)):
        d = os.path.join(fol, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def generate_jsons(train_path, val_path, output_base_path, num_per_class_in_a=50, seed=1234):
    random_state = np.random.RandomState(seed=seed)
    classes, class_to_idx = find_classes(train_path)
    samples = make_dataset(train_path, class_to_idx, IMG_EXTENSIONS)
    random_state.shuffle(samples)

    # Train
    io.write_json(
        {
            "root": train_path,
            "samples": samples,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train.json"),
    )

    # Resampled Train
    class_dict = {}
    for path, class_idx in samples:
        if class_idx not in class_dict:
            class_dict[class_idx] = []
        class_dict[class_idx].append((path, class_idx))

    samples_a, samples_b = [], []
    for class_idx in range(len(class_dict)):
        class_samples = class_dict[class_idx]
        chosen = set(random_state.choice(np.arange(len(class_samples)),
                                         num_per_class_in_a, replace=False))
        for i, sample in enumerate(class_samples):
            if i in chosen:
                samples_a.append(sample)
            else:
                samples_b.append(sample)

    io.write_json(
        {
            "root": train_path,
            "samples": samples_a,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train_val.json"),
    )
    io.write_json(
        {
            "root": train_path,
            "samples": samples_b,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train_train.json"),
    )
    # Shuffled Train
    random_classes = np.random.randint(1000, size=len(samples))
    io.write_json(
        {
            "root": train_path,
            "samples": [(path, int(c)) for (path, _), c in zip(samples, random_classes)],
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train_shuffle.json"),
    )

    # Val
    classes, class_to_idx = find_classes(val_path)
    val_samples = make_dataset(val_path, class_to_idx, IMG_EXTENSIONS)
    io.write_json(
        {
            "root": val_path,
            "samples": val_samples,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "val.json"),
    )


def generate_jsons_with_extended_annot(
        train_path, val_path, val_annotation_path,
        output_base_path,
        extended_annot_base_path=None,
        num_per_class_in_train_val=50, seed=1234):
    os.makedirs(output_base_path, exist_ok=True)
    random_state = np.random.RandomState(seed=seed)
    classes, class_to_idx = find_classes(train_path)
    train_samples = make_dataset(train_path, class_to_idx, IMG_EXTENSIONS)
    random_state.shuffle(train_samples)

    # 1. Train
    io.write_json(
        {
            "root": train_path,
            "samples": train_samples,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train.json"),
    )

    # 2. New Train, TrainVal split
    if extended_annot_base_path is not None:
        annot_data = get_extended_annot_data(extended_annot_base_path)

        # 2a. Put samples that have annotations into class_dict, others into remaining_samples
        class_dict = {}
        remaining_samples = []
        for path, class_idx in train_samples:
            if class_idx not in class_dict:
                class_dict[class_idx] = []
            folder_id, class_id, file_id = split_path(path)
            assert folder_id == class_id
            if (class_id, file_id) in annot_data:
                class_dict[class_idx].append((path, class_idx))
            else:
                remaining_samples.append((path, class_idx))
        print(len(train_samples), len(remaining_samples))

        # 2b. Construct splits
        train_val_samples = []
        train_train_samples: list = remaining_samples.copy()
        for annot_class_samples in class_dict.values():
            random_state.shuffle(annot_class_samples)
            train_val_samples += annot_class_samples[:num_per_class_in_train_val]
            train_train_samples += annot_class_samples[num_per_class_in_train_val:]
        random_state.shuffle(train_val_samples)
        random_state.shuffle(train_train_samples)

        # 2c. Construct annot subsamples
        subsampled_annot_data = {}
        for path, _ in train_val_samples:
            _, class_id, file_id = split_path(path)
            subsampled_annot_data[f"{class_id}_{file_id}"], metadata = export_bboxes.get_gt_boxes(
                ann_path=annot_data[(class_id, file_id)],
                category=class_id,
                break_ratio=False,
                html_lib="html.parser",
            )
        print(len(subsampled_annot_data))

        # 2d. Write all
        io.write_json(
            {
                "root": train_path,
                "samples": train_val_samples,
                "classes": classes,
                "class_to_idx": class_to_idx,
            },
            os.path.join(output_base_path, "train_val.json"),
        )
        io.write_json(
            {
                "root": train_path,
                "samples": train_train_samples,
                "classes": classes,
                "class_to_idx": class_to_idx,
            },
            os.path.join(output_base_path, "train_train.json"),
        )
        io.write_json(
            subsampled_annot_data,
            os.path.join(output_base_path, "train_val_bboxes.json"),
        )

    # 3. Shuffled Train
    random_classes = np.random.randint(1000, size=len(train_samples))
    shuffled_samples = [(path, int(c)) for (path, _), c in zip(train_samples, random_classes)]
    io.write_json(
        {
            "root": train_path,
            "samples": shuffled_samples,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "train_shuffle.json"),
    )

    # Val
    classes, class_to_idx = find_classes(val_path)
    val_samples = make_dataset(val_path, class_to_idx, IMG_EXTENSIONS)
    io.write_json(
        {
            "root": val_path,
            "samples": val_samples,
            "classes": classes,
            "class_to_idx": class_to_idx,
        },
        os.path.join(output_base_path, "val.json"),
    )
    export_bboxes.get_annotations_and_write(
        data_path=val_path,
        annotation_path=val_annotation_path,
        break_ratio=False,
        output_path=os.path.join(output_base_path, "val_bboxes.json"),
    )


def get_extended_annot_data(annot_base_path):
    annot_path_ls = sorted(glob.glob(os.path.join(annot_base_path, "*/*.xml")))
    annot_data = {}
    for path in tqdm.tqdm(annot_path_ls):
        folder_id, class_id, file_id = split_path(path)
        if folder_id == class_id:
            annot_data[class_id, file_id] = path
    return annot_data


def split_path(path):
    tokens = path.split(".")[0].split("/")
    class_id, file_id = tokens[-1].split("_")
    folder_id = tokens[-2]
    return folder_id, class_id, file_id


def main(args: RunConfiguration):
    generate_jsons_with_extended_annot(
        train_path=args.train_path,
        val_path=args.val_path,
        val_annotation_path=args.val_annotation_path,
        output_base_path=args.output_base_path,
        extended_annot_base_path=args.extended_annot_base_path,
        num_per_class_in_train_val=args.num_per_class_in_train_val,
        seed=args.seed,
    )


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
