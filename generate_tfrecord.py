"""Generate TensorFlow Object Detection records from annotated-image CSVs.

Usage: generate_tfrecord.py CSV LABEL_MAP IMAGE_DIR OUTPUT
Images are stored as lossless RGB PNG; numeric class IDs are preserved.
"""
import argparse
import csv
import io
from pathlib import Path
from PIL import Image


def read_rows(csv_path, image_dir):
    """Validate rows and group objects without changing numeric class IDs."""
    groups = {}
    image_dir = Path(image_dir).resolve()
    with Path(csv_path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must contain {sorted(required)}')
        for row in reader:
            path = (image_dir / row['filename']).resolve()
            if not path.is_relative_to(image_dir):
                raise ValueError(f'Image must be inside image_dir: {row["filename"]}')
            if not path.is_file():
                raise FileNotFoundError(path)
            for key in ('width', 'height', 'xmin', 'ymin', 'xmax', 'ymax'):
                row[key] = int(row[key])
            if row['class'] not in {str(i) for i in range(1, 93)}:
                raise ValueError(f'Class must be a numeric string 1..92: {row["class"]}')
            if not (0 <= row['xmin'] < row['xmax'] <= row['width'] and
                    0 <= row['ymin'] < row['ymax'] <= row['height']):
                raise ValueError(f'Invalid box in {row["filename"]}: {row}')
            groups.setdefault(row['filename'], []).append(row)
    if not groups:
        raise ValueError('CSV contains no annotated images.')
    for filename, rows in groups.items():
        with Image.open(image_dir / filename) as image:
            if any((row['width'], row['height']) != image.size for row in rows):
                raise ValueError(f'XML/CSV dimensions differ from image dimensions: {filename}')
    return groups


def create_example(filename, rows, image_dir, label_map, tf):
    with Image.open(Path(image_dir) / filename) as image:
        image = image.convert('RGB')
        encoded = io.BytesIO()
        image.save(encoded, format='PNG')
        width, height = image.size
    def bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
    def int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    def float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': int_feature([height]),
        'image/width': int_feature([width]),
        'image/filename': bytes_feature([filename.encode('utf-8')]),
        'image/source_id': bytes_feature([filename.encode('utf-8')]),
        'image/encoded': bytes_feature([encoded.getvalue()]),
        'image/format': bytes_feature([b'png']),
        'image/object/bbox/xmin': float_feature([row['xmin'] / width for row in rows]),
        'image/object/bbox/xmax': float_feature([row['xmax'] / width for row in rows]),
        'image/object/bbox/ymin': float_feature([row['ymin'] / height for row in rows]),
        'image/object/bbox/ymax': float_feature([row['ymax'] / height for row in rows]),
        'image/object/class/text': bytes_feature([row['class'].encode('utf-8') for row in rows]),
        'image/object/class/label': int_feature([label_map[row['class']] for row in rows]),
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('label_map_path', type=Path)
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('output_path', type=Path)
    args = parser.parse_args()
    if args.output_path.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_path}; choose a new output or move the old record.')
    groups = read_rows(args.csv_path, args.image_dir)
    import tensorflow as tf
    from object_detection.utils import label_map_util
    label_map = label_map_util.get_label_map_dict(str(args.label_map_path))
    if label_map != {str(i): i for i in range(1, 93)}:
        raise ValueError('Label map must preserve the original numeric names and IDs 1..92.')
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to an exclusive temporary file. Do not leave a partial final record on failure.
    import tempfile
    import os
    fd, temporary = tempfile.mkstemp(prefix=args.output_path.name + '.', suffix='.tmp', dir=args.output_path.parent)
    os.close(fd)
    try:
        with tf.io.TFRecordWriter(temporary) as writer:
            for filename in sorted(groups):
                writer.write(create_example(filename, groups[filename], args.image_dir, label_map, tf).SerializeToString())
        if args.output_path.exists():
            raise FileExistsError(args.output_path)
        Path(temporary).replace(args.output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Wrote {len(groups)} images to {args.output_path}')


if __name__ == '__main__':
    main()
