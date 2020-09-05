import paddle

from PIL import Image
import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir):
    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images

def DataReader(root, transforms=None):
    def reader():
        samples = make_dataset(root)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of:"+root))
        for idx in range(len(samples)):
            path, target = samples[idx]
            img =  Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if  transforms is not None:
                for t in transforms:
                    # print(type(img))
                    img = t(img)
            yield img, target
    return reader

def DataLoader(root, transforms=None, batch_size=1, shuffle=False):
    if shuffle:
        return paddle.batch(paddle.reader.shuffle(DataReader(root, transforms), batch_size * 2),batch_size=batch_size)
    else:
        return paddle.batch(DataReader(root, transforms), batch_size=batch_size)