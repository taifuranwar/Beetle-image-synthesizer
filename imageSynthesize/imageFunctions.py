import numpy as np
from scipy.misc import imread, imresize

from . import vggLoader


# loads the image
def load_image(image_path):
    return imread(image_path)


# opens the image in  tensors
def image_preprocessor(x, img_width, img_height):
    img = imresize(x, (img_height, img_width), interp='bicubic').astype('float64')
    img = vggLoader.img_to_vgg_converter(img)
    img = np.expand_dims(img, axis=0)
    return img


# tenor->image
def convert_to_image(x, contrast_percent=0.0, resize=None):
    x = vggLoader.image_from_vgg_converter(x)
    if contrast_percent:
        min_x, max_x = np.percentile(x, (contrast_percent, 100 - contrast_percent))
        x = (x - min_x) * 255.0 / (max_x - min_x)
    x = np.clip(x, 0, 255)
    if resize:
        x = imresize(x, resize, interp='bicubic')
    return x.astype('uint8')
