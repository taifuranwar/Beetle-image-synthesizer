import os
import time

import numpy as np
import scipy.ndimage
from scipy.misc import imsave

from imageSynthesize import imageFunctions, vggLoader
from imageSynthesize.vggOptimizer import vggOptimizer


def main(args, model_class):
    #main loop.
    #please set server script memlimit to high

    if args.num_scales > 1:
        step_scale_factor = (1 - args.min_scale) / (args.num_scales - 1)
    else:
        step_scale_factor = 0.0
        args.min_scale = 1.0
    # input image loading
    full_source_image = imageFunctions.load_image(args.filteredSourceImage)
    full_source_image_mask = imageFunctions.load_image(args.unfilteredSourceImage)
    full_target_image_mask = imageFunctions.load_image(args.unfilteredTargetImage)
    # image ratio
    full_img_width, full_img_height = calculate_image(args, full_target_image_mask)
    img_num_channels = 3
    target_scale_ratio_width = float(full_target_image_mask.shape[1]) / full_img_width
    target_scale_ratio_height = float(full_target_image_mask.shape[0]) / full_img_height
    # does output directory works ?
    output_dir = os.path.dirname(args.filteredTargetImage)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # loop of the image manipulation
    x = None  # output image
    optimizer = vggOptimizer()
    for scale_i in range(args.num_scales):
        scale_factor = (scale_i * step_scale_factor) + args.min_scale
        # scale the source image
        img_width = int(round(full_img_width * scale_factor))
        img_height = int(round(full_img_height * scale_factor))
        # vggOptimizer state
        if x is None:  # we need to create an initial state
            x = np.random.uniform(0, 255, (img_height, img_width, 3))
            x = vggLoader.img_to_vgg_converter(x)
        else:  # resize the last state
            zoom_ratio = img_width / float(x.shape[-1])
            x = scipy.ndimage.zoom(x, (1, zoom_ratio, zoom_ratio), order=1)
            img_height, img_width = x.shape[-2:]
        # determine scaling of source images
        if args.source_scale_mode == 'match':
            source_img_width = img_width
            source_img_height = img_height
        elif args.source_scale_mode == 'none':
            source_img_width = full_source_image_mask.shape[1] * scale_factor
            source_img_height = full_source_image_mask.shape[0] * scale_factor
        else:
            source_img_width = full_source_image_mask.shape[1] * scale_factor * target_scale_ratio_width
            source_img_height = full_source_image_mask.shape[0] * scale_factor * target_scale_ratio_height
        source_img_width = int(round(args.source_scale * source_img_width))
        source_img_height = int(round(args.source_scale * source_img_height))
        # prep the source image
        source_image = imageFunctions.image_preprocessor(full_source_image_mask, source_img_width, source_img_height)
        source_image_mask = imageFunctions.image_preprocessor(full_source_image, source_img_width, source_img_height)
        target_image = imageFunctions.image_preprocessor(full_target_image_mask, img_width, img_height)
        print('Scale factor {} "A" shape {} "B" shape {}'.format(scale_factor, source_image.shape, target_image.shape))
        # create the NN model and load
        net = vggLoader.get_model(img_width, img_height, weights_path=args.vgg_weights, pool_mode=args.pool_mode)
        model = model_class(net, args)
        model.build(source_image, source_image_mask, target_image, (1, img_num_channels, img_height, img_width))

        for i in range(1):
            print('Start of iteration {} x {}'.format(scale_i, i))
            start_time = time.time()

            color_jitter = (0 * 2) * (np.random.random((3, img_height, img_width)) - 0.5)
            x += color_jitter

            jitter = 0 * scale_factor
            ox, oy = np.random.randint(-jitter, jitter+1, 2)
            x = np.roll(np.roll(x, ox, -1), oy, -2) # apply jitter shift
            # actually run the vggOptimizer
            x, min_val, info = optimizer.optimize(x, model)
            print('Current loss value: {}'.format(min_val))
            # unjitter the image
            x = x.reshape((3, img_height, img_width))

            x = np.roll(np.roll(x, -ox, -1), -oy, -2) # unshift image

            x -= color_jitter
            # save the image
            if args.output_full_size:
                out_resize_shape = (full_img_height, full_img_width)
            else:
                out_resize_shape = None
            img = imageFunctions.convert_to_image(np.copy(x), contrast_percent=0.02,resize=out_resize_shape)
            fname = args.filteredTargetImage + '_at_iteration_{}_{}.png'.format(scale_i, i)
            imsave(fname, img)
            end_time = time.time()
            print('Image saved as {}'.format(fname))
            print('Iteration completed in {:.2f} seconds'.format(end_time - start_time,))


def calculate_image(args, full_target_image_mask):
    '''Determine the dimensions of the generated picture.

    Defaults to the size of Image B.
    '''
    full_img_width = full_target_image_mask.shape[1]
    full_img_height = full_target_image_mask.shape[0]
    if args.out_width or args.out_height:
        if args.out_width and args.out_height:
            full_img_width = args.out_width
            full_img_height = args.out_height
        else:
            if args.out_width:
                full_img_height = int(round(args.out_width / float(full_img_width) * full_img_height))
                full_img_width = args.out_width
            else:
                full_img_width = int(round(args.out_height / float(full_img_height) * full_img_width))
                full_img_height = args.out_height
    return full_img_width, full_img_height
