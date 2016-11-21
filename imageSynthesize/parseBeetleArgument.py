import argparse
import os

from keras import backend as K


VGG_ENV_VAR = 'VGG_WEIGHT_PATH'

#transferring argument called in beetle ajax call
class CommaSplitAction(argparse.Action):
    #splitting arguments
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [v.strip() for v in values.split(',')])


def parse_args():
    '''Parses command line arguments for the image synthesize command.'''
    parser = argparse.ArgumentParser(description='Image Synthesize in Beetle app')
    parser.add_argument('unfilteredSourceImage', metavar='ref', type=str,
                        help='Source image path mask')
    parser.add_argument('filteredSourceImage', metavar='base', type=str,
                        help='Source image path ')
    parser.add_argument('unfilteredTargetImage', metavar='ref', type=str,
                        help='Target image mask')
    parser.add_argument('filteredTargetImage', metavar='res_prefix', type=str,
                        help='Output path')
    # size-related
    parser.add_argument('--width', dest='out_width', type=int,
                        default=0, help='Set output width')
    parser.add_argument('--height', dest='out_height', type=int,
                        default=0, help='Set output height')
    parser.add_argument('--scales', dest='num_scales', type=int,
                        default=3, help='Run at N different scales')
    parser.add_argument('--min-scale', dest='min_scale', type=float,
                        default=0.25, help='Smallest scale to iterate')
    parser.add_argument('--source-scale-mode', dest='source_scale_mode', type=str,
                        default='none', help='Method of scaling source image and mask relative to target image')
    parser.add_argument('--source-scale', dest='source_scale', type=float,
                        default=1.0, help='Additional scale factor for source mask  and source image')
    parser.add_argument('--output-full', dest='output_full_size', action='store_true',
                        help='Output all intermediate images at full size regardless of current scale.')

    # VGG
    parser.add_argument('--vgg-weights', dest='vgg_weights', type=str)
    parser.add_argument('--pool-mode', dest='pool_mode', type=str,
                        default='max', help='Pooling mode for VGG ("avg" or "max")')

    args = parser.parse_args()
    # make sure weights are in place
    if not os.path.exists(args.vgg_weights):
        print('Please use vggLoader path')
        return None
    return args
