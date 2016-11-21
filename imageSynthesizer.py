#!/usr/bin/env python
'''
Beetle image syntehsizer

'''
import time

import imageSyntehsize.argparser
import imageSyntehsize.main


if __name__ == '__main__':
    args = imageSyntehsize.argparser.parse_args()
    if args:
        if args.match_model == 'patchmatch':
            print('Using PatchMatch model')
            from imageSyntehsize.models.nnf import NNFModel as model_class
        else:
            print('Using brute-force model')
            from imageSyntehsize.models.analogy import AnalogyModel as model_class
        start_time = time.time()
        try:
            imageSyntehsize.main.main(args, model_class)
        except KeyboardInterrupt:
            print('Shutting down...')
        print('Done after {:.2f} seconds'.format(time.time() - start_time))
