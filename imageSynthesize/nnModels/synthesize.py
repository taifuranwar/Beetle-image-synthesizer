import time

import numpy as np
from keras import backend as K

from imageSynthesize.losses.synthesize import synthesize_loss
from imageSynthesize.losses.core import content_loss
from imageSynthesize.losses.markovRandomField import markovRandomField_loss
from imageSynthesize.losses.neuralConvolutionalStyleLoss import neuralConvolutionalStyleLoss

from .base import BaseModel


class SynthesizeModel(BaseModel):
    #Model for image Synthesize.

    def imageInputLossFuntion(self, source_image, source_image_mask, target_image):
        '''Calculte loss as the function of the x image'''
        print('crating loss function...')
        loss = super(SynthesizeModel, self).imageInputLossFuntion(source_image, source_image_mask, target_image)
        # calculate static features beforehand
        print('calculate static features beforehand...')
        all_source_features, all_source_image_mask_features, all_target_features = self.precompute_static_features(source_image, source_image_mask, target_image)
        print('Building and combining losses...')

        for layer_name in ['conv3_1', 'conv4_1']:
            source_features = all_source_features[layer_name][0]
            source_image_mask_features = all_source_image_mask_features[layer_name][0]
            target_features = all_target_features[layer_name][0]
            # current combined output
            layer_features = self.get_layer_output(layer_name)
            combination_features = layer_features[0, :, :, :]
            al = synthesize_loss(source_features, source_image_mask_features,
                target_features, combination_features,
                use_full_synthesize= False,
                patch_size=1,
                patch_stride=1)
            loss += (1.0 / len(['conv3_1', 'conv4_1'])) * al


        for layer_name in ['conv3_1', 'conv4_1']:
            source_image_mask_features = K.variable(all_source_image_mask_features[layer_name][0])
            layer_features = self.get_layer_output(layer_name)
            # current combined output
            combination_features = layer_features[0, :, :, :]
            sl = markovRandomField_loss(source_image_mask_features, combination_features,
                patch_size=1,
                patch_stride=1)
            loss += (0.5 / len(['conv3_1', 'conv4_1'])) * sl


        for layer_name in ['conv3_1', 'conv4_1']:
            target_features = K.variable(all_target_features[layer_name][0])
            # current combined output
            bp_features = self.get_layer_output(layer_name)
            cl = content_loss(bp_features, target_features)
            loss += 0.0 / len(['conv3_1', 'conv4_1']) * cl


        return loss
