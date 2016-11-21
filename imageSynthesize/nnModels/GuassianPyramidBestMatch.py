import time

import numpy as np
from keras import backend as K

from imageSynthesize.losses.core import content_loss
from imageSynthesize.losses.guassianPyramidBestMatch import guassianPyramidBestMatch_synthesize_loss, NNFState, PatchMatcher
from imageSynthesize.losses.neuralConvolutionalStyleLoss.neuralConvolutionalStyleLoss import neuralConvolutionalStyleLoss

from .base import BaseModel


class GuassianPyramidBestMatch(BaseModel):

    def build(self, source_image, source_image_mask, target_image, output_shape):
        self.output_shape = output_shape
        loss = self.imageInputLossFuntion(source_image, source_image_mask, target_image)
        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, self.net_input)
        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)
        f_inputs = [self.net_input]
        for guassianPyramidBestMatch in self.feature_guassianPyramidBestMatchs:
            f_inputs.append(guassianPyramidBestMatch.placeholder)
        self.f_outputs = K.function(f_inputs, outputs)

    def eval_loss_and_grads(self, x):
        x = x.reshape(self.output_shape)
        f_inputs = [x]

        for guassianPyramidBestMatch in self.feature_guassianPyramidBestMatchs:
            guassianPyramidBestMatch.update(x, num_steps=self.args.markovRandomField_guassianPyramidBestMatch_steps)
            new_target = guassianPyramidBestMatch.matcher.get_reconstruction()
            f_inputs.append(new_target)

        outs = self.f_outputs(f_inputs)
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    def imageInputLossFuntion(self, source_image, source_image_mask, target_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        print('Building loss...')
        loss = super(GuassianPyramidBestMatch, self).imageInputLossFuntion(source_image, source_image_mask, target_image)
        # Precompute static features for performance
        print('Precomputing static features...')
        all_source_features, all_source_image_mask_features, all_target_features = self.precompute_static_features(source_image, source_image_mask, target_image)
        print('Building and combining losses...')
        if 1.0:
            for layer_name in ['conv3_1', 'conv4_1']:
                source_features = all_source_features[layer_name][0]
                source_image_mask_features = all_source_image_mask_features[layer_name][0]
                target_features = all_target_features[layer_name][0]
                
                layer_features = self.get_layer_output(layer_name)
                combination_features = layer_features[0, :, :, :]
                al = guassianPyramidBestMatch_synthesize_loss(
                    source_features, source_image_mask_features, target_features, combination_features,
                    num_steps=self.args.synthesize_guassianPyramidBestMatch_steps, patch_size=1,
                    patch_stride=1, jump_size=1.0)
                loss += (1.0 / len(['conv3_1', 'conv4_1'])) * al

        existing_feature_guassianPyramidBestMatchs = getattr(self, 'feature_guassianPyramidBestMatchs', [None] * len(['conv3_1', 'conv4_1']))
        self.feature_guassianPyramidBestMatchs = []

        for layer_name, existing_guassianPyramidBestMatch in zip(['conv3_1', 'conv4_1'], existing_feature_guassianPyramidBestMatchs):
            source_image_mask_features = all_source_image_mask_features[layer_name][0]
            # current combined output
            layer_features = self.get_layer_output(layer_name)
            combination_features = layer_features[0, :, :, :]
            input_shape = self.get_layer_output_shape(layer_name)
            if existing_guassianPyramidBestMatch and not self.args.randomize_mnf_guassianPyramidBestMatch:
                matcher = existing_guassianPyramidBestMatch.matcher.scale((input_shape[3], input_shape[2], input_shape[1]), source_image_mask_features)
            else:
                matcher = PatchMatcher(
                    (input_shape[3], input_shape[2], input_shape[1]), source_image_mask_features,
                    patch_size=1, jump_size=1.0, patch_stride=1)
            guassianPyramidBestMatch = NNFState(matcher, self.get_f_layer(layer_name))
            self.feature_guassianPyramidBestMatchs.append(guassianPyramidBestMatch)
            sl = content_loss(combination_features, guassianPyramidBestMatch.placeholder)
            loss += (0.5 / len(['conv3_1', 'conv4_1'])) * sl


        for layer_name in ['conv3_1', 'conv4_1']:
            target_features = K.variable(all_target_features[layer_name][0])
            # current combined output
            bp_features = self.get_layer_output(layer_name)
            cl = content_loss(bp_features, target_features)
            loss += 0.0 / len(['conv3_1', 'conv4_1']) * cl


        return loss
