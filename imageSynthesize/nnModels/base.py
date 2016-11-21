import time

import numpy as np
from keras import backend as K

from imageSynthesize.losses.core import totalLoss


class BaseModel(object):
    '''Model to be extended.'''
    def __init__(self, net, args):
        self.set_net(net)
        self.args = args

    def set_net(self, net):
        self.net = net
        self.net_input = net.layers[0].input
        self.layer_map = dict([(layer.name, layer) for layer in self.net.layers])
        self._f_layer_outputs = {}

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
        self.f_outputs = K.function([self.net_input], outputs)

    def imageInputLossFuntion(self, source_image, source_image_mask, target_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        loss = K.variable(0.0)
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        loss += 1.0 * totalLoss(self.net_input, *target_image.shape[2:])
        return loss

    def precompute_static_features(self, source_image, source_image_mask, target_image):
        # figure out which layers we need to extract
        a_layers, ap_layers, b_layers = set(), set(), set()

        for layerset in (a_layers, ap_layers, b_layers):
            layerset.update(['conv3_1', 'conv4_1'])
        ap_layers.update(['conv3_1', 'conv4_1'])
        b_layers.update(['conv3_1', 'conv4_1'])
        ap_layers.update(['conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
        # let's get those features
        all_source_features = self.get_features(source_image, a_layers)
        all_source_image_mask_features = self.get_features(source_image_mask, ap_layers)
        all_target_features = self.get_features(target_image, b_layers)
        return all_source_features, all_source_image_mask_features, all_target_features

    def get_features(self, x, layers):
        if not layers:
            return None
        f = K.function([self.net_input], [self.get_layer_output(layer_name) for layer_name in layers])

        feature_outputs = f([x])
        features = dict(zip(layers, feature_outputs))
        return features

    def get_f_layer(self, layer_name):
        return K.function([self.net_input], [self.get_layer_output(layer_name)])

    def get_layer_output(self, name):
        if not name in self._f_layer_outputs:
            layer = self.layer_map[name]
            self._f_layer_outputs[name] = layer.get_output()
        return self._f_layer_outputs[name]

    def get_layer_output_shape(self, name):
        layer = self.layer_map[name]
        return layer.output_shape

    def evaluateLossAndGradient(self, x):
        x = x.reshape(self.output_shape)
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
