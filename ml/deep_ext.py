# -*- coding: utf-8 -*-
"""
DEEP_EXT: extension functions for various Deep Learning packages.
"""
import lasagne.layers

class LasagneExtensions:
    @staticmethod
    def get_layer_by_name(net, name):
        for i, layer in enumerate(net.get_all_layers()):
            if layer.name == name:
                return layer, i
        return None, None
    
    @staticmethod
    def encode_input(net, layer_name, X):
        encode_layer_index = map(lambda pair : pair[0], net.layers).index(layer_name)
        encode_layer = net.get_all_layers()[encode_layer_index]

        return LasagneExtensions.get_output_from_nn(encode_layer, X)

    @staticmethod    
    def get_output_from_nn(last_layer, X):
        return lasagne.layers.get_output(last_layer,X).eval()
    
    @staticmethod
    def decode_encoded_input(net, X):
        final_layer = net.get_all_layers()[-1]
        return LasagneExtensions.get_output_from_nn(final_layer, X)
