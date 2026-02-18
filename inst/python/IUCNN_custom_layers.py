#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created February 2026

@author: Torsten Hauffe (torsten.hauffe@gmail.com)
"""


import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
