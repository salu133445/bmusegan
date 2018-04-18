"""Classes for the components of the model, including the generator, the
discriminator and the refiner.
"""
from collections import OrderedDict
import tensorflow as tf
from bmusegan.lib.neuralnet import NeuralNet

class Component(object):
    """Base class for components."""
    def __init__(self, tensor_in, condition, slope_tensor=None):
        if not isinstance(tensor_in, tf.Tensor):
            raise TypeError("`tensor_in` must be of tf.Tensor type")

        self.tensor_in = tensor_in
        self.condition = condition
        self.slope_tensor = slope_tensor

        self.scope = None
        self.tensor_out = tensor_in
        self.nets = OrderedDict()
        self.vars = None

    def __repr__(self):
        return "Component({}, input_shape={}, output_shape={})".format(
            self.scope.name, self.tensor_in.get_shape(),
            str(self.tensor_out.get_shape()))

    def get_summary(self):
        """Return the summary string."""
        cleansed_nets = []
        for net in self.nets.values():
            if isinstance(net, NeuralNet):
                if net.scope is not None:
                    cleansed_nets.append(net)
            if isinstance(net, list):
                if net[0].scope is not None:
                    cleansed_nets.append(net[0])
        return '\n'.join(
            ["{:-^80}".format(' ' + self.scope.name + ' '),
             "{:39} {}".format('Input', self.tensor_in.get_shape())]
            + ['-' * 80 + '\n' + (x.get_summary()) for x in cleansed_nets])

class Generator(Component):
    """Class that defines the generator."""
    def __init__(self, tensor_in, config, condition=None, name='Generator',
                 reuse=None):
        super().__init__(tensor_in, condition)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the generator."""
        nets = OrderedDict()

        nets['shared'] = NeuralNet(self.tensor_in, config['net_g']['shared'],
                                   name='shared')

        nets['pitch_time_private'] = [
            NeuralNet(nets['shared'].tensor_out,
                      config['net_g']['pitch_time_private'],
                      name='pt_'+str(idx))
            for idx in range(config['num_track'])
        ]

        nets['time_pitch_private'] = [
            NeuralNet(nets['shared'].tensor_out,
                      config['net_g']['time_pitch_private'],
                      name='tp_'+str(idx))
            for idx in range(config['num_track'])
        ]

        nets['merged_private'] = [
            NeuralNet(tf.concat([nets['pitch_time_private'][idx].tensor_out,
                                 nets['time_pitch_private'][idx].tensor_out],
                                -1),
                      config['net_g']['merged_private'],
                      name='merged_'+str(idx))
            for idx in range(config['num_track'])
        ]

        tensor_out = tf.concat([x.tensor_out for x in nets['merged_private']],
                               -1)
        return tensor_out, nets

class Discriminator(Component):
    """Class that defines the discriminator."""
    def __init__(self, tensor_in, config, condition=None, name='Discriminator',
                 reuse=None):
        super().__init__(tensor_in, condition)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the discriminator."""
        nets = OrderedDict()

        # main stream
        nets['pitch_time_private'] = [
            NeuralNet(tf.expand_dims(self.tensor_in[..., idx], -1),
                      config['net_d']['pitch_time_private'],
                      name='pt_' + str(idx))
            for idx in range(config['num_track'])
        ]

        nets['time_pitch_private'] = [
            NeuralNet(tf.expand_dims(self.tensor_in[..., idx], -1),
                      config['net_d']['time_pitch_private'],
                      name='tp_' + str(idx))
            for idx in range(config['num_track'])
        ]

        nets['merged_private'] = [
            NeuralNet(
                tf.concat([x.tensor_out,
                           nets['time_pitch_private'][idx].tensor_out], -1),
                config['net_d']['merged_private'], name='merged_' + str(idx))
            for idx, x in enumerate(nets['pitch_time_private'])
        ]

        nets['shared'] = NeuralNet(
            tf.concat([x.tensor_out for x in nets['merged_private']], -1),
            config['net_d']['shared'], name='shared'
        )

        # chroma stream
        reshaped = tf.reshape(
            self.tensor_in, (-1, config['num_bar'], config['num_beat'],
                             config['beat_resolution'], config['num_pitch']//12,
                             12, config['num_track'])
        )
        self.chroma = tf.reduce_sum(reshaped, axis=(3, 4))
        nets['chroma'] = NeuralNet(self.chroma, config['net_d']['chroma'],
                                   name='chroma')

        # onset stream
        padded = tf.pad(self.tensor_in[:, :, :-1, :, 1:],
                        [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]])
        self.onset = tf.concat([tf.expand_dims(self.tensor_in[..., 0], -1),
                                self.tensor_in[..., 1:] - padded], -1)
        nets['onset'] = NeuralNet(self.onset, config['net_d']['onset'],
                                  name='onset')

        if (config['net_d']['chroma'] is not None
                or config['net_d']['onset'] is not None):
            to_concat = [nets['shared'].tensor_out]
            if config['net_d']['chroma'] is not None:
                to_concat.append(nets['chroma'].tensor_out)
            if config['net_d']['onset'] is not None:
                to_concat.append(nets['onset'].tensor_out)
            concated = tf.concat(to_concat, -1)
        else:
            concated = nets['shared'].tensor_out

        # merge streams
        nets['merged'] = NeuralNet(concated, config['net_d']['merged'],
                                   name='merged')

        return nets['merged'].tensor_out, nets

class Refiner(Component):
    """Class that defines the refiner."""
    def __init__(self, tensor_in, config, condition=None, slope_tensor=None,
                 name='Refiner', reuse=None):
        super().__init__(tensor_in, condition, slope_tensor)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the refiner."""
        nets = OrderedDict()

        nets['private'] = [
            NeuralNet(tf.expand_dims(self.tensor_in[..., idx], -1),
                      config['net_r']['private'],
                      slope_tensor=self.slope_tensor, name='private'+str(idx))
            for idx in range(config['num_track'])
        ]

        return tf.concat([x.tensor_out for x in nets['private']], -1), nets

class End2EndGenerator(Component):
    """Class that defines the end-to-end generator."""
    def __init__(self, tensor_in, config, condition=None, slope_tensor=None,
                 name='End2EndGenerator', reuse=None):
        super().__init__(tensor_in, condition, slope_tensor)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the end-to-end generator."""
        nets = OrderedDict()

        nets['shared'] = NeuralNet(self.tensor_in, config['net_g']['shared'],
                                   name='shared')

        nets['pitch_time_private'] = [
            NeuralNet(nets['shared'].tensor_out,
                      config['net_g']['pitch_time_private'],
                      name='pt_'+str(idx))
            for idx in range(config['num_track'])
        ]

        nets['time_pitch_private'] = [
            NeuralNet(nets['shared'].tensor_out,
                      config['net_g']['time_pitch_private'],
                      name='tp_'+str(idx))
            for idx in range(config['num_track'])
        ]

        nets['merged_private'] = [
            NeuralNet(tf.concat([nets['pitch_time_private'][idx].tensor_out,
                                 nets['time_pitch_private'][idx].tensor_out],
                                -1),
                      config['net_g']['merged_private'],
                      name='merged_'+str(idx))
            for idx in range(config['num_track'])
        ]

        nets['refiner_private'] = [
            NeuralNet(nets['merged_private'][idx].tensor_out,
                      config['net_r']['private'],
                      slope_tensor=self.slope_tensor,
                      name='refiner_private'+str(idx))
            for idx in range(config['num_track'])
        ]

        return tf.concat([x.tensor_out for x in nets['private']], -1), nets
