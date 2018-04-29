"""Classes for the models
"""
import os
import time
import numpy as np
import tensorflow as tf
import SharedArray as sa
from musegan.bmusegan.components import Discriminator, Generator, Refiner
from musegan.bmusegan.components import End2EndGenerator
from musegan.utils.metrics import Metrics
from musegan.utils import midi_io
from musegan.utils import image_io

class Model(object):
    """Base class for models."""
    def __init__(self, sess, config, name='model'):
        self.sess = sess
        self.name = name
        self.config = config

        self.scope = None
        self.global_step = None
        self.x_ = None
        self.G = None
        self.D_real = None
        self.D_fake = None
        self.components = []
        self.metrics = None
        self.saver = None

    def init_all(self):
        """Initialize all variables in the scope."""
        print('[*] Initializing variables...')
        tf.variables_initializer(tf.global_variables(self.scope.name)).run()

    def get_adversarial_loss(self, scope_to_reuse=None):
        """Return the adversarial losses for the generator and the
        discriminator."""
        if self.config['gan']['type'] == 'gan':
            adv_loss_d = tf.losses.sigmoid_cross_entropy(
                tf.ones_like(self.D_real.tensor_out),
                self.D_real.tensor_out)
            adv_loss_g = tf.losses.sigmoid_cross_entropy(
                tf.zeros_like(self.D_fake.tensor_out),
                self.D_fake.tensor_out)

        if (self.config['gan']['type'] == 'wgan'
                or self.config['gan']['type'] == 'wgan-gp'):
            adv_loss_d = (tf.reduce_mean(self.D_fake.tensor_out)
                          - tf.reduce_mean(self.D_real.tensor_out))
            adv_loss_g = -tf.reduce_mean(self.D_fake.tensor_out)

            if self.config['gan']['type'] == 'wgan-gp':
                eps = tf.random_uniform(
                    [tf.shape(self.x_)[0], 1, 1, 1, 1], 0.0, 1.0)
                inter = eps * self.x_ + (1. - eps) * self.G.tensor_out
                if scope_to_reuse is None:
                    D_inter = Discriminator(inter, self.config, name='D',
                                            reuse=True)
                else:
                    with tf.variable_scope(scope_to_reuse, reuse=True):
                        D_inter = Discriminator(inter, self.config, name='D',
                                                reuse=True)
                gradient = tf.gradients(D_inter.tensor_out, inter)[0]
                slopes = tf.sqrt(1e-8 + tf.reduce_sum(
                    tf.square(gradient),
                    tf.range(1, len(gradient.get_shape()))))
                gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
                adv_loss_d += (self.config['gan']['gp_coefficient']
                               * gradient_penalty)

        return adv_loss_g, adv_loss_d

    def get_optimizer(self):
        """Return a Adam optimizer."""
        return tf.train.AdamOptimizer(
            self.config['optimizer']['lr'],
            self.config['optimizer']['beta1'],
            self.config['optimizer']['beta2'],
            self.config['optimizer']['epsilon'])

    def get_statistics(self):
        """Return model statistics (number of paramaters for each component)."""
        def get_num_parameter(var_list):
            """Given the variable list, return the total number of parameters.
            """
            return int(np.sum([np.product([x.value for x in var.get_shape()])
                               for var in var_list]))
        num_par = get_num_parameter(tf.trainable_variables(
            self.scope.name))
        num_par_g = get_num_parameter(self.G.vars)
        num_par_d = get_num_parameter(self.D_fake.vars)
        return ("Number of parameters: {}\nNumber of parameters in G: {}\n"
                "Number of parameters in D: {}".format(num_par, num_par_g,
                                                       num_par_d))

    def get_summary(self):
        """Return model summary."""
        return '\n'.join(
            ["{:-^80}".format(' < ' + self.scope.name + ' > ')]
            + [(x.get_summary() + '\n' + '-' * 80) for x in self.components])

    def get_global_step_str(self):
        """Return the global step as a string."""
        return str(tf.train.global_step(self.sess, self.global_step))

    def print_statistics(self):
        """Print model statistics (number of paramaters for each component)."""
        print("{:=^80}".format(' Model Statistics '))
        print(self.get_statistics())

    def print_summary(self):
        """Print model summary."""
        print("{:=^80}".format(' Model Summary '))
        print(self.get_summary())

    def save_statistics(self, filepath=None):
        """Save model statistics to file. Default to save to the log directory
        given as a global variable."""
        if filepath is None:
            filepath = os.path.join(self.config['log_dir'],
                                    'model_statistics.txt')
        with open(filepath, 'w') as f:
            f.write(self.get_statistics())

    def save_summary(self, filepath=None):
        """Save model summary to file. Default to save to the log directory
        given as a global variable."""
        if filepath is None:
            filepath = os.path.join(self.config['log_dir'], 'model_summary.txt')
        with open(filepath, 'w') as f:
            f.write(self.get_summary())

    def save(self, filepath=None):
        """Save the model to a checkpoint file. Default to save to the log
        directory given as a global variable."""
        if filepath is None:
            filepath = os.path.join(self.config['checkpoint_dir'],
                                    self.name + '.model')
        print('[*] Saving checkpoint...')
        self.saver.save(self.sess, filepath, self.global_step)

    def load(self, filepath):
        """Load the model from the latest checkpoint in a directory."""
        print('[*] Loading checkpoint...')
        self.saver.restore(self.sess, filepath)

    def load_latest(self, checkpoint_dir=None):
        """Load the model from the latest checkpoint in a directory."""
        if checkpoint_dir is None:
            checkpoint_dir = self.config['checkpoint_dir']
        print('[*] Loading checkpoint...')
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise ValueError("Checkpoint not found")
        self.saver.restore(self.sess, checkpoint_path)

    def save_samples(self, filename, samples, save_midi=False, shape=None,
                     postfix=None):
        """Save samples to an image file (and a MIDI file)."""
        if shape is None:
            shape = self.config['sample_grid']
        if len(samples) > self.config['num_sample']:
            samples = samples[:self.config['num_sample']]
        if postfix is None:
            imagepath = os.path.join(self.config['sample_dir'],
                                     '{}.png'.format(filename))
        else:
            imagepath = os.path.join(self.config['sample_dir'],
                                     '{}.png'.format(filename + postfix))
        image_io.save_image(imagepath, samples, shape)
        if save_midi:
            binarized = (samples > 0)
            midipath = os.path.join(self.config['sample_dir'],
                                    '{}.mid'.format(filename))
            midi_io.save_midi(midipath, binarized, self.config)

    def run_sampler(self, targets, feed_dict, save_midi=False, postfix=None):
        """Run the target operation with feed_dict and save the samples."""
        if not isinstance(targets, list):
            targets = [targets]
        results = self.sess.run(targets, feed_dict)
        results = [result[:self.config['num_sample']] for result in results]
        samples = np.stack(results, 1).reshape((-1,) + results[0].shape[1:])
        shape = [self.config['sample_grid'][0],
                 self.config['sample_grid'][1] * len(results)]
        if postfix is None:
            filename = self.get_global_step_str()
        else:
            filename = self.get_global_step_str() + postfix
        self.save_samples(filename, samples, save_midi, shape)

    def run_eval(self, target, feed_dict, postfix=None):
        """Run evaluation."""
        result = self.sess.run(target, feed_dict)
        binarized = (result > 0)
        if postfix is None:
            filename = self.get_global_step_str()
        else:
            filename = self.get_global_step_str() + postfix
        reshaped = binarized.reshape((-1,) + binarized.shape[2:])
        mat_path = os.path.join(self.config['eval_dir'], filename+'.npy')
        _ = self.metrics.eval(reshaped, mat_path=mat_path)

class GAN(Model):
    """Class that defines the first-stage (without refiner) model."""
    def __init__(self, sess, config, name='GAN', reuse=None):
        super().__init__(sess, config, name)

        print('[*] Building GAN...')
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.build()

    def build(self):
        """Build the model."""
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Create placeholders
        self.z = tf.placeholder(
            tf.float32,
            (self.config['batch_size'], self.config['net_g']['z_dim']), 'z'
        )
        data_shape = (self.config['batch_size'], self.config['num_bar'],
                      self.config['num_timestep'], self.config['num_pitch'],
                      self.config['num_track'])
        self.x = tf.placeholder(tf.bool, data_shape, 'x')
        self.x_ = tf.cast(self.x, tf.float32, 'x_')

        # Components
        self.G = Generator(self.z, self.config, name='G')
        self.test_round = self.G.tensor_out > 0.5
        self.test_bernoulli = self.G.tensor_out > tf.random_uniform(data_shape)

        self.D_fake = Discriminator(self.G.tensor_out, self.config, name='D')
        self.D_real = Discriminator(self.x_, self.config, name='D', reuse=True)
        self.components = (self.G, self.D_fake)

        # Losses
        self.g_loss, self.d_loss = self.get_adversarial_loss()

        # Optimizers
        with tf.variable_scope('Optimizer'):
            self.g_optimizer = self.get_optimizer()
            self.g_step = self.g_optimizer.minimize(
                self.g_loss, self.global_step, self.G.vars)

            self.d_optimizer = self.get_optimizer()
            self.d_step = self.d_optimizer.minimize(
                self.d_loss, self.global_step, self.D_fake.vars)

            # Apply weight clipping
            if self.config['gan']['type'] == 'wgan':
                with tf.control_dependencies([self.d_step]):
                    self.d_step = tf.group(
                        *(tf.assign(var, tf.clip_by_value(
                            var, -self.config['gan']['clip_value'],
                            self.config['gan']['clip_value']))
                          for var in self.D_fake.vars))

        # Metrics
        self.metrics = Metrics(self.config)

        # Saver
        self.saver = tf.train.Saver()

        # Print and save model information
        self.print_statistics()
        self.save_statistics()
        self.print_summary()
        self.save_summary()

    def train(self, x_train, train_config):
        """Train the model."""
        # Initialize sampler
        self.z_sample = np.random.normal(
            size=(self.config['batch_size'], self.config['net_g']['z_dim']))
        self.x_sample = x_train[np.random.choice(
            len(x_train), self.config['batch_size'], False)]
        feed_dict_sample = {self.x: self.x_sample, self.z: self.z_sample}

        # Save samples
        self.save_samples('x_train', x_train, save_midi=True)
        self.save_samples('x_sample', self.x_sample, save_midi=True)

        # Open log files and write headers
        log_step = open(os.path.join(self.config['log_dir'], 'step.log'), 'w')
        log_batch = open(os.path.join(self.config['log_dir'], 'batch.log'), 'w')
        log_epoch = open(os.path.join(self.config['log_dir'], 'epoch.log'), 'w')
        log_step.write('# epoch, step, negative_critic_loss\n')
        log_batch.write('# epoch, batch, time, negative_critic_loss, g_loss\n')
        log_epoch.write('# epoch, time, negative_critic_loss, g_loss\n')

        # Initialize counter
        counter = 0
        epoch_counter = 0
        num_batch = len(x_train) // self.config['batch_size']

        # Start epoch iteration
        print('{:=^80}'.format(' Training Start '))
        for epoch in range(train_config['num_epoch']):

            print('{:-^80}'.format(' Epoch {} Start '.format(epoch)))
            epoch_start_time = time.time()

            # Prepare batched training data
            z_random_batch = np.random.normal(
                size=(num_batch, self.config['batch_size'],
                      self.config['net_g']['z_dim'])
            )
            x_random_batch = np.random.choice(
                len(x_train), (num_batch, self.config['batch_size']), False
            )

            # Start batch iteration
            for batch in range(num_batch):

                feed_dict_batch = {self.x: x_train[x_random_batch[batch]],
                                   self.z: z_random_batch[batch]}

                if (counter < 25) or (counter % 500 == 0):
                    num_critics = 100
                else:
                    num_critics = 5

                batch_start_time = time.time()

                # Update networks
                for _ in range(num_critics):
                    _, d_loss = self.sess.run([self.d_step, self.d_loss],
                                              feed_dict_batch)
                    log_step.write("{}, {:14.6f}\n".format(
                        self.get_global_step_str(), -d_loss
                    ))

                _, d_loss, g_loss = self.sess.run(
                    [self.g_step, self.d_loss, self.g_loss], feed_dict_batch
                )
                log_step.write("{}, {:14.6f}\n".format(
                    self.get_global_step_str(), -d_loss
                ))

                time_batch = time.time() - batch_start_time

                # Print iteration summary
                if train_config['verbose']:
                    if batch < 1:
                        print("epoch |   batch   |  time  |    - D_loss    |"
                              "     G_loss")
                print("  {:2d}  | {:4d}/{:4d} | {:6.2f} | {:14.6f} | "
                      "{:14.6f}".format(epoch, batch, num_batch, time_batch,
                                        -d_loss, g_loss))

                log_batch.write("{:d}, {:d}, {:f}, {:f}, {:f}\n".format(
                    epoch, batch, time_batch, -d_loss, g_loss
                ))

                # run sampler
                if train_config['sample_along_training']:
                    if counter%100 == 0 or (counter < 300 and counter%20 == 0):
                        self.run_sampler(self.G.tensor_out, feed_dict_sample,
                                         False)
                        self.run_sampler(self.test_round, feed_dict_sample,
                                         (counter > 500), postfix='test_round')
                        self.run_sampler(self.test_bernoulli, feed_dict_sample,
                                         (counter > 500),
                                         postfix='test_bernoulli')

                # run evaluation
                if train_config['evaluate_along_training']:
                    if counter%10 == 0:
                        self.run_eval(self.test_round, feed_dict_sample,
                                      postfix='test_round')
                        self.run_eval(self.test_bernoulli, feed_dict_sample,
                                      postfix='test_bernoulli')

                counter += 1

            # print epoch info
            time_epoch = time.time() - epoch_start_time

            if not train_config['verbose']:
                if epoch < 1:
                    print("epoch |    time    |    - D_loss    |     G_loss")
                print("  {:2d}  | {:8.2f} | {:14.6f} | {:14.6f}".format(
                    epoch, time_epoch, -d_loss, g_loss))

            log_epoch.write("{:d}, {:f}, {:f}, {:f}\n".format(
                epoch, time_epoch, -d_loss, g_loss
            ))

            # save checkpoints
            self.save()

            epoch_counter += 1

        print('{:=^80}'.format(' Training End '))
        log_step.close()
        log_batch.close()
        log_epoch.close()

class RefineGAN(Model):
    """Class that defines the second-stage (with refiner) model."""
    def __init__(self, sess, config, pretrained, name='RefineGAN', reuse=None):
        super().__init__(sess, config, name)
        self.pretrained = pretrained

        print('[*] Building RefineGAN...')
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.build()

    def build(self):
        """Build the model."""
        # Create global step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get tensors from the pretrained model
        self.z = self.pretrained.z
        self.x = self.pretrained.x
        self.x_ = self.pretrained.x_

        # Slope tensor for applying slope annealing trick to stochastic neurons
        self.slope_tensor = tf.Variable(1.0)

        # Components
        self.G = Refiner(self.pretrained.G.tensor_out, self.config,
                         slope_tensor=self.slope_tensor, name='R')
        self.D_real = self.pretrained.D_real
        with tf.variable_scope(self.pretrained.scope, reuse=True):
            self.D_fake = Discriminator(self.G.tensor_out, self.config,
                                        name='D')
        self.components = (self.pretrained.G, self.G, self.D_fake)

        # Losses
        self.g_loss, self.d_loss = self.get_adversarial_loss(
            self.pretrained.scope)

        # Optimizers
        with tf.variable_scope('Optimizer'):
            self.g_optimizer = self.get_optimizer()
            if self.config['joint_training']:
                self.g_step = self.g_optimizer.minimize(
                    self.g_loss, self.global_step, (self.G.vars
                                                    + self.pretrained.G.vars))
            else:
                self.g_step = self.g_optimizer.minimize(
                    self.g_loss, self.global_step, self.G.vars)
            self.d_optimizer = self.get_optimizer()
            self.d_step = self.d_optimizer.minimize(
                self.d_loss, self.global_step, self.D_fake.vars)

            # Apply weight clipping
            if self.config['gan']['type'] == 'wgan':
                with tf.control_dependencies([self.d_step]):
                    self.d_step = tf.group(
                        *(tf.assign(var, tf.clip_by_value(
                            var, -self.config['gan']['clip_value'],
                            self.config['gan']['clip_value']))
                          for var in self.D_fake.vars))

        # Metrics
        self.metrics = Metrics(self.config)

        # Saver
        self.saver = tf.train.Saver()

        # Print and save model information
        self.print_statistics()
        self.save_statistics()
        self.print_summary()
        self.save_summary()

    def train(self, x_train, train_config):
        """Train the model."""
        # Initialize sampler
        self.z_sample = np.random.normal(
            size=(self.config['batch_size'], self.config['net_g']['z_dim']))
        self.x_sample = x_train[np.random.choice(
            len(x_train), self.config['batch_size'], False)]
        feed_dict_sample = {self.x: self.x_sample, self.z: self.z_sample}

        # Save samples
        self.save_samples('x_train', x_train, save_midi=True)
        self.save_samples('x_sample', self.x_sample, save_midi=True)

        pretrained_samples = self.sess.run(self.pretrained.G.tensor_out,
                                           feed_dict_sample)
        self.save_samples('pretrained', pretrained_samples)

        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pretrained_threshold = (pretrained_samples > threshold)
            self.save_samples('pretrained_threshold_{}'.format(threshold),
                              pretrained_threshold, save_midi=True)

        for idx in range(5):
            pretrained_bernoulli = np.ceil(
                pretrained_samples
                - np.random.uniform(size=pretrained_samples.shape))
            self.save_samples('pretrained_bernoulli_{}'.format(idx),
                              pretrained_bernoulli, save_midi=True)

        # Open log files and write headers
        log_step = open(os.path.join(self.config['log_dir'], 'step.log'), 'w')
        log_batch = open(os.path.join(self.config['log_dir'], 'batch.log'), 'w')
        log_epoch = open(os.path.join(self.config['log_dir'], 'epoch.log'), 'w')
        log_step.write('# epoch, step, negative_critic_loss\n')
        log_batch.write('# epoch, batch, time, negative_critic_loss, g_loss\n')
        log_epoch.write('# epoch, time, negative_critic_loss, g_loss\n')

        # Define slope annealing op
        if train_config['slope_annealing_rate'] != 1.:
            slope_annealing_op = tf.assign(
                self.slope_tensor,
                self.slope_tensor * train_config['slope_annealing_rate'])

        # Initialize counter
        counter = 0
        epoch_counter = 0
        num_batch = len(x_train) // self.config['batch_size']

        # Start epoch iteration
        print('{:=^80}'.format(' Training Start '))
        for epoch in range(train_config['num_epoch']):

            print('{:-^80}'.format(' Epoch {} Start '.format(epoch)))
            epoch_start_time = time.time()

            # Prepare batched training data
            z_random_batch = np.random.normal(
                size=(num_batch, self.config['batch_size'],
                      self.config['net_g']['z_dim'])
            )
            x_random_batch = np.random.choice(
                len(x_train), (num_batch, self.config['batch_size']), False)

            # Start batch iteration
            for batch in range(num_batch):

                feed_dict_batch = {self.x: x_train[x_random_batch[batch]],
                                   self.z: z_random_batch[batch]}

                if counter % 500 == 0: # (counter < 25)
                    num_critics = 100
                else:
                    num_critics = 5

                batch_start_time = time.time()

                # Update networks
                for _ in range(num_critics):
                    _, d_loss = self.sess.run([self.d_step, self.d_loss],
                                              feed_dict_batch)
                    log_step.write("{}, {:14.6f}\n".format(
                        self.get_global_step_str(), -d_loss
                    ))

                _, d_loss, g_loss = self.sess.run(
                    [self.g_step, self.d_loss, self.g_loss], feed_dict_batch
                )
                log_step.write("{}, {:14.6f}\n".format(
                    self.get_global_step_str(), -d_loss
                ))

                time_batch = time.time() - batch_start_time

                # Print iteration summary
                if train_config['verbose']:
                    if batch < 1:
                        print("epoch |   batch   |  time  |    - D_loss    |"
                              "     G_loss")
                print("  {:2d}  | {:4d}/{:4d} | {:6.2f} | {:14.6f} | "
                      "{:14.6f}".format(epoch, batch, num_batch, time_batch,
                                        -d_loss, g_loss))

                log_batch.write("{:d}, {:d}, {:f}, {:f}, {:f}\n".format(
                    epoch, batch, time_batch, -d_loss, g_loss
                ))

                # run sampler
                if train_config['sample_along_training']:
                    if counter%100 == 0 or (counter < 300 and counter%20 == 0):
                        self.run_sampler(self.G.tensor_out, feed_dict_sample,
                                         (counter > 500))
                        self.run_sampler(self.G.preactivated, feed_dict_sample,
                                         False, postfix='preactivated')

                # run evaluation
                if train_config['evaluate_along_training']:
                    if counter%10 == 0:
                        self.run_eval(self.G.tensor_out, feed_dict_sample)

                counter += 1

            # print epoch info
            time_epoch = time.time() - epoch_start_time

            if not train_config['verbose']:
                if epoch < 1:
                    print("epoch |    time    |    - D_loss    |     G_loss")
                print("  {:2d}  | {:8.2f} | {:14.6f} | {:14.6f}".format(
                    epoch, time_epoch, -d_loss, g_loss))

            log_epoch.write("{:d}, {:f}, {:f}, {:f}\n".format(
                epoch, time_epoch, -d_loss, g_loss
            ))

            # save checkpoints
            self.save()

            if train_config['slope_annealing_rate'] != 1.:
                self.sess.run(slope_annealing_op)

            epoch_counter += 1

        print('{:=^80}'.format(' Training End '))
        log_step.close()
        log_batch.close()
        log_epoch.close()

class End2EndGAN(Model):
    """Class that defines the end-to-end model."""
    def __init__(self, sess, config, name='End2EndGAN', reuse=None):
        super().__init__(sess, config, name)

        print('[*] Building End2EndGAN...')
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.build()

    def build(self):
        """Build the model."""
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Create placeholders
        self.z = tf.placeholder(
            tf.float32,
            (self.config['batch_size'], self.config['net_g']['z_dim']), 'z'
        )
        data_shape = (self.config['batch_size'], self.config['num_bar'],
                      self.config['num_timestep'], self.config['num_pitch'],
                      self.config['num_track'])
        self.x = tf.placeholder(tf.bool, data_shape, 'x')
        self.x_ = tf.cast(self.x, tf.float32, 'x_')

        # Slope tensor for applying slope annealing trick to stochastic neurons
        self.slope_tensor = tf.Variable(1.0)

        # Components
        self.G = End2EndGenerator(self.z, self.config,
                                  slope_tensor=self.slope_tensor, name='G')
        self.D_fake = Discriminator(self.G.tensor_out, self.config, name='D')
        self.D_real = Discriminator(self.x_, self.config, name='D', reuse=True)
        self.components = (self.G, self.D_fake)

        # Losses
        self.g_loss, self.d_loss = self.get_adversarial_loss()

        # Optimizers
        with tf.variable_scope('Optimizer'):
            self.g_optimizer = self.get_optimizer()
            self.g_step = self.g_optimizer.minimize(
                self.g_loss, self.global_step, self.G.vars)

            self.d_optimizer = self.get_optimizer()
            self.d_step = self.d_optimizer.minimize(
                self.d_loss, self.global_step, self.D_fake.vars)

            # Apply weight clipping
            if self.config['gan']['type'] == 'wgan':
                with tf.control_dependencies([self.d_step]):
                    self.d_step = tf.group(
                        *(tf.assign(var, tf.clip_by_value(
                            var, -self.config['gan']['clip_value'],
                            self.config['gan']['clip_value']))
                          for var in self.D_fake.vars))

        # Metrics
        self.metrics = Metrics(self.config)

        # Saver
        self.saver = tf.train.Saver()

        # Print and save model information
        self.print_statistics()
        self.save_statistics()
        self.print_summary()
        self.save_summary()

    def train(self, x_train, train_config):
        """Train the model."""
        # Initialize sampler
        self.z_sample = np.random.normal(
            size=(self.config['batch_size'], self.config['net_g']['z_dim']))
        self.x_sample = x_train[np.random.choice(
            len(x_train), self.config['batch_size'], False)]
        feed_dict_sample = {self.x: self.x_sample, self.z: self.z_sample}

        # Save samples
        self.save_samples('x_train', x_train, save_midi=True)
        self.save_samples('x_sample', self.x_sample, save_midi=True)

        # Open log files and write headers
        log_step = open(os.path.join(self.config['log_dir'], 'step.log'), 'w')
        log_batch = open(os.path.join(self.config['log_dir'], 'batch.log'), 'w')
        log_epoch = open(os.path.join(self.config['log_dir'], 'epoch.log'), 'w')
        log_step.write('# epoch, step, negative_critic_loss\n')
        log_batch.write('# epoch, batch, time, negative_critic_loss, g_loss\n')
        log_epoch.write('# epoch, time, negative_critic_loss, g_loss\n')

        # Define slope annealing op
        if train_config['slope_annealing_rate'] != 1.:
            slope_annealing_op = tf.assign(
                self.slope_tensor,
                self.slope_tensor * train_config['slope_annealing_rate'])

        # Initialize counter
        counter = 0
        epoch_counter = 0
        num_batch = len(x_train) // self.config['batch_size']

        # Start epoch iteration
        print('{:=^80}'.format(' Training Start '))
        for epoch in range(train_config['num_epoch']):

            print('{:-^80}'.format(' Epoch {} Start '.format(epoch)))
            epoch_start_time = time.time()

            # Prepare batched training data
            z_random_batch = np.random.normal(
                size=(num_batch, self.config['batch_size'],
                      self.config['net_g']['z_dim'])
            )
            x_random_batch = np.random.choice(
                len(x_train), (num_batch, self.config['batch_size']), False)

            # Start batch iteration
            for batch in range(num_batch):

                feed_dict_batch = {self.x: x_train[x_random_batch[batch]],
                                   self.z: z_random_batch[batch]}

                if (counter < 25) or (counter % 500 == 0):
                    num_critics = 100
                else:
                    num_critics = 5

                batch_start_time = time.time()

                # Update networks
                for _ in range(num_critics):
                    _, d_loss = self.sess.run([self.d_step, self.d_loss],
                                              feed_dict_batch)
                    log_step.write("{}, {:14.6f}\n".format(
                        self.get_global_step_str(), -d_loss
                    ))

                _, d_loss, g_loss = self.sess.run(
                    [self.g_step, self.d_loss, self.g_loss], feed_dict_batch
                )
                log_step.write("{}, {:14.6f}\n".format(
                    self.get_global_step_str(), -d_loss
                ))

                time_batch = time.time() - batch_start_time

                # Print iteration summary
                if train_config['verbose']:
                    if batch < 1:
                        print("epoch |   batch   |  time  |    - D_loss    |"
                              "     G_loss")
                print("  {:2d}  | {:4d}/{:4d} | {:6.2f} | {:14.6f} | "
                      "{:14.6f}".format(epoch, batch, num_batch, time_batch,
                                        -d_loss, g_loss))

                log_batch.write("{:d}, {:d}, {:f}, {:f}, {:f}\n".format(
                    epoch, batch, time_batch, -d_loss, g_loss
                ))

                # run sampler
                if train_config['sample_along_training']:
                    if counter%100 == 0 or (counter < 300 and counter%20 == 0):
                        self.run_sampler(self.G.tensor_out, feed_dict_sample,
                                         (counter > 500))
                        self.run_sampler(self.G.preactivated, feed_dict_sample,
                                         False, postfix='preactivated')

                # run evaluation
                if train_config['evaluate_along_training']:
                    if counter%10 == 0:
                        self.run_eval(self.G.tensor_out, feed_dict_sample)

                counter += 1

            # print epoch info
            time_epoch = time.time() - epoch_start_time

            if not train_config['verbose']:
                if epoch < 1:
                    print("epoch |    time    |    - D_loss    |     G_loss")
                print("  {:2d}  | {:8.2f} | {:14.6f} | {:14.6f}".format(
                    epoch, time_epoch, -d_loss, g_loss))

            log_epoch.write("{:d}, {:f}, {:f}, {:f}\n".format(
                epoch, time_epoch, -d_loss, g_loss
            ))

            # save checkpoints
            self.save()

            if train_config['slope_annealing_rate'] != 1.:
                self.sess.run(slope_annealing_op)

            epoch_counter += 1

        print('{:=^80}'.format(' Training End '))
        log_step.close()
        log_batch.close()
        log_epoch.close()
