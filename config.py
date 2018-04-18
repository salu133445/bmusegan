"""Define configuration variables in experiment, model and training levels.

Quick Setup
===========
Change the values in the dictionary `SETUP` for a quick setup.
Documentation is provided right after each key.

Configuration
=============
More configuration options are provided. Three dictionaries `EXP_CONFIG`,
`MODEL_CONFIG` and `TRAIN_CONFIG` define configuration variables in
experiment, model and training levels, respectively. The automatically-
determined experiment name is based only on the values defined in the
dictionary `SETUP`, so remember to provide the experiment name manually.
"""
import os
import shutil
import distutils.dir_util
import numpy as np
import tensorflow as tf

# Quick setup
SETUP = {
    'exp_name': None,
    # The experiment name. Also the name of the folder that will be created
    # in './exp/' and all the experiment-related files are saved in that
    # folder. None to determine automatically. The automatically-
    # determined experiment name is based only on the values defined in the
    # dictionary `SETUP`, so remember to provide the experiment name manually.

    'train_x': 'herman_alternative_13b_phrase_lastfm_',
    # Filename of the training data for `sa.attach()`. The training data
    # must be saved to the shared memory using Shared Array package first.
    # Note that some default values in `MODEL_CONFIG` assume using LPD-13,
    # so remember to modify them if not using LPD-13.

    'gpu': '0',
    # The GPU index in os.environ['CUDA_VISIBLE_DEVICES'] to use.

    'prefix': 'alternative',
    # Prefix for the experiment name. Useful when training with different
    # training data to avoid replacing the previous experiment outputs.

    'mode': 'phrase',
    # {'bar', 'phrase', None}
    # Use the two common modes which come with several presets and
    # pretrained models or set to None and setup `MODEL_CONFIG['num_bar']`
    # to define the number of bars to output.

    'two_stage_training': True,
    # True to train the model in a two-stage training setting. False to
    # train the model in an end-to-end manner.

    'training_phase': 'train',
    # {'train', 'pretrain'}
    # The training phase in a two-stage training setting. Only effective
    # when `two_stage_training` is True.

    'joint_training': False,
    # True to train the generator and the refiner jointly. Only effective
    # when `two_stage_training` is True and `training_phase` is 'train'.

    'pretrained_dir': None,
    # The directory containing the pretrained model. None to retrain the
    # model from scratch.

    'first_stage_dir': None,
    # The directory containing the pretrained first-stage model. None to
    # determine automatically (assuming default `exp_name`). Only effective
    # when `two_stage_training` is True and `training_phase` is 'train'.

    'preset_g': 'proposed',
    # {'proposed', 'proposed_smaller', None}
    # Use a preset network architecture for the generator or set to None and
    # setup `MODEL_CONFIG['net_g']` to define the network architecture.

    'preset_d': 'proposed',
    # {'proposed', 'proposed_smaller', 'ablated', 'baseline', None}
    # Use a preset network architecture for the discriminator or set to None
    # and setup `MODEL_CONFIG['net_d']` to define the network architecture.

    'preset_r': 'preactivation_round_3x12'
    # {'round', 'bernoulli', 'round_3x12', 'bernoulli_3x12',
    #  'preactivation_round', 'preactivation_bernoulli',
    #  'preactivation_round_3x12', 'preactivation_bernoulli_3x12'}
    # Use a preset network architecture for the refiner or set to None and
    # setup `MODEL_CONFIG['net_r']` to define the network architecture.
}

#===============================================================================
#=========================== TensorFlow Configuration ==========================
#===============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = SETUP['gpu']
TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.allow_growth = True

#===============================================================================
#======================= Experiment-level Configuration ========================
#===============================================================================
EXP_CONFIG = {
    'exp_name': None,
    'pretrained_dir': None,
    'first_stage_dir': None
}

if EXP_CONFIG['exp_name'] is None:
    if SETUP['exp_name'] is not None:
        EXP_CONFIG['exp_name'] = SETUP['exp_name']
    elif not SETUP['two_stage_training']:
        EXP_CONFIG['exp_name'] = '_'.join(
            ('end2end', SETUP['prefix'], SETUP['preset_d'], SETUP['preset_r'])
        )
    elif SETUP['training_phase'] == 'pretrain':
        EXP_CONFIG['exp_name'] = '_'.join(
            (SETUP['phase'], SETUP['prefix'], SETUP['preset_d'])
        )
    elif SETUP['training_phase'] == 'train':
        if SETUP['joint_training']:
            EXP_CONFIG['exp_name'] = '_'.join(
                (SETUP['training_phase'], 'jointly_training',
                 SETUP['prefix'], SETUP['preset_d'], SETUP['preset_r'])
            )
        else:
            EXP_CONFIG['exp_name'] = '_'.join(
                (SETUP['training_phase'], SETUP['prefix'], SETUP['preset_d'],
                 SETUP['preset_r'])
            )

if EXP_CONFIG['two_stage_training'] is None:
    EXP_CONFIG['two_stage_training'] = SETUP['two_stage_training']
if EXP_CONFIG['pretrained_dir'] is None:
    EXP_CONFIG['pretrained_dir'] = SETUP['pretrained_dir']
if EXP_CONFIG['first_stage_dir'] is None:
    if SETUP['first_stage_dir'] is not None:
        EXP_CONFIG['first_stage_dir'] = SETUP['first_stage_dir']
    else:
        EXP_CONFIG['first_stage_dir'] = "./exp/pretrain_{}/checkpoints".format(
            '_'.join((SETUP['prefix'], SETUP['preset_d']))
        )

#===============================================================================
#======================== Training-level Configuration =========================
#===============================================================================
TRAIN_CONFIG = {
    'two_stage_training': None,
    'training_phase': None,
    'train_x': None,
    'num_epoch': 20,
    'slope_annealing_rate': 1.1,
    'verbose': True
}

if TRAIN_CONFIG['training_phase'] is None:
    TRAIN_CONFIG['training_phase'] = SETUP['training_phase']

if TRAIN_CONFIG['train_x'] is None:
    TRAIN_CONFIG['train_x'] = SETUP['train_x']

#===============================================================================
#========================= Model-level Configuration ===========================
#===============================================================================
MODEL_CONFIG = {
    # Models
    'joint_training': None,

    # Parameters
    'batch_size': 16, # Note: tf.layers.conv3d_transpose requires a fixed batch
                      # size in TensorFlow < 1.6
    'z_dim': 128,
    'gan': {
        'type': 'wgan-gp', # 'gan', 'wgan'
        'clip_value': .01,
        'gp_coefficient': 10.
    },
    'optimizer': {
        # Parameters for Adam optimizers
        'lr': .002,
        'beta1': .5,
        'beta2': .9,
        'epsilon': 1e-8
    },

    # Data
    'num_bar': None,
    'num_beat': 4,
    'num_pitch': 84,
    'num_track': 8,
    'num_timestep': 96,
    'beat_resolution': 24,
    'lowest_pitch': 24, # MIDI note number of the lowest pitch in data tensors

    # Tracks
    'track_names': (
        'Drums', 'Piano', 'Guitar', 'Bass', 'Ensemble', 'Reed', 'Synth Lead',
        'Synth Pad'
    ),
    'programs': (0, 0, 24, 32, 48, 64, 80, 88),
    'is_drums': (True, False, False, False, False, False, False, False),

    # Network architectures (define them here if not using the presets)
    'net_g': None,
    'net_d': None,
    'net_r': None,

    # Playback
    'pause_between_samples': 96,
    'tempo': 90.,

    # Samples
    'num_sample': 16,
    'sample_grid': (2, 8),

    # Metrics
    'metric_map': np.array([
        # indices of tracks for the metrics to compute
        [True] * 8, # empty bar rate
        [True] * 8, # number of pitch used
        [False] + [True] * 7, # qualified note rate
        [False] + [True] * 7, # polyphonicity
        [False] + [True] * 7, # in scale rate
        [True] + [False] * 7, # in drum pattern rate
        [False] + [True] * 7  # number of chroma used
    ], dtype=bool),
    'tonal_distance_pairs': [(1, 2)], # pairs to compute the tonal distance
    'scale_mask': list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])),
    'drum_filter': np.tile([1., .1, 0., 0., 0., .1], 16),
    'tonal_matrix_coefficient': (1., 1., .5),

    # Directories
    'checkpoint_dir': os.path.join('exp', EXP_CONFIG['exp_name'],
                                   'checkpoints'),
    'sample_dir': os.path.join('exp', EXP_CONFIG['exp_name'], 'samples'),
    'eval_dir': os.path.join('exp', EXP_CONFIG['exp_name'], 'eval'),
    'log_dir': os.path.join('exp', EXP_CONFIG['exp_name'], 'logs'),
    'src_dir': os.path.join('exp', EXP_CONFIG['exp_name'], 'src')
}

if MODEL_CONFIG['num_bar'] is None:
    if SETUP['mode'] == 'bar':
        MODEL_CONFIG['num_bar'] = 1
    elif SETUP['mode'] == 'phrase':
        MODEL_CONFIG['num_bar'] =  4
if MODEL_CONFIG['joint_training'] is None:
    MODEL_CONFIG['joint_training'] = SETUP['joint_training']

# Import preset network architectures
if MODEL_CONFIG['net_g'] is None:
    if SETUP['mode'] == 'bar':
        if SETUP['preset_g'] == 'proposed':
            from musegan2.presets.bar.generator.proposed import NET_G
    elif SETUP['mode'] == 'phrase':
        if SETUP['preset_g'] == 'proposed':
            from musegan2.presets.phrase.generator.proposed import NET_G
        elif SETUP['preset_g'] == 'proposed_smaller':
            from musegan2.presets.phrase.generator.proposed_smaller import NET_G
    MODEL_CONFIG['net_g'] = NET_G

if MODEL_CONFIG['net_d'] is None:
    if SETUP['mode'] == 'bar':
        if SETUP['preset_d'] == 'proposed':
            from musegan2.presets.bar.discriminator.proposed import NET_D
        elif SETUP['preset_d'] == 'ablated':
            from musegan2.presets.bar.discriminator.ablated import NET_D
        elif SETUP['preset_d'] == 'baseline':
            from musegan2.presets.bar.discriminator.baseline import NET_D
    elif SETUP['mode'] == 'phrase':
        if SETUP['preset_d'] == 'proposed':
            from musegan2.presets.phrase.discriminator.proposed import NET_D
        elif SETUP['preset_d'] == 'proposed_smaller':
            from musegan2.presets.phrase.discriminator.proposed_smaller \
                import NET_D
        elif SETUP['preset_d'] == 'ablated':
            from musegan2.presets.phrase.discriminator.ablated import NET_D
        elif SETUP['preset_d'] == 'baseline':
            from musegan2.presets.phrase.discriminator.baseline import NET_D
    MODEL_CONFIG['net_d'] = NET_D

if MODEL_CONFIG['net_r'] is None:
    if SETUP['mode'] == 'bar':
        if SETUP['preset_r'] == 'resnet_round':
            from musegan2.presets.bar.refiner.round import NET_R
        elif SETUP['preset_r'] == 'resnet_bernoulli':
            from musegan2.presets.bar.refiner.bernoulli import NET_R
    elif SETUP['mode'] == 'phrase':
        if SETUP['preset_r'] == 'round':
            from musegan2.presets.phrase.refiner.round import NET_R
        elif SETUP['preset_r'] == 'bernoulli':
            from musegan2.presets.phrase.refiner.bernoulli import NET_R
        elif SETUP['preset_r'] == 'preactivation_round':
            from musegan2.presets.phrase.refiner.preactivation_round \
                import NET_R
        elif SETUP['preset_r'] == 'preactivation_bernoulli':
            from musegan2.presets.phrase.refiner.preactivation_bernoulli \
                import NET_R
        elif SETUP['preset_r'] == 'round_3x12':
            from musegan2.presets.phrase.refiner.round_3x12 import NET_R
        elif SETUP['preset_r'] == 'bernoulli_3x12':
            from musegan2.presets.phrase.refiner.bernoulli_3x12 import NET_R
        elif SETUP['preset_r'] == 'preactivation_round_3x12':
            from musegan2.presets.phrase.refiner.preactivation_round_3x12 \
                import NET_R
        elif SETUP['preset_r'] == 'preactivation_bernoulli_3x12':
            from musegan2.presets.phrase.refiner.preactivation_bernoulli_3x12 \
            import NET_R
    MODEL_CONFIG['net_r'] = NET_R

#===============================================================================
#=================== Make directories & Backup source code =====================
#===============================================================================
# Make sure directories exist
for path in (MODEL_CONFIG['checkpoint_dir'], MODEL_CONFIG['sample_dir'],
             MODEL_CONFIG['eval_dir'], MODEL_CONFIG['log_dir'],
             MODEL_CONFIG['src_dir']):
    if not os.path.exists(path):
        os.makedirs(path)

# Backup source code
for path in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if os.path.isfile(path):
        if path.endswith('.py'):
            shutil.copyfile(
                os.path.basename(path),
                os.path.join(MODEL_CONFIG['src_dir'], os.path.basename(path))
            )

distutils.dir_util.copy_tree(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'musegan2'),
    os.path.join(MODEL_CONFIG['src_dir'], 'musegan2')
)
