"""Main function
"""
import numpy as np
import tensorflow as tf
import SharedArray as sa
from musegan.bmusegan.models import GAN, RefineGAN, End2EndGAN
from config import EXP_CONFIG, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from config import TF_CONFIG

def load_data():
    """Load and return the training data."""
    print('[*] Loading data...')
    if DATA_CONFIG['training_data_location'] == 'sa':
        x_train = sa.attach(DATA_CONFIG['training_data'])
    elif DATA_CONFIG['training_data_location'] == 'hd':
        x_train = np.load(DATA_CONFIG['training_data'])
    x_train = x_train.reshape(
        -1, MODEL_CONFIG['num_bar'], MODEL_CONFIG['num_timestep'],
        MODEL_CONFIG['num_pitch'], MODEL_CONFIG['num_track']
    )
    print('Training set size:', len(x_train))
    return x_train

def pretrain():
    """Create and pretrain a two-stage model"""
    x_train = load_data()
    with tf.Session(config=TF_CONFIG) as sess:
        gan = GAN(sess, MODEL_CONFIG)
        gan.init_all()
        if EXP_CONFIG['pretrained_dir'] is not None:
            gan.load_latest(EXP_CONFIG['pretrained_dir'])
        gan.train(x_train, TRAIN_CONFIG)

def train():
    """Load the pretrained model and run the second-stage training"""
    x_train = load_data()
    with tf.Session(config=TF_CONFIG) as sess:
        gan = GAN(sess, MODEL_CONFIG)
        gan.init_all()
        gan.load_latest(EXP_CONFIG['first_stage_dir'])

        refine_gan = RefineGAN(sess, MODEL_CONFIG, gan)
        refine_gan.init_all()
        if EXP_CONFIG['pretrained_dir'] is not None:
            refine_gan.load_latest(EXP_CONFIG['pretrained_dir'])
        refine_gan.train(x_train, TRAIN_CONFIG)

def train_end2end():
    """Create and train an end-to-end model"""
    x_train = load_data()
    with tf.Session(config=TF_CONFIG) as sess:
        end2end_gan = End2EndGAN(sess, MODEL_CONFIG)
        end2end_gan.init_all()
        if EXP_CONFIG['pretrained_dir'] is not None:
            end2end_gan.load_latest(EXP_CONFIG['pretrained_dir'])
        end2end_gan.train(x_train, TRAIN_CONFIG)

if __name__ == '__main__':
    print("Start experiment: {}".format(EXP_CONFIG['exp_name']))
    if not EXP_CONFIG['two_stage_training']:
        train_end2end()
    elif TRAIN_CONFIG['training_phase'] == 'pretrain':
        pretrain()
    elif TRAIN_CONFIG['training_phase'] == 'train':
        train()
