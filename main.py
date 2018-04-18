"""Main function
"""
import tensorflow as tf
from bmusegan.models import GAN, RefineGAN, End2EndGAN
from config import TF_CONFIG, EXP_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

def pretrain():
    """Create and pretrain a two-stage model"""
    with tf.Session(config=TF_CONFIG) as sess:
        gan = GAN(sess, MODEL_CONFIG)
        gan.init_all()
        if EXP_CONFIG['pretrained_dir'] is not None:
            gan.load_latest(EXP_CONFIG['pretrained_dir'])
        gan.train(TRAIN_CONFIG)

def train():
    """Load the pretrained model and run the second-stage training"""
    with tf.Session(config=TF_CONFIG) as sess:
        gan = GAN(sess, MODEL_CONFIG)
        gan.init_all()
        gan.load_latest(EXP_CONFIG['first_stage_dir'])

        refine_gan = RefineGAN(sess, MODEL_CONFIG, gan)
        refine_gan.init_all()
        if EXP_CONFIG['pretrained_dir'] is not None:
            refine_gan.load_latest(EXP_CONFIG['pretrained_dir'])
        refine_gan.train(TRAIN_CONFIG)

def train_end2end():
    """Create and train an end-to-end model"""
    with tf.Session(config=TF_CONFIG) as sess:
        end2end_gan = End2EndGAN(sess, MODEL_CONFIG)
        end2end_gan.init_all()
        if EXP_CONFIG['pretrained_dir'] is not None:
            end2end_gan.load_latest(EXP_CONFIG['pretrained_dir'])
        end2end_gan.train(TRAIN_CONFIG)

if __name__ == '__main__':
    print("Start experiment: {}".format(EXP_CONFIG['exp_name']))
    if not EXP_CONFIG['two_stage_training']:
        train_end2end()
    elif TRAIN_CONFIG['phase'] == 'pretrain':
        pretrain()
    elif TRAIN_CONFIG['phase'] == 'train':
        train()
