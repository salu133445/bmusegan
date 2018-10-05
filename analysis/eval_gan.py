import os.path
import numpy as np
import tensorflow as tf
import SharedArray as sa
from musegan2.models import GAN, RefineGAN, End2EndGAN
from config import TF_CONFIG, EXP_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

NUM_BATCH = 50

def main():
    with tf.Session(config=TF_CONFIG) as sess:
        gan = GAN(sess, MODEL_CONFIG)
        gan.init_all()
        gan.load_latest('../checkpoints/')

        # Prepare feed dictionaries
        print("[*] Preparing data...")
        z_sample = np.random.normal(size=(NUM_BATCH, gan.config['batch_size'],
                                          gan.config['z_dim']))
        feed_dict_sample = [{gan.z: z_sample[i]} for i in range(NUM_BATCH)]

        # Run sampler
        print("[*] Running sampler...")
        results = np.array([
            sess.run(gan.G.tensor_out, feed_dict_sample[i])
            for i in range(NUM_BATCH)
        ])
        reshaped = results.reshape((-1,) + results.shape[3:])
        results_round = (reshaped > 0.5)
        results_bernoulli = np.ceil(
            reshaped - np.random.uniform(size=reshaped.shape)
        )

        # Run evaluation
        print("[*] Running evaluation...")
        mat_path = os.path.join(gan.config['eval_dir'], 'round.npy')
        _ = gan.metrics.eval(results_round, mat_path=mat_path)
        mat_path = os.path.join(gan.config['eval_dir'], 'bernoulli.npy')
        _ = gan.metrics.eval(results_bernoulli, mat_path=mat_path)

if __name__ == "__main__":
    main()
