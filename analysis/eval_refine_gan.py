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

        refine_gan = RefineGAN(sess, MODEL_CONFIG, gan)
        refine_gan.init_all()
        refine_gan.load_latest('../checkpoints')

        # Prepare feed dictionaries
        print("[*] Preparing data...")
        z_sample = np.random.normal(size=(NUM_BATCH,
                                          refine_gan.config['batch_size'],
                                          refine_gan.config['z_dim']))
        feed_dict_sample = [{refine_gan.z: z_sample[i]}
                            for i in range(NUM_BATCH)]

        # Run sampler
        print("[*] Running sampler...")
        results = np.array([
            sess.run(refine_gan.G.tensor_out, feed_dict_sample[i])
            for i in range(NUM_BATCH)
        ])
        reshaped = results.reshape((-1,) + results.shape[3:])

        # Run evaluation
        print("[*] Running evaluation...")
        mat_path = os.path.join(refine_gan.config['eval_dir'], 'test.npy')
        _ = refine_gan.metrics.eval(reshaped, mat_path=mat_path)

if __name__ == "__main__":
    main()
