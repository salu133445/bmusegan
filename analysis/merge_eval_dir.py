import os
import joblib
import numpy as np

# EVAL_DIRS = [
#     # ]

# FILEPATHS = [
#
# ]

EVAL_DIRS = [
    # "/home/salu133445/NAS/salu133445/git/ismir2018/exp/train_False_alternative_proposed_resnet_preactivation_round_3x12/eval/",
    # "/home/salu133445/NAS/salu133445/git/ismir2018/exp/train_False_alternative_proposed_resnet_preactivation_bernoulli_3x12/eval/",

    # "/home/salu133445/NAS/salu133445/git/ismir2018/exp/train_True_alternative_proposed_preactivation_round_3x12/eval/",
    # "/home/salu133445/NAS/salu133445/git/ismir2018/exp/train_True_alternative_proposed_preactivation_bernoulli_3x12/eval/",

    # ("/home/salu133445/NAS/salu133445/git/ismir2018/exp/train_False_alternative_proposed_resnet_preactivation_round_3x12/"
    #  "keep_train/exp/train_False_alternative_proposed_resnet_preactivation_round_3x12/eval/"),
    # ("/home/salu133445/NAS/salu133445/git/ismir2018/exp/train_False_alternative_proposed_resnet_preactivation_bernoulli_3x12/"
    #  "keep_train/exp/train_False_alternative_proposed_resnet_preactivation_bernoulli_3x12/eval/"),

    # "/home/salu133445/NAS/salu133445/git/ismir2018/exp/end2end_alternative_proposed_smaller_resnet_preactivation_round/eval/",
    # "/home/salu133445/NAS/salu133445/git/ismir2018/exp/end2end_alternative_proposed_smaller_resnet_preactivation_bernoulli/eval/",

    ("/home/salu133445/NAS/salu133445/git/ismir2018/exp/end2end_alternative_proposed_smaller_resnet_preactivation_round/keep_train/"
     "exp/end2end_alternative_proposed_smaller_resnet_preactivation_round/eval/"),
    ("/home/salu133445/NAS/salu133445/git/ismir2018/exp/end2end_alternative_proposed_smaller_resnet_preactivation_bernoulli/keep_train/"
     "exp/end2end_alternative_proposed_smaller_resnet_preactivation_bernoulli/eval/"),
]

FILEPATHS = [
    # "./data/train_alternative_proposed_round.npz",
    # "./data/train_alternative_proposed_bernoulli.npz",

    # "./data/train_alternative_proposed_round_joint.npz",
    # "./data/train_alternative_proposed_bernoulli_joint.npz",

    # "./data/train_alternative_proposed_round_keep_train.npz",
    # "./data/train_alternative_proposed_bernoulli_keep_train.npz",

    # "./data/end2end_alternative_proposed_smaller_round.npz",
    # "./data/end2end_alternative_proposed_smaller_bernoulli.npz",

    "./data/end2end_alternative_proposed_smaller_round_keep_train.npz",
    "./data/end2end_alternative_proposed_smaller_bernoulli_keep_train.npz",
]


def get_npy_files(target_dir):
    filepaths = []
    for path in os.listdir(target_dir):
        if path.endswith('.npy'):
            filepaths.append(path)
    return filepaths

def load(filepath, eval_dir):
    step = int(os.path.splitext(filepath)[0])
    data = np.load(os.path.join(eval_dir, filepath))
    return (step, data[()]['score_matrix_mean'],
            data[()]['score_pair_matrix_mean'])

def main():
    """Main function
    """
    for idx, eval_dir in enumerate(EVAL_DIRS):
        filepaths = get_npy_files(eval_dir)
        collected = joblib.Parallel(n_jobs=30, verbose=5)(
            joblib.delayed(load)(filepath, eval_dir) for filepath in filepaths)

        steps = []
        score_matrix_means = []
        score_pair_matrix_means = []

        for x in collected:
            steps.append(x[0])
            score_matrix_means.append(x[1])
            score_pair_matrix_means.append(x[2])

        steps = np.array(steps)
        score_matrix_means = np.stack(score_matrix_means)
        score_pair_matrix_means = np.stack(score_pair_matrix_means)

        argsort = steps.argsort()
        steps = steps[argsort]
        score_matrix_means = score_matrix_means[argsort]
        score_pair_matrix_means = score_pair_matrix_means[argsort]

        np.savez(FILEPATHS[idx], steps=steps,
                 score_matrix_means=score_matrix_means,
                 score_pair_matrix_means=score_pair_matrix_means)

if __name__ == "__main__":
    main()
