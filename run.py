import tensorflow as tf
from mamltf2 import MAML, FirstOrderMAML, Reptile, IMAML, PretrainedModel, SinusoidRegressionTaskDistribution
import argparse


def main(name, method, batches, tasks, samples, epochs, larger_network, save_preliminary):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(40, activation='relu', input_shape=(1,))
    ] + ([
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dense(80, activation='relu')
    ] if larger_network else []) + [
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    taskDistribution = SinusoidRegressionTaskDistribution()

    if method == 'maml':
        metaModel = MAML(model, taskDistribution)
    elif method =='fomaml':
        metaModel = FirstOrderMAML(model, taskDistribution)
    elif method == 'reptile':
        metaModel = Reptile(model, taskDistribution, outerLearningRate = 0.1, innerLearningRate = 0.02)
    elif method == 'imaml':
        metaModel = IMAML(model, taskDistribution, outerLearningRate = 0.1, innerLearningRate = 0.001)
    elif method == 'pretrained':
        metaModel = PretrainedModel(model, taskDistribution)
    else:
        raise ValueError('Method %s is not supported' % method)

    for epoch in range(epochs):
        meanBatchLoss = metaModel.trainBatch(samples, tasks, batches)

        print(
            "Meta loss @ epoch %d, batch size %d: %.4f"
            % (epoch, batches, float(meanBatchLoss))
        )

        # Last log is the final model so just drop the epoch
        if epoch < epochs - 1:
            if save_preliminary:
                metaModel.saveKeras('./models/' + name + ('_epoch_' + str(epoch)))
        else:
            metaModel.saveKeras('./models/' + name)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="""Tensorflow 2 implementation of Finn et al. 2017 (MAML) run file.
    Trained models can be loaded and visualized in visualize.py.""")

    parser.add_argument('name', metavar='name', type=str,
                        help='You can later find the model under models/name/...')

    parser.add_argument('method', type=str, default="maml",
                        help="""Name of the method you want to apply. Supported are 'pretrained' and 'maml'.""")

    parser.add_argument('--batches', default=1000, type=int,
                        help='Size of one batch/epoch')

    parser.add_argument('--tasks', default=25, type=int,
                        help='Number of training tasks per iteration')

    parser.add_argument('--samples', default=10, type=int,
                        help='Number of samples per task per iteration')

    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of epochs')

    parser.add_argument('--larger-network', default=False, type=bool,
                        help='If set to true, a larger network will be used (two dense layers with size 80 each).')

    parser.add_argument('--save-preliminary', default=False, type=bool,
                        help='If set to true, preliminary models after each epoch are saved too.')

    args=parser.parse_args()
    main(**vars(args))
