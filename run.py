import tensorflow as tf
import tensorflowjs as tfjs

from tasks import SinusoidRegressionTaskDistribution
from maml import RegressionMAML
import argparse


def main(name, nBatch, nTasks, nSamples, nEpochs, useLargerNetwork):
    metaModel = tf.keras.models.Sequential([
        tf.keras.layers.Dense(40, activation='relu', input_shape=(1,))
    ] + ([
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dense(80, activation='relu')
    ] if useLargerNetwork else []) + [
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    taskDistribution = SinusoidRegressionTaskDistribution()
    maml = RegressionMAML(metaModel, taskDistribution)

    for epoch in range(nEpochs):
        meanBatchLoss = maml.trainBatch(nSamples, nTasks, nBatch)

        print(
            "Meta loss @ epoch %d, batch size %d: %.4f"
            % (epoch, nBatch, float(meanBatchLoss))
        )

        # Last log is the final model so just drop the epoch
        tfjs.converters.save_keras_model(
            metaModel, './models/' + name + ('_epoch_' + str(epoch) if epoch < nEpochs - 1 else ''))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Tensorflow 2 implementation of Finn et al. 2017 (MAML) run file. 
    Trained models can be loaded and visualized in visualize.py.""")
    parser.add_argument('name', metavar='name', type=str, nargs=1,
                        help='You can later find the model under models/name/...')

    parser.add_argument('--batch-size', default=1000, type=int,
                        help='Size of one batch/epoch')

    parser.add_argument('--training-tasks', default=25, type=int,
                        help='Number of training tasks per iteration')

    parser.add_argument('--task-sample-size', default=10, type=int,
                        help='Number of samples per task per iteration')

    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of epochs')

    parser.add_argument('--use-larger-network', default=False, type=bool,
                        help='If set to true, a larger network will be used (two dense layers with size 80 each).')

    args = parser.parse_args()
    main(args.name[0], args.batch_size, args.training_tasks,
         args.task_sample_size, args.epochs, args.use_larger_network)