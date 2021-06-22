import tensorflow as tf
from mamltf2 import RegressionMAML, RegressionFirstOrderMAML, RegressionReptile, PretrainedModel, SinusoidRegressionTaskDistribution
import argparse


def main(name, method, nBatch, nTasks, nSamples, nEpochs, useLargerNetwork, savePreliminary):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(40, activation='relu', input_shape=(1,))
    ] + ([
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dense(80, activation='relu')
    ] if useLargerNetwork else []) + [
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    taskDistribution = SinusoidRegressionTaskDistribution()

    if method == 'maml':
        metaModel = RegressionMAML(model, taskDistribution)
    elif method =='fomaml':
        metaModel = RegressionFirstOrderMAML(model, taskDistribution)
    elif method == 'reptile':
        metaModel = RegressionReptile(model, taskDistribution)
    elif method == 'pretrained':
        metaModel = PretrainedModel(model, taskDistribution)
    else:
        raise ValueError('Method %s is not supported' % method)

    for epoch in range(nEpochs):
        meanBatchLoss = metaModel.trainBatch(nSamples, nTasks, nBatch)

        print(
            "Meta loss @ epoch %d, batch size %d: %.4f"
            % (epoch, nBatch, float(meanBatchLoss))
        )

        # Last log is the final model so just drop the epoch
        if epoch < nEpochs - 1:
            if savePreliminary:
                metaModel.saveKeras('./models/' + name + ('_epoch_' + str(epoch)))
        else:
            metaModel.saveKeras('./models/' + name)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="""Tensorflow 2 implementation of Finn et al. 2017 (MAML) run file.
    Trained models can be loaded and visualized in visualize.py.""")
    parser.add_argument('name', metavar='name', type=str, nargs=1,
                        help='You can later find the model under models/name/...')

    parser.add_argument('method', type=str, default="maml", nargs=1,
                        help="""Name of the method you want to apply. Supported are 'pretrained' and 'maml'.""")

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

    parser.add_argument('--save-preliminary', default=False, type=bool,
                        help='If set to true, preliminary models after each epoch are saved too.')

    args=parser.parse_args()
    main(args.name[0], args.method[0], args.batch_size, args.training_tasks,
         args.task_sample_size, args.epochs, args.use_larger_network, args.save_preliminary)
