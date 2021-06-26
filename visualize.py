import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

from mamltf2 import RegressionMAML, RegressionFirstOrderMAML, RegressionReptile, PretrainedModel, SinusoidRegressionTaskDistribution

def main(name, method, gradientSteps, nSamples, sampleLower, sampleUpper):
    taskDistribution = SinusoidRegressionTaskDistribution()

    modelPath = './models/' + name + '/model.json'
    if method == 'maml':
        metaModel = RegressionMAML(modelPath, taskDistribution)
    elif method =='fomaml':
        metaModel = RegressionFirstOrderMAML(modelPath, taskDistribution)
    elif method == 'reptile':
        metaModel = RegressionReptile(modelPath, taskDistribution)
    elif method == 'pretrained':
        metaModel = PretrainedModel(modelPath, taskDistribution)
    else:
        raise ValueError('Method %s is not supported' % method)

    task = taskDistribution.sampleTask()
    ys, xs = task.sampleFromTask(nSamples, sampleLower, sampleUpper)

    x = tf.reshape(tf.linspace(-5., 5., 1000), (1000, 1))

    # Nice when saving the figure
    plt.subplots(num='%s %s %d %d %d' % (name, str(gradientSteps), nSamples, sampleLower, sampleUpper))
    plt.plot(x, task(x), label="True sine")

    for nSteps in ([0] + gradientSteps if not 0 in gradientSteps else gradientSteps):
        prediction = metaModel.steps(ys, xs, nSteps)(x)
        print('Mean squared error from -5 to 5 with %d steps: %.4f' %
              (nSteps, metaModel.mse(task(x), prediction)))
        plt.plot(x, prediction, label='%d gradient steps' % nSteps)

    plt.scatter(xs, ys)
    plt.title('Comparing gradient steps for model ' + name)
    plt.ylim([-5, 5])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Tensorflow 2 implementation of Finn et al. 2017 (MAML) visualization file.
    Loads a trained model from ./models/, samples a random task and then performs a varying number of gradient steps with that model.
    The results are compared, 0 gradient steps is always included but you can savely add it to the list, if you want.""")
    parser.add_argument('name', metavar='name', type=str, nargs=1,
                        help='The model you want to load from models/name/...')

    parser.add_argument('method', type=str, default="maml", nargs=1,
                        help="""Name of the method you want to apply. Supported are 'pretrained' and 'maml'.""")

    parser.add_argument('gradsteps', default=[1], type=int, nargs='*',
                        help='A list of numbers of gradient steps that should be compared')

    parser.add_argument('--samples', default=5, type=int,
                        help='Number of task samples for gradient updates.')

    parser.add_argument('--sample-lower', default=-5, type=float,
                        help="""Lower bound on task samples. Use to verify that the model extrapolates
                        structure even for areas where no samples exist.""")

    parser.add_argument('--sample-upper', default=5, type=float,
                        help="""Upper bound on task samples. Use to verify that the model extrapolates
                        structure even for areas where no samples exist.""")

    args = parser.parse_args()
    main(args.name[0], args.method[0], args.gradsteps, args.samples,
         args.sample_lower, args.sample_upper)
