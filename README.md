# maml-tf2
A fast and comprehensive implementation of MAML (Finn et al. 2017) in Tensorflow 2.

## Fast
The implementation makes use of the `@tf.function` decorator to precompile the update steps in training which significantly speeds up training compared to eager execution.

## Comprehensive
The actual algorithm comprises only roughly 60 loc and uses a straightforward interface. It is easy to read and straightforward to extend.

## Scripts
The codebase comes with two scripts (`run.py` and `visualize.py`) which, apart from serving as examples of use, provide a CLI to train and visualize MAML, e.g. train a model with different parameters or comparing a different number of fine-tuning steps in meta-validation. Models are persisted during training and can be loaded during meta-validation/visualization via the `keras` model interface provided by `tensorflowjs`. This has the benefit of producing models that can be loaded directly both in `python` and `javascript`. To reproduce e.g. the experiments from Finn et al. 2017 you can run the following command, which will run for a total of 70000 iterations, distributed over 70 epochs, such that you receive an average loss every 1000 iterations:

```
python3 run.py maml-reproduce maml --epochs=70 --batch-size=1000 --training-tasks=25 --task-sample=10
```

To visualize the model on a randomly sampled task by comparing predicitions for 0, 1 and 10 gradient steps with `K = 10` you can run:

```
python3 visualize.py maml-reproduce 0 1 10 --samples=10
```

## Getting Started
The codebase can be used in two ways: To conduct experiments using the `run.py` CLI and a predefined set of parameters or to integrate the algorithm into a bigger pipeline or framework. The following shows a minimal example of how to train a model for the sinusoid regression task.

```python
import tensorflow as tf
import tensorflowjs as tfjs
from mamltf2 import MAML, SinusoidRegressionTaskDistribution

ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(40, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(1)
])

taskDistribution = SinusoidRegressionTaskDistribution()
maml = MAML(ann, taskDistribution)

# Trains a batch of 1000 iterations with 10 task samples each, distributed over 25 tasks
maml.trainBatch(nSamples = 10, nTasks = 25, nBatch = 1000)
```

You can now fine-tune the model by calling

```python
fineTunedClone = maml.steps(ys, xs, nSteps = 5)
```

where `[ys, xs]` are samples drawn from a random task. This will return a clone of the meta-trained model with an additional
5 gradient steps on `[ys, xs]`. Note that the connections of this clone and the meta-trained model are cut completely, such that you can reuse the meta-learned initial parameters as often as you need.

The codebase uses the `tensorflowjs` API to save and load models. The easiest way to make use of the API is to save a model like this:

```python
maml.saveKeras('/path/to/destination/')
```

and load it again via

```python
maml = MAML('/path/to/destination/', taskDistribution)
```

which stores the actual `tensorflow` model in `maml.metaModel`. However, the format makes it also possible to load the model directly in the browser, via e.g. in `node`:

```javascript
import * as tf from '@tensorflow/tfjs'

load = async () => {
    const model = await tf.loadLayersModel('https://path/to/destintation/model.json');
}
```

## Contribute:
See how to contribute [here](https://github.com/pupuis/maml-tf2/blob/main/doc/CONTRIBUTING.md).
