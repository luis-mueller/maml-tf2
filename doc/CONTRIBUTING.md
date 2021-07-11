## Contribute with Issues or Bugfixes 
Pretty much self-explanatory.

## Contribute with an algorithm implementation
You can write a new meta-learning method (somehow related to MAML would be good) by subclassing the `Model` class in 
`mamltf2/model.py`. Please place the file in `mamltf2/methods` and import your method in `mamltf2/__init__.py`.

Consider to add a converge-behaviour unit-test to the `test`-folder in `mamltf2/methods`. See more below.

Apart from the `__init__` method where you can define your custom state, there are two methods you can redefine/implement:

### Implementing `update`
Example:

```python 

from mamltf import Model
import tensorflow as tf

class MyModel(Model):

    @tf.function
    def update(self, batch):
        # extract the batch
        y_test, x_test, y_train, x_train = batch

        # compute the update step...
        ...

        loss = ...
        
        ...

        # return the loss 
        return loss
```

If you need want to adjust the signature of the `batch` variable, you can redefine `trainBatch`:

### Redefining `trainBatch`
Lets you make adjustments to the signature of batch (or to set some variables, or ...). Example can be found in the `reptile` implementation.
As a special case, if you do not need test samples you can simply do:

```python 

from mamltf import Model
import tensorflow as tf

class MyModel(Model):

    @tf.function
    def update(self, batch):
        # extract the batch
        y_train, x_train = batch

        # compute the update step...
        ...

        loss = ...
        
        ...

        # return the loss 
        return loss

    def trainBatch(self, nSamples, nTasks, nBatch):
        ...

        return super().trainBatch(nSamples, nTasks, nBatch, alsoSampleTest=False)
```

## Contribute with writing unit tests
You can write tests for everything contained but make sure that the tests are fast, precise, test one thing, etc.

### Convergence behaviour
As a heuristic test, it is nice to write down an expected average loss after running one of the methods for a fixed number of epochs and 
a fixed set of parameters (task and method) and then calling the method in a unit test to see if the loss is met. This might fail here and there 
but it gives a rough idea whether the method still runs as intended. You can set the average loss very pessimistic, to avoid 'false positives', i.e. the Github action rejects your commit, eventhough your method just barely didn't meet expectations.

Here the blueprint for such an implementation (best add it to `mamltf2/methods/test/test_methods_behaviour.py`):

```python 
    def test_mymodel_convbehav(self):
        model = MyModel(self.model, self.taskDistribution, lossfn='mse', myparams...) # fixed beforehand
        
        loss = self.doTraining('mymodel', model, nEpochs = 3, nSamples = 10, nTasks = 5, nBatch = 1000) # fixed beforehand
        self.assertLessEqual(loss, my_threshold) # fixed beforehand
```

The method `doTraining` already abstracts the training loop but feel free to write your own.
