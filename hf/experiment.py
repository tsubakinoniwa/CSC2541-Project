import haiku as hk
import optax
import tensorflow_datasets as tfds
import jax
import jax.numpy as np
import memory_profiler

from jax import jit, value_and_grad
from functools import partial
from hf.optimizer import hf
from tqdm.notebook import trange
from jax.config import config

config.update("jax_enable_x64", True)


def get_datasets_64():
    ds_builder = tfds.image_classification.Cifar10()
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='test', batch_size=-1))

    X_train = (np.float64(train_ds['image']) / 255)
    X_test = (np.float64(test_ds['image']) / 255)

    pixel_avg = np.average(X_train, axis=0)
    X_train = X_train - pixel_avg
    X_test = X_test - pixel_avg

    y_train = jax.nn.one_hot(train_ds['label'], num_classes=10)
    y_test = test_ds['label']

    return (X_train, y_train), (X_test, y_test)


def forward(X, is_training):
    model = hk.nets.ResNet18(num_classes=10)
    return model(X, is_training=is_training)


if __name__ == '__main__':
    clf = hk.transform_with_state(forward)


    @partial(jit, static_argnames=('is_training',))
    def loss(params, state, batch, labels, is_training=True):
        logits, state = clf.apply(params, state, None, batch, is_training)
        return np.average(optax.softmax_cross_entropy(logits, labels)), state


    @partial(jit, static_argnames=('is_training',))
    def dloss(params, state, batch, labels, is_training=True):
        return value_and_grad(loss, has_aux=True)(
            params, state, batch, labels, True)


    @jit
    def get_acc(params, state, batch, labels):
        return np.average(np.argmax(
            clf.apply(params, state, None, batch, False)[0], axis=-1) == labels)


    opt = hf(clf, loss, dloss)
    (X_train, y_train), (X_test, y_test) = get_datasets_64()

    N = len(X_train)
    BATCH_SIZE = 1000
    NUM_BATCHES = int(np.ceil(N / BATCH_SIZE))

    prng_key = jax.random.PRNGKey(37528349823)
    init_key, shuffle_key = jax.random.split(prng_key)

    params, state = clf.init(
        init_key, X_train[:BATCH_SIZE], is_training=True)
    opt_state = opt.init(
        params, xi=0.5, lambd=1.0, alpha=0.75, max_iter=5,
        line_search=True, fname='cg.txt',
        use_momentum=False)

    train_loss_hist = []
    val_acc_hist = []
    lambdas = []


    @memory_profiler.profile
    def loop(shuffle_key, X_train, y_train, params, state):
        shuffle_key, rep_key = jax.random.split(shuffle_key)
        inds = jax.random.permutation(rep_key, N)
        X_train, y_train = X_train[inds], y_train[inds]

        for batch in trange(NUM_BATCHES, leave=False):
            batch_start = batch * BATCH_SIZE
            batch_end = min((batch + 1) * BATCH_SIZE, N)

            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            (batch_loss, new_state), batch_grad = dloss(
                params, state, X_batch, y_batch, True)
            updates, opt_state = opt.update(
                batch_grad, opt_state, params, state, X_batch, y_batch)

            params = optax.apply_updates(params, updates)
            state = new_state

        return shuffle_key, params, state


    for rep in trange(20):
        shuffle_key, params, state = loop(
            shuffle_key, X_train, y_train, params, state)
