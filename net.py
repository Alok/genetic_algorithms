#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from math import ceil, floor

import keras
import numpy as np
from keras.activations import relu, sigmoid, softmax, tanh
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adamax
from keras.utils.np_utils import to_categorical as one_hot

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train, y_test = one_hot(y_train), one_hot(y_test)

def normalize_input(x):
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2]
                     ).astype('float32') / 255


x_train, x_test = normalize_input(x_train), normalize_input(x_test)

# ######################### HYPERPARAMETERS ###################################

ACTIVATION = 'relu'
CALLBACKS = [EarlyStopping()]
MAX_HIDDEN_DEPTH = 5
MAX_HIDDEN_WIDTH = 1000
MUTATION_RATE = 0.15
NUM_EPOCHS = 2
OPTIMIZER = Adam()
POPULATION_SIZE = 2
NUM_PARENTS = min(5, POPULATION_SIZE)
REPRIEVE_RATE = 0.1
SURVIVAL_RATE = 0.3

VALIDATION_SPLIT = 0.3
VERBOSE = True

# #############################################################################


def individual(
    hidden_depth=MAX_HIDDEN_DEPTH,
    hidden_width=MAX_HIDDEN_WIDTH,
    epochs=NUM_EPOCHS):

    if hidden_depth > MAX_HIDDEN_DEPTH:
        hidden_depth = MAX_HIDDEN_DEPTH
    if hidden_width > MAX_HIDDEN_WIDTH:
        hidden_width = MAX_HIDDEN_WIDTH

    model = Sequential()

    # initial layer
    model.add(
        Dense(
            hidden_width,
            activation=ACTIVATION,
            input_shape=(x_train.shape[1], )))

    for _ in range(hidden_depth):
        model.add(Dense(hidden_width, activation=ACTIVATION))

    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(
        optimizer=OPTIMIZER,
        loss=categorical_crossentropy,
        metrics=['accuracy'], )

    if VERBOSE:
        model.summary()

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_split=VALIDATION_SPLIT,
        callbacks=CALLBACKS,
        verbose=VERBOSE)

    return model


def breed(parents, mutation_rate=MUTATION_RATE):
    ''' TODO add mutation rates for each hyperparameter
    creates new child from parents
    '''

    # TODO define

    child_width = (random.choice(parents).get_config())[0]['units']
    # len() is number of layers, and we subtract 2, one each for input and output
    child_depth = len(random.choice(parents).get_config()) - 2

    # TODO mutation strategies here
    if random.random() < MUTATION_RATE:
        child_depth = max(
            1,
            floor(truncnorm.rvs(-MAX_HIDDEN_DEPTH / 2, MAX_HIDDEN_DEPTH / 2)))
        child_width = max(
            1,
            floor(
                truncnorm.rvs(-MAX_HIDDEN_WIDTH / 10, MAX_HIDDEN_WIDTH / 10)))

    child = individual(hidden_width=child_width, hidden_depth=child_depth)
    return child


def loss(individual):
    ''' Returns loss function value on test set.
    XXX should this give accuracy on validation data?
    '''
    return individual.evaluate(x_test, y_test)[0]


def avg_loss(population):
    '''XXX parallelize'''
    return sum(loss(individual) for individual in population)


def create_population(count=POPULATION_SIZE):
    return [
        individual(
            hidden_depth=random.randint(1, MAX_HIDDEN_DEPTH),
            hidden_width=random.randint(1, MAX_HIDDEN_WIDTH))
        for _ in range(POPULATION_SIZE)]


def cull(population):

    population = sorted(population, key=loss)
    split_idx = ceil(SURVIVAL_RATE * len(population))

    # they survive
    fittest = population[:split_idx]
    survivors = random.sample(
        population[split_idx:],
        floor(REPRIEVE_RATE * len(population[split_idx:])))

    return fittest + survivors


# pop -> pop
def create_new_generation(population):

    num_parents = len(population)
    parents = population[:num_parents]
    assert len(population) <= POPULATION_SIZE

    while len(population) < POPULATION_SIZE:
        child = breed(random.sample(parents, k=max(1,NUM_PARENTS)))
        population += child
    return population


def evolve():
    population = create_population()
    LOSS = avg_loss(population)

    iter = 0

    while LOSS > 0 and iter < 10:
        population = create_new_generation(cull(population))
        LOSS = avg_loss(population)

        iter += 1

        print('LOSS = %f, iter = %d' % (LOSS, iter))

    return population


if __name__ == '__main__':
    pass
    evolve()
