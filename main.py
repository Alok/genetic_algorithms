#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
make list of N numbers that sum to X
'''

import random
from math import floor

import numpy as np

N = 5
TARGET = 200
max_val = 1e1
min_val = -max_val

POPULATION_SIZE = 100
MUTATION_RATE = .05
NOISE_FACTOR = 2 ** 3
NUM_PARENTS = 2
CULL_RATE = 0.8
REPRIEVE_RATE = 0.05


def individual(length=N, min=min_val, max=max_val):
    '''Each suggested solution for a genetic algorithm is referred to as an individual'''
    return np.random.uniform(low=min, high=max, size=(length, ))


def population(count, length=N, min=min_val, max=max_val):
    return np.array([individual(length, min, max) for _ in range(count)])


def loss(individual):
    ''' metric between individual and target'''
    return np.linalg.norm(np.sum(individual) - TARGET)


def avg_loss(population):
    return np.sum(loss(individual)
                  for individual in population) / POPULATION_SIZE


def breed(parents, mutation_rate=MUTATION_RATE):

    child = np.zeros((N, ))
    assert len(parents) <= N
    step = N // len(parents)

    for i, parent in enumerate(np.random.permutation(parents)):
        child[i * step:(i + 1) * step] = parent[i * step:(i + 1) * step]

    # mutate

    if random.random() < mutation_rate:
        perturbation = NOISE_FACTOR * np.random.randn(*child.shape)
        child += perturbation

    return child


def cull(population):
    population = np.array(sorted(population, key=loss))
    split_index = floor((1 - CULL_RATE) * len(population))

    # they survive
    fittest = population[:split_index]
    rest = population[split_index:]

    survivors = rest[np.random.choice(
        rest.shape[0], floor(REPRIEVE_RATE * rest.shape[0])), :]

    return np.concatenate((fittest, survivors))


def create_new_generation(population):
    num_parents = len(population)
    parents = population[:num_parents]
    assert len(population) <= POPULATION_SIZE
    while len(population) < POPULATION_SIZE:
        child = breed(
            parents[np.random.choice(parents.shape[0], size=NUM_PARENTS)])
        population = np.concatenate((population, child.reshape(1, N)))
    return population


def evolve():
    global NOISE_FACTOR
    pop = population(POPULATION_SIZE, length=N)
    LOSS = avg_loss(pop)
    iter = 0

    print(f'NOISE = {NOISE_FACTOR}')
    print(f'CULL = {CULL_RATE}')
    print('--------------------------------------------------------------------------------')
    for _ in range(2000):
        pop = cull(pop)
        pop = create_new_generation(pop)
        LOSS = avg_loss(pop)

        if iter % 100 == 0:
            NOISE_FACTOR = min(1 / 100 * LOSS, 20)
            print(f'LOSS = {LOSS}, iter = {iter}')

        iter += 1

    print('--------------------------------------------------------------------------------')

    return pop


if __name__ == '__main__':
    pop_p = evolve()

    for ind in (sorted(pop_p, key=loss))[:3]:
        print(f'individual: {ind}')
