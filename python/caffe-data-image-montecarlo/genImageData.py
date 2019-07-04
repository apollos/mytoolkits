import os
import sys
import datetime

from multiprocessing import Process

import numpy as np
from matplotlib import pyplot

LATTICE_SIZE = 100
SAMPLE_SIZE = 12000
STEP_ORDER_RANGE = [3, 7]
SAMPLE_FOLDER = 'samples'

#----------------------------------------------------------------------#
#   Check periodic boundary conditions
#----------------------------------------------------------------------#
def bc(i):
    if i+1 > LATTICE_SIZE-1:
        return 0
    if i-1 < 0:
        return LATTICE_SIZE - 1
    else:
        return i

#----------------------------------------------------------------------#
#   Calculate internal energy
#----------------------------------------------------------------------#
def energy(system, N, M):
    return -1 * system[N,M] * (system[bc(N-1), M] \
                               + system[bc(N+1), M] \
                               + system[N, bc(M-1)] \
                               + system[N, bc(M+1)])

#----------------------------------------------------------------------#
#   Build the system
#----------------------------------------------------------------------#
def build_system():
    system = np.random.random_integers(0, 1, (LATTICE_SIZE, LATTICE_SIZE))
    system[system==0] = - 1

    return system

#----------------------------------------------------------------------#
#   The Main monte carlo loop
#----------------------------------------------------------------------#
def main(T, index):

    score = np.random.random()
    order = score*(STEP_ORDER_RANGE[1]-STEP_ORDER_RANGE[0]) + STEP_ORDER_RANGE[0]
    stop = np.int(np.round(np.power(10.0, order)))
    print('Running sample: {}, stop @ {}'.format(index, stop))
    sys.stdout.flush()

    system = build_system()

    for step in range(stop):
        M = np.random.randint(0, LATTICE_SIZE)
        N = np.random.randint(0, LATTICE_SIZE)

        E = -2. * energy(system, N, M)

        if E <= 0.:
            system[N,M] *= -1
        elif np.exp(-1./T*E) > np.random.rand():
            system[N,M] *= -1

        #if step % 100000 == 0:
        #    print('.'),
        #    sys.stdout.flush()

    filename = '{}/'.format(SAMPLE_FOLDER) + '{:0>5d}'.format(index) + '_{}.jpg'.format(score)
    pyplot.imsave(filename, system, cmap='gray')
    print('Saved to {}!\n'.format(filename))
    sys.stdout.flush()

#----------------------------------------------------------------------#
#   Run the menu for the monte carlo simulation
#----------------------------------------------------------------------#

def run_main(index, length):
    np.random.seed(datetime.datetime.now().microsecond)
    for i in xrange(index, index+length):
        main(0.1, i)

def run():

    cmd = 'mkdir -p {}'.format(SAMPLE_FOLDER)
    os.system(cmd)

    n_processes = 8
    length = int(SAMPLE_SIZE/n_processes)
    processes = [Process(target=run_main, args=(x, length)) for x in np.arange(n_processes)*length]

    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

if __name__ == '__main__':
    run()

