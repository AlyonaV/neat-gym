from time import time
import os
import numpy as np
import neat
from neat.reporting import StdOutReporter, BaseReporter
from neat.math_util import mean, stdev

class SaveReporter(BaseReporter):

    def __init__(self, env_name, checkpoint, novelty):

        BaseReporter.__init__(self)

        self.best_fitness = -np.inf
        self.checkpoint = checkpoint

        # Make directories for results
        SaveReporter.mkdir('models')
        SaveReporter.mkdir('visuals')
        SaveReporter.mkdir('runs')

        # Create CSV file for history and write its header
        self.csvfile = open('runs/%s.csv' % env_name, 'w')
        self.csvfile.write('Gen,Time,MeanFit,StdFit,MaxFit')
        if novelty:
            self.csvfile.write(',MeanNov,StdNov,MaxNov')
        self.csvfile.write('\n')

        # Start timing for CSV file data
        self.start = time()

    def post_evaluate(self, config, population, species, best_genome):

        fits = [c.actual_fitness for c in population.values()]

        # Save current generation info to history file
        fit_max = max(fits)
        self.csvfile.write('%d,%f,%+5.3f,%+5.3f,%+5.3f' %
                           (config.gen,
                            time()-self.start,
                            mean(fits),
                            stdev(fits),
                            fit_max))

        if config.is_novelty():
            novs = [c.fitness for c in population.values()]
            self.csvfile.write(',%+5.3f,%+5.3f,%+5.3f' %
                               (mean(novs), stdev(novs), max(novs)))

        self.csvfile.write('\n')
        self.csvfile.flush()

        # Track best
        if self.checkpoint and fit_max > self.best_fitness:
            self.best_fitness = fit_max
            print('############# Saving new best %f ##############' %
                  self.best_fitness)
            config.save_genome(best_genome)

    def mkdir(name):
        os.makedirs(name, exist_ok=True)


class _StdOutReporter(StdOutReporter):

    def __init__(self, show_species_detail):

        StdOutReporter.__init__(self, show_species_detail)

    def post_evaluate(self, config, population, species, best_genome):

        # Special report for novelty search
        if config.is_novelty():

            novelties = [c.fitness for c in population.values()]
            nov_mean = mean(novelties)
            nov_std = stdev(novelties)
            best_species_id = species.get_species_id(best_genome.key)
            print('Population\'s average novelty: %3.5f stdev: %3.5f' %
                  (nov_mean, nov_std))
            print('Best novelty: %3.5f - size: (%d,%d) - species %d - id %d' %
                  (best_genome.fitness,
                   best_genome.size()[0],
                   best_genome.size()[1],
                   best_species_id,
                   best_genome.key))
            print('Best actual fitness: %f ' % best_genome.actual_fitness)

        # Ordinary report otherwise
        else:

            StdOutReporter.post_evaluate(
                    self,
                    config,
                    population,
                    species,
                    best_genome)

        print('Evaluations this generation: %d' % config.current_evaluations)
        print('Total evaluations: %d' % config.total_evaluations)
