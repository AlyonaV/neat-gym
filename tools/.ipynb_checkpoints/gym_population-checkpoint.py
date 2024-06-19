from time import time
import numpy as np
import matplotlib.pyplot as plt
import neat
from neat.population import Population, CompleteExtinctionException

class GymPopulation(Population):
    '''
    Supports genomes that report their number of evaluations
    '''

    def __init__(self, config, stats):

        Population.__init__(self, config)

        self.config = config

        self.stats = stats

    def run(self, fitness_function, ngen, maxtime, config):

        print("maxtime is ", maxtime)
        gen = 0
        start = time()
        
        count = 0

        while ((ngen is None or gen < ngen)
               and (maxtime is None or time()-start < maxtime)):

            self.config.gen = gen

            gen += 1

            self.config.current_evaluations = 0

            self.reporters.start_generation(self.generation)
            # Evaluate all genomes using the user-provided function.
            try:
                for i, p in self.population.items():
                    print("Pop is ", type(i), type(p))
                fitness_function(list(self.population.items()), self.config)
            except:
                print("fitness_function call failed!")
                continue
            print("fitness_function succeeded")
            # Gather and report statistics.
            best = None
            for g in self.population.values():

                if g.fitness is None:
                    raise RuntimeError('Fitness not assigned to genome %d' %
                                       g.key)
                # Break out fitness tuple into actual fitness, evaluations
                g.fitness, g.actual_fitness, evaluations = (
                        self.parse_fitness(g.fitness))
                # Accumulate evaluations
                self.config.current_evaluations += evaluations
                self.config.total_evaluations += evaluations
                if best is None:
                    best = g

                else:
                    if g.actual_fitness > best.actual_fitness:
                        best = g

            self.reporters.post_evaluate(self.config,
                                         self.population,
                                         self.species,
                                         best)

            # Track the best genome ever seen.
            if (self.best_genome is None or
                    best.actual_fitness > self.best_genome.actual_fitness):
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.actual_fitness
                                            for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config,
                                                  self.generation,
                                                  best)
                    break

            # Create the next generation from the current generation.
            self.reproduce()

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.create_new_pop()
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config,
                                  self.population,
                                  self.generation)

            self.reporters.end_generation(self.config,
                                          self.population,
                                          self.species)

            self.generation += 1
            
            if(count%100 == 0):
                config.save_genome(self.best_genome, count)
                self.plot_species()
            count += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config,
                                          self.generation,
                                          self.best_genome)

        self.plot_species()

        return self.best_genome

    def reproduce(self):
        self.population = \
                 self.reproduction.reproduce(self.config, self.species,
                                             self.config.pop_size,
                                             self.generation)

    def create_new_pop(self):
        self.population = \
                self.reproduction.create_new(self.config.genome_type,
                                             self.config.genome_config,
                                             self.config.pop_size)

    def parse_fitness(self, fitness):
        '''
        Break out fitness tuple into
        (fitness for selection, actual fitness, evaluations)
        '''
        return fitness[0], fitness[0], fitness[1]

    def plot_species(self):
        """ Visualizes speciation throughout evolution. """

        species_sizes = self.stats.get_species_sizes()
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        fig, ax = plt.subplots()
        ax.stackplot(range(num_generations), *curves)

        filename = self.config.make_name(self.best_genome)

        plt.title(filename)
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        plt.savefig('visuals/%s-species.pdf' % filename)
        print("savefile ", 'visuals/%s-species.pdf' % filename)

        plt.close()