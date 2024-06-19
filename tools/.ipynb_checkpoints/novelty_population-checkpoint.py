from gym_population import * 

class _NoveltyPopulation(_GymPopulation):
    '''
    Supports genomes that report their novelty
    '''

    def __init__(self, config, stats):

        _GymPopulation.__init__(self, config, stats)

    def parse_fitness(self, fitness):
        '''
        Break out fitness tuple into
        (fitness for selection, actual fitness, evaluations)
        '''

        # Use actual_fitness to encode ignored objective, and replace genome's
        # fitness with its novelty, summed over behaviors.  If the behavior is
        # None, we treat its sparsity as zero.
        actual_fitness, behaviors, evaluations = fitness

        fitness = np.sum([0 if behavior is None
                          else self.config.novelty.add(behavior)
                          for behavior in behaviors])

        return fitness, actual_fitness, evaluations