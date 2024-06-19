from gym_hyper_config import *
from pureples.es_hyperneat.es_hyperneat import ESNetwork

class _GymEsHyperConfig(_GymHyperConfig):

    def __init__(self, args):

        _GymHyperConfig.__init__(self, args, substrate=())

        es = self.params['ES']

        self.es_params = {
                'initial_depth': int(es['initial_depth']),
                'max_depth': int(es['max_depth']),
                'variance_threshold': float(es['variance_threshold']),
                'band_threshold': float(es['band_threshold']),
                'iteration_level': int(es['iteration_level']),
                'division_threshold': float(es['division_threshold']),
                'max_weight': float(es['max_weight']),
                'activation': es['activation']
                }

    def save_genome(self, genome, count):

        cppn, _, net = self.make_nets(genome)
        self.save_nets(genome, cppn, net, suffix='-eshyper', cnt=count)

    def make_nets(self, genome):

        cppn = neat.nn.FeedForwardNetwork.create(genome, self)
        esnet = ESNetwork(self.substrate, cppn, self.es_params)
        net = esnet.create_phenotype_network()
        return cppn, esnet, net

    @staticmethod
    def eval_genome(genome, config):

        _, esnet, net = config.make_nets(genome)
        return config.eval_net_mean(net, genome)