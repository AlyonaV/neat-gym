import neat
from tools.gym_neat_config import *

from pureples.shared.visualize import draw_net
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.substrate import Substrate

class GymHyperConfig(GymNeatConfig):

    def __init__(self, args, substrate=None):

        GymNeatConfig.__init__(self, args, layout=(5, 1))

        # Attempt to get substrate info from environment
        if hasattr(self.env, 'get_substrate'):
            actfun, inp, hid, out = self.env.get_substrate()

        # Default to substrate info from config file
        else:
            subs = self.params['Substrate']
            inp = eval(subs['input'])
            hid = eval(subs['hidden']) if substrate is None else substrate
            out = eval(subs['output'])
            actfun = subs['function']

        self.substrate = Substrate(inp, out, hid)
        self.actfun = actfun

        # For recurrent nets
        self.activations = len(self.substrate.hidden_coordinates) + 2

        # Output of CPPN is recurrent, so negate indices
        self.node_names = {j: self.node_names[k]
                           for j, k in enumerate(self.node_names)}

        # CPPN itself always has the same input and output nodes
        self.cppn_node_names = {-1: 'x1',
                                -2: 'y1',
                                -3: 'x2',
                                -4: 'y2',
                                -5: 'bias',
                                0: 'weight'}

    def save_genome(self, genome, count):

        cppn, net = self.make_nets(genome)
        self.save_nets(genome, cppn, net, cnt=count)

    def save_nets(self, genome, cppn, net, suffix='-hyper'):
        pickle.dump((net, self.env_name),
                    open('models/%s.dat' %
                         self.make_name(genome, suffix=suffix), 'wb'))
        _GymNeatConfig.draw_net(cppn,
                                'visuals/%s' %
                                self.make_name(genome, suffix='-cppn'),
                                self.cppn_node_names)
        self.draw_net(net,
                      'visuals/%s' %
                      self.make_name(genome, suffix=suffix),
                      self.node_names)

    def make_nets(self, genome):

        cppn = neat.nn.FeedForwardNetwork.create(genome, self)
        return (cppn,
                create_phenotype_network(cppn,
                                         self.substrate,
                                         self.actfun))

    @staticmethod
    def eval_genome(genome, config):

        cppn, net = config.make_nets(genome)
        return config.eval_net_mean(net, genome)