from configparser import ConfigParser

import os
import random
import pickle
import neat
from neat.nn import FeedForwardNetwork
from neat.math_util import mean, stdev
from neat.config import ConfigParameter

from neat_gym import _gym_make, _is_discrete, eval_net
from pureples.shared.visualize import draw_net

class GymNeatConfig(object):
    '''
    # class for helping Gym work with NEAT
    '''

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self, args, layout=None):

        # Check config file exists
        if not os.path.isfile(args.configfile):
            print('No such config file: %s' %
                  os.path.abspath(args.configfile))
            exit(1)

        # Use default NEAT settings
        self.genome_type = neat.DefaultGenome
        self.reproduction_type = neat.DefaultReproduction
        self.species_set_type = neat.DefaultSpeciesSet
        self.stagnation_type = neat.DefaultStagnation

        parameters = ConfigParser()
        with open(args.configfile) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

            self.node_names = {}

            try:
                names = parameters['Names']
                for idx, name in enumerate(eval(names['input'])):
                    self.node_names[-idx-1] = name
                for idx, name in enumerate(eval(names['output'])):
                    self.node_names[idx] = name
            except Exception:
                pass

        param_list_names = []

        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn('Using default %s for %s' %
                                  (p.default, p.name), DeprecationWarning)
            param_list_names.append(p.name)

        # Bozo filter for missing sections
        self.check_params(args.configfile, parameters, 'NEAT')
        self.check_params(args.configfile, parameters, 'Gym')

        # Get number of episode repetitions
        gympar = parameters['Gym']
        env_name = gympar['environment']
        self.reps = int(gympar['episode_reps'])

        # Make gym environment form name in command-line arguments
        env = _gym_make(env_name)

        # Get input/output layout from environment, or from layout for Hyper
        if layout is None:
            num_inputs = env.observation_space.shape[0]
            #num_inputs = env.observation_space['observation'].shape[0]
            if _is_discrete(env):
                num_outputs = env.action_space.n
            else:
                num_outputs = env.action_space.shape[0]
        else:
            num_inputs, num_outputs = layout
        print('num_inputs is ', num_inputs)
        print('num_outputs is ', num_outputs)
        # Parse type sections.
        genome_dict = dict(parameters.items(self.genome_type.__name__))

        genome_dict['num_inputs'] = num_inputs
        genome_dict['num_outputs'] = num_outputs

        self.genome_config = self.genome_type.parse_config(genome_dict)

        stagnation_dict = dict(parameters.items(self.stagnation_type.__name__))
        self.stagnation_config = \
            self.stagnation_type.parse_config(stagnation_dict)

        self.species_set_dict = \
            dict(parameters.items(self.species_set_type.__name__))
        self.species_set_config = \
            self.species_set_type.parse_config(self.species_set_dict)

        self.reproduction_dict = \
            dict(parameters.items(self.reproduction_type.__name__))
        self.reproduction_config = \
            self.reproduction_type.parse_config(self.reproduction_dict)

        # Store environment name for saving results
        self.env_name = env_name

        # Get number of generations and random seed from config;
        # use defaults if missing
        neatpar = parameters['NEAT']
        self.ngen = self.get_with_default(neatpar, 'generations',
                                          lambda s: int(s), None)
        self.seed = self.get_with_default(neatpar, 'seed',
                                          lambda s: int(s), None)
        self.checkpoint = self.get_with_default(neatpar, 'checkpoint',
                                                lambda s: bool(s), False)

        # Set random seed (including None)
        random.seed(self.seed)

        # Set max episode steps from spec in __init__.py
        self.max_episode_steps = env.spec.max_episode_steps
        if self.max_episode_steps is None:
            self.max_episode_steps = 400

        # Store environment for later
        self.env = env

        # Track evaluations
        self.current_evaluations = 0
        self.total_evaluations = 0

        # Support novelty search
        self.novelty = GymNeatConfig.parse_novelty(args.configfile) \
            if args.novelty else None

        # Store config parameters for subclasses
        self.params = parameters

        # For debugging
        self.gen = 0

        # Default to non-recurrent net
        self.activations = 1

    def eval_net_mean(self, net, genome):

        return (self.eval_net_mean_novelty(net, genome)
                if self.is_novelty()
                else self.eval_net_mean_reward(net, genome))

    def eval_net_mean_reward(self, net, genome):
        reward_sum = 0
        total_steps = 0

        for _ in range(self.reps):
            reward, steps = eval_net(net,
                                     self.env,
                                     activations=self.activations,
                                     seed=self.seed,
                                     max_episode_steps=self.max_episode_steps,
                                     report=False)
                                     #csvfilename='eval_log.csv')

            reward_sum += reward
            total_steps += steps

        return reward_sum/self.reps, total_steps

    def eval_net_mean_novelty(self, net, genome):
        reward_sum = 0
        total_steps = 0

        # No behaviors yet
        behaviors = [None] * self.reps

        for j in range(self.reps):

            reward, behavior, steps = self.eval_net_novelty(net, genome)

            reward_sum += reward

            behaviors[j] = behavior

            total_steps += steps

        return reward_sum/self.reps, behaviors, total_steps

    def eval_net_novelty(self, net, genome):
        #print("eval_net_novelty startd")
        env = self.env
        env.seed(self.seed)
        state = env.reset()
        steps = 0

        is_discrete = _is_discrete(env)

        total_reward = 0

        while self.max_episode_steps is None or steps < self.max_episode_steps:

            # Support recurrent nets
            for k in range(self.activations):
                action = net.activate(state)

            # Support both discrete and continuous actions
            action = (np.argmax(action)
                      if is_discrete
                      else action * env.action_space.high)

            state, reward, done, info = env.step(action)

            #behavior = info['behavior']
            behavior = info['current_position']

            # Accumulate reward, but not novelty
            total_reward += reward

            if done:
                break

            steps += 1
        if steps == self.max_episode_steps:
            total_reward = total_reward - 10
        env.close()

        # Return total reward and final behavior
        return total_reward, behavior, steps

    def save_genome(self, genome, count=0):

        name = self.make_name(genome, count)
        net = FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.env_name), open('models/%s.dat' % name, 'wb'))
        GymNeatConfig.draw_net(net,
                                'visuals/%s-network' % name,
                                self.node_names)

    def is_novelty(self):

        return self.novelty is not None

    def make_name(self, genome, count=0, suffix=''):

        return '%s_s%s_c%d_f%+010.3f' % \
               (self.env_name, suffix, count, genome.actual_fitness)

    def get_with_default(self, params, name, fun, default):
        return fun(params[name]) if name in params else default

    def check_params(self, filename, params, section_name):
        if not params.has_section(section_name):
            self.error('%s section missing from configuration file %s' %
                       (section_name, filename))

    def error(self, msg):
        print('ERROR: ' + msg)
        exit(1)

    @staticmethod
    def draw_net(net, filename, node_names):

        # Create PDF using PUREPLES function
        draw_net(net, filename=filename, node_names=node_names)

        # Delete text
        os.remove(filename)

    @staticmethod
    def eval_genome(genome, config):
        '''
        The result of this function gets assigned to the genome's fitness.
        '''
        print("eval_genome started")
        net = FeedForwardNetwork.create(genome, config)
        return config.eval_net_mean(net, genome)

    @staticmethod
    def parse_novelty(cfgfilename):

        novelty = None

        parameters = ConfigParser()

        with open(cfgfilename) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

            try:
                names = parameters['Novelty']
                novelty = Novelty(eval(names['k']),
                                  eval(names['threshold']),
                                  eval(names['limit']),
                                  eval(names['ndims']))
            except Exception:
                print('File %s has no [Novelty] section' % cfgfilename)
                exit(1)

        return novelty
    