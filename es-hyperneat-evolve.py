#!/usr/bin/env python3
'''
ES-HyperNEAT evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from neat_gym import _GymEsHyperConfig, _evolve_cmdline

_evolve_cmdline(_GymEsHyperConfig.make_config)
