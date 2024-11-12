from nni.experiment import Experiment

search_space = {
    'lr': {'_type': 'loguniform', '_value': [0.01, 0.1]},
    'lamd': {'_type': 'loguniform', '_value': [20, 50]},
    'beta': {'_type': 'loguniform', '_value': [1, 10]},
    'sigma': {'_type': 'longuniform', '_value': [2, 8]}
}

experiment = Experiment('local')

experiment.config.trial_command = 'python hynni.py'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 30
experiment.config.trial_concurrency = 2
experiment.run(8080)
experiment.stop()
