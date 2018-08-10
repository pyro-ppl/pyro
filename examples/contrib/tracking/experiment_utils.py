import json
import argparse
import pandas
import os


def args2json(args, filename):
    args_dict = vars(args)
    with open(filename, 'w') as fp:
        json.dump(args_dict, fp)


def json2args(filename):
    with open(filename) as handle:
        args_dict = json.loads(handle.read())
    args = argparse.ArgumentParser().parse_args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    return args


def tabulate_results(exp_dir='.', filename=None):
    from glob import glob
    experiments = glob(os.path.join(exp_dir, "exp*", ""))
    all_results = pandas.DataFrame()
    for e in experiments:
        config_fpath = os.path.join(e, 'config.json')
        result_fpath = os.path.join(e, 'results.csv')
        if os.path.exists(config_fpath) and os.path.exists(result_fpath):
            with open(config_fpath, 'r') as handle:
                config_dict = json.loads(handle.read())
            row_name = config_dict.pop('exp_name')
            for k, v in config_dict.items():
                config_dict[k] = [v]
            config = pandas.DataFrame(config_dict, index=[row_name])
            results = pandas.read_csv(result_fpath, index_col=0)
            results = results.rename(index={'acc': row_name})
            entry = pandas.concat([config, results], axis=1)
            all_results = pandas.concat([all_results, entry], axis=0)
    if filename is not None:
        all_results.to_csv(filename)
    else:
        return all_results
