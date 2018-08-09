import json
import argparse


def args2json(args, filename):
    args_dict = vars(args)
    with open(filename, 'w') as fp:
        json.dump(args_dict, fp)


def json2args(filename):
    with open(filename) as handle:
        args_dict = json.loads(handle.read())[0]
    args = argparse.ArgumentParser().parse_args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    return args
