# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import bz2
import csv
import datetime
import logging
import multiprocessing
import os
import subprocess
import sys
import urllib

import torch

from pyro.contrib.examples.util import get_data_directory, _mkdir_p

DATA = get_data_directory(__file__)

# https://www.bart.gov/about/reports/ridership
SOURCE_DIR = "http://64.111.127.166/origin-destination/"
SOURCE_FILES = [
    "date-hour-soo-dest-2011.csv.gz",
    "date-hour-soo-dest-2012.csv.gz",
    "date-hour-soo-dest-2013.csv.gz",
    "date-hour-soo-dest-2014.csv.gz",
    "date-hour-soo-dest-2015.csv.gz",
    "date-hour-soo-dest-2016.csv.gz",
    "date-hour-soo-dest-2017.csv.gz",
    "date-hour-soo-dest-2018.csv.gz",
    "date-hour-soo-dest-2019.csv.gz",
]
CACHE_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/bart_full.pkl.bz2"


def _load_hourly_od(basename):
    filename = os.path.join(DATA, basename.replace(".csv.gz", ".pkl"))
    if os.path.exists(filename):
        return filename

    # Download source files.
    gz_filename = os.path.join(DATA, basename)
    if not os.path.exists(gz_filename):
        url = SOURCE_DIR + basename
        logging.debug("downloading {}".format(url))
        urllib.request.urlretrieve(url, gz_filename)
    csv_filename = gz_filename[:-3]
    assert csv_filename.endswith(".csv")
    if not os.path.exists(csv_filename):
        logging.debug("unzipping {}".format(gz_filename))
        subprocess.check_call(["gunzip", "-k", gz_filename])
    assert os.path.exists(csv_filename)

    # Convert to PyTorch.
    logging.debug("converting {}".format(csv_filename))
    start_date = datetime.datetime.strptime("2000-01-01", "%Y-%m-%d")
    stations = {}
    num_rows = sum(1 for _ in open(csv_filename))
    logging.info("Formatting {} rows".format(num_rows))
    rows = torch.empty((num_rows, 4), dtype=torch.long)
    with open(csv_filename) as f:
        for i, (date, hour, origin, destin, trip_count) in enumerate(csv.reader(f)):
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
            date += datetime.timedelta(hours=int(hour))
            rows[i, 0] = int((date - start_date).total_seconds() / 3600)
            rows[i, 1] = stations.setdefault(origin, len(stations))
            rows[i, 2] = stations.setdefault(destin, len(stations))
            rows[i, 3] = int(trip_count)
            if i % 10000 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()

    # Save data with metadata.
    dataset = {
        "basename": basename,
        "start_date": start_date,
        "stations": stations,
        "rows": rows,
        "schema": ["time_hours", "origin", "destin", "trip_count"],
    }
    dataset["rows"]
    logging.debug("saving {}".format(filename))
    torch.save(dataset, filename)
    return filename


def load_bart_od():
    """
    Load a dataset of hourly origin-destination ridership counts for every pair
    of BART stations during the years 2011-2019.

    **Source** https://www.bart.gov/about/reports/ridership

    This downloads the dataset the first time it is called. On subsequent calls
    this reads from a local cached file ``.pkl.bz2``. This attempts to
    download a preprocessed compressed cached file maintained by the Pyro team.
    On cache hit this should be very fast. On cache miss this falls back to
    downloading the original data source and preprocessing the dataset,
    requiring about 350MB of file transfer, storing a few GB of temp files, and
    taking upwards of 30 minutes.

    :returns: a dataset is a dictionary with fields:

        -   "stations": a list of strings of station names
        -   "start_date": a :py:class:`datetime.datetime` for the first observaion
        -   "counts": a ``torch.FloatTensor`` of ridership counts, with shape
            ``(num_hours, len(stations), len(stations))``.
    """
    _mkdir_p(DATA)
    filename = os.path.join(DATA, "bart_full.pkl.bz2")
    # Work around apparent bug in torch.load(),torch.save().
    pkl_file = filename.rsplit(".", 1)[0]
    if not os.path.exists(pkl_file):
        try:
            urllib.request.urlretrieve(CACHE_URL, filename)
            logging.debug("cache hit, uncompressing")
            with bz2.BZ2File(filename) as src, open(filename[:-4], "wb") as dst:
                dst.write(src.read())
        except urllib.error.HTTPError:
            logging.debug("cache miss, preprocessing from scratch")
    if os.path.exists(pkl_file):
        return torch.load(pkl_file)

    filenames = multiprocessing.Pool(len(SOURCE_FILES)).map(_load_hourly_od, SOURCE_FILES)
    datasets = list(map(torch.load, filenames))

    stations = sorted(set().union(*(d["stations"].keys() for d in datasets)))
    min_time = min(int(d["rows"][:, 0].min()) for d in datasets)
    max_time = max(int(d["rows"][:, 0].max()) for d in datasets)
    num_rows = max_time - min_time + 1
    start_date = datasets[0]["start_date"] + datetime.timedelta(hours=min_time),
    logging.info("Loaded data from {} stations, {} hours"
                 .format(len(stations), num_rows))

    result = torch.zeros(num_rows, len(stations), len(stations))
    for dataset in datasets:
        part_stations = sorted(dataset["stations"], key=dataset["stations"].__getitem__)
        part_to_whole = torch.tensor(list(map(stations.index, part_stations)))
        time = dataset["rows"][:, 0] - min_time
        origin = part_to_whole[dataset["rows"][:, 1]]
        destin = part_to_whole[dataset["rows"][:, 2]]
        count = dataset["rows"][:, 3].float()
        result[time, origin, destin] = count
        dataset.clear()
    logging.info("Loaded {} shaped data of mean {:0.3g}"
                 .format(result.shape, result.mean()))

    dataset = {
        "stations": stations,
        "start_date": start_date,
        "counts": result,
    }
    torch.save(dataset, pkl_file)
    subprocess.check_call(["bzip2", "-k", pkl_file])
    assert os.path.exists(filename)
    return dataset


def load_fake_od():
    """
    Create a tiny synthetic dataset for smoke testing.
    """
    dataset = {
        "stations": ["12TH", "EMBR", "SFIA"],
        "start_date": datetime.datetime.strptime("2000-01-01", "%Y-%m-%d"),
        "counts": torch.distributions.Poisson(100).sample([24 * 7 * 8, 3, 3]),
    }
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART data preprocessor")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(relativeCreated) 9d %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    load_bart_od()
