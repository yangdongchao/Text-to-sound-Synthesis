#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

from pathlib import Path
import os
import csv
import pickle


def write_csv_file(csv_obj, file_name):

    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, csv_obj[0].keys())
        writer.writeheader()
        writer.writerows(csv_obj)
    print(f'Write to {file_name} successfully.')


def load_csv_file(file_name):

    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj


def load_pickle_file(file_name):

    with open(file_name, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj


def write_pickle_file(obj, file_name):

    Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Write to {file_name} successfully.')
