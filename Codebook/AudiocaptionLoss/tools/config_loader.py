#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import yaml
from dotmap import DotMap


def get_config():

    with open('settings/setting2.yaml', 'r') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    return config
