# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:59:38 2021

@author: Tony_Tien
"""

import logging
from logging import handlers
import time
import os

from Config import log_path

class Log(object):

    def __init__(self, logger=None, log_cate='ActivationPredict_Log'):

        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        self.log_time = time.strftime("%Y_%m_%d")
        file_dir = log_path
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir
        self.log_name = self.log_path + "/" + log_cate + "_" + self.log_time + '.log'

        fh = logging.handlers.RotatingFileHandler(self.log_name, 'a', maxBytes=10485760, backupCount=5) 
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        fh.close()
        ch.close()

    def getlog(self):
        return self.logger