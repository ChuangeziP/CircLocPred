#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np

from feature_extracting.DACC.descnucleotide import *
from feature_extracting.DACC.pubscripts import *


def get_DACC():
    my_property_name, my_property_value = check_parameters.check_acc_arguments()
    file = '../../data/dataset/input.fa'
    fastas = read_fasta_sequences.read_nucleotide_sequences(file)
    lag = 10
    encodings = ACC.make_acc_vector(fastas, my_property_name, my_property_value, lag, 2)

    print(encodings)
    np.save(f'../../data/DACC_{lag}', arr=encodings)
    print('完成DACC特征的提取')


if __name__ == '__main__':
    get_DACC()
