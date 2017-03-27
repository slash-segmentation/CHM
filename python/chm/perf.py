#!/usr/bin/env python2
"""
Calculate the accuracy/performance of a set of images compared to the ground truth data.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

def main():
    from .utils import calc_confusion_matrix, calc_accuracy, calc_fvalue, calc_gmean
    from pysegtools.images.io import FileImageStack
    from pysegtools.images.filters.threshold import ThresholdImageStack

    import sys
    if len(sys.argv) != 3:
        print("Requires 2 arguments: a stack of predicted images and a stack of ground-truth images")
        sys.exit(1)
    
    predicted = ThresholdImageStack(FileImageStack.open_cmd(sys.argv[1]), 'auto-stack')
    ground_truth = ThresholdImageStack(FileImageStack.open_cmd(sys.argv[2]), 1)
    
    confusion_matrix = calc_confusion_matrix(predicted, ground_truth)
    print("Accuracy = %f"%calc_accuracy(*confusion_matrix))
    print("F-value  = %f"%calc_fvalue(*confusion_matrix))
    print("G-mean   = %f"%calc_gmean(*confusion_matrix))

if __name__ == "__main__": main()
