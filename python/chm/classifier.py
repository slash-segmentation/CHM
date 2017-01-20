"""
Classifier abstract class.

Jeffrey Bush, 2015-2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractproperty, abstractmethod
class Classifier(object):
    __metaclass__ = ABCMeta    

    @abstractproperty
    def learned(self):
        """Returns True if the classifier has been learned."""
        return False

    @abstractproperty
    def features(self):
        """Number of features learned on or None if not learned yet."""
        return None
    
    @property
    def extra_features(self):
        """
        The number of extra, un-initialized, 'features' this classifier requires that must be 
        in the last rows of the feature matrices given to evaluate and learn.
        """
        return 0
    
    @abstractmethod
    def evaluate(self, X, nthreads=1):
        """
        Evaluates the feature matrix X (features by sample) with the classifier. The array X has
        must be 2D float64 with the first dimension equal to self.features. The classifier must
        already be loaded or learned before this is called.
        """
        pass

    @abstractproperty
    def evaluation_memory(self):
        """
        The amount of memory required to evaluate the model per pixel. The value is undefined if
        the model is not learned.
        """
        return 0

    @abstractmethod
    def learn(self, X, Y, nthreads=1):
        """
        Learns the feature matrix X (features by sample) with Y as the labels with a length of
        samples. The array X must be 2D float64 with the last self.extra_features rows 
        uninitialized and Y must be a 1D bool array of the same length as X as the 2nd dimension of
        X. The classifier must NOT already be loaded or learned before this is called.
        """
        pass
