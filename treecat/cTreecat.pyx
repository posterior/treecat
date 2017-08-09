# distutils: language = c++
# distutils: sources = treecat/treecat.cpp

cimport numpy as np
import numpy as np

from eigency.core cimport MatrixXi
from eigency.core cimport VectorXi
from eigency.core cimport Map

from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector

ctypedef map[string, int64_t] Config


cdef extern from 'treecat.hpp' namespace 'treecat':

    string _echo 'treecat::echo' (string& arg)

    cdef struct Suffstats:
        int row
        VectorXi cell
        MatrixXi feature
        MatrixXi latent
        MatrixXi feature_latent
        MatrixXi latent_latent

    cdef struct Model:
        pass

    bool _train_model 'treecat::train_model' \
        (Map[MatrixXi]& data, Config& config, Model& model, string& error)


def echo(arg):
    return _echo(arg.encode('utf-8')).decode('utf-8')


def train_model(data, mask, config):
    cdef Config _config
    cdef Model _model
    # TODO convert config
    cdef string error
    if not _train_model(Map[MatrixXi](data), _config, _model, error):
        raise ValueError(error)
    return {
        'config': config,
        'suffstats': None,  # TODO
        'assignments': None,  # TODO
    }
