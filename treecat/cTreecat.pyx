# distutils: language = c++
# distutils: sources = treecat/treecat.cpp

cimport numpy as np
import numpy as np

from eigency.core cimport MatrixXi
from eigency.core cimport VectorXi
from eigency.core cimport Map

from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

ctypedef map[string, string] Config


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
        (Map[MatrixXi]& data, Config& config, Model& model)


def echo(arg):
    return _echo(bytes(arg, encoding='utf-8')).decode('utf-8')


def train_model(data, mask, config):
    cdef Config _config
    cdef Model _model
    # TODO convert config
    status = _train_model(Map[MatrixXi](data), _config, _model)
    assert status
    return {
        'config': config,
        'suffstats': None,  # TODO
        'assignments': None,  # TODO
    }
