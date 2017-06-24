#pragma once

#include <cmath>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <utility>

namespace treecat {

std::string echo(const std::string& message);

typedef std::map<std::string, int64_t> Config;

struct SuffStats {
    int row;
    Eigen::VectorXi cell;
    Eigen::MatrixXi feature;
    Eigen::MatrixXi latent;
    Eigen::MatrixXi feature_latent;
    Eigen::MatrixXi latent_latent;
};

struct Model {
    Config config;
    std::vector<std::pair<int, int>> tree;
    SuffStats suffstats;
    Eigen::MatrixXi assignments;
};

bool train_model(const Eigen::Map<Eigen::MatrixXi>& data, const Config& config,
                 Model& model, std::string& error);

}  // namespace treecat
