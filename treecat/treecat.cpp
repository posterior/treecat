#include <treecat/treecat.hpp>
#include <cstdlib>
#include <random>

#define TREECAT_ASSERT(cond, message) \
    {                                 \
        if (!(cond)) {                \
            error = message;          \
            return false;             \
        }                             \
    }

namespace Eigen {
typedef Matrix<bool, Dynamic, 1> VectorXb;
}  // namespace Eigen

namespace treecat {

static inline int64_t map_find(const Config& config, const std::string& key,
                               int64_t default_value = 0) {
    auto i = config.find(key);
    return (i == config.end()) ? i->second : default_value;
}

std::string echo(const std::string& message) { return message; }

bool train_model(const Eigen::Map<Eigen::MatrixXi>& data, const Config& config,
                 Model& model, std::string& error) {
    const size_t N = data.rows();
    const size_t V = data.cols();
    const size_t E = V * (V - 1) / 2;
    const size_t M = map_find(config, "model_num_clusters");
    const size_t C = map_find(config, "model_num_categories");
    TREECAT_ASSERT(M, "Missing config field: model_num_clusters");
    TREECAT_ASSERT(C, "Missing config field: model_num_categories");

    SuffStats& ss = model.suffstats;
    auto& assignments = model.assignments;

    ss.row = 0;
    ss.cell.resize(N);
    ss.cell.setZero();
    ss.feature.resize(V * M, M);
    ss.feature.setZero();
    ss.latent.resize(V, M);
    ss.latent.setZero();
    ss.feature_latent.resize(V * C, M);
    ss.feature_latent.setZero();
    ss.latent_latent.resize(E * M, M);
    ss.latent_latent.setZero();

    assignments.resize(N, V);
    Eigen::VectorXb is_assigned(V);
    is_assigned.setZero();

    while (ss.row < N) {
        // TODO
    }

    return false;
}

}  // namespace treecat
