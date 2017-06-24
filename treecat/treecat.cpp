#include <treecat/treecat.hpp>

namespace treecat {

std::string echo(const std::string& message) { return message; }

bool train_model(const Eigen::Map<Eigen::MatrixXi>& data, const Config& config,
                 Model& model) {
    // TODO
    return false;
}

}  // namespace treecat
