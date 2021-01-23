// clang-format off
#include "utils/differentiation.hpp"  // differentiation.hpp has to go first
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/kinematics.hpp"
#include "math/tiny/tiny_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "multi_body.hpp"
#include "visualizer/meshcat/meshcat_urdf_visualizer.h"
#include "urdf/urdf_parser.hpp"
#include "urdf/urdf_to_multi_body.hpp"
#include "utils/dataset.hpp"
#include "utils/experiment.hpp"
#include "utils/file_utils.hpp"
// clang-format on

using namespace TINY;
using namespace tds;

const int dof = 2;
const std::size_t num_layers = 2;  // TOTAL number of layers
const std::size_t num_hidden_units = 4;

constexpr StaticNeuralNetworkSpecification<num_layers> policy_nn_spec() {
  StaticNeuralNetworkSpecification<num_layers> augmentation;
  for (std::size_t i = 0; i < num_layers; ++i) {
    augmentation.layers[i] = num_hidden_units;
    augmentation.use_bias[i] = true;
    augmentation.activations[i] = tds::NN_ACT_ELU;
  }
  augmentation.layers.front() = dof;
  augmentation.layers.back() = dof;

  augmentation.use_bias.front() = true;
  augmentation.activations.back() = tds::NN_ACT_IDENTITY;
  return augmentation;
}

constexpr int param_dim = policy_nn_spec().num_parameters();

/**
 * Gym environment that computes the cost (i.e. opposite of reward) to be
 * minimized by the policy optimizer.
 */
template<typename Algebra>
struct CostFunctor {
  using Scalar = typename Algebra::Scalar;
  // XXX this is important for the AD libraries
  static const int kDim = param_dim;
  int timesteps{200};
  Scalar dt{Algebra::from_double(1e-2)};

  mutable tds::NeuralNetwork<Algebra> policy;
  World<Algebra> world;
  MultiBody<Algebra>* robot_mb;
  UrdfStructures<Algebra> urdf_structures;
  MeshcatUrdfVisualizer<Algebra>* meshcat_viz_;

  CostFunctor(MeshcatUrdfVisualizer<Algebra>* meshcat_viz = nullptr) :
      policy(policy_nn_spec()), meshcat_viz_(meshcat_viz) {
    robot_mb = world.create_multi_body();
    urdf_structures = LoadRobotUrdf();
    UrdfToMultiBody<Algebra>::convert_to_multi_body(
        urdf_structures, world, *robot_mb, 0);
//    robot_mb->initialize();

    if (meshcat_viz_ != nullptr) {
      robot_mb->print_state();
      meshcat_viz->convert_visuals(urdf_structures, "", robot_mb);
      meshcat_viz_->sync_visual_transforms(robot_mb);
    }
  }

  /**
   * Rollout function that, given the policy parameters x, computes the cost.
   */
  Scalar operator()(const std::vector<Scalar>& x) const {
    robot_mb->initialize();

    policy.set_parameters(x);

    std::vector<Scalar> policy_input(dof), policy_output(dof);
    Scalar cost = Algebra::zero();

    // Maybe add velocity as inputs
    for (int t = 0; t < timesteps; ++t) {
      for (int i = 0; i < dof; ++i) {
        policy_input[i] = robot_mb->q(i);
      }
      policy.compute(policy_input, policy_output);
      // only 1 tau, or 2.
      // apply policy NN output actions as joint forces
      for (int i = 0; i < dof; ++i) {
        robot_mb->tau(i) = policy_output[i];
      }
      tds::forward_dynamics(*robot_mb, world.get_gravity());
//      // clip velocities and accelerations to avoid NaNs over long rollouts
//      for (int i = 0; i < dof; ++i) {
//        system->qd(i) = Algebra::min(Algebra::from_double(4), system->qd(i));
//        system->qd(i) = Algebra::max(Algebra::from_double(-4), system->qd(i));
//        system->qdd(i) = Algebra::min(Algebra::from_double(14), system->qdd(i));
//        system->qdd(i) =
//            Algebra::max(Algebra::from_double(-14), system->qdd(i));
//      }
      tds::integrate_euler(*robot_mb, dt);

      // the last sphere should be as high as possible, so cost is negative z
      // translation
      cost += -robot_mb->links().back().X_world.translation[2];

      // cost += -system->links().back().X_world.translation[2]
      // + pow(system.q(0), 2)

      if (meshcat_viz_ != nullptr) {
        meshcat_viz_->sync_visual_transforms(robot_mb);
      }
    }
    return cost;
  }

  UrdfStructures<Algebra> LoadRobotUrdf() {
    UrdfParser<Algebra> parser;

    char search_path[TINY_MAX_EXE_PATH_LEN];
    std::string file_name;
    FileUtils::find_file("cartpole.urdf", file_name);
    FileUtils::extract_path(file_name.c_str(), search_path,
                            TINY_MAX_EXE_PATH_LEN);
    std::ifstream ifs(file_name);
    std::string urdf_string;
    if (!ifs.is_open()) {
      std::cout << "Error, cannot open file_name: " << file_name << std::endl;
      exit(-1);
    }
    urdf_string = std::string((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
    StdLogger logger;
    UrdfStructures<Algebra> urdf_structures;
    int flags = 0;
    parser.load_urdf_from_string(urdf_string, flags, logger, urdf_structures);

    return urdf_structures;
  }
};

MeshcatUrdfVisualizer<tds::EigenAlgebra> meshcat_viz;

struct PolicyOptExperiment : public tds::Experiment {
  CostFunctor<tds::EigenAlgebra> cost;
  PolicyOptExperiment() : tds::Experiment("policy_optimization") {
    cost = CostFunctor<tds::EigenAlgebra>(&meshcat_viz);
  }

  /**
   * Visualize the current best policy after each evolution.
   */
  void after_iteration(const std::vector<double>& x) override {
    cost(x);
  }
};

int main(int argc, char* argv[]) {
//  MeshcatUrdfVisualizer<tds::EigenAlgebra> meshcat_viz;
  std::cout << "Waiting for meshcat server" << std::endl;
  meshcat_viz.delete_all();

  tds::OptimizationProblem<tds::DIFF_CERES, CostFunctor> problem;
  for (std::size_t i = 0; i < param_dim; ++i) {
    // assign parameter names, bounds, and initial values
    problem[i].name = "parameter_" + std::to_string(i);
    problem[i].minimum = -0.5;
    problem[i].maximum = 0.5; // check if we need to change this
    problem[i].value = problem[i].random_value();
  }
  PolicyOptExperiment experiment;
  // more settings can be made here for the experiment
  // it will use LBFGS from NLOPT with Pagmo by default
  //   experiment["settings"]["pagmo"]["nlopt"]["solver"] = "slsqp";
  experiment["settings"]["pagmo"]["num_evolutions"] = 1;
  experiment.run(problem);

  return EXIT_SUCCESS;
}