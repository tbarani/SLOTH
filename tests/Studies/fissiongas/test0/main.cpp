/**
 * @file main.cpp
 * @author Tommaso Barani
 * @brief Calibration of the interfacial parameters of the UO2 fission gas model.
 *
 * A single flat grain boundary between two grains is relaxed towards equilibrium. The interfacial
 * energy computed by SLOTH ("Sigma", twice the integral of the gradient energy density, equal at
 * equilibrium to the total excess energy) is compared with the target grain boundary energy of the
 * model of L.K. Aagesen et al., Comput. Mater. Sci. 161 (2019) 35-45.
 *
 * The purpose of this test is to verify, before any coupled simulation is run, that the multi-well
 * free energy generated from coefficients.json reproduces the intended interfacial energy, i.e.
 * that no factor is lost in the interaction term of the grand-potential functional.
 *
 * Units are those of the non-dimensionalisation used in the reference: lengths in nm, times in
 * 0.1 s, energy densities in 64e9 J/m3. In these units
 *
 *   kappa = 0.52734375, m = 4.6875e-3      (l_int = sqrt(8 kappa/m) = 30 nm)
 *   sigma_mm = (sqrt(2)/3) sqrt(kappa m) = 0.0234375   (i.e. 1.5 J/m2)
 *
 * so that the expected value of "Sigma" is sigma_mm * Ly.
 *
 * Copyright CEA (c) 2026
 *
 */
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "./FissionGasCoefficients.hpp"
#include "Sloth/sloth.hpp"
#include "Sloth/tests.hpp"

struct TestParameters {
  // Time
  double control_time_step = 10.;
  double control_initial_time = 0.;
  double control_final_time = 2.e4;
  // Initial interface thickness (nm). Deliberately different from the equilibrium value so that
  // the relaxation towards the equilibrium profile is an actual test.
  double control_initial_thickness = 15.;
  // Levels of uniform refinement, used to check that the measured interfacial energy converges
  // towards the analytical value.
  int control_refinement_level = 0;
  // Relax a matrix-bubble interface (gamma = 0.922) instead of a grain boundary (gamma = 1.5).
  bool control_matrix_bubble = false;
};

void common_parameters(mfem::OptionsParser& args, TestParameters& p) {
  args.AddOption(&p.control_initial_time, "-i", "--initial_time",
                 "Initial time of the simulation.");
  args.AddOption(&p.control_final_time, "-t", "--final_time", "Final time of the simulation.");
  args.AddOption(&p.control_time_step, "-dt", "--time_step",
                 "Constant time-step of the simulation.");
  args.AddOption(&p.control_initial_thickness, "-e", "--initial_thickness",
                 "Thickness of the interface in the initial conditions.");
  args.AddOption(&p.control_refinement_level, "-l", "--refinement_level",
                 "Number of levels of uniform refinement of the mesh.");
  args.AddOption(&p.control_matrix_bubble, "-mb", "--matrix_bubble", "-mm", "--matrix_matrix",
                 "Relax a matrix-bubble interface instead of a grain boundary.");

  args.Parse();

  if (!args.Good()) {
    if (mfem::Mpi::WorldRank() == 0) {
      args.PrintUsage(mfem::out);
      std::exit(EXIT_FAILURE);
    }
  }
  if (mfem::Mpi::WorldRank() == 0) args.PrintOptions(mfem::out);
}

///---------------
/// Main program
///---------------
int main(int argc, char* argv[]) {
  //---------------------------------------
  // Initialize MPI and HYPRE
  //---------------------------------------
  setVerbosity(Verbosity::Quiet);

  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  //---------------------------------------
  // Profiling start
  Profiling::getInstance().enable();

  // ################ //
  //   Read options   //
  // ################ //
  TestParameters p;
  mfem::OptionsParser args(argc, argv);
  common_parameters(args, p);

  /////////////////////////
  const int DIM = 2;
  using FECollection = Test<DIM>::FECollection;
  using VARS = Test<DIM>::VARS;
  using VAR = Test<DIM>::VAR;
  using PST = Test<DIM>::PST;
  using SPA = Test<DIM>::SPA;
  /////////////////////////
  using OPE = TransientOperator<FECollection, DIM>;
  using PB = Problem<OPE, VARS, PST>;

  // ###########################################
  //         Spatial Discretization           //
  // ###########################################
  const std::string& main_folder_path = "Saves";

  // ##############################
  //           Meshing           //
  // ##############################
  const std::string mesh_type = "InlineSquareWithQuadrangles";
  const int order_fe = 1;
  const int refinement_level = p.control_refinement_level;

  // 5 nm elements: the interface (30 nm) is described by 6 elements, i.e. finer than the 10 nm
  // used in production so that the measured energy is not limited by the discretisation.
  const int nx = 60;
  const int ny = 6;
  const double lx = 300.;
  const double ly = 30.;
  const std::tuple<int, int, double, double>& tuple_of_dimensions =
      std::make_tuple(nx, ny, lx, ly);

  SPA spatial(mesh_type, order_fe, refinement_level, tuple_of_dimensions);

  // ##############################
  //     Boundary conditions     //
  // ##############################
  auto boundaries = {Boundary("lower", 0, "Neumann"), Boundary("right", 1, "Neumann"),
                     Boundary("upper", 2, "Neumann"), Boundary("left", 3, "Neumann")};
  auto bcs = BoundaryConditions<FECollection, DIM>(&spatial, boundaries);

  // ###########################################
  //            Physical models               //
  // ###########################################
  // ####################
  //     Coefficients    //
  // ####################
  // Gradient energy coefficient, non-dimensionalised: kappa = (3/4) sigma_mm l_int
  const double kappa = 0.52734375;
  // Order parameter mobility, non-dimensionalised
  const double mobility_value = 0.1;

  // With gamma = 1.5 the equilibrium profile is known analytically and sigma = (sqrt(2)/3)sqrt(kappa
  // m). For any other gamma the interfacial energy is only known through the fitted function g(gamma)
  // of the reference (its Eq. 22), which does not reproduce its own special case: evaluated at
  // g = sqrt(2)/3 it returns gamma = 1.90 instead of 1.5. The interfacial energy of the
  // matrix-bubble interface is therefore measured here rather than taken from that fit, and the
  // semi-dihedral angle that the model actually imposes is deduced from Young's equation
  // sigma_mm = 2 sigma_mb cos(theta/2).
  Coefficient ac_energy =
      p.control_matrix_bubble
          ? Coefficient(Glossary::FreeEnergy, Scheme::Implicit, FgMultiWellMB())
          : Coefficient(Glossary::FreeEnergy, Scheme::Implicit, FgMultiWell());
  Coefficient ac_grad_energy(Glossary::GradEnergy, Scheme::Implicit, FgGradEnergy());
  Coefficient ac_capillary(Glossary::Capillary, kappa);
  Coefficient ac_mobility(Glossary::Mobility, mobility_value);
  Coefficients ac_coef(ac_energy, ac_capillary, ac_mobility, ac_grad_energy);

  // ####################
  //     variables     //
  // ####################
  const double interface_position = 0.5 * lx;
  const double initial_thickness = p.control_initial_thickness;

  auto ic_grain_1 = std::function<double(const mfem::Vector&, double)>(
      [interface_position, initial_thickness](const mfem::Vector& x,
                                              [[maybe_unused]] double time) {
        return 0.5 - 0.5 * std::tanh(2. * (x[0] - interface_position) / initial_thickness);
      });
  auto ic_grain_2 = std::function<double(const mfem::Vector&, double)>(
      [interface_position, initial_thickness](const mfem::Vector& x,
                                              [[maybe_unused]] double time) {
        return 0.5 + 0.5 * std::tanh(2. * (x[0] - interface_position) / initial_thickness);
      });

  auto ac_v1 = VAR(&spatial, bcs, "eta_1", Glossary::PhaseField, 2,
                   AnalyticalFunctions<DIM>(ic_grain_1));
  ac_v1.set_additional_information("eta");
  auto ac_v2 = VAR(&spatial, bcs, "eta_2", Glossary::PhaseField, 2,
                   AnalyticalFunctions<DIM>(ic_grain_2));
  ac_v2.set_additional_information("eta");

  auto ac_vars = VARS(ac_v1, ac_v2);

  // ###########################################
  //      Post-processing                     //
  // ###########################################
  const int level_of_detail = 1;
  const int frequency = 100;

  auto ac_p_pst = Parameters(Parameter("main_folder_path", main_folder_path),
                             Parameter("calculation_path", "AllenCahn"),
                             Parameter("frequency", frequency),
                             Parameter("level_of_detail", level_of_detail),
                             Parameter("enable_compute_energies", true),
                             Parameter("enable_save_specialized_at_iter", true));

  // ####################
  //     operators     //
  // ####################
  std::vector<SPA*> ac_spatials{&spatial, &spatial};
  OPE ac_oper(ac_spatials, {"AllenCahn"}, TimeScheme::EulerImplicit, "TimeDerivative");
  ac_oper.overload_nl_solver(
      NLSolverType::NEWTON,
      Parameters(Parameter("description", "Newton solver "), Parameter("print_level", -1),
                 Parameter("rel_tol", 1.e-10), Parameter("abs_tol", 1.e-14)));

  auto ac_pst = PST(&spatial, ac_p_pst);
  PB ac_pb("AllenCahn", ac_oper, ac_vars, {ac_coef, ac_coef}, ac_pst);

  auto cc = Coupling("Grain boundary relaxation", ac_pb);

  // ###########################################
  //            Time-integration              //
  // ###########################################
  auto time_params = Parameters(Parameter("initial_time", p.control_initial_time),
                                Parameter("final_time", p.control_final_time),
                                Parameter("time_step", p.control_time_step));
  auto time = TimeDiscretization(time_params, cc);

  time.solve();

  //---------------------------------------
  // Profiling stop
  //---------------------------------------
  Profiling::getInstance().print();

  //---------------------------------------
  // Finalize MPI
  //---------------------------------------
  MPI_Finalize();
  //---------------------------------------
  return 0;
}
