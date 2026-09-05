/**
 * @file main.cpp
 * @author Tommaso Barani
 * @brief Growth of intergranular fission gas bubbles on a flat grain boundary in UO2.
 *
 * Two-dimensional reduction of the model of L.K. Aagesen, D. Schwen, M.R. Tonks, Y. Zhang,
 * "Phase-field modeling of fission gas bubble growth on grain boundaries and triple junctions in
 * UO2 nuclear fuel", Comput. Mater. Sci. 161 (2019) 35-45.
 *
 * Three coupled problems are solved in a staggered way at each time step:
 *
 *  1. "OrderParameters" : Allen-Cahn equations for the two grains and for the bubble phase, driven
 *     by the grand-potential difference between the phases (the free energy given to the
 *     integrator is the integrand of Eq. 1 without the gradient term, which the AllenCahn
 *     integrator adds through the Capillary coefficient).
 *
 *  2. "MuVac" and 3. "MuGas" : conservation equations for U vacancies and for Xe on U sites,
 *     written in terms of their chemical potentials (Eqs. 28-29). The time integrator supplies
 *     chi dmu/dt, the Fourier integrator the divergence of D chi grad(mu), the PhaseChangeSource
 *     integrator the partitioning term (the variation of the density at frozen mu, caused by the
 *     motion of the interfaces), and the MassSource integrator the production term, which is
 *     restricted to the fuel matrix by the interpolation function h_m.
 *
 * Units are those of the non-dimensionalisation used in the reference: lengths in nm, times in
 * 0.1 s, energy densities in 64e9 J/m3. Production rates are those of Table 1 of the reference,
 * i.e. representative of normal LWR operation, so that the physical time needed to saturate the
 * grain boundary is of the order of 1e8 in these units.
 *
 * Copyright CEA (c) 2026
 *
 */
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "./FissionGasCoefficients.hpp"
#include "AnalyticalFunctions/AnalyticalFunctions.hpp"
#include "Sloth/sloth.hpp"
#include "Sloth/tests.hpp"

struct TestParameters {
  double control_initial_time = 0.;
  double control_final_time = 1.e6;
  // Time step ramp: dt = dt_ratio * t, bounded by dt_min and dt_max.
  double control_dt_ratio = 1.e-2;
  double control_dt_min = 1.;
  double control_dt_max = 1.e5;
  // Number of bubbles on the grain boundary. The periodic cell is sized so that the linear density
  // of bubbles along the boundary is always the one of the reference, 1/l_GB = 4.46/um: reducing
  // this number gives a smaller cell of the same microstructure, not a different one.
  int control_nb_bubbles = 5;
  // Scaling applied to both production rates. Setting it to zero isolates the conservation
  // properties of the scheme from the production term.
  double control_source_factor = 1.;
  // Semi-implicit treatment of the products of order parameters (see below).
  bool control_semi_implicit = true;
  // Levels of uniform refinement of the mesh.
  int control_refinement_level = 0;
  // Solver diagnostics and controls.
  bool control_verbose = false;
  int control_newton_iter = 100;
  double control_newton_rtol = 1.e-10;
  // Number of time steps between two saves of the fields.
  int control_save_frequency = 50;
  // Time past which the step drops to dt_min (0 disables). Used to resolve an event whose time is
  // roughly known, such as the coalescence of two impinging bubbles, without paying the small step
  // over the whole calculation.
  double control_dt_switch = 0.;
  // Scaling applied to the diffusivity of both species. The equilibrium shape of a bubble does not
  // depend on it, only the time taken to reach it does: reshaping a bubble at constant volume
  // means moving matter over its own size, which takes a time of the order of r^2 / D, about
  // 2.e5 tau* for the radius and the diffusivity of the reference. Raising this factor reaches the
  // equilibrium shape in an affordable time without changing what that shape is.
  double control_diffusivity_factor = 1.;
  // Initial radius of the bubbles. The reference uses 44 nm, which is only 1.5 times the width of
  // the diffuse interface: the region where the boundary meets the surface of a bubble is then as
  // large as the bubble itself, and the balance of the three interfacial energies that sets the
  // dihedral angle cannot be resolved. Raising it tests whether the angle is recovered once the
  // bubble is large compared with the interface.
  double control_bubble_radius = 44.;
  // Size of the periodic cell along the boundary, per bubble. Defaults to l_GB of the reference.
  double control_l_gb = 224.;
  // Size of the domain across the boundary.
  double control_lx = 600.;
  // Size of an element. The width of the diffuse interface is sqrt(8 kappa / m) = 30 nm, so the
  // default of 10 nm puts only three elements across it, which is the bare minimum: the region
  // where the boundary meets the surface of a bubble, where three interfaces meet, needs more.
  double control_dx = 10.;
};

void common_parameters(mfem::OptionsParser& args, TestParameters& p) {
  args.AddOption(&p.control_initial_time, "-i", "--initial_time",
                 "Initial time of the simulation.");
  args.AddOption(&p.control_final_time, "-t", "--final_time", "Final time of the simulation.");
  args.AddOption(&p.control_dt_ratio, "-dtr", "--time_step_ratio",
                 "Ratio between the time-step and the current time.");
  args.AddOption(&p.control_dt_min, "-dtmin", "--time_step_min", "Smallest time-step.");
  args.AddOption(&p.control_dt_max, "-dtmax", "--time_step_max", "Largest time-step.");
  args.AddOption(&p.control_nb_bubbles, "-nb", "--nb_bubbles",
                 "Number of bubbles on the grain boundary (sets the size of the periodic cell).");
  args.AddOption(&p.control_source_factor, "-sf", "--source_factor",
                 "Scaling applied to the production rates of gas atoms and vacancies.");
  args.AddOption(&p.control_semi_implicit, "-si", "--semi_implicit", "-fi", "--fully_implicit",
                 "Treat the products of distinct order parameters semi-implicitly.");
  args.AddOption(&p.control_refinement_level, "-l", "--refinement_level",
                 "Number of levels of uniform refinement of the mesh.");
  args.AddOption(&p.control_verbose, "-v", "--verbose", "-q", "--quiet",
                 "Print the convergence history of the nonlinear solvers.");
  args.AddOption(&p.control_newton_iter, "-nit", "--newton_iter",
                 "Maximum number of iterations of the nonlinear solvers.");
  args.AddOption(&p.control_diffusivity_factor, "-df", "--diffusivity_factor",
                 "Scaling applied to the diffusivity of both species.");
  args.AddOption(&p.control_bubble_radius, "-r", "--bubble_radius",
                 "Initial radius of the bubbles.");
  args.AddOption(&p.control_l_gb, "-lgb", "--l_gb", "Cell size along the boundary, per bubble.");
  args.AddOption(&p.control_lx, "-lx", "--lx", "Domain size across the boundary.");
  args.AddOption(&p.control_dx, "-dx", "--element_size", "Size of an element.");
  args.AddOption(&p.control_save_frequency, "-freq", "--save_frequency",
                 "Number of time steps between two saves of the fields.");
  args.AddOption(&p.control_dt_switch, "-dts", "--dt_switch",
                 "Time past which the time step drops to its minimum (0 disables).");
  args.AddOption(&p.control_newton_rtol, "-nrt", "--newton_rtol",
                 "Relative tolerance of the nonlinear solvers.");

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
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  //---------------------------------------
  Profiling::getInstance().enable();

  TestParameters p;
  mfem::OptionsParser args(argc, argv);
  common_parameters(args, p);

  setVerbosity(p.control_verbose ? Verbosity::Debug : Verbosity::Quiet);

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

  const std::string& main_folder_path = "Saves";

  // ###########################################
  //         Spatial Discretization           //
  // ###########################################
  // x is normal to the grain boundary, y runs along it and is periodic.
  const std::string mesh_type = "InlineSquareWithQuadrangles";
  const int order_fe = 1;
  const int refinement_level = p.control_refinement_level;
  // Characteristic spacing of the grain boundary bubbles, l_GB = 1/sqrt(N_GB) with
  // N_GB = 20/um2 (Section 3.1 of the reference). The cell holds control_nb_bubbles of them.
  const double l_gb = p.control_l_gb;
  const int nb_bubbles = p.control_nb_bubbles;
  const double lx = p.control_lx;
  const double ly = l_gb * nb_bubbles;
  const int nx = static_cast<int>(std::round(lx / p.control_dx));
  const int ny = static_cast<int>(std::round(l_gb / p.control_dx)) * nb_bubbles;

  mfem::Vector y_translation({0.0, ly});
  std::vector<mfem::Vector> translations = {y_translation};
  const std::tuple<int, int, double, double>& tuple_of_dimensions = std::make_tuple(nx, ny, lx, ly);

  SPA spatial(mesh_type, order_fe, refinement_level, tuple_of_dimensions, true);
  SPA spatial_muv(spatial.get_mesh(), order_fe);
  SPA spatial_mug(spatial.get_mesh(), order_fe);
  SPA spatial_v1(spatial.get_mesh(), order_fe);
  SPA spatial_v2(spatial.get_mesh(), order_fe);
  // ##############################
  //     Boundary conditions     //
  // ##############################
  auto boundaries = {Boundary("lower", 0, "Neumann"), Boundary("right", 1, "Neumann"),
                     Boundary("upper", 2, "Neumann"), Boundary("left", 3, "Neumann")};
  auto op_bcs = BoundaryConditions<FECollection, DIM>(&spatial, boundaries);
  auto eta1_bcs = BoundaryConditions<FECollection, DIM>(&spatial_v1, boundaries);
  auto eta2_bcs = BoundaryConditions<FECollection, DIM>(&spatial_v2, boundaries);
  auto muv_bcs = BoundaryConditions<FECollection, DIM>(&spatial_muv, boundaries);
  auto mug_bcs = BoundaryConditions<FECollection, DIM>(&spatial_mug, boundaries);

  // ###########################################
  //            Physical models               //
  // ###########################################
  // Non-dimensionalised model parameters (Table 1 of the reference)
  const double kappa = 0.52734375;                                  // (3/4) sigma_mm l_int
  const double mobility_value = 0.1;                                // order parameter mobility
  const double diffusivity = 1.e-2 * p.control_diffusivity_factor;  // D_g = D_v = 0.1 nm2/s
  const double source_gas = 2.35e-10 * p.control_source_factor;     // s_g^0 = 2.35e12 at/(cm3 s)
  const double source_vac = 2.35e-9 * p.control_source_factor;      // s_v^0 = 10 s_g^0

  // Order parameters.
  //
  // Two discretisations of the same free energy are available. In the fully implicit one every
  // order parameter is taken at the new time step. In the semi-implicit one, every term that
  // couples distinct order parameters keeps one factor at the new time step and evaluates the
  // others at the old one:
  //
  //   gamma eta_i^2 eta_j^2   ->   gamma ( eta_i^2 etan_j^2 + eta_j^2 etan_i^2 - etan_i^2 etan_j^2
  //   ) h_a(eta) omega_a        ->   the interpolation is expanded around eta^n, keeping only the
  //                                numerator of h_a at the new time step.
  //
  // Both are built so that, for every block, the derivative with respect to that block reduces to
  // the fully implicit one when eta^{n+1} = eta^n: the two schemes therefore have exactly the same
  // equilibria and differ only by a first-order-in-dt transient, the same order as the backward
  // Euler scheme they are used with. The purely explicit terms subtracted above play no part in the
  // derivatives; they are there so that the value of the free energy, which the energy
  // post-processing integrates, is the physical one.
  //
  // The benefit is that the Hessian of the semi-implicit form is diagonal: the Newton system
  // decouples into one scalar problem per order parameter, and the strongly nonlinear rational
  // function 1/(sum eta^2)^2 coming from the interpolation functions no longer appears in the
  // Jacobian.
  Coefficient op_energy =
      p.control_semi_implicit
          ? Coefficient(Glossary::FreeEnergy, Scheme::SemiImplicit, FgFreeEnergySI())
          : Coefficient(Glossary::FreeEnergy, Scheme::Implicit, FgFreeEnergy());
  Coefficient op_grad_energy(Glossary::GradEnergy, Scheme::Implicit, FgGradEnergy());
  Coefficient op_capillary(Glossary::Capillary, kappa);
  Coefficient op_mobility(Glossary::Mobility, mobility_value);
  Coefficients op_coef(op_energy, op_capillary, op_mobility, op_grad_energy);

  // Vacancies. The susceptibility plays the role of the heat capacity, and the product of the
  // diffusion coefficient by the susceptibility that of the thermal conductivity.
  Coefficient muv_unit(Glossary::Concentration, 1.0);
  Coefficient muv_chi(Glossary::Cp, Scheme::Implicit, FgSusceptibility());
  Coefficient muv_cond(Glossary::Conductivity, Scheme::Implicit, FgSusceptibility(diffusivity));
  // The density is evaluated explicitly, i.e. at the previous value of the chemical potential.
  // Summing the residual over the domain then telescopes exactly to rho^{n+1} - rho^n, so the
  // scheme conserves the species exactly. Evaluating it implicitly would leave a defect
  // (chi^{n+1} - chi^n)(mu^{n+1} - mu^n) per time step.
  Coefficient muv_density(Glossary::DefectDensity, Scheme::Explicit, FgDensityVac());
  Coefficient muv_source(Glossary::Source, Scheme::Implicit, FgMatrixFraction(source_vac));
  // Declared as a free energy only so that the energy post-processing integrates it over the
  // domain: the "Density" column of the problem is then the total number of vacancies, which is
  // what the conservation of the species has to be checked on.
  Coefficient muv_total(Glossary::FreeEnergy, Scheme::Implicit, FgDensityVac());
  Coefficients muv_coef(muv_unit, muv_chi, muv_cond, muv_density, muv_source, muv_total);

  // Gas atoms
  Coefficient mug_unit(Glossary::Concentration, 1.0);
  Coefficient mug_chi(Glossary::Cp, Scheme::Implicit, FgSusceptibility());
  Coefficient mug_cond(Glossary::Conductivity, Scheme::Implicit, FgSusceptibility(diffusivity));
  Coefficient mug_density(Glossary::DefectDensity, Scheme::Explicit, FgDensityGas());
  Coefficient mug_source(Glossary::Source, Scheme::Implicit, FgMatrixFraction(source_gas));
  Coefficient mug_total(Glossary::FreeEnergy, Scheme::Implicit, FgDensityGas());
  Coefficients mug_coef(mug_unit, mug_chi, mug_cond, mug_density, mug_source, mug_total);

  // ####################
  //     variables     //
  // ####################
  // Grain boundary position and diffuse interface width
  const double x_gb = 0.5 * lx;
  const double width = 30.;
  // Initial bubbles on the grain boundary. The linear density along the boundary is the
  // two-dimensional counterpart of N_GB = 20/um2, i.e. 1/l_GB = 4.46/um, and the initial radius is
  // the one used in the reference.
  const double bubble_radius = p.control_bubble_radius;
  // One bubble per l_GB, displaced inside its cell by a fixed, reproducible jitter so that the
  // arrangement is not perfectly regular, as in the reference where the positions are random. A
  // single bubble is left at the centre of the cell instead: it is then the only feature of the
  // calculation, which is what the measurement of the equilibrium shape needs.
  std::vector<double> bubble_y;
  for (int k = 0; k < nb_bubbles; ++k) {
    const double jitter =
        (nb_bubbles == 1) ? 0. : 0.18 * std::sin(7.3 * static_cast<double>(k) + 1.1);
    bubble_y.emplace_back((static_cast<double>(k) + 0.5 + jitter) * l_gb);
  }

  auto bubble_field = [bubble_y, bubble_radius, x_gb, width, ly](const mfem::Vector& x) {
    double eta_b = 0.;
    for (const auto& yb : bubble_y) {
      // Nearest periodic image along y
      double dy = x[1] - yb;
      dy -= ly * std::round(dy / ly);
      const double dx = x[0] - x_gb;
      const double r = std::sqrt(dx * dx + dy * dy);
      eta_b = std::max(eta_b, 0.5 - 0.5 * std::tanh(2. * (r - bubble_radius) / width));
    }
    return eta_b;
  };

  auto ic_grain_1 = std::function<double(const mfem::Vector&, double)>(
      [bubble_field, x_gb, width](const mfem::Vector& x, [[maybe_unused]] double time) {
        const double eta_b = bubble_field(x);
        const double s = 0.5 - 0.5 * std::tanh(2. * (x[0] - x_gb) / width);
        return (1. - eta_b) * s;
      });
  auto ic_grain_2 = std::function<double(const mfem::Vector&, double)>(
      [bubble_field, x_gb, width](const mfem::Vector& x, [[maybe_unused]] double time) {
        const double eta_b = bubble_field(x);
        const double s = 0.5 + 0.5 * std::tanh(2. * (x[0] - x_gb) / width);
        return (1. - eta_b) * s;
      });
  auto ic_bubble = std::function<double(const mfem::Vector&, double)>(
      [bubble_field](const mfem::Vector& x, [[maybe_unused]] double time) {
        return bubble_field(x);
      });

  auto op_v1 = VAR(&spatial_v1, eta1_bcs, "eta_m1", Glossary::PhaseField, 2,
                   AnalyticalFunctions<DIM>(ic_grain_1));
  op_v1.set_additional_information("eta");
  auto op_v2 = VAR(&spatial_v2, eta2_bcs, "eta_m2", Glossary::PhaseField, 2,
                   AnalyticalFunctions<DIM>(ic_grain_2));
  op_v2.set_additional_information("eta");
  auto op_v3 =
      VAR(&spatial, op_bcs, "eta_b0", Glossary::PhaseField, 2, AnalyticalFunctions<DIM>(ic_bubble));
  op_v3.set_additional_information("eta");
  auto op_vars = VARS(op_v1, op_v2, op_v3);

  // mu = 0 puts each phase at its own equilibrium composition, i.e. the bubbles start at
  // c_g^{b,eq} = 0.454 and the matrix at exp(-E_f/kT), as prescribed in the reference.
  auto ic_muv = std::function<double(const mfem::Vector&, double)>(
      [](const mfem::Vector& x, [[maybe_unused]] double time) { return 0.; });
  auto muv_v = VAR(&spatial_muv, muv_bcs, "mu_v", Glossary::ChemicalPotential, 2,
                   AnalyticalFunctions<DIM>(ic_muv));
  muv_v.set_additional_information("mu");
  auto muv_vars = VARS(muv_v);

  auto mug_v = VAR(&spatial_mug, mug_bcs, "mu_g", Glossary::ChemicalPotential, 2,
                   AnalyticalFunctions<DIM>(ic_muv));
  mug_v.set_additional_information("mu");
  auto mug_vars = VARS(mug_v);

  // ###########################################
  //      Post-processing                     //
  // ###########################################
  const int level_of_detail = 1;
  const int frequency = p.control_save_frequency;

  // Integrated over the whole range of the order parameter, not over the region where it exceeds
  // 0.5: a threshold would follow the reshaping of the diffuse profile rather than the amount of
  // bubble phase, which is what the conservation of the species has to be compared with.
  std::map<std::string, std::tuple<double, double>> bubble_integral = {{"eta_b0", {-1.1, 1.1}}};
  // Wide bounds, so that the integral and the average are taken over the whole domain. In the
  // matrix and before any transport takes place, mu = s^0 t / chi_m, which is an analytical check
  // of the production term and of the susceptibility.
  std::map<std::string, std::tuple<double, double>> muv_integral = {{"mu_v", {-1.e30, 1.e30}}};
  std::map<std::string, std::tuple<double, double>> mug_integral = {{"mu_g", {-1.e30, 1.e30}}};
  std::map<std::string, std::tuple<double, double>> var_integral = {
      {"eta_b0", {-1.1, 1.1}}, {"mu_v", {-1.e30, 1.e30}}, {"mu_g", {-1.e30, 1.e30}}};

  auto v_p_pst = Parameters(
      Parameter("main_folder_path", main_folder_path), Parameter("calculation_path", "Bubbles"),
      Parameter("frequency", frequency), Parameter("level_of_detail", level_of_detail),
      Parameter("enable_compute_energies", true), Parameter("integral_to_compute", var_integral),
      Parameter("enable_save_specialized_at_iter", true));
  auto v_pst = PST(&spatial, v_p_pst);

  // ####################
  //     operators     //
  // ####################
  auto newton_params =
      Parameters(Parameter("description", "Newton solver "),
                 Parameter("print_level", p.control_verbose ? 1 : -1),
                 Parameter("iter_max", p.control_newton_iter),
                 Parameter("rel_tol", p.control_newton_rtol), Parameter("abs_tol", 1.e-14));

  std::vector<SPA*> op_spatials{&spatial, &spatial_v1, &spatial_v2};
  OPE op_oper(op_spatials, {"AllenCahn"}, TimeScheme::EulerImplicit, "TimeDerivative");
  op_oper.overload_nl_solver(NLSolverType::NEWTON, newton_params);
  const auto& solver = HypreSolverType::HYPRE_GMRES;
  const auto& precond = HyprePreconditionerType::HYPRE_ILU;
  op_oper.overload_solver(solver);
  op_oper.overload_preconditioner(precond);

  std::vector<SPA*> mu_spatials{&spatial_muv};
  OPE muv_oper(mu_spatials, {"Fourier", "PhaseChangeSource", "MassSource"},
               TimeScheme::EulerImplicit, "HeatTimeDerivative");
  muv_oper.overload_nl_solver(NLSolverType::NEWTON, newton_params);
  const auto& ch_solver = HypreSolverType::HYPRE_GMRES;
  const auto& ch_precond = HyprePreconditionerType::HYPRE_ILU;
  muv_oper.overload_solver(ch_solver);
  muv_oper.overload_preconditioner(ch_precond);

  std::vector<SPA*> mug_spatials{&spatial_mug};
  OPE mug_oper(mug_spatials, {"Fourier", "PhaseChangeSource", "MassSource"},
               TimeScheme::EulerImplicit, "HeatTimeDerivative");
  mug_oper.overload_nl_solver(NLSolverType::NEWTON, newton_params);
  mug_oper.overload_solver(ch_solver);
  mug_oper.overload_preconditioner(ch_precond);

  // ####################
  //     problems      //
  // ####################
  // The order parameters are solved first, so that the chemical potential problems see the
  // auxiliary variables at the new time step and their previous values at the old one, which is
  // what the PhaseChangeSource integrator needs.
  PB op_pb("OrderParameters", op_oper, op_vars, {op_coef, op_coef, op_coef}, v_pst, muv_vars,
           mug_vars);
  PB muv_pb("MuVac", muv_oper, muv_vars, {muv_coef}, v_pst, op_vars);
  PB mug_pb("MuGas", mug_oper, mug_vars, {mug_coef}, v_pst, op_vars);
  // AMR
  /////////////////////////////////
  ///  AC
  mfem::ConstantCoefficient amr_coef_ac{1.0};
  mfem::DiffusionIntegrator amr_integ_ac{amr_coef_ac};
  SlothErrorEstimators estimator_ac(ErrorEstimatorType::KELLY, &amr_integ_ac);

  MultiVariableMaxAMR<VARS> amr_ac(*spatial.get_mesh(), spatial.is_nc_simplices());

  auto amr_params = Parameters(Parameter("max_elem_error", 1.e-4), Parameter("amr_max_level", 3),
                               Parameter("nc_limit", 0), Parameter("max_preref_cycles", 3));

  amr_ac.SetCriteria(/*estimator*/ &estimator_ac, amr_params);
  op_pb.set_amr(&amr_ac);
  /////////////////////////////////
  // AMR
  auto cc = Coupling("Intergranular fission gas bubbles", op_pb, muv_pb, mug_pb);

  // The order parameters and the chemical potentials are coupled both ways: the driving force of
  // the Allen-Cahn equations is the difference of grand potential between the phases, which depends
  // on the chemical potentials, and the motion of the interfaces feeds the conservation equations
  // through the partitioning term. The problems are staggered once per time step, which leaves that
  // coupling explicit and stable below a time step of the order of chi dx / (L drho^2).

  // ###########################################
  //            Time-integration              //
  // ###########################################
  // The production rates are those of normal operation, so the physical times of interest are very
  // large. A time step proportional to the current time keeps the number of steps moderate while
  // resolving the fast initial equilibration of the bubble shapes.
  const double dt_ratio = p.control_dt_ratio;
  const double dt_min = p.control_dt_min;
  const double dt_max = p.control_dt_max;
  const double dt_switch = p.control_dt_switch;
  // Past dt_switch the step drops to its minimum. Events like the coalescence of two bubbles are
  // far faster than the growth that leads to them: the bridge of matrix between two impinging
  // bubbles snaps in a few tau* after tens of thousands of tau* of quiet growth, and the Newton
  // solver of the fully implicit scheme stops converging on the large steps sized for the growth.
  auto user_time_step =
      std::function<double(double)>([dt_ratio, dt_min, dt_max, dt_switch](double time) {
        if (dt_switch > 0. && time >= dt_switch) {
          return dt_min;
        }
        return std::clamp(dt_ratio * time, dt_min, dt_max);
      });

  auto time_params =
      Parameters(Parameter("initial_time", p.control_initial_time),
                 Parameter("final_time", p.control_final_time), Parameter("vtk_unified", true));
  auto time = TimeDiscretization(user_time_step, time_params, cc);

  time.solve();

  //---------------------------------------
  Profiling::getInstance().print();
  //---------------------------------------
  MPI_Finalize();
  //---------------------------------------
  return 0;
}
