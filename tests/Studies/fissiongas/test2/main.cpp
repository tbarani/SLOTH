/**
 * @file main.cpp
 * @author Tommaso Barani
 * @brief Intergranular fission gas bubbles in a Merope polycrystal: the 2D polycrystal version
 * of test1, in the spirit of Fig. 4 of D.-U. Kim et al., Mater. Theory 6 (2022) 7.
 *
 * The microstructure is not built analytically but read from a gmsh mesh produced by Merope
 * itself (see README.md in this directory and merope2d/make_mesh.sh): Merope's geo2d tool
 * (LaguerreTess_2D on sac_de_billes seeds) writes a native 2D .geo with one
 * Physical Surface("grain_k") per crystallite and Periodic Line pairs on the box edges, and
 * gmsh turns it into poly.msh. SLOTH loads poly.msh with mesh_type "GMSH" (the $Periodic
 * section of the file makes the mesh periodic) and links each crystallite to one phase field:
 * the variable eta_gi is initialized to 1 on the attribute set {"grain_i"} and to 0 elsewhere
 * (Variable's attribute-based initial condition), so grain i is exactly the domain of eta_gi.
 *
 * On top of that microstructure the same three-problem staggered model as test1 is solved:
 * Allen-Cahn for the NB_GRAINS grain order parameters plus the bubble order parameter eta_b,
 * and the two chemical-potential problems for vacancies and gas. The initial GB bubbles are
 * placed on grain boundaries: the midpoints of the mesh faces separating two grains are collected
 * from a serial copy of the mesh file (identical on every MPI rank, no communication needed) and
 * a fixed number of them, evenly spaced along the sorted list, host a tanh bubble.
 *
 * Units and model parameters are those of test1 (non-dimensionalised Aagesen et al. CMS 161
 * (2019), with the single soft curvature of Kim et al. 2022): lengths in nm, times in 0.1 s,
 * energy densities in 64e9 J/m3. The thermodynamics for NB_GRAINS grains is generated at
 * configure time from coefficients.json (generate_coefficients.py -ng 5).
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
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "./FissionGasCoefficients.hpp"
#include "AnalyticalFunctions/AnalyticalFunctions.hpp"
#include "Coefficients/Coefficient.hpp"
#include "Glossary/Glossary.hpp"
#include "Options/TimeOptions.hpp"
#include "Sloth/sloth.hpp"
#include "Sloth/tests.hpp"

// Number of grains of the polycrystal = number of grain order parameters. Must match the number
// of "grain_k" attribute sets of poly.msh (and the -ng argument of generate_coefficients.py).
constexpr int NB_GRAINS = 5;
constexpr int NB_OPS = NB_GRAINS + 1;  // grains + bubble

struct TestParameters {
  double control_initial_time = 0.;
  double control_final_time = 1.e6;
  // Same time-step ramp as test1: dt = dt_ratio * t, bounded by dt_min and dt_max.
  double control_dt_ratio = 1.e-2;
  double control_dt_min = 1.;
  double control_dt_max = 4.e3;
  // Number of initial bubbles on the grain boundaries, and their radius.
  int control_nb_bubbles = 4;
  double control_bubble_radius = 22.;
  // Scaling applied to both production rates (0 isolates the conservation properties).
  double control_source_factor = 1.;
  // Semi-implicit treatment of the products of order parameters (see test1).
  bool control_semi_implicit = true;
  // Levels of uniform refinement of the mesh.
  int control_refinement_level = 0;
  // AMR on the order-parameter problem (the bubble/grain-boundary problem).
  double error_el_amr = 1e-4;
  int max_level_amr = 2;
  bool control_amr = true;
  // Solver diagnostics and controls.
  bool control_verbose = false;
  int control_newton_iter = 100;
  double control_newton_rtol = 1.e-10;
  // Number of time steps between two saves of the fields.
  int control_save_frequency = 50;
  double control_dt_switch = 0.;
  double control_diffusivity_factor = 1.;
  double control_mobility_factor = 1.;
  // Mesh file, produced from the Merope microstructure (see README.md).
  std::string control_mesh_file = "poly.msh";
  // Treat the mesh as periodic (needs a "Periodic" boundary flag).
  bool control_periodic = true;
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
                 "Number of initial bubbles placed on the grain boundaries.");
  args.AddOption(&p.control_bubble_radius, "-r", "--bubble_radius", "Initial radius of the bubbles.");
  args.AddOption(&p.control_source_factor, "-sf", "--source_factor",
                 "Scaling applied to the production rates of gas atoms and vacancies.");
  args.AddOption(&p.control_semi_implicit, "-si", "--semi_implicit", "-fi", "--fully_implicit",
                 "Treat the products of distinct order parameters semi-implicitly.");
  args.AddOption(&p.control_refinement_level, "-l", "--refinement_level",
                 "Number of levels of uniform refinement of the mesh.");
  args.AddOption(&p.error_el_amr, "-erm_amr", "--error_max_amr", "Max error for AMR");
  args.AddOption(&p.max_level_amr, "-elm_amr", "--max_element_amr",
                 "Max #elements for AMR subvision");
  args.AddOption(&p.control_amr, "-amr", "--amr", "-noamr", "--no_amr",
                 "Enable AMR on the order-parameter problem.");
  args.AddOption(&p.control_verbose, "-v", "--verbose", "-q", "--quiet",
                 "Print the convergence history of the nonlinear solvers.");
  args.AddOption(&p.control_newton_iter, "-nit", "--newton_iter",
                 "Maximum number of iterations of the nonlinear solvers.");
  args.AddOption(&p.control_newton_rtol, "-nrt", "--newton_rtol",
                 "Relative tolerance of the nonlinear solvers.");
  args.AddOption(&p.control_diffusivity_factor, "-df", "--diffusivity_factor",
                 "Scaling applied to the diffusivity of both species.");
  args.AddOption(&p.control_mobility_factor, "-lf", "--mobility_factor",
                 "Scaling applied to the mobility of the order parameters.");
  args.AddOption(&p.control_save_frequency, "-freq", "--save_frequency",
                 "Number of time steps between two saves of the fields.");
  args.AddOption(&p.control_dt_switch, "-dts", "--dt_switch",
                 "Time past which the time step drops to its minimum (0 disables).");
  args.AddOption(&p.control_mesh_file, "-m", "--mesh_file",
                 "gmsh mesh of the polycrystal (per-grain physical groups, periodic).");
  args.AddOption(&p.control_periodic, "-p", "--periodic", "-np", "--no_periodic",
                 "Treat the mesh as periodic.");

  args.Parse();

  if (!args.Good()) {
    if (mfem::Mpi::WorldRank() == 0) {
      args.PrintUsage(mfem::out);
      std::exit(EXIT_FAILURE);
    }
  }
  if (mfem::Mpi::WorldRank() == 0) args.PrintOptions(mfem::out);
}

/// Collect the midpoints of the interior faces separating two grains, from a serial copy of the
/// mesh file. Every rank reads the same file and computes the same list, so no MPI communication
/// is needed and the initial condition is identical on all ranks.
std::vector<std::pair<double, double>> grain_boundary_midpoints(const std::string& mesh_file) {
  mfem::Mesh smesh(mesh_file, 1, 1);
  smesh.FinalizeTopology();
  // Map each interior edge (sorted vertex pair) to the first element that owns it; an edge seen
  // with two different grain attributes is a grain boundary. Vertex-based, so no MFEM face
  // tables are needed (and the $Periodic vertex merging makes the box edges consistent).
  std::map<std::pair<int, int>, std::pair<int, int>> edge_owner;  // edge -> (element, attribute)
  std::map<std::pair<long, long>, std::pair<double, double>> uniq;
  mfem::Array<int> verts;
  for (int el = 0; el < smesh.GetNE(); el++) {
    const int attr = smesh.GetAttribute(el);
    smesh.GetElementVertices(el, verts);
    const int nv = verts.Size();
    for (int k = 0; k < nv; k++) {
      const int v1 = verts[k], v2 = verts[(k + 1) % nv];
      const auto key = std::minmax(v1, v2);
      const auto it = edge_owner.find(key);
      if (it == edge_owner.end()) {
        edge_owner[key] = {el, attr};
        continue;
      }
      if (it->second.second != attr) {
        const double x = 0.5 * (smesh.GetVertex(v1)[0] + smesh.GetVertex(v2)[0]);
        const double y = 0.5 * (smesh.GetVertex(v1)[1] + smesh.GetVertex(v2)[1]);
        uniq[{std::llround(x * 1.e9), std::llround(y * 1.e9)}] = {x, y};
      }
    }
  }
  std::vector<std::pair<double, double>> gb;
  gb.reserve(uniq.size());
  for (const auto& kv : uniq) gb.push_back(kv.second);
  std::sort(gb.begin(), gb.end());
  return gb;
}

/// Bounding box of the (periodic) mesh, used for the periodic images of the bubbles.
std::pair<double, double> domain_size(const std::string& mesh_file) {
  mfem::Mesh smesh(mesh_file, 1, 1);
  smesh.FinalizeTopology();
  double xmin = 1.e300, xmax = -1.e300, ymin = 1.e300, ymax = -1.e300;
  for (int v = 0; v < smesh.GetNV(); v++) {
    xmin = std::min(xmin, smesh.GetVertex(v)[0]);
    xmax = std::max(xmax, smesh.GetVertex(v)[0]);
    ymin = std::min(ymin, smesh.GetVertex(v)[1]);
    ymax = std::max(ymax, smesh.GetVertex(v)[1]);
  }
  return {xmax - xmin, ymax - ymin};
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
  // The mesh comes from the Merope microstructure: per-grain physical groups "grain_1..N"
  // (element attributes), $Periodic on the box faces. The last flag tells SLOTH the mesh is
  // periodic.
  const std::string& mesh_file = p.control_mesh_file;
  const int order_fe = 1;
  const int refinement_level = p.control_refinement_level;

  // periodic_mesh (the $Periodic section of the file), enable_nc_mesh = true so that
  // AMR can refine non-conformingly.
  const bool periodic = p.control_periodic;
  SPA spatial("GMSH", order_fe, refinement_level, mesh_file, periodic, true);
  // The shared-mesh constructor does not inherit the periodic flag: pass it explicitly.
  SPA spatial_muv(spatial.get_mesh(), order_fe, periodic);
  SPA spatial_mug(spatial.get_mesh(), order_fe, periodic);
  // One spatial discretization per grain order parameter, all sharing the mesh.
  std::vector<std::unique_ptr<SPA>> grain_spas;
  for (int i = 0; i < NB_GRAINS; i++) {
    grain_spas.push_back(std::make_unique<SPA>(spatial.get_mesh(), order_fe, periodic));
  }
  // ##############################
  //     Boundary conditions     //
  // ##############################
  // The only boundary elements are the outer frame of the box, physical name "box".
  const Boundary box_bdr =
      periodic ? Boundary("box", 0, "Periodic") : Boundary("box", 0, "Neumann");
  auto boundaries = {box_bdr};
  auto op_bcs = BoundaryConditions<FECollection, DIM>(&spatial, boundaries);
  auto muv_bcs = BoundaryConditions<FECollection, DIM>(&spatial_muv, boundaries);
  auto mug_bcs = BoundaryConditions<FECollection, DIM>(&spatial_mug, boundaries);

  // ###########################################
  //            Physical models               //
  // ###########################################
  // Same non-dimensionalised parameters as test1 (Table 1 of Aagesen et al. CMS 161 (2019), with
  // the single soft curvature k of Kim et al. 2022); the coefficients that depend on the number
  // of order parameters are in FissionGasCoefficients.json / the generated header.
  const double kappa = 0.52734375;                                  // (3/4) sigma_mm l_int
  const double mobility_value = 0.1 * p.control_mobility_factor;    // L = 1.56e-11 m3/(J s)
  const double diffusivity = 1.e-2 * p.control_diffusivity_factor;  // D_g = D_v = 0.1 nm2/s
  const double source_gas = 2.35e-10 * p.control_source_factor;     // s_g^0
  const double source_vac = 2.35e-9 * p.control_source_factor;      // s_v^0 = 10 s_g^0

  // Order parameters: same free energy as test1, generated for NB_OPS order parameters.
  Coefficient op_energy =
      p.control_semi_implicit
          ? Coefficient(Glossary::FreeEnergy, Scheme::SemiImplicit, FgFreeEnergySI())
          : Coefficient(Glossary::FreeEnergy, Scheme::Implicit, FgFreeEnergy());
  Coefficient op_grad_energy(Glossary::GradEnergy, Scheme::Implicit, FgGradEnergy());
  Coefficient op_capillary(Glossary::Capillary, kappa);
  Coefficient op_mobility(Glossary::Mobility, mobility_value);
  Coefficients op_coef(op_energy, op_capillary, op_mobility, op_grad_energy);

  // Vacancies and gas atoms: same as test1; the matrix fraction h_m = sum eta_g^2 / sum eta^2
  // now runs over NB_GRAINS grains.
  Coefficient muv_unit(Glossary::Concentration, 1.0);
  Coefficient muv_chi(Glossary::Cp, Scheme::Implicit, FgSusceptibility());
  Coefficient muv_cond(Glossary::Conductivity, Scheme::Implicit, FgSusceptibility(diffusivity));
  Coefficient muv_density(Glossary::DefectDensity, Scheme::Explicit, FgDensityVac());
  Coefficient muv_source(Glossary::Source, Scheme::Implicit, FgMatrixFraction(source_vac));
  Coefficient muv_total(Glossary::FreeEnergy, Scheme::Implicit, FgDensityVac());
  Coefficients muv_coef(muv_unit, muv_chi, muv_cond, muv_density, muv_source, muv_total);

  Coefficient mug_unit(Glossary::Concentration, 1.0);
  Coefficient mug_chi(Glossary::Cp, Scheme::Implicit, FgSusceptibility());
  Coefficient mug_cond(Glossary::Conductivity, Scheme::Implicit, FgSusceptibility(diffusivity));
  Coefficient mug_density(Glossary::DefectDensity, Scheme::Explicit, FgDensityGas());
  Coefficient mug_source(Glossary::Source, Scheme::Implicit, FgMatrixFraction(source_gas));
  Coefficient mug_total(Glossary::FreeEnergy, Scheme::Implicit, FgDensityGas());
  Coefficients mug_coef(mug_unit, mug_chi, mug_cond, mug_density, mug_source, mug_total);

  // post processing
  Coefficient switch_bbl(Glossary::PhaseField, Scheme::Implicit, SwitchingFunctionBubble());
  switch_bbl.set_name("h_b");
  Coefficient switch_mat(Glossary::PhaseField, Scheme::Implicit, FgMatrixFraction());
  switch_mat.set_name("h_m");

  // ####################
  //     variables     //
  // ####################
  // Diffuse interface width of the model (test0): l_int = sqrt(8 kappa / m) = 30 nm.
  const double width = 30.;
  const double bubble_radius = p.control_bubble_radius;

  // Initial bubbles on the grain boundaries: evenly spaced midpoints of the GB faces.
  const auto gb = grain_boundary_midpoints(mesh_file);
  const auto [Lx, Ly] = domain_size(mesh_file);
  const int nb_bubbles = std::min(p.control_nb_bubbles, static_cast<int>(gb.size()));
  std::vector<std::pair<double, double>> bubbles;
  for (int k = 0; k < nb_bubbles; k++) {
    bubbles.push_back(gb[static_cast<size_t>(k) * gb.size() / nb_bubbles]);
  }
  if (mfem::Mpi::WorldRank() == 0) {
    std::cout << "Polycrystal " << mesh_file << ": " << NB_GRAINS << " grains, " << gb.size()
              << " GB face midpoints, " << nb_bubbles << " initial bubbles of radius "
              << bubble_radius << " nm on a " << Lx << " x " << Ly << " nm periodic box"
              << std::endl;
  }

  auto bubble_field = [bubbles, bubble_radius, width, Lx, Ly](const mfem::Vector& x) {
    double eta_b = 0.;
    for (const auto& b : bubbles) {
      double dx = x[0] - b.first;
      dx -= Lx * std::round(dx / Lx);
      double dy = x[1] - b.second;
      dy -= Ly * std::round(dy / Ly);
      const double r = std::sqrt(dx * dx + dy * dy);
      eta_b = std::max(eta_b, 0.5 - 0.5 * std::tanh(2. * (r - bubble_radius) / width));
    }
    return eta_b;
  };

  // Bubble phase: tanh bubbles on the grain boundaries, everywhere in the domain.
  auto ic_bubble = std::function<double(const mfem::Vector&, double)>(
      [bubble_field](const mfem::Vector& x, [[maybe_unused]] double time) {
        return bubble_field(x);
      });
  // Grain i: 1 inside grain i (attribute set "grain_i"), cut by the bubbles, 0 elsewhere. The
  // attribute-based initial condition projects the function on "grain_i" only and leaves the
  // rest of the domain at zero.
  auto ic_grain = std::function<double(const mfem::Vector&, double)>(
      [bubble_field](const mfem::Vector& x, [[maybe_unused]] double time) {
        return 1. - bubble_field(x);
      });

  VARS op_vars;
  std::vector<SPA*> op_spatials;
  for (int i = 1; i <= NB_GRAINS; i++) {
    SPA* s = grain_spas[i - 1].get();
    auto bcs = BoundaryConditions<FECollection, DIM>(s, boundaries);
    const std::string name = "eta_g" + std::to_string(i);
    const std::set<std::string> attr = {"grain_" + std::to_string(i)};
    op_vars.add(VAR(s, bcs, name, Glossary::PhaseField, 2, AnalyticalFunctions<DIM>(ic_grain), attr));
    op_vars.get_variable(name).set_additional_information("eta");
    op_spatials.push_back(s);
  }
  auto op_vb = VAR(&spatial, op_bcs, "eta_b", Glossary::PhaseField, 2,
                   AnalyticalFunctions<DIM>(ic_bubble));
  op_vb.set_additional_information("eta");
  op_vars.add(op_vb);
  op_spatials.push_back(&spatial);

  // mu = 0 puts each phase at its own equilibrium composition, as in test1.
  auto ic_mu = std::function<double(const mfem::Vector&, double)>(
      [](const mfem::Vector& x, [[maybe_unused]] double time) { return 0.; });
  auto muv_v = VAR(&spatial_muv, muv_bcs, "mu_v", Glossary::ChemicalPotential, 2,
                   AnalyticalFunctions<DIM>(ic_mu));
  muv_v.set_additional_information("mu");
  auto muv_vars = VARS(muv_v);

  auto mug_v = VAR(&spatial_mug, mug_bcs, "mu_g", Glossary::ChemicalPotential, 2,
                   AnalyticalFunctions<DIM>(ic_mu));
  mug_v.set_additional_information("mu");
  auto mug_vars = VARS(mug_v);

  // ###########################################
  //      Post-processing                     //
  // ###########################################
  const int level_of_detail = 1;
  const int frequency = p.control_save_frequency;

  // Integrated over the whole range of the order parameter (see test1 for why), and wide bounds
  // for the chemical potentials: the last column of the mu problems is the species inventory.
  std::map<std::string, std::tuple<double, double>> var_integral = {
      {"eta_b", {-1.1, 1.1}}, {"mu_v", {-1.e30, 1.e30}}, {"mu_g", {-1.e30, 1.e30}}};

  auto v_p_pst = Parameters(
      Parameter("main_folder_path", main_folder_path), Parameter("calculation_path", "Polycrystal"),
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

  auto lin_params = Parameters(Parameter("print_level", p.control_verbose ? 3 : -1),
                               Parameter("iter_max", 10000));
  OPE op_oper(op_spatials, {"AllenCahn"}, TimeScheme::EulerImplicit, "TimeDerivative");
  op_oper.overload_nl_solver(NLSolverType::NEWTON, newton_params);
  op_oper.overload_solver(HypreSolverType::HYPRE_GMRES, lin_params);
  op_oper.overload_preconditioner(HyprePreconditionerType::HYPRE_ILU);

  std::vector<SPA*> mu_spatials{&spatial_muv};
  OPE muv_oper(mu_spatials, {"Fourier", "PhaseChangeSource", "MassSource"},
               TimeScheme::EulerImplicit, "HeatTimeDerivative");
  muv_oper.overload_nl_solver(NLSolverType::NEWTON, newton_params);
  muv_oper.overload_solver(HypreSolverType::HYPRE_GMRES, lin_params);
  muv_oper.overload_preconditioner(HyprePreconditionerType::HYPRE_ILU);

  std::vector<SPA*> mug_spatials{&spatial_mug};
  OPE mug_oper(mug_spatials, {"Fourier", "PhaseChangeSource", "MassSource"},
               TimeScheme::EulerImplicit, "HeatTimeDerivative");
  mug_oper.overload_nl_solver(NLSolverType::NEWTON, newton_params);
  mug_oper.overload_solver(HypreSolverType::HYPRE_GMRES, lin_params);
  mug_oper.overload_preconditioner(HyprePreconditionerType::HYPRE_ILU);

  // ####################
  //     problems      //
  // ####################
  // One coefficient block per order parameter (NB_GRAINS grains + the bubble).
  PB op_pb("OrderParameters", op_oper, op_vars, std::vector<Coefficients>(NB_OPS, op_coef), v_pst,
           muv_vars, mug_vars);
  PB muv_pb("MuVac", muv_oper, muv_vars, {muv_coef}, v_pst, op_vars);
  PB mug_pb("MuGas", mug_oper, mug_vars, {mug_coef}, v_pst, op_vars);

  mug_pb.set_vtk_coefficients({switch_mat, switch_bbl});

  // AMR on the order parameters (grains + bubbles), driven by the Kelly error estimator of the
  // diffusion problem on each field, maximized over the fields. As in test1.
  mfem::ConstantCoefficient amr_coef_ac{1.0};
  mfem::DiffusionIntegrator amr_integ_ac{amr_coef_ac};
  SlothErrorEstimators estimator_ac(ErrorEstimatorType::KELLY, &amr_integ_ac);

  MultiVariableMaxAMR<VARS> amr_ac(*spatial.get_mesh(), spatial.is_nc_simplices());

  auto amr_params = Parameters(Parameter("max_elem_error", p.error_el_amr),
                               Parameter("amr_max_level", p.max_level_amr),
                               Parameter("nc_limit", 0), Parameter("max_preref_cycles", 3));

  amr_ac.SetCriteria(/*estimator*/ &estimator_ac, amr_params);
  if (p.control_amr) {
    op_pb.set_amr(&amr_ac);
  }

  auto cc = Coupling("Polycrystal intergranular fission gas bubbles", op_pb, muv_pb, mug_pb);

  // ###########################################
  //            Time-integration              //
  // ###########################################
  const double dt_ratio = p.control_dt_ratio;
  const double dt_min = p.control_dt_min;
  const double dt_max = p.control_dt_max;
  const double dt_switch = p.control_dt_switch;
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
