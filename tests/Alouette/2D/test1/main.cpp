/**
 * @file main.cpp
 * @author Tommaso Barani (tommaso.barani@cea.fr)
 * @brief S0 coupled SLOTH/alouette test: HeatTransfer/2D/test1 with the R2
 *        multiscale conductivity provider.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright CEA (C) 2026
 *
 * This file is part of SLOTH.
 *
 * SLOTH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SLOTH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * WHAT THIS TEST IS. The standalone `HeatTransfer/2D/test1` problem, run
 * through the REAL `Coupling`/`TimeDiscretization` loop with one extra member
 * in front: a `Property_problem` whose property is the S0 multiscale manager.
 * The heat problem's mesh, boundary conditions, source, coefficients, scheme,
 * PST parameters and time loop are copied verbatim, so the coupled run must
 * write a byte-identical `Saves/Heat` tree. That identity is the point: the
 * R2 provider changes the CODE PATH of the conductivity (element lookup
 * instead of the state-only callback) and must change no number. The
 * `MultiscaleProviderProbe` additionally proves the provider was actually
 * visited on every element with one constant value (a path that is declared
 * but never reached is not tested).
 *
 * WHY THE MANAGER RUNS FIRST IN THE COUPLING. `Coupling::execute` runs its
 * members in order and calls `Problem::update()` after each, so the manager
 * publishes `lambda_bar` into its live grid function before the heat problem
 * assembles. That is the whole S0 data flow: manager table -> published
 * variable -> R2 provider -> conductivity at the element.
 *
 * THE PST-LESS-FIRST-MEMBER TRAP, stated where it bites. The manager has no
 * PST (it must write nothing to `Saves`, or the identity gate would compare
 * apples with oranges). `Coupling::need_to_be_saved()` and
 * `Coupling::get_shared_dc()` dereference the FIRST member's PST without a
 * guard. The only caller is `TimeDiscretization::save_vtk_unified()`, which
 * returns before that dereference when `vtk_unified` is false, and false is
 * the default and is what this test uses. If this test is ever switched to a
 * unified VTK save, the manager MUST get a PST or the coupling order must
 * change: an empty `std::optional` dereference is UB, not an exception.
 *
 * THE MFEM-FIRST CONTRACT. MFEM is built WITHOUT exceptions and
 * `Mpi::Init` starts with `MFEM_VERIFY(!IsInitialized(), "MPI already
 * initialized!")`, which aborts. The host must therefore initialise MPI (and
 * MFEM) BEFORE `alouette_micro_init`. `--wrong-order` exercises the refusal
 * on purpose: it calls `alouette_micro_init` first (alouette's `Runtime::init`
 * does `MPI_Init` when MPI is not up and owns it), then `mfem::Mpi::Init`
 * aborts. The mode exists so the abort path is EVIDENCE, not an assertion in a
 * document.
 *
 * TEARDOWN ORDER. Handles (none at S0) -> `alouette_micro_finalize` -> the
 * host's `MPI_Finalize`. `alouette_micro_finalize` tears down only what
 * alouette owns; with MPI already up at `alouette_micro_init` it owns Kokkos
 * alone, so the host's communicator survives.
 */
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Coefficients/MultiscaleCoefficient.hpp"
#include "MultiscaleManager.hpp"
#include "Sloth/sloth.hpp"
#include "Sloth/tests.hpp"
#include "alouette/coupling/alouette_micro.h"

namespace {

/// @brief The MPI thread level as its MPI name (for the report line).
const char* thread_level_name(int level) {
  switch (level) {
    case MPI_THREAD_SINGLE:
      return "MPI_THREAD_SINGLE";
    case MPI_THREAD_FUNNELED:
      return "MPI_THREAD_FUNNELED";
    case MPI_THREAD_SERIALIZED:
      return "MPI_THREAD_SERIALIZED";
    case MPI_THREAD_MULTIPLE:
      return "MPI_THREAD_MULTIPLE";
    default:
      return "unknown";
  }
}

}  // namespace

///---------------
/// Main program
///---------------
int main(int argc, char* argv[]) {
  //---------------------------------------
  // Test options (parsed before MPI: the wrong-order mode must run before it)
  //---------------------------------------
  double conductivity_value = 2.0;
  bool wrong_order = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--wrong-order") {
      wrong_order = true;
    } else if (argument == "--conductivity") {
      MFEM_VERIFY(i + 1 < argc, "Alouette2Dtest1: --conductivity expects a value.");
      conductivity_value = std::stod(argv[++i]);
    } else {
      std::cerr << "Alouette2Dtest1: unknown argument '" << argument << "'\n";
      return 2;
    }
  }
  MFEM_VERIFY(std::isfinite(conductivity_value) && conductivity_value > 0.0,
              "Alouette2Dtest1: the conductivity must be finite and positive, got "
                  << conductivity_value << ".");

  if (wrong_order) {
    // The negative control: alouette FIRST, MFEM second. `Runtime::init`
    // initialises MPI itself here and owns it; the following `Mpi::Init` must
    // abort on `MFEM_VERIFY(!IsInitialized(), ...)`.
    std::cout << "[wrong-order] calling alouette_micro_init() before mfem::Mpi::Init()"
              << std::endl;
    const int status = alouette_micro_init();
    MFEM_VERIFY(status == AL_MICRO_OK,
                "[wrong-order] alouette_micro_init() failed with status " << status << ".");
    std::cout << "[wrong-order] alouette_micro_init() succeeded (version "
              << alouette_micro_version() << "); expecting the MFEM abort" << std::endl;
  }

  //---------------------------------------
  // Initialize MPI and HYPRE
  //---------------------------------------
  setVerbosity(Verbosity::Debug);
  // The thread contract of the coupling is single-threaded, so ask for
  // MPI_THREAD_SINGLE and record what the runtime actually provides.
  int provided = -1;
  mfem::Mpi::Init(argc, argv, MPI_THREAD_SINGLE, &provided);
  if (wrong_order) {
    // Reaching this line means MFEM did not enforce the contract: report it
    // with a distinguishable status instead of running a meaningless case.
    std::cerr << "[wrong-order] mfem::Mpi::Init() did NOT abort; the MFEM-first contract "
                 "is not enforced"
              << std::endl;
    return 3;
  }
  int queried = -1;
  const int query_status = MPI_Query_thread(&queried);
  MFEM_VERIFY(query_status == MPI_SUCCESS, "MPI_Query_thread failed.");
  mfem::Hypre::Init();

  const int rank = mfem::Mpi::WorldRank();
  if (rank == 0) {
    std::cout << "MPI thread level: requested=MPI_THREAD_SINGLE provided=" << provided << " ("
              << thread_level_name(provided) << ") queried=" << queried << " ("
              << thread_level_name(queried) << ")" << std::endl;
  }

  //---------------------------------------
  // alouette micro runtime: MPI (and MFEM) are up, so alouette owns Kokkos only
  //---------------------------------------
  const int micro_status = alouette_micro_init();
  MFEM_VERIFY(micro_status == AL_MICRO_OK,
              "alouette_micro_init() failed with status " << micro_status << ".");
  if (rank == 0) {
    std::cout << "alouette_micro_init OK, ABI " << ALOUETTE_MICRO_ABI_VERSION << ", version "
              << alouette_micro_version() << std::endl;
  }
  //
  //---------------------------------------
  // Profiling
  Profiling::getInstance().enable();
  //---------------------------------------
  /////////////////////////
  using namespace Sloth2D;

  // ###########################################
  // ###########################################
  //         Spatial Discretization           //
  // ###########################################
  // ###########################################
  // ##############################
  //           Meshing           //
  // ##############################
  auto refinement_level = 0;
  SPA spatial("GMSH", 1, refinement_level, "camembert2D.msh", false);

  // ##############################
  //     Boundary conditions     //
  // ##############################
  auto Tboundaries = {Boundary("lower", 0, "Neumann"), Boundary("external", 2, "Dirichlet", 750.),
                      Boundary("upper", 1, "Neumann")};
  auto Tbcs = BCS(&spatial, Tboundaries);

  // ###########################################
  // ###########################################
  //      Multiscale manager (null micro)      //
  // ###########################################
  // ###########################################
  // The manager owns the material-point table and publishes its constant fold
  // into its own output variable. The variable is the one channel that
  // survives the property problem's construction: `Parameters` carries
  // primitives only, so the provider cannot receive the table directly.
  auto manager_parameters =
      Parameters(Parameter("description", "S0 multiscale manager (null micro)"),
                 Parameter("n_material_points", 4), Parameter("lambda", conductivity_value));
  VARS manager_vars(VAR(&spatial, Tbcs, MultiscaleManager::kOutputName, Glossary::Conductivity, 2,
                        conductivity_value));

  // The probe counts provider calls, the elements it visited and the value
  // range it returned. It is the evidence that the R2 path ran.
  MultiscaleProviderProbe probe;
  probe.reset(spatial.get_mesh()->GetNE());
  // The provider reads the LIVE manager variable (`manager_vars[0]`, the
  // container's own copy -- `Variables` stores by value), which
  // `Problem::update()` refreshes after each manager step.
  auto multiscale_provider = make_multiscale_provider(manager_vars[0], probe);
  auto multiscale_coefficient =
      std::make_shared<MultiscaleCoefficient>(std::move(multiscale_provider));

  // ###########################################
  // ###########################################
  //            Physical models               //
  // ###########################################
  // ###########################################
  // ####################
  //     parameters    //
  // ####################
  // Heat
  const auto& rho(1.e3);
  const auto& cp(10.);
  Coefficient density(Glossary::Concentration, rho);
  Coefficient heat_capacity(Glossary::Cp, cp);
  // Scheme::Explicit on purpose: a coefficient that must see the element is
  // never evaluable through the state-only `compute()` path, and Explicit
  // routes a missed branch to `FunctionCoefficient::F()` (which aborts) rather
  // than to a silently wrong scalar. The integrator's two conductivity call
  // sites branch to `eval_at(Tr)` through `as_multiscale()`.
  Coefficient conductivity(Glossary::Conductivity, Scheme::Explicit, multiscale_coefficient);

  // ############################
  //     variables IC + SRC    //
  // ###########################
  const auto& pellet_radius = 0.00465;
  // Heat
  auto heat_vars = VARS(VAR(&spatial, Tbcs, "T", Glossary::Temperature, 2, 750.));
  auto pl = 4.e4;
  auto src_func = std::function<double(const mfem::Vector&, double)>(
      [pl, pellet_radius]([[maybe_unused]] const mfem::Vector& vcoord,
                          [[maybe_unused]] double time) {
        const auto func = pl / (M_PI * pellet_radius * pellet_radius);

        return func;
      });

  // ###########################################
  // ###########################################
  //      Post-processing                     //
  // ###########################################
  // ###########################################
  // Unchanged from the standalone test, so the `Saves/Heat` tree is the
  // reference the coupled run is compared against byte for byte.
  const std::string& main_folder_path = "Saves";
  const auto& level_of_detail = 1;
  const auto& frequency = 1;
  // Heat
  const std::string& calculation_path = "Heat";
  auto p_pst = Parameters(
      Parameter("main_folder_path", main_folder_path),
      Parameter("calculation_path", calculation_path), Parameter("frequency", frequency),
      Parameter("level_of_detail", level_of_detail), Parameter("enable_compute_energies", false));
  auto pst = PST(&spatial, p_pst);

  // ####################
  //     Problems      //
  // ####################

  // Multiscale manager: PST-less on purpose (it writes nothing to `Saves`).
  // `Property_problem` runs it once per macro step, before the heat problem,
  // and `Coupling::execute` updates the manager variable before the heat
  // assembly.
  Property_problem<MultiscaleManager, VARS, PST> manager_pb("MultiscaleManager", manager_parameters,
                                                            manager_vars);

  // Heat: identical to the standalone test, except for `conductivity`.
  Coefficients coef_pb(density, heat_capacity, conductivity);
  std::vector<AnalyticalFunctions<DIM> > src_term;
  src_term.emplace_back(AnalyticalFunctions<DIM>(src_func));
  std::vector<SPA*> spatials{&spatial};
  TransientOPE oper(spatials, {"Fourier"}, TimeScheme::EulerImplicit, "HeatTimeDerivative",
                    src_term);

  oper.overload_nl_solver(NLSolverType::NEWTON,
                          Parameters(Parameter("description", "Newton solver "),
                                     Parameter("print_level", 1), Parameter("abs_tol", 1.e-10)));
  TransientPB Heat_pb("Heat", oper, heat_vars, {coef_pb}, pst);

  // Coupling: the manager FIRST, so it publishes before the heat problem runs.
  auto cc = Coupling("Heat transfer with multiscale manager", manager_pb, Heat_pb);
  // ###########################################
  // ###########################################
  //            Time-integration              //
  // ###########################################
  // ###########################################
  const auto& t_initial = 0.0;
  const auto& t_final = 1.;
  const auto& dt = 0.1;
  auto time_params = Parameters(Parameter("initial_time", t_initial),
                                Parameter("final_time", t_final), Parameter("time_step", dt));
  auto time = TimeDiscretization(time_params, cc);

  time.solve();

  //---------------------------------------
  // Provider evidence: the R2 path must have run, on every element, with the
  // published constant. A path that is declared but not visited is not tested.
  //---------------------------------------
  const long long local_calls = probe.call_count;
  const long long local_visits = probe.element_visit_count;
  const long long local_elements = static_cast<long long>(spatial.get_mesh()->GetNE());
  long long global_calls = 0;
  long long global_visits = 0;
  long long global_elements = 0;
  double global_min = probe.min_value;
  double global_max = probe.max_value;
  MPI_Allreduce(&local_calls, &global_calls, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_visits, &global_visits, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_elements, &global_elements, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&probe.min_value, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&probe.max_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "MULTISCALE PROVIDER: calls=" << global_calls << " elements_visited="
              << global_visits << "/" << global_elements << " value=[" << global_min << ", "
              << global_max << "] expected=" << conductivity_value << std::endl;
  }
  MFEM_VERIFY(global_calls > 0,
              "The R2 multiscale provider was never called: the conductivity branch did not run.");
  MFEM_VERIFY(global_visits == global_elements,
              "The R2 multiscale provider visited " << global_visits << " of " << global_elements
                                                    << " elements: the element lookup is incomplete.");
  MFEM_VERIFY(global_min == global_max,
              "The published field is constant, so every provider value must be identical; got ["
                  << global_min << ", " << global_max << "].");
  const double relative_gap =
      std::abs(global_min - conductivity_value) / std::abs(conductivity_value);
  MFEM_VERIFY(relative_gap <= 1.0e-15,
              "The provider returned " << global_min << " where the manager published "
                                       << conductivity_value << " (relative gap "
                                       << relative_gap << ").");

  //---------------------------------------
  // Profiling stop
  //---------------------------------------
  Profiling::getInstance().print();
  //---------------------------------------
  // Teardown: no alouette handle exists at S0, then the micro runtime, then
  // the host's MPI (alouette owns only Kokkos here and never the communicator).
  //---------------------------------------
  const int finalize_status = alouette_micro_finalize();
  MFEM_VERIFY(finalize_status == AL_MICRO_OK,
              "alouette_micro_finalize() failed with status " << finalize_status << ".");
  if (rank == 0) {
    std::cout << "alouette_micro_finalize OK" << std::endl;
  }
  //---------------------------------------
  // Finalize MPI
  //---------------------------------------
  MPI_Finalize();
  //---------------------------------------
  return 0;
}
