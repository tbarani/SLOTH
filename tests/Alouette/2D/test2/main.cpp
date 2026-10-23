/**
 * @file main.cpp
 * @author Tommaso Barani (tommaso.barani@cea.fr)
 * @brief S1 offline SLOTH/alouette test: the layered macro problem run three
 *        ways -- dump, offline table, constant lambda* -- with exact analytic
 *        anchors.
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
 * WHAT THIS TEST IS. The S1 stage of the coupling (docs/DESIGN_SLOTH_COUPLING.md
 * 3.10 and 6): the OFFLINE, file-exchanged variant. This program is the macro
 * side. It never links alouette and never shares a runtime with it -- it writes
 * a macro dump, an external `alouette_offline_batch` process turns it into a
 * response table, and this program reads the table back and runs the macro
 * problem with the lambda* it found. The chain script
 * (`run_layered_macro.sh`) drives the three runs and compares them.
 *
 * WHY THE MACRO PROBLEM IS THE LAYERED ONE. The design note's 4.1 redefined the
 * first analytic gate: a laminate is maximally anisotropic, so the scalar
 * projection the v1 macro operator uses matches NEITHER the series nor the
 * parallel response of a single layered microstructure. The adopted repair
 * layers HOMOGENEOUS cells instead: every material point (macro element) is a
 * uniform-lambda microstructure, whose lambda* is exactly lambda*I and is
 * therefore representable by the scalar R2 provider. The gate then tests the
 * macro assembly and the coupling plumbing against the exact series/parallel
 * COMPOSITE response, and the micro laminate tensor stays pinned where it is
 * already exact (`m4.laminate_2d`, re-checked through the coupling by
 * `t.micro_laminate`).
 *
 * WHY THE STEADY PROFILE ALONE IS NOT THE ANCHOR. A source-free strip with
 * Dirichlet ends has a LINEAR steady profile whatever the conductivity is: the
 * temperature probes cannot distinguish lambda = 2 from lambda = 40. The note
 * says so and requires the FLUX and a conductivity-sensitive TRANSIENT instead.
 * Hence three cases:
 *
 *   series     layers along x, lambda 40 below and 2 above, gradient along x
 *              -> the HARMONIC (series) mean, lambda_eff = 3.8095...
 *   parallel   layers along y, lambda 2 below and 40 above, gradient along x
 *              -> the ARITHMETIC (parallel) mean, lambda_eff = 21
 *   transient  layers along y, the sine IC of the exact solution
 *              700 + 100x + 20 sin(2 pi x) exp(-lambda_layer (2 pi)^2 t)
 *              -> the DECAY RATE per layer is the conductivity, and it is the
 *              only anchor a wrong lambda cannot fake: the two layers decay at
 *              2 and at 40, a factor 20 apart.
 *
 * The two steady cases differ ONLY in which layer sits where, so a swapped
 * layer assignment flips 3.81 into 21 (and the parallel case into 3.81): the
 * pair pins the assignment, not just the arithmetic.
 *
 * THE THREE MODES, AND WHAT EACH ONE PROVES.
 *
 *   dump      runs the macro loop with the analytically known layer
 *             conductivities and writes the macro dump (plus a summary). The
 *             dump's loads do not depend on the answer for the STEADY cases,
 *             and for the transient case the layer values are the case's own
 *             input data, so nothing circular is being assumed: the offline
 *             table's lambda* is load- and time-independent by construction
 *             (the batch solves one steady cell per material point).
 *   constant  runs the same macro loop with the layer values applied directly
 *             through the R2 provider -- the S0 path, no file involved.
 *   table     runs it with lambda* read from alouette's response table.
 *
 * `constant` and `table` must produce the SAME summary: for a homogeneous cell
 * the M4 cell's answer is lambda* = lambda exactly, so the offline round trip
 * may not move a digit. The chain asserts it at 1e-12 relative.
 *
 * WHY THE PER-STEP DRIVER IS NOT `TimeDiscretization`. The dump needs the macro
 * state at the START of every step, and `TimeDiscretization::solve()` owns both
 * the initialization and the step; there is no hook between them. This program
 * therefore makes exactly the calls `TimeDiscretization::solve()` makes for one
 * step, in the same order -- `Problem::initialize` once, then
 * `Problem::execute` + `Problem::update` per step -- and PROVES the equivalence
 * in dump mode: `driver_gap` is the largest difference between this driver's
 * final probes and a single long `TimeDiscretization` run over the same clock.
 * The two differ only by the last step's clock clamp (the reference accumulates
 * `dt` and clamps `final - current`), which is why the bound is 1e-9 K and not
 * zero.
 *
 * WHY lambda* IS A SCALAR HERE. The R2 provider is `double(int element)`, so
 * the macro operator takes ONE conductivity per material point. The response
 * table carries a 2x2 tensor; the reader refuses a table that is not lambda*I
 * instead of silently projecting it (OfflineTable.hpp). For S1's homogeneous
 * cells lambda* IS lambda*I, and the refusal is what keeps that honest.
 *
 * THE DUMP IS SAMPLED AT THE QUADRATURE POINTS, in the (element, qp) row-major
 * order the format fixes. A Q1 field's gradient is constant per element only
 * when the exact field is linear there; the steady cases are, the transient
 * case's sine is not, so the batch averages the quadrature-point gradients
 * arithmetically (2x2 Gauss weights are equal, so that mean is the
 * quadrature-weighted mean). The steady cases assert their dump gradients
 * against the closed-form values, which is what makes a transposed sample or a
 * swapped axis detectable at the writer.
 */
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Coefficients/MultiscaleCoefficient.hpp"
#include "OfflineTable.hpp"
#include "Sloth/sloth.hpp"
#include "Sloth/tests.hpp"

namespace {

// ---------------------------------------------------------------------------
// The case definition and its exact solution
// ---------------------------------------------------------------------------

/// The macro mesh: 20 x 2 Q1 elements on the unit square. 20 along x puts the
/// interface (x = 0.5 or y = 0.5) on a NODE line, which is what lets each
/// material point be a homogeneous layer: the batch refuses an element whose
/// quadrature points straddle the interface.
constexpr int kNx = 20;
/// The number of elements along y.
constexpr int kNy = 2;
/// The domain's extent along x.
constexpr double kLx = 1.0;
/// The domain's extent along y.
constexpr double kLy = 1.0;
/// The finite element order (Q1, the heat test's template).
constexpr int kFeOrder = 1;
/// The cold Dirichlet temperature (the left side).
constexpr double kTCold = 700.0;
/// The hot Dirichlet temperature (the right side).
constexpr double kTHot = 800.0;
/// The applied temperature difference, which is also the v1 macro load.
constexpr double kDeltaT = kTHot - kTCold;
/// The transient IC's sine amplitude (K).
constexpr double kAmplitude = 20.0;
/// pi, so the exact solution does not depend on M_PI being defined.
constexpr double kPi = 3.14159265358979323846;
/// (2 pi)^2, the decay-rate factor of the sine mode on a unit-length cell.
constexpr double kTwoPiSquared = 4.0 * kPi * kPi;
/// The density of the macro problem (kg/m^3); constant, so it only scales the
/// transient and cannot affect a steady anchor.
constexpr double kDensity = 1.0e3;
/// The heat capacity of the macro problem (J/kg/K), likewise constant.
constexpr double kHeatCapacity = 10.0;

/// The steady temperature probes' tolerance (K). The Q1 space CONTAINS the
/// exact piecewise-linear solution, so the only error is the Newton solve's.
constexpr double kSteadyAnchorTolerance = 1.0e-6;
/// The transient probes' tolerance (K): 1% of the applied difference, wide
/// enough for backward Euler's per-step gap against the exponential and tight
/// enough that a wrong decay rate (a factor 20 between the layers) cannot pass.
constexpr double kTransientAnchorTolerance = 1.0;
/// The flux anchors' relative tolerance. The FE flux is exact for these cases
/// (the volume average telescopes per layer), so this is roundoff plus the
/// solve's.
constexpr double kFluxAnchorTolerance = 1.0e-6;
/// The per-step driver's equivalence bound against the single long run (K).
constexpr double kDriverGapTolerance = 1.0e-9;
/// The tolerance on the prescribed Dirichlet values being attained (K).
constexpr double kBoundaryTolerance = 1.0e-6;

/**
 * @brief Which layered case this run is.
 */
enum class CaseKind {
  /// Layers along x, the harmonic (series) composite response.
  Series,
  /// Layers along y, the arithmetic (parallel) composite response.
  Parallel,
  /// Layers along y, the conductivity-sensitive sine decay.
  Transient
};

/**
 * @brief The case's full definition: geometry, layers, clock and IC.
 */
struct CaseSpec {
  /// Which case.
  CaseKind kind = CaseKind::Series;
  /// The case's name on the command line.
  std::string name;
  /// The layer axis: 0 = x, 1 = y. The gradient is always along x.
  int layer_axis = 0;
  /// The conductivity below the interface.
  double lambda_below = 0.0;
  /// The conductivity above the interface.
  double lambda_above = 0.0;
  /// The interface position along the layer axis.
  double interface = 0.0;
  /// The number of macro steps.
  int n_steps = 0;
  /// The macro time step.
  double dt = 0.0;
  /// Whether the IC is the exact solution at t = 0 (the transient case) or a
  /// uniform 700 K state (the steady cases, which must FIND the answer rather
  /// than start on it).
  bool exact_initial_condition = false;

  /// @brief The final time.
  /// @return `n_steps * dt`.
  double final_time() const { return static_cast<double>(n_steps) * dt; }
};

/**
 * @brief Resolve a case name.
 * @param name `series`, `parallel` or `transient`.
 * @return The case; an unknown name is refused.
 */
CaseSpec case_spec(const std::string& name) {
  CaseSpec spec;
  spec.name = name;
  if (name == "series") {
    spec.kind = CaseKind::Series;
    spec.layer_axis = 0;
    spec.lambda_below = 40.0;
    spec.lambda_above = 2.0;
    spec.interface = 0.5;
    // The steady cases must actually REACH the steady state, and their clock
    // is chosen for that, not for accuracy: dt * r_min >> 1 turns implicit
    // Euler into a steady solve, and ten steps drive the initial error below
    // machine precision. The slowest relaxation rate of the composite is
    // r_min ~ lambda* pi^2 / (rho c_p) = 3.8e-3 1/s here, so dt * r_min is
    // ~380 per step. A "physically scaled" dt (0.1 s) would leave the initial
    // 700 K state 96% undecayed after 50 steps and the anchor would fail --
    // correctly, because the field would not be steady.
    spec.n_steps = 30;
    spec.dt = 1.0e3;
  } else if (name == "parallel") {
    spec.kind = CaseKind::Parallel;
    spec.layer_axis = 1;
    spec.lambda_below = 2.0;
    spec.lambda_above = 40.0;
    spec.interface = 0.5;
    // Same argument, with r_min ~ 21 pi^2 / (rho c_p) = 2.1e-2 1/s.
    spec.n_steps = 30;
    spec.dt = 1.0e3;
  } else if (name == "transient") {
    spec.kind = CaseKind::Transient;
    spec.layer_axis = 1;
    // HOMOGENEOUS on purpose. A layered transient has no closed form: the two
    // layers would decay at different rates, which makes the field y-dependent
    // and drives a lateral flux between them. The first revision ignored that
    // flux and called the decoupled field "exact"; it missed its own anchors
    // by 2.0 K and 1.3 K (measured). The LAYERED response is anchored by the
    // two steady cases (series 3.81, parallel 21, exact to 1e-10); this case
    // anchors the conductivity-sensitive TIME dependence: the sine mode
    // decays at `lambda (2 pi)^2 / (rho c_p)`, so t_end = 125 s is one decay
    // time (amplitude e^-1). That is where the probe is most sensitive to
    // lambda -- 3.7 K per unit lambda -- so a table carrying the harmonic
    // 3.81 instead of 2 misses by 4.4 K, and the arithmetic 21 by 7.4 K,
    // both far outside the 1 K anchor. Backward Euler at dt = 0.5 is 0.015 K
    // from the exact amplitude (measured).
    spec.lambda_below = 2.0;
    spec.lambda_above = 2.0;
    spec.interface = 0.5;
    spec.n_steps = 250;
    spec.dt = 0.5;
    spec.exact_initial_condition = true;
  } else {
    throw std::runtime_error("unknown case '" + name +
                             "' (expected series, parallel or transient)");
  }
  return spec;
}

/**
 * @brief The layer conductivity at a coordinate along the layer axis.
 * @param spec      The case.
 * @param coordinate The coordinate along the layer axis.
 * @return The conductivity of the layer that coordinate sits in.
 */
double layer_lambda(const CaseSpec& spec, double coordinate) {
  return coordinate <= spec.interface ? spec.lambda_below : spec.lambda_above;
}

/**
 * @brief The exact temperature of the case.
 *
 * The three closed forms:
 * - series: the 1D series solution, `T = T_cold + q x / lambda(x)` with
 *   `q = dT / (d_below/lambda_below + d_above/lambda_above)`;
 * - parallel: `T = T_cold + dT x` (the layers are in parallel with the
 *   gradient, so the field is the same linear ramp in both);
 * - transient: `T_cold + dT x + A sin(2 pi x) exp(-D_layer (2 pi)^2 t)` with
 *   `D_layer = lambda_layer / (rho c_p)`, which satisfies the heat equation in
 *   each layer and is continuous across the interface (it has no y dependence,
 *   so the normal flux is zero on both sides).
 *
 * @param spec The case.
 * @param x    The x coordinate.
 * @param y    The y coordinate.
 * @param t    The time.
 * @return The exact temperature (K).
 */
double exact_temperature(const CaseSpec& spec, double x, double y, double t) {
  const double coordinate = spec.layer_axis == 0 ? x : y;
  const double lambda = layer_lambda(spec, coordinate);
  switch (spec.kind) {
    case CaseKind::Series: {
      const double series_resistance = spec.interface / spec.lambda_below +
                                       (kLx - spec.interface) / spec.lambda_above;
      const double flux = kDeltaT / series_resistance;
      if (x <= spec.interface) {
        return kTCold + flux * x / spec.lambda_below;
      }
      return kTCold + flux * spec.interface / spec.lambda_below +
             flux * (x - spec.interface) / spec.lambda_above;
    }
    case CaseKind::Parallel:
      return kTCold + kDeltaT * x;
    case CaseKind::Transient:
      // The decay rate is the DIFFUSIVITY lambda / (rho c_p) times (2 pi)^2,
      // not lambda (2 pi)^2: the heat equation is
      // rho c_p dT/dt = div(lambda grad T). The first revision dropped the
      // 1/(rho c_p) and decayed 1e4 times too fast, which made every probe
      // anchor meaningless rather than wrong-looking.
      return kTCold + kDeltaT * x +
             kAmplitude * std::sin(2.0 * kPi * x) *
                 std::exp(-(lambda / (kDensity * kHeatCapacity)) * kTwoPiSquared * t);
  }
  return 0.0;
}

/**
 * @brief The exact effective conductivity of the case.
 *
 * Series (layers along x, gradient along x): the harmonic mean weighted by the
 * layer thicknesses. Parallel (layers along y, gradient along x): the
 * arithmetic mean, because the layers are in parallel with the gradient.
 *
 * @param spec The case.
 * @return The exact lambda*_xx.
 */
double exact_lambda_eff(const CaseSpec& spec) {
  const double below = spec.interface;
  const double above = kLx - spec.interface;
  if (spec.kind == CaseKind::Series) {
    return 1.0 / (below / spec.lambda_below + above / spec.lambda_above);
  }
  return below * spec.lambda_below + above * spec.lambda_above;
}

/**
 * @brief The exact volume-averaged current along x.
 *
 * The v1 macro convention is the M4/ABI one: `<J> = lambda* . E` with the
 * POSITIVE conductivity, so `<J_x> = lambda*_xx * dT/L`. The physical Fourier
 * flux carries the opposite sign; the gate reports the convention the ABI
 * reports.
 *
 * @param spec The case.
 * @return The exact mean current along x.
 */
double exact_mean_current_x(const CaseSpec& spec) {
  return exact_lambda_eff(spec) * kDeltaT / kLx;
}

/**
 * @brief The analytic lambda* of every material point: the layer its centre
 *        sits in.
 *
 * The macro mesh resolves the layer boundary (the interface is a node line), so
 * every element lies inside one layer, and a homogeneous cell's lambda* is that
 * layer's value EXACTLY. That is the reference the offline table is checked
 * against, and it is also the conductivity the dump run itself used -- which is
 * what lets the table's currents be replayed step by step.
 *
 * @param spec The case.
 * @param mesh The macro mesh.
 * @return One value per element.
 */
std::vector<double> analytic_lambda_star(const CaseSpec& spec, mfem::ParMesh& mesh) {
  std::vector<double> values(static_cast<std::size_t>(mesh.GetNE()), 0.0);
  mfem::Vector centre;
  for (int e = 0; e < mesh.GetNE(); ++e) {
    mesh.GetElementCenter(e, centre);
    const double coordinate = spec.layer_axis == 0 ? centre(0) : centre(1);
    values[static_cast<std::size_t>(e)] = layer_lambda(spec, coordinate);
  }
  return values;
}

// ---------------------------------------------------------------------------
// One macro run
// ---------------------------------------------------------------------------

/**
 * @brief One quadrature-point sample of the macro state (the dump's row).
 */
struct Sample {
  /// The macro element (the material point).
  int element = 0;
  /// The quadrature point within the element.
  int qp = 0;
  /// The physical coordinates.
  double x = 0.0;
  /// The physical y coordinate.
  double y = 0.0;
  /// The physical z coordinate (0 in 2D).
  double z = 0.0;
  /// The temperature gradient, in the macro units.
  double gradient[3] = {0.0, 0.0, 0.0};
  /// The temperature at the point (K).
  double temperature = 0.0;
  /// The quadrature weight times the transformation's Jacobian determinant:
  /// the physical volume this point represents. 2x2 Gauss weights are equal,
  /// so this is also the weight the batch's arithmetic mean of the
  /// quadrature-point gradients implies.
  double weight = 0.0;
};

/**
 * @brief The macro problem for one case, with a live per-element lambda* table.
 *
 * The R2 provider reads `lambda_star_` at every assembly, so changing the
 * vector between steps is the whole "the manager republished the property"
 * mechanism of S0. Here it is written once, before the first step: S1's
 * lambda* is frozen.
 *
 * The problem is built WITHOUT a PST on purpose. `TimeDiscretization` and
 * `Coupling::initialize` save VTK and CSV frames through the PST when one
 * exists, and this test's outputs are the dump, the table and the summary --
 * a `Saves/` tree would be noise that the chain's working directory does not
 * need. Without a PST every save is a documented no-op.
 */
class MacroRun {
 public:
  /**
   * @brief Build the mesh, the boundary conditions, the variables and the
   *        problem for a case.
   *
   * @param spec        The case.
   * @param lambda_star The per-element lambda* table, borrowed: the provider
   *                    reads it live, so it must outlive this object.
   */
  MacroRun(const CaseSpec& spec, std::vector<double>& lambda_star)
      : spec_(spec),
        spatial_("InlineSquareWithQuadrangles", kFeOrder, 0,
                 std::make_tuple(kNx, kNy, kLx, kLy)),
        bcs_(&spatial_, Boundary("bottom", 0, "Neumann"),
             Boundary("right", 1, "Dirichlet", kTHot), Boundary("top", 2, "Neumann"),
             Boundary("left", 3, "Dirichlet", kTCold)),
        lambda_star_(lambda_star),
        multiscale_(std::make_shared<MultiscaleCoefficient>(
            [this](int element) { return element_value(element); })),
        conductivity_(Glossary::Conductivity, Scheme::Explicit, multiscale_),
        density_(Glossary::Concentration, kDensity),
        heat_capacity_(Glossary::Cp, kHeatCapacity),
        vars_(Sloth2D::VAR(&spatial_, bcs_, "T", Glossary::Temperature, 2,
                           AnalyticalFunctions<Sloth2D::DIM>(initial_condition(spec)))),
        oper_(std::vector<Sloth2D::SPA*>{&spatial_}, {"Fourier"}, TimeScheme::EulerImplicit,
              "HeatTimeDerivative") {
    // `Problem` stores its operator BY VALUE (a `Problem<OPE,...>` member is an
    // `OPE`), so a solver configured on `oper_` AFTER the problem exists is
    // configured on a copy nobody runs. Test1's order (configure, then build
    // the problem) is the only order that works; that is why `problem_` is an
    // `optional` emplaced here rather than a member-init entry.
    //
    // The Newton tolerances are SLOTH's defaults, loosened on purpose. The
    // default pair (`rel_tol = abs_tol = 1e-13`) is INCONSISTENT with the
    // inner linear solver: each Newton iteration re-solves with HypreGMRES at
    // a 1e-12 relative tolerance, so the residual cannot fall below
    // ~1e-12 * ||F(0)|| and a 1e-13 goal is unreachable. The steady cases sit
    // exactly there: after one step the field is steady to roundoff, ||F(0)||
    // is tiny, and the default run burns 100 iterations and then aborts. The
    // Newton goal must be LOOSER than the linear one, hence 1e-11; `abs_tol`
    // is the floor for the near-zero-residual steady steps. The system is
    // linear, so this is a stopping rule on a solve that is exact up to the
    // linear solver -- the analytic anchors below are what proves the answer.
    oper_.overload_nl_solver(NLSolverType::NEWTON,
                             Parameters(Parameter("description", "Newton solver "),
                                        Parameter("rel_tol", 1.0e-11),
                                        Parameter("abs_tol", 1.0e-8)));
    problem_.emplace(std::string("Heat"), oper_, vars_,
                     std::vector<Coefficients>{
                         Coefficients(density_, heat_capacity_, conductivity_)});
    MFEM_VERIFY(static_cast<int>(lambda_star_.size()) == spatial_.get_mesh()->GetNE(),
                "MacroRun: the lambda* table has " << lambda_star_.size() << " entries for "
                                                   << spatial_.get_mesh()->GetNE() << " elements.");
  }

  MacroRun(const MacroRun&) = delete;
  MacroRun& operator=(const MacroRun&) = delete;
  MacroRun(MacroRun&&) = delete;
  MacroRun& operator=(MacroRun&&) = delete;

  ~MacroRun() = default;

  /**
   * @brief Apply the Dirichlet data and build the operator's transient state.
   *
   * This is the `initialize` half of one `TimeDiscretization` step, with the
   * initial time the driver's own clock.
   */
  void initialize() { problem_->initialize(0.0, spec_.dt); }

  /**
   * @brief Take one macro step of `spec.dt`, starting at `step * dt`.
   * @param step The step index (the clock is `step * dt`).
   */
  void step(int step) {
    const double time = static_cast<double>(step) * spec_.dt;
    double next_time = time;
    problem_->execute(step + 1, next_time, time, spec_.dt);
    problem_->update();
  }

  /**
   * @brief Run the whole case through a single `TimeDiscretization` loop.
   *
   * This is the REFERENCE the per-step driver is compared against, not the
   * driver itself: it exercises the real clock bookkeeping (including its
   * last-step clamp), and `driver_gap` pins that the driver's shortcut changes
   * nothing that matters.
   */
  void run_single_loop() {
    auto coupling = Coupling("Heat transfer", *problem_);
    auto time_params = Parameters(Parameter("initial_time", 0.0),
                                  Parameter("final_time", spec_.final_time()),
                                  Parameter("time_step", spec_.dt));
    auto time = TimeDiscretization(time_params, coupling);
    time.solve();
  }

  /**
   * @brief The mesh.
   * @return The distributed mesh.
   */
  mfem::ParMesh* mesh() { return spatial_.get_mesh(); }

  /**
   * @brief The temperature field (the problem's own live grid function).
   * @return The grid function.
   */
  mfem::ParGridFunction& temperature() { return vars_[0].get_ref_gf(); }

  /**
   * @brief Sample every element's quadrature points, in (element, qp) order.
   * @return The samples.
   */
  std::vector<Sample> sample() {
    mfem::ParMesh* const mesh = spatial_.get_mesh();
    mfem::ParGridFunction& field = temperature();
    const int n_elements = mesh->GetNE();
    const int n_qp = quadrature(mesh->GetElementGeometry(0)).GetNPoints();
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>(n_elements) * static_cast<std::size_t>(n_qp));
    mfem::Vector gradient(Sloth2D::DIM);
    for (int e = 0; e < n_elements; ++e) {
      const mfem::IntegrationRule& ir = quadrature(mesh->GetElementGeometry(e));
      MFEM_VERIFY(ir.GetNPoints() == n_qp,
                  "sample(): element " << e << " has " << ir.GetNPoints() << " quadrature points, "
                                       << "element 0 has " << n_qp
                                       << "; the dump format has ONE n_qp for all elements.");
      mfem::ElementTransformation* const transformation = mesh->GetElementTransformation(e);
      mfem::Vector point(Sloth2D::DIM);
      for (int q = 0; q < n_qp; ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        // `GetGradient` documents that the integration point must already be
        // set on the transformation; `GetValue` documents that it does not
        // reset it. Both are therefore driven from this one SetIntPoint, and
        // the Jacobian is read here, while the point is the current one.
        transformation->SetIntPoint(&ip);
        const double weight = ip.weight * transformation->Weight();
        field.GetGradient(*transformation, gradient);
        transformation->Transform(ip, point);
        Sample sample;
        sample.element = e;
        sample.qp = q;
        sample.x = point(0);
        sample.y = point(1);
        sample.z = 0.0;
        sample.gradient[0] = gradient(0);
        sample.gradient[1] = gradient(1);
        sample.gradient[2] = 0.0;
        sample.temperature = field.GetValue(*transformation, ip);
        sample.weight = weight;
        samples.push_back(sample);
      }
    }
    return samples;
  }

  /**
   * @brief The case this run belongs to.
   * @return The case.
   */
  const CaseSpec& spec() const { return spec_; }

  /**
   * @brief The element a point falls in, for a probe.
   * @param x The x coordinate.
   * @param y The y coordinate.
   * @return The element index.
   */
  int find_element(double x, double y) {
    mfem::DenseMatrix points(Sloth2D::DIM, 1);
    points(0, 0) = x;
    points(1, 0) = y;
    mfem::Array<int> element_ids;
    mfem::Array<mfem::IntegrationPoint> points_ref;
    // `warn = false`: the refusal below is the message, and a warning per rank
    // in a chain log is noise.
    const int found = spatial_.get_mesh()->FindPoints(points, element_ids, points_ref, false);
    MFEM_VERIFY(found == 1, "probe (" << x << ", " << y << ") was not located in exactly one "
                                      << "element (found " << found << ").");
    return element_ids[0];
  }

  /**
   * @brief The temperature at a point.
   * @param x The x coordinate.
   * @param y The y coordinate.
   * @return The temperature (K).
   */
  double probe(double x, double y) {
    mfem::DenseMatrix points(Sloth2D::DIM, 1);
    points(0, 0) = x;
    points(1, 0) = y;
    mfem::Array<int> element_ids;
    mfem::Array<mfem::IntegrationPoint> points_ref;
    const int found = spatial_.get_mesh()->FindPoints(points, element_ids, points_ref, false);
    MFEM_VERIFY(found == 1, "probe (" << x << ", " << y << ") was not located in exactly one "
                                      << "element (found " << found << ").");
    mfem::ElementTransformation* const transformation =
        spatial_.get_mesh()->GetElementTransformation(element_ids[0]);
    transformation->SetIntPoint(&points_ref[0]);
    return temperature().GetValue(*transformation, points_ref[0]);
  }

  /**
   * @brief The number of elements (the number of material points).
   * @return The element count.
   */
  int n_elements() { return spatial_.get_mesh()->GetNE(); }

  /**
   * @brief The number of quadrature points per element (the dump's `n_qp`).
   *
   * The dump format carries ONE count for all elements, so a mesh whose
   * elements disagree is refused at sampling time rather than written.
   *
   * @return The count of element 0's rule.
   */
  int quadrature_points_per_element() {
    return quadrature(spatial_.get_mesh()->GetElementGeometry(0)).GetNPoints();
  }

  /**
   * @brief The lambda* the provider would return for an element.
   * @param element The element.
   * @return The value.
   */
  double element_value(int element) const {
    MFEM_VERIFY(element >= 0 && element < static_cast<int>(lambda_star_.size()),
                "the R2 provider was asked for element " << element << " of "
                                                         << lambda_star_.size() << ".");
    return lambda_star_[static_cast<std::size_t>(element)];
  }

 private:
  /**
   * @brief The 2x2 Gauss rule of the element geometry.
   *
   * `2 * kFeOrder` is the polynomial order the rule integrates exactly: a Q1
   * field's gradient is degree 1 per direction, so 2x2 Gauss is exact and its
   * four weights are equal -- which is what makes the batch's arithmetic mean
   * of the quadrature gradients the quadrature-weighted mean.
   *
   * @param geometry The element geometry.
   * @return The rule.
   */
  static const mfem::IntegrationRule& quadrature(mfem::Geometry::Type geometry) {
    return mfem::IntRules.Get(geometry, 2 * kFeOrder);
  }

  /**
   * @brief The case's initial condition.
   * @param spec The case.
   * @return The IC as an analytical function.
   */
  static AnalyticalFunctions<Sloth2D::DIM> initial_condition(const CaseSpec& spec) {
    if (spec.exact_initial_condition) {
      return AnalyticalFunctions<Sloth2D::DIM>(
          std::function<double(const mfem::Vector&, double)>([](const mfem::Vector& x, double) {
            return kTCold + kDeltaT * x(0) + kAmplitude * std::sin(2.0 * kPi * x(0));
          }));
    }
    return AnalyticalFunctions<Sloth2D::DIM>(
        std::function<double(const mfem::Vector&, double)>(
            [](const mfem::Vector&, double) { return kTCold; }));
  }

  /// The case.
  const CaseSpec spec_;
  /// The mesh and its finite element space.
  Sloth2D::SPA spatial_;
  /// The four boundary conditions.
  Sloth2D::BCS bcs_;
  /// The per-element lambda* table, borrowed (the provider reads it live).
  std::vector<double>& lambda_star_;
  /// The R2 provider object; the Coefficient below shares it by pointer.
  std::shared_ptr<MultiscaleCoefficient> multiscale_;
  /// The conductivity the Fourier integrator reads.
  Coefficient conductivity_;
  /// The density.
  Coefficient density_;
  /// The heat capacity.
  Coefficient heat_capacity_;
  /// The temperature variable.
  Sloth2D::VARS vars_;
  /// The transient heat operator.
  Sloth2D::TransientOPE oper_;
  /// The heat problem, PST-less.
  std::optional<Sloth2D::TransientPB> problem_;
};

// ---------------------------------------------------------------------------
// The dump, the summary and the comparison
// ---------------------------------------------------------------------------

/**
 * @brief Write the macro dump in the S1 format (docs 3.10).
 *
 * The row order is CHECKED, not assumed: the format fixes (step, element, qp)
 * row-major precisely so that a missing or repeated row is detectable, and a
 * writer that emitted rows in another order would produce a file the batch
 * refuses (or, worse, one it reads with the gradients attached to the wrong
 * material point).
 *
 * @param path              The output file.
 * @param spec              The case (the clock).
 * @param n_material_points The number of macro elements.
 * @param n_qp              The quadrature points per element.
 * @param samples           One entry per step, each in (element, qp) order.
 */
void write_dump(const std::string& path, const CaseSpec& spec, int n_material_points, int n_qp,
                const std::vector<std::vector<Sample>>& samples) {
  MFEM_VERIFY(samples.size() == static_cast<std::size_t>(spec.n_steps),
              "write_dump: " << samples.size() << " sampled steps for " << spec.n_steps
                             << " macro steps.");
  MFEM_VERIFY(n_material_points > 0 && n_qp > 0, "write_dump: empty sample grid.");
  const std::size_t rows_per_step =
      static_cast<std::size_t>(n_material_points) * static_cast<std::size_t>(n_qp);
  for (const std::vector<Sample>& rows : samples) {
    MFEM_VERIFY(rows.size() == rows_per_step,
                "write_dump: a step carries " << rows.size() << " rows for "
                                              << n_material_points << " material points times "
                                              << n_qp << " quadrature points.");
    for (int e = 0; e < n_material_points; ++e) {
      for (int q = 0; q < n_qp; ++q) {
        const Sample& sample =
            rows[static_cast<std::size_t>(e) * static_cast<std::size_t>(n_qp) +
                 static_cast<std::size_t>(q)];
        MFEM_VERIFY(sample.element == e && sample.qp == q,
                    "write_dump: row " << (static_cast<std::size_t>(e) * n_qp + q)
                                       << " carries (element " << sample.element << ", qp "
                                       << sample.qp << ") where the format fixes (" << e << ", "
                                       << q << ").");
      }
    }
  }
  std::ofstream out(path);
  MFEM_VERIFY(static_cast<bool>(out), "cannot open the dump '" << path << "' for writing.");
  out.precision(17);
  out << "alouette-macro-dump 1\n";
  out << "dim 2\n";
  out << "extent " << kLx << ' ' << kLy << ' ' << 1.0 << '\n';
  out << "n_steps " << spec.n_steps << '\n';
  out << "n_material_points " << n_material_points << '\n';
  out << "n_qp " << n_qp << '\n';
  for (int s = 0; s < spec.n_steps; ++s) {
    out << "step " << s << " time " << static_cast<double>(s) * spec.dt << " dt " << spec.dt
        << '\n';
    for (const Sample& sample : samples[static_cast<std::size_t>(s)]) {
      out << "qp " << sample.element << ' ' << sample.qp << ' ' << sample.x << ' ' << sample.y
          << ' ' << sample.z << ' ' << sample.gradient[0] << ' ' << sample.gradient[1] << ' '
          << sample.gradient[2] << ' ' << sample.temperature << '\n';
    }
  }
  out.flush();
  MFEM_VERIFY(static_cast<bool>(out), "writing the dump '" << path << "' failed.");
}

/**
 * @brief The summary's numbers, as named fields.
 *
 * Named fields rather than a vector indexed by position: the writer's order
 * and the assertion's field must be the same object, or a reordering of one
 * silently changes the meaning of the other.
 */
struct Summary {
  /// The volume-averaged current along x (the ABI's positive-conductivity
  /// convention, `<J> = lambda* . E`).
  double mean_current_x = 0.0;
  /// The volume-averaged current along y (zero by symmetry in these cases).
  double mean_current_y = 0.0;
  /// `mean_current_x * L / dT`.
  double lambda_eff = 0.0;
  /// The cold Dirichlet data (the case's input, reported so the summary is
  /// self-contained).
  double T_left = kTCold;
  /// The hot Dirichlet data (likewise an input).
  double T_right = kTHot;
  /// The temperature at (0.25, 0.25) (K).
  double T_probe_quarter = 0.0;
  /// The temperature at (0.5, 0.5) (K).
  double T_probe_centre = 0.0;
  /// The temperature at (0.25, 0.75) (K).
  double T_probe_quarter_top = 0.0;
  /// The largest probe-vs-exact gap at the final time (K).
  double anchor_error = 0.0;
  /// The per-step driver's gap against a single long run (K); dump mode only.
  double driver_gap = 0.0;

  /**
   * @brief The `key value` rows, in the fixed order the chain's assertions read.
   * @param with_driver_gap Whether to append `driver_gap` (dump mode only).
   * @return The rows.
   */
  std::vector<std::pair<std::string, double>> rows(bool with_driver_gap) const {
    std::vector<std::pair<std::string, double>> rows{{"mean_current_x", mean_current_x},
                                                     {"mean_current_y", mean_current_y},
                                                     {"lambda_eff", lambda_eff},
                                                     {"T_left", T_left},
                                                     {"T_right", T_right},
                                                     {"T_probe_quarter", T_probe_quarter},
                                                     {"T_probe_centre", T_probe_centre},
                                                     {"T_probe_quarter_top", T_probe_quarter_top},
                                                     {"anchor_error", anchor_error}};
    if (with_driver_gap) {
      rows.emplace_back("driver_gap", driver_gap);
    }
    return rows;
  }
};

/**
 * @brief Write a summary file (`key value` per line, `#` comments skipped).
 * @param path    The output file.
 * @param header  A one-line provenance comment (case and mode).
 * @param rows    The numbers, in the order they are written.
 */
void write_summary(const std::string& path, const std::string& header,
                   const std::vector<std::pair<std::string, double>>& rows) {
  std::ofstream out(path);
  MFEM_VERIFY(static_cast<bool>(out), "cannot open the summary '" << path << "' for writing.");
  out.precision(17);
  out << "# alouette offline chain summary: " << header << '\n';
  for (const auto& [key, value] : rows) {
    out << key << ' ' << value << '\n';
  }
  out.flush();
  MFEM_VERIFY(static_cast<bool>(out), "writing the summary '" << path << "' failed.");
}

/**
 * @brief Read a summary file.
 * @param path The file.
 * @return The numbers, keyed by name.
 */
std::map<std::string, double> read_summary(const std::string& path) {
  std::ifstream in(path);
  MFEM_VERIFY(static_cast<bool>(in), "cannot open the summary '" << path << "'.");
  std::map<std::string, double> values;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }
    std::istringstream stream(line);
    std::string key;
    double value = 0.0;
    MFEM_VERIFY(static_cast<bool>(stream >> key >> value),
                "summary '" << path << ":" << line_no << "' is not a 'key value' line.");
    std::string extra;
    MFEM_VERIFY(!(stream >> extra), "summary '" << path << ":" << line_no
                                                << "' has trailing content '" << extra << "'.");
    MFEM_VERIFY(values.emplace(key, value).second,
                "summary '" << path << "' repeats the key '" << key << "'.");
  }
  MFEM_VERIFY(!values.empty(), "summary '" << path << "' is empty.");
  return values;
}

/**
 * @brief Compare two summaries: identical key sets, then a tight agreement.
 * @param path_a The first summary.
 * @param path_b The second summary.
 * @return The largest relative gap.
 */
double compare_summaries(const std::string& path_a, const std::string& path_b) {
  const std::map<std::string, double> a = read_summary(path_a);
  const std::map<std::string, double> b = read_summary(path_b);
  MFEM_VERIFY(a.size() == b.size(), "the summaries have different key sets (" << a.size()
                                                                              << " vs " << b.size()
                                                                              << " keys).");
  double worst = 0.0;
  for (const auto& [key, value_a] : a) {
    const auto found = b.find(key);
    MFEM_VERIFY(found != b.end(), "the second summary has no key '" << key << "'.");
    const double value_b = found->second;
    const double scale = std::max(1.0, std::max(std::abs(value_a), std::abs(value_b)));
    const double gap = std::abs(value_a - value_b) / scale;
    worst = std::max(worst, gap);
    std::printf("COMPARE %-22s a=%-22.15g b=%-22.15g rel=%.3e %s\n", key.c_str(), value_a, value_b,
                gap, gap <= 1.0e-12 ? "ok" : "FAILED");
  }
  return worst;
}

/**
 * @brief The command line.
 */
struct Options {
  /// The case name.
  std::string case_name = "series";
  /// `dump`, `table`, `constant` or `compare`.
  std::string mode;
  /// The dump to write (dump mode).
  std::string dump;
  /// The response table to read (table mode).
  std::string table;
  /// The summary to write (dump, table and constant modes).
  std::string summary;
  /// The two summaries to compare (compare mode).
  std::string compare_a;
  /// The second summary to compare (compare mode).
  std::string compare_b;
};

/**
 * @brief Print the usage.
 * @param program `argv[0]`.
 */
void print_usage(const char* program) {
  std::printf(
      "usage: %s --case <series|parallel|transient> --mode <dump|table|constant>\n"
      "          [--dump <file>] [--table <file>] --summary <file>\n"
      "       %s --mode compare --compare <summary a> <summary b>\n"
      "\n"
      "The S1 offline chain's macro side (docs/DESIGN_SLOTH_COUPLING.md 3.10).\n"
      "  dump      run the macro loop and write the 'alouette-macro-dump 1' file\n"
      "  constant  run it with the analytic layer lambda* (the S0 path)\n"
      "  table     run it with the lambda* read from alouette's response table\n"
      "  compare   require two summaries to agree to 1e-12 relative\n",
      program, program);
}

/**
 * @brief Parse the command line.
 * @param argc The argument count.
 * @param argv The arguments.
 * @return The options; a malformed line is refused.
 */
Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("the option " + flag + " expects " + what + ".");
      }
      return argv[++i];
    };
    if (flag == "--help" || flag == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (flag == "--case") {
      options.case_name = value("a case name");
    } else if (flag == "--mode") {
      options.mode = value("a mode");
    } else if (flag == "--dump") {
      options.dump = value("a path");
    } else if (flag == "--table") {
      options.table = value("a path");
    } else if (flag == "--summary") {
      options.summary = value("a path");
    } else if (flag == "--compare") {
      options.compare_a = value("the first summary");
      options.compare_b = value("the second summary");
    } else {
      throw std::runtime_error("unknown argument '" + flag + "' (see --help)");
    }
  }
  if (options.mode.empty()) {
    throw std::runtime_error("--mode is required (dump, table, constant or compare)");
  }
  if (options.mode == "dump") {
    if (options.dump.empty() || options.summary.empty()) {
      throw std::runtime_error("--mode dump needs --dump and --summary");
    }
  } else if (options.mode == "table") {
    if (options.table.empty() || options.summary.empty()) {
      throw std::runtime_error("--mode table needs --table and --summary");
    }
  } else if (options.mode == "constant") {
    if (options.summary.empty()) {
      throw std::runtime_error("--mode constant needs --summary");
    }
  } else if (options.mode == "compare") {
    if (options.compare_a.empty() || options.compare_b.empty()) {
      throw std::runtime_error("--mode compare needs --compare <a> <b>");
    }
  } else {
    throw std::runtime_error("unknown mode '" + options.mode +
                             "' (expected dump, table, constant or compare)");
  }
  return options;
}

// ---------------------------------------------------------------------------
// The runs
// ---------------------------------------------------------------------------

/// The three probe points, as (x, y).
constexpr double kProbeX[3] = {0.25, 0.5, 0.25};
/// The three probe points' y coordinates. All three sit on node lines of the
/// 20 x 2 mesh; the field is continuous, so any adjacent element gives the
/// same value, and `FindPoints` resolves the location.
constexpr double kProbeY[3] = {0.25, 0.5, 0.75};
/// The probe names, in the summary's order.
const char* const kProbeKey[3] = {"T_probe_quarter", "T_probe_centre", "T_probe_quarter_top"};

/**
 * @brief The summary of a finished run: the anchors, the probes, the flux.
 * @param run  The run (its final state).
 * @param spec The case.
 * @return The summary.
 */
Summary make_summary(MacroRun& run, const CaseSpec& spec) {
  const double final_time = spec.final_time();
  Summary summary;
  // The probes, and the anchor error they carry.
  double anchor_error = 0.0;
  double probe_value[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    probe_value[i] = run.probe(kProbeX[i], kProbeY[i]);
    const double exact = exact_temperature(spec, kProbeX[i], kProbeY[i], final_time);
    anchor_error = std::max(anchor_error, std::abs(probe_value[i] - exact));
  }
  summary.T_probe_quarter = probe_value[0];
  summary.T_probe_centre = probe_value[1];
  summary.T_probe_quarter_top = probe_value[2];
  summary.anchor_error = anchor_error;
  // The flux anchors. `<J_x>` is the volume average of lambda dT/dx, computed
  // from the SAME per-element table the operator assembled and the field's own
  // gradients -- the reference is the scheme's own bookkeeping, not a
  // continuum formula.
  const std::vector<Sample> samples = run.sample();
  double current_x = 0.0;
  double current_y = 0.0;
  double volume = 0.0;
  for (const Sample& sample : samples) {
    current_x += sample.weight * run.element_value(sample.element) * sample.gradient[0];
    current_y += sample.weight * run.element_value(sample.element) * sample.gradient[1];
    volume += sample.weight;
  }
  MFEM_VERIFY(!samples.empty() && volume > 0.0,
              "make_summary: the quadrature volume is not positive.");
  summary.mean_current_x = current_x / volume;
  summary.mean_current_y = current_y / volume;
  summary.lambda_eff = summary.mean_current_x * kLx / kDeltaT;
  // The prescribed Dirichlet data. These are the case's INPUTS, reported so
  // the summary is self-contained (lambda_eff is computable from it alone).
  // That the assembled field actually attains them is asserted separately
  // below, on the field's own extremes.
  summary.T_left = kTCold;
  summary.T_right = kTHot;
  return summary;
}

}  // namespace

/**
 * @brief The test's entry point.
 * @param argc The argument count.
 * @param argv The arguments.
 * @return 0 on success, non-zero on a refusal or a failed anchor.
 */
int main(int argc, char* argv[]) {
  setVerbosity(Verbosity::Error);
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  // MATools' timers are LIVE inside the operator (`Catch_Time_Section` in
  // `SetTransientParameters`, the integrators and the solver), and they
  // dereference a null root node unless the singleton has been constructed.
  // `disable()` constructs it and turns OFF the timetable and the profiling
  // file: this test's outputs are the dump, the table and the summary, and a
  // timing report in the chain's workdir is noise. (`enable()` would print and
  // write a file; the construction is what the timers need, not the output.)
  Profiling::getInstance().disable();

  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::WorldRank() == 0) {
      std::fprintf(stderr,
                   "Alouette2Dtest2: the S1 offline chain runs SERIAL (this run has %d ranks). "
                   "The macro dump's element numbers are local indices, so a parallel macro mesh "
                   "would write a dump whose material points do not match the batch's reading.\n",
                   mfem::Mpi::WorldSize());
    }
    MPI_Finalize();
    return 2;
  }

  int status = 0;
  try {
    const Options options = parse_options(argc, argv);
    if (options.mode == "compare") {
      const double worst = compare_summaries(options.compare_a, options.compare_b);
      std::printf("%s: worst relative gap %.3e\n",
                  worst <= 1.0e-12 ? "PASS" : "FAILED", worst);
      MPI_Finalize();
      return worst <= 1.0e-12 ? 0 : 1;
    }

    const CaseSpec spec = case_spec(options.case_name);
    // The per-element lambda* table. In `constant` and `dump` modes it is the
    // case's own layer value (the analytic lambda* of a homogeneous cell); in
    // `table` mode it is what alouette's response table reports, checked
    // against the layer value below.
    std::vector<double> lambda_star(static_cast<std::size_t>(kNx * kNy), 0.0);
    {
      // The layer assignment comes from the mesh, not from the element
      // numbering: a renumbered mesh must not silently swap the layers.
      MacroRun assignment(spec, lambda_star);
      lambda_star = analytic_lambda_star(spec, *assignment.mesh());
    }
    // What the dump run itself used. Kept because `table` mode overwrites
    // `lambda_star` with the table's values and the round-trip replay below
    // must reproduce the DUMP's trajectory, not the table's.
    const std::vector<double> analytic_lambda(lambda_star);

    if (options.mode == "table") {
      const alouette_offline::Table table = alouette_offline::read_table(options.table);
      MFEM_VERIFY(table.n_material_points == kNx * kNy,
                  "the response table has " << table.n_material_points << " material points for "
                                            << kNx * kNy << " macro elements.");
      // The table must be aligned with THIS case's clock: a table written for
      // another run would otherwise be applied as if it were this one's.
      for (int s = 0; s < table.n_steps; ++s) {
        const double expected = static_cast<double>(s) * spec.dt;
        const double actual = table.steps[static_cast<std::size_t>(s)].time;
        const double scale = std::max(1.0, std::abs(expected));
        MFEM_VERIFY(std::abs(actual - expected) <= 1.0e-12 * scale,
                    "the response table's step " << s << " is at t=" << actual
                                                << " but the case's clock says " << expected << ".");
      }
      MFEM_VERIFY(table.n_steps == spec.n_steps,
                  "the response table has " << table.n_steps << " steps for " << spec.n_steps
                                            << " macro steps.");
      for (int e = 0; e < table.n_material_points; ++e) {
        lambda_star[static_cast<std::size_t>(e)] = table.lambda_star(e);
      }
    }

    MacroRun run(spec, lambda_star);
    run.initialize();

    // The dump is sampled at the START of every step: the macro state the
    // batch is asked about is the one the step begins from.
    std::vector<std::vector<Sample>> dumped;
    if (options.mode == "dump") {
      dumped.reserve(static_cast<std::size_t>(spec.n_steps));
    }
    for (int s = 0; s < spec.n_steps; ++s) {
      if (options.mode == "dump") {
        dumped.push_back(run.sample());
      }
      run.step(s);
    }

    Summary summary = make_summary(run, spec);

    // The Dirichlet data must be ATTAINED by the assembled field, not merely
    // declared: the extremes of the exact solution are exactly the two
    // boundary values (the sine mode has no interior extremum at these
    // times), so a field that misses them never had its boundary conditions
    // applied.
    const double minimum = run.temperature().Min();
    const double maximum = run.temperature().Max();
    MFEM_VERIFY(std::abs(minimum - kTCold) <= kBoundaryTolerance,
                "the field's minimum is " << minimum << " K, not the cold Dirichlet value "
                                          << kTCold << " K.");
    MFEM_VERIFY(std::abs(maximum - kTHot) <= kBoundaryTolerance,
                "the field's maximum is " << maximum << " K, not the hot Dirichlet value " << kTHot
                                          << " K.");

    if (options.mode == "table") {
      // The batch computed <J> = lambda* . E from the dump's own gradients.
      // Replaying the dump run here (same analytic layer values the dump was
      // written with) and comparing EVERY step closes the round trip: a
      // transposed gradient, a dropped component, an out-of-order row or a
      // step whose rows were written twice would not survive it. Checking only
      // step 0 would be nearly vacuous for the steady cases -- their initial
      // field is uniform, so its gradient is zero.
      const alouette_offline::Table table = alouette_offline::read_table(options.table);
      // The table's lambda* must BE the homogeneous cell's value: this is the
      // micro answer, not the macro one, and a silent offset here would move
      // every current below it.
      double worst_lambda = 0.0;
      for (int e = 0; e < table.n_material_points; ++e) {
        const double expected = analytic_lambda[static_cast<std::size_t>(e)];
        const double actual = table.lambda_star(e);
        const double scale = std::max(1.0, std::abs(expected));
        worst_lambda = std::max(worst_lambda, std::abs(actual - expected) / scale);
      }
      std::printf("ROUND TRIP: table lambda* vs the homogeneous cell's layer value, "
                  "worst relative gap %.3e\n", worst_lambda);
      MFEM_VERIFY(worst_lambda <= 1.0e-10,
                  "the response table's lambda* is not the homogeneous cell's layer value (worst "
                  "relative gap "
                      << worst_lambda << ").");
      std::vector<double> replay_table(analytic_lambda);
      MacroRun replay(spec, replay_table);
      replay.initialize();
      const int n_qp = replay.quadrature_points_per_element();
      double worst = 0.0;
      for (int s = 0; s < table.n_steps; ++s) {
        const std::vector<Sample> samples = replay.sample();
        for (int e = 0; e < table.n_material_points; ++e) {
          double mean[2] = {0.0, 0.0};
          for (int q = 0; q < n_qp; ++q) {
            const Sample& sample =
                samples[static_cast<std::size_t>(e) * static_cast<std::size_t>(n_qp) +
                        static_cast<std::size_t>(q)];
            mean[0] += sample.gradient[0];
            mean[1] += sample.gradient[1];
          }
          mean[0] /= static_cast<double>(n_qp);
          mean[1] /= static_cast<double>(n_qp);
          const alouette_offline::Response& response =
              table.steps[static_cast<std::size_t>(s)].rows[static_cast<std::size_t>(e)];
          for (int component = 0; component < 2; ++component) {
            const double expected = response.lambda[0] * mean[component];
            const double scale = std::max(1.0, std::abs(expected));
            const double gap = std::abs(response.current[component] - expected) / scale;
            worst = std::max(worst, gap);
          }
        }
        replay.step(s);
      }
      std::printf("ROUND TRIP: table current vs lambda*.E on every replayed step, "
                  "worst relative gap %.3e\n", worst);
      MFEM_VERIFY(worst <= 1.0e-12,
                  "the response table's current does not match lambda* times the macro side's "
                  "mean gradient (worst relative gap "
                      << worst << "): the dump's rows and the batch's reading disagree.");
    }

    // The anchors. The tolerance is the case's own: a steady solve converges to
    // the exact discrete solution (the Q1 space contains it), while the
    // transient carries backward Euler's per-step gap against the exponential.
    const double anchor_tolerance = spec.kind == CaseKind::Transient ? kTransientAnchorTolerance
                                                                    : kSteadyAnchorTolerance;
    const double exact_current = exact_mean_current_x(spec);
    const double exact_lambda = exact_lambda_eff(spec);
    const double current_error = std::abs(summary.mean_current_x - exact_current) /
                                 std::max(1.0, std::abs(exact_current));
    const double lambda_error =
        std::abs(summary.lambda_eff - exact_lambda) / std::max(1.0, std::abs(exact_lambda));
    const double probe_computed[3] = {summary.T_probe_quarter, summary.T_probe_centre,
                                      summary.T_probe_quarter_top};
    std::printf("CASE %s (%s): t_end=%g\n", spec.name.c_str(), options.mode.c_str(),
                spec.final_time());
    std::printf("  lambda_eff   computed=%.15g exact=%.15g rel=%.3e %s\n", summary.lambda_eff,
                exact_lambda, lambda_error,
                lambda_error <= kFluxAnchorTolerance ? "ok" : "FAILED");
    std::printf("  <J_x>        computed=%.15g exact=%.15g rel=%.3e %s\n", summary.mean_current_x,
                exact_current, current_error,
                current_error <= kFluxAnchorTolerance ? "ok" : "FAILED");
    for (int i = 0; i < 3; ++i) {
      const double exact =
          exact_temperature(spec, kProbeX[i], kProbeY[i], spec.final_time());
      std::printf("  %-20s computed=%.15g exact=%.15g abs=%.3e %s\n", kProbeKey[i],
                  probe_computed[i], exact, std::abs(probe_computed[i] - exact),
                  std::abs(probe_computed[i] - exact) <= anchor_tolerance ? "ok" : "FAILED");
    }
    MFEM_VERIFY(summary.anchor_error <= anchor_tolerance,
                "the temperature anchors miss by " << summary.anchor_error << " K, above the case's "
                                                   << anchor_tolerance << " K tolerance.");
    MFEM_VERIFY(lambda_error <= kFluxAnchorTolerance,
                "lambda_eff misses by a relative " << lambda_error << ".");
    MFEM_VERIFY(current_error <= kFluxAnchorTolerance,
                "the mean current misses by a relative " << current_error << ".");

    if (options.mode == "dump") {
      // The driver's equivalence: a single long `TimeDiscretization` run over
      // the same clock, on a FRESH state (so the reference starts from the IC,
      // exactly like the driver did).
      std::vector<double> reference_table(lambda_star);
      MacroRun reference(spec, reference_table);
      reference.run_single_loop();
      double driver_gap = 0.0;
      for (int i = 0; i < 3; ++i) {
        driver_gap = std::max(driver_gap, std::abs(reference.probe(kProbeX[i], kProbeY[i]) -
                                                   probe_computed[i]));
      }
      std::printf("  driver_gap   %.3e (per-step driver vs one long TimeDiscretization run) %s\n",
                  driver_gap, driver_gap <= kDriverGapTolerance ? "ok" : "FAILED");
      MFEM_VERIFY(driver_gap <= kDriverGapTolerance,
                  "the per-step driver differs from a single long run by " << driver_gap
                                                                           << " K.");
      summary.driver_gap = driver_gap;
      write_dump(options.dump, spec, run.n_elements(), run.quadrature_points_per_element(), dumped);
      std::printf("  dump         %s (%zu steps, %zu rows per step)\n", options.dump.c_str(),
                  dumped.size(), dumped.empty() ? 0UL : dumped.front().size());
    }

    write_summary(options.summary, "case=" + spec.name + " mode=" + options.mode,
                  summary.rows(options.mode == "dump"));
    std::printf("SUMMARY %s\n", options.summary.c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Alouette2Dtest2: %s\n", error.what());
    status = 1;
  }

  MPI_Finalize();
  return status;
}
