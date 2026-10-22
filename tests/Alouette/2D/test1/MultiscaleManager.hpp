/**
 * @file MultiscaleManager.hpp
 * @author Tommaso Barani (tommaso.barani@cea.fr)
 * @brief S0 multiscale manager: a kernel-free property problem that owns the
 *        material-point table and publishes it as its own output variable.
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
 * WHY A PROPERTY PROBLEM AND NOT A BESPOKE PROBLEM. The manager must run once
 * per macro step, BEFORE the thermal problem, and it must write a field the
 * conductivity provider can read. `Property_problem` already does exactly
 * that: it calls `PropertyBase::compute(outputs, inputs)` with the problem's
 * own primary variables as writable vectors, and `Coupling::execute` calls
 * `Problem::update()` after each member. Running the manager first in the
 * coupling therefore publishes the table before the heat problem assembles.
 * No kernel change is involved, which is the S0 scope.
 *
 * WHY THE PROPERTY WRITES ITS OWN VARIABLE INSTEAD OF A PRIVATE BUFFER. The
 * provider is created before the property problem and is handed to the
 * diffusion coefficient; the property object itself is constructed inside
 * `Property_problem`, and `Parameters` carries only the primitive variant
 * (int/double/string/...), so a `std::vector` or a callback cannot be passed
 * through it. The published output variable is the one channel that already
 * survives that construction: `Variable::update()` refreshes the live
 * `ParGridFunction` every step, and the provider reads it by element index.
 * At S0 every entry of the table is the same constant, so the fold is exact
 * and the identity comparison against the standalone heat test is byte for
 * byte (`m3`-style element-to-dof scatter is the S2 step).
 *
 * THE PST IS DELIBERATELY ABSENT. A manager that saved frames would change
 * the reference tree of the identity gate; `ProblemBase::save_vtk` is a no-op
 * without a PST, so the manager writes nothing. The one trap is recorded in
 * main.cpp: `Coupling::need_to_be_saved()` and `Coupling::get_shared_dc()`
 * dereference the FIRST member's PST unguardedly, so a PST-less first member
 * is safe only while `vtk_unified` stays false (its default), because
 * `TimeDiscretization::save_vtk_unified()` is the only caller.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "Parameters/Parameter.hpp"
#include "Parameters/Parameters.hpp"
#include "Property/PropertyBase.hpp"
#include "Sloth/sloth.hpp"
#include "Variables/Variable.hpp"
#include "mfem.hpp"  // NOLINT [no include the directory when naming mfem include file]

/**
 * @class MultiscaleManager
 * @brief Owns the S0 material-point table and publishes its constant fold.
 *
 * The table is `n_material_points` doubles, every entry equal to the
 * `"lambda"` parameter at S0 (the null-micro constant-conductivity state).
 * `get_property` writes the fold into the single output variable
 * (`"lambda_bar"`), which `Variable::update` copies into its
 * `ParGridFunction`; the R2 provider reads that grid function by element.
 */
class MultiscaleManager : public PropertyBase {
 public:
  /// The name of the one output variable this property writes.
  static constexpr const char* kOutputName = "lambda_bar";

  /**
   * @brief Build the manager from its `Parameters`.
   *
   * @param params `"n_material_points"` (int, >= 1, default 1) and
   *               `"lambda"` (double, default 2.0), plus an optional
   *               `"description"`.
   */
  explicit MultiscaleManager(const Parameters& params);

  /// The material-point table this manager owns (S0: constant).
  const std::vector<double>& material_points() const { return material_points_; }

  /// The value published to the provider at the last `get_property` call.
  double published_value() const { return published_value_; }

  /// The number of material points in the owned table.
  int n_material_points() const { return static_cast<int>(material_points_.size()); }

  /**
   * @brief The constant fold of the table (the value published each step).
   *
   * A table whose entries are all equal returns that entry EXACTLY, not its
   * floating-point mean: at S0 the table is `n` copies of one constant, and
   * the identity gate against the standalone heat test must be bit-for-bit.
   * `(v + v + ... + v) / n` is not exact for every `v` and `n` (0.1 and 3
   * suffice), so relying on the mean would make the identity a rounding
   * coincidence rather than a property. A genuinely non-constant table (S2
   * onward) returns the arithmetic mean, which is the documented projection.
   *
   * @return The published value.
   */
  double fold() const {
    const double first = material_points_.front();
    for (const double value : material_points_) {
      if (value != first) {
        const double sum = std::accumulate(material_points_.begin(), material_points_.end(), 0.0);
        return sum / static_cast<double>(material_points_.size());
      }
    }
    return first;
  }

 protected:
  /**
   * @brief Verify the property contract: one output, the expected name, no inputs.
   *
   * A mismatch is a programming error in the test app, and a silent default
   * would publish into the wrong variable (a plausible-looking wrong answer),
   * so every check aborts.
   */
  void check_variables_consistency(
      std::vector<std::tuple<std::vector<std::string>, std::reference_wrapper<mfem::Vector>>>&
          output_system,
      std::vector<std::tuple<std::vector<std::string>, mfem::Vector>> input_system) override;

  /**
   * @brief Publish the constant fold of the table into the output variable.
   *
   * @param output_system The property problem's primary variables (one).
   * @param input_system  Empty at S0: the manager has no coupling inputs yet.
   */
  void get_property(
      std::vector<std::tuple<std::vector<std::string>, std::reference_wrapper<mfem::Vector>>>&
          output_system,
      std::vector<std::tuple<std::vector<std::string>, mfem::Vector>> input_system) override;

 private:
  /// The owned material-point table (S0: `n_material_points` equal entries).
  std::vector<double> material_points_;
  /// The fold written at the last call; also the value the provider must see.
  double published_value_{0.0};
  /// One-time announcement guard (rank 0 only).
  bool announced_{false};
};

/**
 * @struct MultiscaleProviderProbe
 * @brief Counters proving the R2 provider was exercised and what it returned.
 *
 * The identity gate's whole risk is a provider path that is DECLARED but
 * never reached (the M3-kernel lesson: a path that is not visited is not
 * tested). The probe records every call, the number of distinct elements
 * visited on this rank, and the value range actually returned; `main`
 * reduces them globally and refuses if the range is not the published
 * constant or if no element was visited.
 */
struct MultiscaleProviderProbe {
  /// Number of `eval_at` provider calls (one per quadrature point and block).
  long long call_count{0};
  /// Number of distinct local elements the provider was called for.
  long long element_visit_count{0};
  /// Smallest value returned (initialised to +inf by `reset`).
  double min_value{0.0};
  /// Largest value returned (initialised to -inf by `reset`).
  double max_value{0.0};
  /// Per-local-element visited flags, sized to the local mesh's element count.
  std::vector<unsigned char> visited;

  /**
   * @brief Size the visited flags and clear every counter.
   *
   * @param local_elements The local `ParMesh::GetNE()`.
   */
  void reset(int local_elements) {
    call_count = 0;
    element_visit_count = 0;
    min_value = std::numeric_limits<double>::infinity();
    max_value = -std::numeric_limits<double>::infinity();
    visited.assign(static_cast<std::size_t>(std::max(local_elements, 0)), 0);
  }

  /**
   * @brief Record one provider evaluation.
   *
   * @param element_no The local element index the provider was asked for.
   * @param value      The value it returned.
   */
  void note(int element_no, double value) {
    ++call_count;
    if (element_no >= 0 && static_cast<std::size_t>(element_no) < visited.size() &&
        visited[static_cast<std::size_t>(element_no)] == 0) {
      visited[static_cast<std::size_t>(element_no)] = 1;
      ++element_visit_count;
    }
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }
};

/**
 * @brief Build the R2 element-index provider over a manager output variable.
 *
 * The returned callable is what `MultiscaleCoefficient` stores. It reads the
 * LIVE grid function every call (never a snapshot): the manager updates it
 * each macro step, and `Coupling::execute` updates the manager before the
 * heat problem assembles, so the heat assembly always sees the current step's
 * table.
 *
 * The element value is the MEAN of the element's nodal values. At S0 the
 * published field is constant, so a constant element is returned EXACTLY (the
 * mean of `n` equal doubles is not exact for every value; see
 * `MultiscaleManager::fold`), which is what makes the identity against the
 * standalone heat test bit-for-bit rather than a rounding coincidence. A
 * non-constant per-element table would need the S2 element-to-dof scatter;
 * the mean is then the documented projection, not the raw table entry.
 *
 * The fespace is the one the heat problem assembles on (both variables come
 * from the same `SPA`), so `Tr.ElementNo` and the local element index used
 * here are the same numbering by construction.
 *
 * @tparam FEC The finite element collection of the variable.
 * @tparam DIM The spatial dimension.
 * @param manager_variable The manager's output `Variable` (outlives the provider).
 * @param probe            Call counters, owned by the caller.
 * @return `std::function<double(int)>` suitable for `MultiscaleCoefficient`.
 */
template <class FEC, int DIM>
std::function<double(int)> make_multiscale_provider(Variable<FEC, DIM>& manager_variable,
                                                    MultiscaleProviderProbe& probe) {
  return [&manager_variable, &probe](int element_no) {
    mfem::ParFiniteElementSpace* fespace = manager_variable.get_fespace();
    MFEM_VERIFY(fespace != nullptr,
                "make_multiscale_provider: the manager variable has no finite element space.");
    MFEM_VERIFY(element_no >= 0 && element_no < fespace->GetNE(),
                "make_multiscale_provider: element " << element_no
                                                     << " is outside this rank's [0, "
                                                     << fespace->GetNE() << ") elements.");
    mfem::Array<int> dofs;
    fespace->GetElementDofs(element_no, dofs);
    MFEM_VERIFY(dofs.Size() > 0, "make_multiscale_provider: element "
                                     << element_no << " has no degrees of freedom.");
    mfem::Vector values;
    manager_variable.get_ref_gf().GetSubVector(dofs, values);
    const double reference = values(0);
    double value = reference;
    for (int i = 1; i < values.Size(); ++i) {
      if (values(i) != reference) {
        double sum = 0.0;
        for (int j = 0; j < values.Size(); ++j) {
          sum += values(j);
        }
        value = sum / static_cast<double>(values.Size());
        break;
      }
    }
    probe.note(element_no, value);
    return value;
  };
}

/// @cond INTERNAL
inline MultiscaleManager::MultiscaleManager(const Parameters& params) : PropertyBase(params) {
  const int n_points = params.template get_param_value_or_default<int>("n_material_points", 1);
  const double lambda = params.template get_param_value_or_default<double>("lambda", 2.0);
  MFEM_VERIFY(n_points >= 1, "MultiscaleManager: n_material_points must be at least 1, got "
                                 << n_points << ".");
  MFEM_VERIFY(std::isfinite(lambda) && lambda > 0.0,
              "MultiscaleManager: lambda must be finite and positive, got " << lambda << ".");
  material_points_.assign(static_cast<std::size_t>(n_points), lambda);
  published_value_ = this->fold();
}

inline void MultiscaleManager::check_variables_consistency(
    std::vector<std::tuple<std::vector<std::string>, std::reference_wrapper<mfem::Vector>>>&
        output_system,
    std::vector<std::tuple<std::vector<std::string>, mfem::Vector>> input_system) {
  MFEM_VERIFY(output_system.size() == 1,
              "MultiscaleManager: exactly one output variable is expected, got "
                  << output_system.size() << ".");
  const auto& infos = std::get<0>(output_system.front());
  MFEM_VERIFY(infos.size() == 1 && infos.front() == MultiscaleManager::kOutputName,
              "MultiscaleManager: the output variable must be named '" << kOutputName
                                                                       << "', got "
                                                                       << (infos.empty() ? "<none>"
                                                                                         : infos.front())
                                                                       << ".");
  MFEM_VERIFY(input_system.empty(),
              "MultiscaleManager: S0 has no auxiliary inputs, got " << input_system.size() << ".");
}

inline void MultiscaleManager::get_property(
    std::vector<std::tuple<std::vector<std::string>, std::reference_wrapper<mfem::Vector>>>&
        output_system,
    std::vector<std::tuple<std::vector<std::string>, mfem::Vector>> input_system) {
  MFEM_VERIFY(output_system.size() == 1 && input_system.empty(),
              "MultiscaleManager::get_property: the property contract was not checked.");
  this->published_value_ = this->fold();
  // One output variable, all its true degrees of freedom: the published field
  // is the constant fold, and `Variable::update` scatters it into the grid
  // function the provider reads.
  std::get<1>(output_system.front()).get() = this->published_value_;
  if (!announced_) {
    announced_ = true;
    if (mfem::Mpi::WorldRank() == 0) {
      std::cout << "MULTISCALE MANAGER: " << material_points_.size()
                << " material point(s), published constant fold " << published_value_ << '\n';
    }
  }
}
/// @endcond

