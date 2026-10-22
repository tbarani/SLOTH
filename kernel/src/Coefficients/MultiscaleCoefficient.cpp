/**
 * @file MultiscaleCoefficient.cpp
 * @author Tommaso Barani (tommaso.barani@cea.fr)
 * @brief FunctionCoefficient whose value is a lookup by mesh element index (the R2 multiscale
 *        provider).
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
 * The three inherited evaluators are deliberately unanswerable: the state
 * spans they receive carry no element identity, so any value they could
 * return would be an invented one. MFEM_VERIFY(false, ...) is used instead of
 * MFEM_ASSERT because it survives NDEBUG (the suite runs Release builds) and
 * instead of an exception because MFEM is built WITHOUT exceptions here: the
 * abort is the loud refusal, and the integrator branch is what keeps it
 * unreachable on the supported path.
 */
#include "Coefficients/MultiscaleCoefficient.hpp"

#include <functional>
#include <span>
#include <vector>

#include "Coefficients/FunctionCoefficient.hpp"
#include "mfem.hpp"  // NOLINT [no include the directory when naming mfem include file]

/* The `return` statements after each MFEM_VERIFY(false, ...) are dead code:
 * mfem_error is [[noreturn]] and the abort is the only reachable outcome.
 * They exist so the compiler's data-flow analysis does not depend on that
 * attribute through the macro expansion. */

/**
 * @brief Refusing evaluator: a multiscale coefficient has no value without the element context.
 *
 * @return Never returns; aborts through MFEM_VERIFY.
 */
std::function<double(const std::span<const double>&, const std::span<const double>&,
                     const std::span<const double>&, const unsigned int dimension)>
MultiscaleCoefficient::F() {
  auto func = []([[maybe_unused]] const std::span<const double>& input_vector,
                 [[maybe_unused]] const std::span<const double>& exp_input_vector,
                 [[maybe_unused]] const std::span<const double>& auxiliary_vector,
                 [[maybe_unused]] const unsigned int dimension) {
    MFEM_VERIFY(false,
                "MultiscaleCoefficient::F() was called: a multiscale coefficient is a lookup by "
                "element index and cannot be evaluated from state values. The diffusion "
                "integrator must branch through SlothBaseCoefficient::as_multiscale() and call "
                "eval_at(Tr) instead.");
    return 0.0;
  };
  return func;
}

/**
 * @brief Refusing gradient evaluator: the frozen table has no state derivative.
 *
 * @return Never returns; aborts through MFEM_VERIFY.
 */
std::function<std::vector<double>(const std::span<const double>&, const std::span<const double>&,
                                  const std::span<const double>&, const unsigned int dimension)>
MultiscaleCoefficient::GradientF() {
  auto func = []([[maybe_unused]] const std::span<const double>& input_vector,
                 [[maybe_unused]] const std::span<const double>& exp_input_vector,
                 [[maybe_unused]] const std::span<const double>& auxiliary_vector,
                 [[maybe_unused]] const unsigned int dimension) {
    MFEM_VERIFY(false,
                "MultiscaleCoefficient::GradientF() was called: a multiscale coefficient is a "
                "lookup by element index; its state derivative is identically zero and the "
                "integrator's multiscale branch pins grad_diffu = 0 instead of asking for it.");
    return std::vector<double>{};
  };
  return func;
}

/**
 * @brief Refusing Hessian evaluator: the frozen table has no state derivative.
 *
 * @return Never returns; aborts through MFEM_VERIFY.
 */
std::function<std::vector<double>(const std::span<const double>&, const std::span<const double>&,
                                  const std::span<const double>&, const unsigned int dimension)>
MultiscaleCoefficient::HessianF() {
  auto func = []([[maybe_unused]] const std::span<const double>& input_vector,
                 [[maybe_unused]] const std::span<const double>& exp_input_vector,
                 [[maybe_unused]] const std::span<const double>& auxiliary_vector,
                 [[maybe_unused]] const unsigned int dimension) {
    MFEM_VERIFY(false,
                "MultiscaleCoefficient::HessianF() was called: a multiscale coefficient is a "
                "lookup by element index and has no state Hessian.");
    return std::vector<double>{};
  };
  return func;
}
