/**
 * @file MultiscaleCoefficient.hpp
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
 * WHY THIS CLASS EXISTS. The coefficient callback
 * (FunctionCoefficient.hpp, F()/GradientF()/HessianF()) receives only the
 * state spans and the dimension: it can never see the element or the
 * quadrature point. A multiscale (FE2) coefficient is a property of the
 * MATERIAL POINT, i.e. of the element, so it cannot be expressed through that
 * callback at all. It needs a context-aware provider.
 *
 * WHERE THE STATE LIVES, AND WHY IT IS NOT A SlothBaseCoefficient SUBCLASS.
 * `Coefficients` stores `std::vector<Coefficient>` BY VALUE and
 * `compute_coefficient(Coefficient coef, ...)` copies again, so a subclass of
 * SlothBaseCoefficient would be SLICED on every copy. The surviving object is
 * the `std::shared_ptr<FunctionCoefficient>` inside the base, so the
 * multiscale state lives HERE, in a FunctionCoefficient subclass, and the two
 * conductivity call sites of DiffusionNLFormIntegrator.tpp branch to
 * `eval_at(Tr)` through the non-virtual `SlothBaseCoefficient::as_multiscale()`
 * predicate. SlothBaseCoefficient is the right NATURAL home for the
 * non-virtual hook because it is the object actually stored; it is not a
 * subclassing target.
 *
 * THIS IS A LOOKUP, NOT A COMPUTE. The manager has already solved the micro
 * cells and published lambda*; `eval_at` reads the frozen table and nothing
 * else. Evaluating a micro cell inside an assembly loop would (a) recompute
 * for every quadrature point, and (b) make the Newton Jacobian inconsistent
 * with the residual it differentiates. The frozen table also makes
 * d(lambda*)/du identically zero, which is why the integrator's `grad_diffu`
 * term is pinned to zero in the multiscale branch and the staggered tangent
 * stays exactly the existing one.
 *
 * THE INHERITED EVALUATORS REFUSE LOUDLY. F()/GradientF()/HessianF() cannot
 * answer without the element context, so every callable they return aborts
 * through MFEM_VERIFY(false, ...) (which is active in release builds, unlike
 * MFEM_ASSERT). A silent zero here would be a wrong answer that looks
 * plausible; a hard abort names the defect at its site. The integrator branch
 * is what makes this unreachable on the supported path.
 */
#pragma once
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include "Coefficients/FunctionCoefficient.hpp"
#include "mfem.hpp"  // NOLINT [no include the directory when naming mfem include file]

/**
 * @class MultiscaleCoefficient
 * @brief Coefficient whose value is provided by element index instead of by the state values.
 *
 * A thin FunctionCoefficient subclass holding a caller-supplied provider
 * `double(int element_no)`. The provider is expected to read a table the
 * coupling manager has already published (frozen over the macro step), not to
 * solve anything.
 */
class MultiscaleCoefficient : public FunctionCoefficient {
 public:
  /// The element-index provider: given a mesh element number, return the
  /// material-point value (lambda*, in the thermal case).
  using ElementValueProvider = std::function<double(int element_no)>;

  explicit MultiscaleCoefficient(ElementValueProvider provider)
      : element_provider_(std::move(provider)) {}

  ~MultiscaleCoefficient() override = default;

  MultiscaleCoefficient(const MultiscaleCoefficient&) = default;
  MultiscaleCoefficient& operator=(const MultiscaleCoefficient&) = default;

  /**
   * @brief Evaluate the provider at a volume element.
   *
   * @param Tr The element transformation; must be a volume element
   *           (ElementTransformation::ELEMENT). A boundary element here is a
   *           programming error in the calling integrator, not a recoverable
   *           condition, so it aborts.
   * @return The provider's value for `Tr.ElementNo`.
   */
  double eval_at(const mfem::ElementTransformation& Tr) const {
    MFEM_VERIFY(Tr.ElementType == mfem::ElementTransformation::ELEMENT,
                "MultiscaleCoefficient::eval_at(...) expects a volume element transformation, "
                "got ElementType = "
                    << Tr.ElementType << " (ELEMENT = "
                    << mfem::ElementTransformation::ELEMENT << ").");
    MFEM_VERIFY(Tr.ElementNo >= 0,
                "MultiscaleCoefficient::eval_at(...) got a negative element number: "
                    << Tr.ElementNo << ".");
    return this->element_provider_(Tr.ElementNo);
  }

 protected:
  std::function<double(const std::span<const double>&, const std::span<const double>&,
                       const std::span<const double>&, const unsigned int dimension)>
  F() final;
  std::function<std::vector<double>(const std::span<const double>&, const std::span<const double>&,
                                    const std::span<const double>&, const unsigned int dimension)>
  GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&, const std::span<const double>&,
                                    const std::span<const double>&, const unsigned int dimension)>
  HessianF() final;

 private:
  ElementValueProvider element_provider_;
};
