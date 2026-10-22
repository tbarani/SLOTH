/**
 * @file SlothBaseCoefficient.hpp
 * @author Clément Introïni (clement.introini@cea.fr)
 * @brief Base class definining a coefficient from an analytical formula
 * @remark Depends on ExprTk library (MIT license)
 * @version 0.1
 * @date 2025-11-06
 *
 * Copyright CEA (C) 2025
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
 */
#pragma once
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "Coefficients/ConstantCoefficient.hpp"
#include "Coefficients/FunctionCoefficient.hpp"
#include "Glossary/Glossary.hpp"
#include "Options/TimeOptions.hpp"
#include "mfem.hpp"  // NOLINT [no include the directory when naming mfem include file]

class MultiscaleCoefficient;

/**
 * @brief Base class defining a coefficient from an analytical formula.
 */
class SlothBaseCoefficient {
 private:
  std::string name_;
  GlossaryQuantity coefficient_quantity_;
  std::shared_ptr<FunctionCoefficient> coefficient_;
  std::vector<int> bdr_index_;

  Scheme scheme_;

 public:
  SlothBaseCoefficient(GlossaryQuantity type, Scheme scheme,
                       std::shared_ptr<FunctionCoefficient> coef);
  SlothBaseCoefficient(GlossaryQuantity type, double coef);

  virtual ~SlothBaseCoefficient() = default;

  // Scalar
  double compute();

  // Implicit/Explicit without auxiliary variables
  double compute(const std::span<const double>& values,
                 std::optional<int> dimension = std::nullopt);

  // Semi-implicit without auxiliary variables
  // and Implicit/Explicit with auxiliary variables
  double compute(const std::span<const double>& values,
                 const std::span<const double>& auxiliary_values,
                 std::optional<int> dimension = std::nullopt);

  // Semi-implicit with auxiliary variables
  double compute(const std::span<const double>& values, const std::span<const double>& exp_values,
                 const std::span<const double>& auxiliary_values,
                 std::optional<int> dimension = std::nullopt);

  // Implicit/Explicit without auxiliary variables
  double compute_gradient(const int i, const std::span<const double>& values,
                          std::optional<int> dimension = std::nullopt);

  // Semi-implicit without auxiliary variables
  // and Implicit/Explicit with auxiliary variables
  double compute_gradient(const int i, const std::span<const double>& values,
                          const std::span<const double>& auxiliary_values,
                          std::optional<int> dimension = std::nullopt);

  // Semi-implicit with auxiliary variables
  double compute_gradient(const int i, const std::span<const double>& values,
                          const std::span<const double>& exp_values,
                          const std::span<const double>& auxiliary_values,
                          std::optional<int> dimension = std::nullopt);

  // Implicit/Explicit without auxiliary variables
  double compute_hessian(const int i, const int j, const std::span<const double>& values,
                         std::optional<int> dimension = std::nullopt);

  // Semi-implicit without auxiliary variables
  // and Implicit/Explicit with auxiliary variables
  double compute_hessian(const int i, const int j, const std::span<const double>& values,
                         const std::span<const double>& auxiliary_values,
                         std::optional<int> dimension = std::nullopt);

  // Semi-implicit with auxiliary variables
  double compute_hessian(const int i, const int j, const std::span<const double>& values,
                         const std::span<const double>& exp_values,
                         const std::span<const double>& auxiliary_values,
                         std::optional<int> dimension = std::nullopt);

  GlossaryType get_type() const;
  const GlossaryQuantity get_quantity();
  unsigned int get_id() const;
  bool is_implicit() const;
  bool is_explicit() const;
  bool is_semi_implicit() const;
  bool is_scalar() const;

  /**
   * @brief The multiscale provider behind this coefficient, or nullptr.
   *
   * The R2 hook (docs/DESIGN_SLOTH_COUPLING.md 3.6): a coefficient that must
   * see the element context cannot be evaluated through compute(), so the
   * diffusion integrator asks this predicate at its two conductivity call
   * sites and branches to MultiscaleCoefficient::eval_at(Tr) when it is
   * non-null. NON-VIRTUAL on purpose: SlothBaseCoefficient is stored BY VALUE
   * inside Coefficients and the copies slice a derived virtual, which is why
   * the multiscale state lives in the shared FunctionCoefficient instead. The
   * dynamic_cast is defined in the companion .cpp, where
   * MultiscaleCoefficient is a complete type.
   *
   * @return Pointer to the multiscale coefficient, or nullptr when this
   *         coefficient is an ordinary analytical/constant one.
   */
  MultiscaleCoefficient* as_multiscale() const;

  /**
   * @brief The underlying FunctionCoefficient (the object the copies share).
   *
   * @return The shared coefficient implementation.
   */
  std::shared_ptr<FunctionCoefficient> get_function_coefficient() const;

  void set_time(double time);

  std::string get_name();
  void set_name(const std::string& name);
  void set_bdr_index_coef(std::vector<int> ids);
  std::vector<int> get_bdr_index_coef() const;
};
