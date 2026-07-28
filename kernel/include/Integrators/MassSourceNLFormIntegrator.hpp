/**
 * @file MassSourceNLFormIntegrator.hpp
 * @author Tommaso Barani
 * @brief Contribution of a volumetric production rate that depends on the solved and auxiliary
 *        variables to a conservation equation.
 * @version 0.1
 * @date 2026-07-28
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
 */
#pragma once
#include <algorithm>
#include <list>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Coefficients/SlothBaseCoefficient.hpp"
#include "Integrators/SlothNLFormIntegrator.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Parameters/Parameter.hpp"
#include "Parameters/Parameters.hpp"
#include "Utils/Utils.hpp"
#include "mfem.hpp"  // NOLINT [no include the directory when naming mfem include file]

/**
 * @brief Volumetric production contribution to a conservation equation.
 *
 * Assembles, for each block,
 * @f[
 *   -\int s\,\psi ,
 * @f]
 * where the production rate @f$ s @f$ is an ordinary coefficient of type GlossaryType::Source.
 * Unlike the source term handled by the Operators, which is a function of space and time only, it
 * is evaluated with the solved and auxiliary variables at each integration point, so it may for
 * instance be restricted to one phase by an interpolation function of the order parameters,
 * @f$ s = s^0\, h_m(\eta) @f$.
 *
 * @remark The production rate coefficient is of type GlossaryType::Source and is mandatory: an
 *         operator that has no such production has no reason to carry this integrator.
 *
 * @remark The production rate is assumed to be independent of the solved variables of its own
 *         problem: the Jacobian contribution is zero.
 *
 * @tparam VARS Template parameter defining the variables used
 *              in the integrator.
 */
template <class VARS>
class MassSourceNLFormIntegrator : public SlothNLFormIntegrator<VARS> {
 private:
  std::list<GlossaryType> expected_list_{GlossaryType::Source};
  mfem::Vector Psi;

 protected:
  Coefficients source;

  std::vector<mfem::ParGridFunction> vaux_gf_;

  void get_coefficients() override;

  virtual double get_rate_at_ip(unsigned int blk, const std::span<const double>& values,
                                const std::span<const double>& aux_values);

 public:
  MassSourceNLFormIntegrator(Geometry geometry, const double time_step,
                             const std::vector<mfem::ParGridFunction>& u_old,
                             const std::vector<mfem::ParGridFunction>& aux_old,
                             const Parameters& params, std::vector<VARS*> auxvars,
                             const std::vector<Coefficients>& coefficients);

  void AssembleElementVector(const mfem::Array<const mfem::FiniteElement*>& el,
                             mfem::ElementTransformation& Tr,
                             const mfem::Array<const mfem::Vector*>& elfun,
                             const mfem::Array<mfem::Vector*>& elvec) override;

  void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement*>& el,
                           mfem::ElementTransformation& Tr,
                           const mfem::Array<const mfem::Vector*>& elfun,
                           const mfem::Array2D<mfem::DenseMatrix*>& elmats) override;
  void init() override;
};

#include "Integrators/MassSourceNLFormIntegrator.tpp"
