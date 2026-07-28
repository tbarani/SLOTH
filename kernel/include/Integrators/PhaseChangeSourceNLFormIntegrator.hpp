/**
 * @file PhaseChangeSourceNLFormIntegrator.hpp
 * @author Tommaso Barani
 * @brief Contribution of a phase change to the conservation equation of a species solved in terms
 *        of its chemical potential.
 * @version 0.2
 * @date 2026-07-26
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
 * @brief Phase-change contribution to a conservation equation written in terms of a chemical
 *        potential.
 *
 * Given a species whose density @f$ \rho @f$ depends both on the solved variable @f$ \mu @f$ and on
 * auxiliary variables @f$ \eta @f$ (typically phase-field order parameters), the conservation
 * equation
 * @f[
 *   \frac{\partial \rho(\mu,\eta)}{\partial t} = \nabla\cdot(D\chi\nabla\mu) + s
 * @f]
 * is solved with @f$ \chi = \partial\rho/\partial\mu @f$ carried by the time integrator
 * (HeatTimeDerivative), which provides @f$ \chi\,\partial\mu/\partial t @f$. The remaining part of
 * the total derivative, i.e. the variation of the density at frozen @f$ \mu @f$,
 * @f[
 *   \sum_{\alpha i} \frac{\partial \rho}{\partial \eta_{\alpha i}}
 *                   \frac{\partial \eta_{\alpha i}}{\partial t}
 * @f]
 * is assembled here by finite difference of the density coefficient between the current and the
 * previous values of the auxiliary variables, at frozen solved variable:
 * @f[
 *   \frac{\rho(\mu^{n+1},\eta^{n+1}) - \rho(\mu^{n+1},\eta^{n})}{\Delta t}.
 * @f]
 *
 * The production rate @f$ s @f$ is handled separately by MassSourceNLFormIntegrator.
 *
 * @remark The density coefficient is of type GlossaryType::DefectDensity and is mandatory.
 *
 * @tparam VARS Template parameter defining the variables used
 *              in the integrator.
 */
template <class VARS>
class PhaseChangeSourceNLFormIntegrator : public SlothNLFormIntegrator<VARS> {
 private:
  std::list<GlossaryType> expected_list_{GlossaryType::DefectDensity};
  mfem::Vector Psi;

 protected:
  Coefficients density;

  std::vector<mfem::ParGridFunction> vaux_gf_;
  std::vector<mfem::ParGridFunction> vaux_old_gf_;

  void get_coefficients() override;

  virtual double get_rate_at_ip(unsigned int blk, const std::span<const double>& values,
                                const std::span<const double>& aux_values,
                                const std::span<const double>& aux_old_values);

  virtual double get_rate_derivative_at_ip(unsigned int blk, const std::span<const double>& values,
                                           const std::span<const double>& aux_values,
                                           const std::span<const double>& aux_old_values);

 public:
  PhaseChangeSourceNLFormIntegrator(Geometry geometry, const double time_step,
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

#include "Integrators/PhaseChangeSourceNLFormIntegrator.tpp"
