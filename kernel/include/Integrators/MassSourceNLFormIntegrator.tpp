/**
 * @file MassSourceNLFormIntegrator.tpp
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

#include "Coefficients/AxiCylindricalCoefficient.hpp"
#include "Integrators/SlothNLFormIntegrator.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Parameters/Parameter.hpp"
#include "Parameters/Parameters.hpp"
#include "Utils/Utils.hpp"
#include "mfem.hpp"  // NOLINT [no include the directory when naming mfem include file]

/**
 * @brief Construct a new MassSourceNLFormIntegrator object.
 *
 * This constructor initializes the nonlinear form integrator. It forwards the provided previous
 * solution fields, simulation parameters, auxiliary variables, and coefficients to the base SLOTH
 * nonlinear form integrator.
 *
 * @tparam VARS Template parameter defining the variables used
 *              in the integrator.
 *
 * @param u_old        Vector of previous-time-step solution fields.
 * @param params       Paramters that can be used with the integrator.
 * @param auxvars      Auxiliary variables required by the inetgrator.
 * @param coefficients List of coefficients defining material properties.
 *
 */
template <class VARS>
MassSourceNLFormIntegrator<VARS>::MassSourceNLFormIntegrator(
    Geometry geometry, const double time_step, const std::vector<mfem::ParGridFunction>& u_old,
    const std::vector<mfem::ParGridFunction>& aux_old, const Parameters& params,
    std::vector<VARS*> auxvars, const std::vector<Coefficients>& coefficients)
    : SlothNLFormIntegrator<VARS>(geometry, time_step, u_old, aux_old, params, auxvars,
                                  coefficients) {
  this->integrator_name_ = "MassSource";
}

/**
 * @brief Initialize the integrator.
 *
 * This method performs all necessary setup steps for the integrator:
 * 1. Checks that all coefficients contain the expected types.
 * 2. Retrieves and stores the coefficients internally.
 * 3. Stores the auxiliary variables.
 *
 * @tparam VARS Template parameter defining the variables used
 *              in the integrator.
 *
 * @pre The integrator's 'expected_list_' is populated with the
 *      required coefficient types for this integrator.
 *
 */
template <class VARS>
void MassSourceNLFormIntegrator<VARS>::init() {
  this->check_coefficient_types(this->expected_list_);
  this->get_coefficients();

  for (std::size_t i = 0; i < this->aux_infos_.size(); ++i) {
    this->vaux_gf_.emplace_back(std::move(this->aux_gf_[i]));
  }
}

/**
 * @brief Assemble the element-level residual vector for the nonlinear problem.
 *
 * The production rate is evaluated with the solved and auxiliary variables at each integration
 * point and subtracted from the residual.
 *
 * @tparam VARS Template parameter defining the variables used in the integrator.
 *
 * @param el      Array of pointers to finite elements.
 * @param Tr      Element transformation.
 * @param elfun   Array of local finite element solution vectors.
 * @param elvect  Array of vectors where the computed element residual contributions
 *                will be stored.
 *
 * @note Users typically do not call this function directly; it is invoked
 *       internally during the assembly of the global nonlinear form.
 */
template <class VARS>
void MassSourceNLFormIntegrator<VARS>::AssembleElementVector(
    const mfem::Array<const mfem::FiniteElement*>& el, mfem::ElementTransformation& Tr,
    const mfem::Array<const mfem::Vector*>& elfun, const mfem::Array<mfem::Vector*>& elvect) {
  int num_blocks = el.Size();
  std::vector<double> u_values(2 * num_blocks);
  std::vector<double> vaux_gf_at_ip(this->nb_vaux_);

  for (int blk = 0; blk < num_blocks; ++blk) {
    int nd = el[blk]->GetDof();
    Psi.SetSize(nd);
    elvect[blk]->SetSize(nd);
    *elvect[blk] = 0.;

    const mfem::IntegrationRule* ir =
        &mfem::IntRules.Get(el[blk]->GetGeomType(), 2 * el[blk]->GetOrder() + Tr.OrderW());

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const mfem::IntegrationPoint& ip = ir->IntPoint(i);
      el[blk]->CalcShape(ip, Psi);
      Tr.SetIntPoint(&ip);

      for (size_t k = 0; k < this->nb_vaux_; ++k) {
        vaux_gf_at_ip[k] = this->vaux_gf_[k].GetValue(Tr, ip);
      }
      for (int off_blk = 0; off_blk < num_blocks; ++off_blk) {
        u_values[off_blk] = (*elfun[off_blk]) * Psi;
        u_values[off_blk + num_blocks] = this->u_old_[off_blk].GetValue(Tr, ip);
      }

      double weight_coef = ip.weight * Tr.Weight();
      if (this->isAxisymmetric()) {
        weight_coef *= AxiCylindricalCoefficient().Eval(Tr, ip);
      }

      const double rate = this->get_rate_at_ip(blk, std::span<const double>(u_values),
                                               std::span<const double>(vaux_gf_at_ip)) *
                          weight_coef;
      add(*elvect[blk], rate, Psi, *elvect[blk]);
    }
  }
}

/**
 * @brief Assemble the element-level Jacobian matrix for the nonlinear problem.
 *
 * The production rate is assumed to be independent of the solved variables of its own problem, so
 * the Jacobian contribution is zero.
 *
 * @remark All the blocks are sized and zeroed so that the integrator remains well defined when
 *         several variables are solved together.
 *
 * @tparam VARS Template parameter defining the variables used in the integrator.
 *
 * @param el      Array of pointers to finite elements.
 * @param Tr      Element transformation.
 * @param elfun   Array of local finite element solution vectors.
 * @param elmats  Array of dense matrices where the computed element Jacobian contributions
 *                will be stored.
 *
 * @note Users typically do not call this function directly; it is invoked
 *       internally during the assembly of the global nonlinear form.
 */
template <class VARS>
void MassSourceNLFormIntegrator<VARS>::AssembleElementGrad(
    const mfem::Array<const mfem::FiniteElement*>& el, mfem::ElementTransformation& Tr,
    const mfem::Array<const mfem::Vector*>& elfun,
    const mfem::Array2D<mfem::DenseMatrix*>& elmats) {
  int num_blocks = el.Size();
  for (int blk = 0; blk < num_blocks; ++blk) {
    const int nd = el[blk]->GetDof();
    for (int jblk = 0; jblk < num_blocks; ++jblk) {
      elmats(blk, jblk)->SetSize(nd, el[jblk]->GetDof());
      *elmats(blk, jblk) = 0.0;
    }
  }
}

/**
 * @brief Retrieve and store the coefficients required by the integrator.
 *
 * This method collects the coefficient of type `Source` of each block and adds it to the internal
 * storage. Only coefficients with ID 0 are considered for each block.
 *
 * @remark This method can be overridden in derived classes to provide
 *         custom behavior for retrieving coefficients.
 *
 * @tparam VARS Template parameter defining the variabls used in the integrator.
 *
 */
template <class VARS>
void MassSourceNLFormIntegrator<VARS>::get_coefficients() {
  for (unsigned int i = 0; i < this->nb_blk_; i++) {
    if (this->get_coefficient(i, GlossaryType::Source, 0).has_value()) {
      source.add(*(this->get_coefficient(i, GlossaryType::Source, 0)));
    }
  }
}

/**
 * @brief Compute the production contribution at an integration point.
 *
 * @tparam VARS Template parameter defining the variables used in the integrator.
 *
 * @param blk         Index of the block.
 * @param values      Values of the current and previous solutions at the integration point.
 * @param aux_values  Values of the auxiliary variables at the integration point.
 *
 * @return The computed contribution at the integration point.
 */
template <class VARS>
double MassSourceNLFormIntegrator<VARS>::get_rate_at_ip(
    unsigned int blk, const std::span<const double>& values,
    const std::span<const double>& aux_values) {
  return -this->compute_coefficient(source[blk], values, aux_values);
}
