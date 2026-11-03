/**
 *
 * Copyright CEA (C) 2026
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

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <span>
#include <vector>


#include "Options/PhysicalPropertiesOptions.hpp"
  
#include "Coefficients/FunctionCoefficient.hpp"
  

#pragma once

/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = eta9**2*(-cgbeq*mug/Va - cvbeq*muv/Va - muv**2/(2*Va**2*kvb) - mug**2/(2*Va**2*kgb))/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2) + mbar*(eta1**4/4 - eta1**2/2 + eta2**4/4 - eta2**2/2 + eta3**4/4 - eta3**2/2 + eta4**4/4 - eta4**2/2 + eta5**4/4 - eta5**2/2 + eta6**4/4 - eta6**2/2 + eta7**4/4 - eta7**2/2 + eta8**4/4 - eta8**2/2 + eta9**4/4 + eta9**2*gmb*(eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2) - eta9**2/2 + gmm*(-eta1**4 - eta2**4 - eta3**4 - eta4**4 - eta5**4 - eta6**4 - eta7**4 - eta8**4 + (eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)**2)/2 + 1/4) + (-cgmeq*mug/Va - cvmeq*muv/Va - muv**2/(2*Va**2*kvm) - mug**2/(2*Va**2*kgm))*(eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2)
 */
class FgFreeEnergy : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgFreeEnergy() : prefactor_(1.0) {}
 explicit FgFreeEnergy(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgFreeEnergy() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgFreeEnergy::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double eta3 = input_vector[2];
    double eta4 = input_vector[3];
    double eta5 = input_vector[4];
    double eta6 = input_vector[5];
    double eta7 = input_vector[6];
    double eta8 = input_vector[7];
    double eta9 = input_vector[8];
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double kvm = 0.04015625;
    double kgm = 0.04015625;
    double kvb = 0.04015625;
    double kgb = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
 double   F = std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*((1.0/4.0)*std::pow(eta1, 4) - 1.0/2.0*std::pow(eta1, 2) + (1.0/4.0)*std::pow(eta2, 4) - 1.0/2.0*std::pow(eta2, 2) + (1.0/4.0)*std::pow(eta3, 4) - 1.0/2.0*std::pow(eta3, 2) + (1.0/4.0)*std::pow(eta4, 4) - 1.0/2.0*std::pow(eta4, 2) + (1.0/4.0)*std::pow(eta5, 4) - 1.0/2.0*std::pow(eta5, 2) + (1.0/4.0)*std::pow(eta6, 4) - 1.0/2.0*std::pow(eta6, 2) + (1.0/4.0)*std::pow(eta7, 4) - 1.0/2.0*std::pow(eta7, 2) + (1.0/4.0)*std::pow(eta8, 4) - 1.0/2.0*std::pow(eta8, 2) + (1.0/4.0)*std::pow(eta9, 4) + std::pow(eta9, 2)*gmb*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)) - 1.0/2.0*std::pow(eta9, 2) + (1.0/2.0)*gmm*(-std::pow(eta1, 4) - std::pow(eta2, 4) - std::pow(eta3, 4) - std::pow(eta4, 4) - std::pow(eta5, 4) - std::pow(eta6, 4) - std::pow(eta7, 4) - std::pow(eta8, 4) + std::pow(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2), 2)) + 1.0/4.0) + (-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2));
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgFreeEnergy::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double eta3 = input_vector[2];
    double eta4 = input_vector[3];
    double eta5 = input_vector[4];
    double eta6 = input_vector[5];
    double eta7 = input_vector[6];
    double eta8 = input_vector[7];
    double eta9 = input_vector[8];
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double kvm = 0.04015625;
    double kgm = 0.04015625;
    double kvb = 0.04015625;
    double kgb = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> gradient(9);
    gradient[0] = this->prefactor_ * (-2*eta1*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta1*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta1*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta1, 3) + 2*eta1*std::pow(eta9, 2)*gmb - eta1 + (1.0/2.0)*gmm*(-4*std::pow(eta1, 3) + 4*eta1*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[1] = this->prefactor_ * (-2*eta2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta2, 3) + 2*eta2*std::pow(eta9, 2)*gmb - eta2 + (1.0/2.0)*gmm*(-4*std::pow(eta2, 3) + 4*eta2*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[2] = this->prefactor_ * (-2*eta3*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta3, 3) + 2*eta3*std::pow(eta9, 2)*gmb - eta3 + (1.0/2.0)*gmm*(-4*std::pow(eta3, 3) + 4*eta3*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[3] = this->prefactor_ * (-2*eta4*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta4, 3) + 2*eta4*std::pow(eta9, 2)*gmb - eta4 + (1.0/2.0)*gmm*(-4*std::pow(eta4, 3) + 4*eta4*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[4] = this->prefactor_ * (-2*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta5, 3) + 2*eta5*std::pow(eta9, 2)*gmb - eta5 + (1.0/2.0)*gmm*(-4*std::pow(eta5, 3) + 4*eta5*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[5] = this->prefactor_ * (-2*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta6, 3) + 2*eta6*std::pow(eta9, 2)*gmb - eta6 + (1.0/2.0)*gmm*(-4*std::pow(eta6, 3) + 4*eta6*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[6] = this->prefactor_ * (-2*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta7, 3) + 2*eta7*std::pow(eta9, 2)*gmb - eta7 + (1.0/2.0)*gmm*(-4*std::pow(eta7, 3) + 4*eta7*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[7] = this->prefactor_ * (-2*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + mbar*(std::pow(eta8, 3) + 2*eta8*std::pow(eta9, 2)*gmb - eta8 + (1.0/2.0)*gmm*(-4*std::pow(eta8, 3) + 4*eta8*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))));
    gradient[8] = this->prefactor_ * (-2*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) - 2*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(std::pow(eta9, 3) + 2*eta9*gmb*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)) - eta9));
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgFreeEnergy::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double eta3 = input_vector[2];
    double eta4 = input_vector[3];
    double eta5 = input_vector[4];
    double eta6 = input_vector[5];
    double eta7 = input_vector[6];
    double eta8 = input_vector[7];
    double eta9 = input_vector[8];
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double kvm = 0.04015625;
    double kgm = 0.04015625;
    double kvb = 0.04015625;
    double kgb = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> hessian(81);
    hessian[0] = this->prefactor_ * (8*std::pow(eta1, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta1, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta1, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta1, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta2, 2) + 4*std::pow(eta3, 2) + 4*std::pow(eta4, 2) + 4*std::pow(eta5, 2) + 4*std::pow(eta6, 2) + 4*std::pow(eta7, 2) + 4*std::pow(eta8, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[1] = this->prefactor_ * (8*eta1*eta2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta2*gmm*mbar + 8*eta1*eta2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[2] = this->prefactor_ * (8*eta1*eta3*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta3*gmm*mbar + 8*eta1*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[3] = this->prefactor_ * (8*eta1*eta4*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta4*gmm*mbar + 8*eta1*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[4] = this->prefactor_ * (8*eta1*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta5*gmm*mbar + 8*eta1*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[5] = this->prefactor_ * (8*eta1*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta6*gmm*mbar + 8*eta1*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[6] = this->prefactor_ * (8*eta1*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta7*gmm*mbar + 8*eta1*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[7] = this->prefactor_ * (8*eta1*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta8*gmm*mbar + 8*eta1*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[8] = this->prefactor_ * (8*eta1*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta9*gmb*mbar - 4*eta1*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta1*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta1*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[9] = this->prefactor_ * (8*eta1*eta2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta2*gmm*mbar + 8*eta1*eta2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[10] = this->prefactor_ * (8*std::pow(eta2, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta2, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta2, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta2, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta1, 2) + 4*std::pow(eta3, 2) + 4*std::pow(eta4, 2) + 4*std::pow(eta5, 2) + 4*std::pow(eta6, 2) + 4*std::pow(eta7, 2) + 4*std::pow(eta8, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[11] = this->prefactor_ * (8*eta2*eta3*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta3*gmm*mbar + 8*eta2*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[12] = this->prefactor_ * (8*eta2*eta4*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta4*gmm*mbar + 8*eta2*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[13] = this->prefactor_ * (8*eta2*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta5*gmm*mbar + 8*eta2*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[14] = this->prefactor_ * (8*eta2*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta6*gmm*mbar + 8*eta2*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[15] = this->prefactor_ * (8*eta2*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta7*gmm*mbar + 8*eta2*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[16] = this->prefactor_ * (8*eta2*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta8*gmm*mbar + 8*eta2*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[17] = this->prefactor_ * (8*eta2*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta9*gmb*mbar - 4*eta2*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta2*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta2*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[18] = this->prefactor_ * (8*eta1*eta3*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta3*gmm*mbar + 8*eta1*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[19] = this->prefactor_ * (8*eta2*eta3*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta3*gmm*mbar + 8*eta2*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta3*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[20] = this->prefactor_ * (8*std::pow(eta3, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta3, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta3, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta3, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta1, 2) + 4*std::pow(eta2, 2) + 4*std::pow(eta4, 2) + 4*std::pow(eta5, 2) + 4*std::pow(eta6, 2) + 4*std::pow(eta7, 2) + 4*std::pow(eta8, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[21] = this->prefactor_ * (8*eta3*eta4*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta4*gmm*mbar + 8*eta3*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[22] = this->prefactor_ * (8*eta3*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta5*gmm*mbar + 8*eta3*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[23] = this->prefactor_ * (8*eta3*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta6*gmm*mbar + 8*eta3*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[24] = this->prefactor_ * (8*eta3*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta7*gmm*mbar + 8*eta3*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[25] = this->prefactor_ * (8*eta3*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta8*gmm*mbar + 8*eta3*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[26] = this->prefactor_ * (8*eta3*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta9*gmb*mbar - 4*eta3*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta3*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta3*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[27] = this->prefactor_ * (8*eta1*eta4*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta4*gmm*mbar + 8*eta1*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[28] = this->prefactor_ * (8*eta2*eta4*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta4*gmm*mbar + 8*eta2*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[29] = this->prefactor_ * (8*eta3*eta4*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta4*gmm*mbar + 8*eta3*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta4*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[30] = this->prefactor_ * (8*std::pow(eta4, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta4, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta4, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta4, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta1, 2) + 4*std::pow(eta2, 2) + 4*std::pow(eta3, 2) + 4*std::pow(eta5, 2) + 4*std::pow(eta6, 2) + 4*std::pow(eta7, 2) + 4*std::pow(eta8, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[31] = this->prefactor_ * (8*eta4*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta5*gmm*mbar + 8*eta4*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[32] = this->prefactor_ * (8*eta4*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta6*gmm*mbar + 8*eta4*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[33] = this->prefactor_ * (8*eta4*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta7*gmm*mbar + 8*eta4*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[34] = this->prefactor_ * (8*eta4*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta8*gmm*mbar + 8*eta4*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[35] = this->prefactor_ * (8*eta4*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta9*gmb*mbar - 4*eta4*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta4*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta4*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[36] = this->prefactor_ * (8*eta1*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta5*gmm*mbar + 8*eta1*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[37] = this->prefactor_ * (8*eta2*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta5*gmm*mbar + 8*eta2*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[38] = this->prefactor_ * (8*eta3*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta5*gmm*mbar + 8*eta3*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[39] = this->prefactor_ * (8*eta4*eta5*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta5*gmm*mbar + 8*eta4*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta5*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[40] = this->prefactor_ * (8*std::pow(eta5, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta5, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta5, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta5, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta1, 2) + 4*std::pow(eta2, 2) + 4*std::pow(eta3, 2) + 4*std::pow(eta4, 2) + 4*std::pow(eta6, 2) + 4*std::pow(eta7, 2) + 4*std::pow(eta8, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[41] = this->prefactor_ * (8*eta5*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta6*gmm*mbar + 8*eta5*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta5*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[42] = this->prefactor_ * (8*eta5*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta7*gmm*mbar + 8*eta5*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta5*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[43] = this->prefactor_ * (8*eta5*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta8*gmm*mbar + 8*eta5*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta5*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[44] = this->prefactor_ * (8*eta5*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta9*gmb*mbar - 4*eta5*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta5*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta5*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[45] = this->prefactor_ * (8*eta1*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta6*gmm*mbar + 8*eta1*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[46] = this->prefactor_ * (8*eta2*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta6*gmm*mbar + 8*eta2*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[47] = this->prefactor_ * (8*eta3*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta6*gmm*mbar + 8*eta3*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[48] = this->prefactor_ * (8*eta4*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta6*gmm*mbar + 8*eta4*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[49] = this->prefactor_ * (8*eta5*eta6*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta6*gmm*mbar + 8*eta5*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta5*eta6*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[50] = this->prefactor_ * (8*std::pow(eta6, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta6, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta6, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta6, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta1, 2) + 4*std::pow(eta2, 2) + 4*std::pow(eta3, 2) + 4*std::pow(eta4, 2) + 4*std::pow(eta5, 2) + 4*std::pow(eta7, 2) + 4*std::pow(eta8, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[51] = this->prefactor_ * (8*eta6*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta6*eta7*gmm*mbar + 8*eta6*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta6*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[52] = this->prefactor_ * (8*eta6*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta6*eta8*gmm*mbar + 8*eta6*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta6*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[53] = this->prefactor_ * (8*eta6*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta6*eta9*gmb*mbar - 4*eta6*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta6*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta6*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[54] = this->prefactor_ * (8*eta1*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta7*gmm*mbar + 8*eta1*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[55] = this->prefactor_ * (8*eta2*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta7*gmm*mbar + 8*eta2*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[56] = this->prefactor_ * (8*eta3*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta7*gmm*mbar + 8*eta3*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[57] = this->prefactor_ * (8*eta4*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta7*gmm*mbar + 8*eta4*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[58] = this->prefactor_ * (8*eta5*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta7*gmm*mbar + 8*eta5*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta5*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[59] = this->prefactor_ * (8*eta6*eta7*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta6*eta7*gmm*mbar + 8*eta6*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta6*eta7*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[60] = this->prefactor_ * (8*std::pow(eta7, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta7, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta7, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta7, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta1, 2) + 4*std::pow(eta2, 2) + 4*std::pow(eta3, 2) + 4*std::pow(eta4, 2) + 4*std::pow(eta5, 2) + 4*std::pow(eta6, 2) + 4*std::pow(eta8, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[61] = this->prefactor_ * (8*eta7*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta7*eta8*gmm*mbar + 8*eta7*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta7*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[62] = this->prefactor_ * (8*eta7*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta7*eta9*gmb*mbar - 4*eta7*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta7*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta7*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[63] = this->prefactor_ * (8*eta1*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta8*gmm*mbar + 8*eta1*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta1*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[64] = this->prefactor_ * (8*eta2*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta8*gmm*mbar + 8*eta2*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta2*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[65] = this->prefactor_ * (8*eta3*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta8*gmm*mbar + 8*eta3*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta3*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[66] = this->prefactor_ * (8*eta4*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta8*gmm*mbar + 8*eta4*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta4*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[67] = this->prefactor_ * (8*eta5*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta8*gmm*mbar + 8*eta5*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta5*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[68] = this->prefactor_ * (8*eta6*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta6*eta8*gmm*mbar + 8*eta6*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta6*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[69] = this->prefactor_ * (8*eta7*eta8*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta7*eta8*gmm*mbar + 8*eta7*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*eta7*eta8*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[70] = this->prefactor_ * (8*std::pow(eta8, 2)*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 8*std::pow(eta8, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 8*std::pow(eta8, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) - 2*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + mbar*(3*std::pow(eta8, 2) + 2*std::pow(eta9, 2)*gmb + (1.0/2.0)*gmm*(4*std::pow(eta1, 2) + 4*std::pow(eta2, 2) + 4*std::pow(eta3, 2) + 4*std::pow(eta4, 2) + 4*std::pow(eta5, 2) + 4*std::pow(eta6, 2) + 4*std::pow(eta7, 2)) - 1) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    hessian[71] = this->prefactor_ * (8*eta8*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta8*eta9*gmb*mbar - 4*eta8*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta8*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta8*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[72] = this->prefactor_ * (8*eta1*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta1*eta9*gmb*mbar - 4*eta1*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta1*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta1*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[73] = this->prefactor_ * (8*eta2*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta2*eta9*gmb*mbar - 4*eta2*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta2*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta2*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[74] = this->prefactor_ * (8*eta3*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta3*eta9*gmb*mbar - 4*eta3*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta3*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta3*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[75] = this->prefactor_ * (8*eta4*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta4*eta9*gmb*mbar - 4*eta4*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta4*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta4*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[76] = this->prefactor_ * (8*eta5*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta5*eta9*gmb*mbar - 4*eta5*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta5*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta5*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[77] = this->prefactor_ * (8*eta6*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta6*eta9*gmb*mbar - 4*eta6*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta6*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta6*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[78] = this->prefactor_ * (8*eta7*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta7*eta9*gmb*mbar - 4*eta7*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta7*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta7*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[79] = this->prefactor_ * (8*eta8*std::pow(eta9, 3)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + 4*eta8*eta9*gmb*mbar - 4*eta8*eta9*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*eta8*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 4*eta8*eta9*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    hessian[80] = this->prefactor_ * (8*std::pow(eta9, 4)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) - 10*std::pow(eta9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2) + 8*std::pow(eta9, 2)*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 3) + mbar*(3*std::pow(eta9, 2) + 2*gmb*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)) - 1) + 2*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) - 2*(-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2), 2));
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = mbar*(gmb*(en9**2*(-en1**2 - en2**2 - en3**2 - en4**2 - en5**2 - en6**2 - en7**2 - en8**2) + en9**2*(eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2) + eta9**2*(en1**2 + en2**2 + en3**2 + en4**2 + en5**2 + en6**2 + en7**2 + en8**2)) + gmm*(en1**4/2 - en1**2*eta1**2 + en2**4/2 - en2**2*eta2**2 + en3**4/2 - en3**2*eta3**2 + en4**4/2 - en4**2*eta4**2 + en5**4/2 - en5**2*eta5**2 + en6**4/2 - en6**2*eta6**2 + en7**4/2 - en7**2*eta7**2 + en8**4/2 - en8**2*eta8**2 - (en1**2 + en2**2 + en3**2 + en4**2 + en5**2 + en6**2 + en7**2 + en8**2)**2/2 + (en1**2 + en2**2 + en3**2 + en4**2 + en5**2 + en6**2 + en7**2 + en8**2)*(eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2))) + mbar*(eta1**4/4 - eta1**2/2 + eta2**4/4 - eta2**2/2 + eta3**4/4 - eta3**2/2 + eta4**4/4 - eta4**2/2 + eta5**4/4 - eta5**2/2 + eta6**4/4 - eta6**2/2 + eta7**4/4 - eta7**2/2 + eta8**4/4 - eta8**2/2 + eta9**4/4 - eta9**2/2 + 1/4) + (en9**2*(-cgbeq*mug/Va - cvbeq*muv/Va - muv**2/(2*Va**2*kvb) - mug**2/(2*Va**2*kgb)) + (-cgmeq*mug/Va - cvmeq*muv/Va - muv**2/(2*Va**2*kvm) - mug**2/(2*Va**2*kgm))*(en1**2 + en2**2 + en3**2 + en4**2 + en5**2 + en6**2 + en7**2 + en8**2))/(en1**2 + en2**2 + en3**2 + en4**2 + en5**2 + en6**2 + en7**2 + en8**2 + en9**2 + eps) + (en9**2*(eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2) + eta9**2*(-en1**2 - en2**2 - en3**2 - en4**2 - en5**2 - en6**2 - en7**2 - en8**2))*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - muv**2/(2*Va**2*kvm) + muv**2/(2*Va**2*kvb) - mug**2/(2*Va**2*kgm) + mug**2/(2*Va**2*kgb))/(en1**2 + en2**2 + en3**2 + en4**2 + en5**2 + en6**2 + en7**2 + en8**2 + en9**2 + eps)**2
 */
class FgFreeEnergySI : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgFreeEnergySI() : prefactor_(1.0) {}
 explicit FgFreeEnergySI(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgFreeEnergySI() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgFreeEnergySI::F() {
  auto func = [&](const std::span<const double>& input_vector, const std::span<const double>& exp_input_vector, const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double eta3 = input_vector[2];
    double eta4 = input_vector[3];
    double eta5 = input_vector[4];
    double eta6 = input_vector[5];
    double eta7 = input_vector[6];
    double eta8 = input_vector[7];
    double eta9 = input_vector[8];
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double en1 = exp_input_vector[0];
    double en2 = exp_input_vector[1];
    double en3 = exp_input_vector[2];
    double en4 = exp_input_vector[3];
    double en5 = exp_input_vector[4];
    double en6 = exp_input_vector[5];
    double en7 = exp_input_vector[6];
    double en8 = exp_input_vector[7];
    double en9 = exp_input_vector[8];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double kvm = 0.04015625;
    double kgm = 0.04015625;
    double kvb = 0.04015625;
    double kgb = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
 double   F = mbar*(gmb*(std::pow(en9, 2)*(-std::pow(en1, 2) - std::pow(en2, 2) - std::pow(en3, 2) - std::pow(en4, 2) - std::pow(en5, 2) - std::pow(en6, 2) - std::pow(en7, 2) - std::pow(en8, 2)) + std::pow(en9, 2)*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)) + std::pow(eta9, 2)*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2))) + gmm*((1.0/2.0)*std::pow(en1, 4) - std::pow(en1, 2)*std::pow(eta1, 2) + (1.0/2.0)*std::pow(en2, 4) - std::pow(en2, 2)*std::pow(eta2, 2) + (1.0/2.0)*std::pow(en3, 4) - std::pow(en3, 2)*std::pow(eta3, 2) + (1.0/2.0)*std::pow(en4, 4) - std::pow(en4, 2)*std::pow(eta4, 2) + (1.0/2.0)*std::pow(en5, 4) - std::pow(en5, 2)*std::pow(eta5, 2) + (1.0/2.0)*std::pow(en6, 4) - std::pow(en6, 2)*std::pow(eta6, 2) + (1.0/2.0)*std::pow(en7, 4) - std::pow(en7, 2)*std::pow(eta7, 2) + (1.0/2.0)*std::pow(en8, 4) - std::pow(en8, 2)*std::pow(eta8, 2) - 1.0/2.0*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2), 2) + (std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2))*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)))) + mbar*((1.0/4.0)*std::pow(eta1, 4) - 1.0/2.0*std::pow(eta1, 2) + (1.0/4.0)*std::pow(eta2, 4) - 1.0/2.0*std::pow(eta2, 2) + (1.0/4.0)*std::pow(eta3, 4) - 1.0/2.0*std::pow(eta3, 2) + (1.0/4.0)*std::pow(eta4, 4) - 1.0/2.0*std::pow(eta4, 2) + (1.0/4.0)*std::pow(eta5, 4) - 1.0/2.0*std::pow(eta5, 2) + (1.0/4.0)*std::pow(eta6, 4) - 1.0/2.0*std::pow(eta6, 2) + (1.0/4.0)*std::pow(eta7, 4) - 1.0/2.0*std::pow(eta7, 2) + (1.0/4.0)*std::pow(eta8, 4) - 1.0/2.0*std::pow(eta8, 2) + (1.0/4.0)*std::pow(eta9, 4) - 1.0/2.0*std::pow(eta9, 2) + 1.0/4.0) + (std::pow(en9, 2)*(-cgbeq*mug/Va - cvbeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgb)) + (-cgmeq*mug/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm))*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))/(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps) + (std::pow(en9, 2)*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2)) + std::pow(eta9, 2)*(-std::pow(en1, 2) - std::pow(en2, 2) - std::pow(en3, 2) - std::pow(en4, 2) - std::pow(en5, 2) - std::pow(en6, 2) - std::pow(en7, 2) - std::pow(en8, 2)))*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2);
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgFreeEnergySI::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, const std::span<const double>& exp_input_vector, const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double eta3 = input_vector[2];
    double eta4 = input_vector[3];
    double eta5 = input_vector[4];
    double eta6 = input_vector[5];
    double eta7 = input_vector[6];
    double eta8 = input_vector[7];
    double eta9 = input_vector[8];
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double en1 = exp_input_vector[0];
    double en2 = exp_input_vector[1];
    double en3 = exp_input_vector[2];
    double en4 = exp_input_vector[3];
    double en5 = exp_input_vector[4];
    double en6 = exp_input_vector[5];
    double en7 = exp_input_vector[6];
    double en8 = exp_input_vector[7];
    double en9 = exp_input_vector[8];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double kvm = 0.04015625;
    double kgm = 0.04015625;
    double kvb = 0.04015625;
    double kgb = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> gradient(9);
    gradient[0] = this->prefactor_ * (2*std::pow(en9, 2)*eta1*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta1, 3) - eta1) + mbar*(2*std::pow(en9, 2)*eta1*gmb + gmm*(-2*std::pow(en1, 2)*eta1 + 2*eta1*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[1] = this->prefactor_ * (2*std::pow(en9, 2)*eta2*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta2, 3) - eta2) + mbar*(2*std::pow(en9, 2)*eta2*gmb + gmm*(-2*std::pow(en2, 2)*eta2 + 2*eta2*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[2] = this->prefactor_ * (2*std::pow(en9, 2)*eta3*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta3, 3) - eta3) + mbar*(2*std::pow(en9, 2)*eta3*gmb + gmm*(-2*std::pow(en3, 2)*eta3 + 2*eta3*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[3] = this->prefactor_ * (2*std::pow(en9, 2)*eta4*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta4, 3) - eta4) + mbar*(2*std::pow(en9, 2)*eta4*gmb + gmm*(-2*std::pow(en4, 2)*eta4 + 2*eta4*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[4] = this->prefactor_ * (2*std::pow(en9, 2)*eta5*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta5, 3) - eta5) + mbar*(2*std::pow(en9, 2)*eta5*gmb + gmm*(-2*std::pow(en5, 2)*eta5 + 2*eta5*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[5] = this->prefactor_ * (2*std::pow(en9, 2)*eta6*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta6, 3) - eta6) + mbar*(2*std::pow(en9, 2)*eta6*gmb + gmm*(-2*std::pow(en6, 2)*eta6 + 2*eta6*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[6] = this->prefactor_ * (2*std::pow(en9, 2)*eta7*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta7, 3) - eta7) + mbar*(2*std::pow(en9, 2)*eta7*gmb + gmm*(-2*std::pow(en7, 2)*eta7 + 2*eta7*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[7] = this->prefactor_ * (2*std::pow(en9, 2)*eta8*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta8, 3) - eta8) + mbar*(2*std::pow(en9, 2)*eta8*gmb + gmm*(-2*std::pow(en8, 2)*eta8 + 2*eta8*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)))));
    gradient[8] = this->prefactor_ * (2*eta9*gmb*mbar*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)) + 2*eta9*(-std::pow(en1, 2) - std::pow(en2, 2) - std::pow(en3, 2) - std::pow(en4, 2) - std::pow(en5, 2) - std::pow(en6, 2) - std::pow(en7, 2) - std::pow(en8, 2))*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(std::pow(eta9, 3) - eta9));
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgFreeEnergySI::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, const std::span<const double>& exp_input_vector, const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double eta3 = input_vector[2];
    double eta4 = input_vector[3];
    double eta5 = input_vector[4];
    double eta6 = input_vector[5];
    double eta7 = input_vector[6];
    double eta8 = input_vector[7];
    double eta9 = input_vector[8];
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double en1 = exp_input_vector[0];
    double en2 = exp_input_vector[1];
    double en3 = exp_input_vector[2];
    double en4 = exp_input_vector[3];
    double en5 = exp_input_vector[4];
    double en6 = exp_input_vector[5];
    double en7 = exp_input_vector[6];
    double en8 = exp_input_vector[7];
    double en9 = exp_input_vector[8];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double kvm = 0.04015625;
    double kgm = 0.04015625;
    double kvb = 0.04015625;
    double kgb = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> hessian(81);
    hessian[0] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta1, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en2, 2) + 2*std::pow(en3, 2) + 2*std::pow(en4, 2) + 2*std::pow(en5, 2) + 2*std::pow(en6, 2) + 2*std::pow(en7, 2) + 2*std::pow(en8, 2))));
    hessian[1] = this->prefactor_ * (0);
    hessian[2] = this->prefactor_ * (0);
    hessian[3] = this->prefactor_ * (0);
    hessian[4] = this->prefactor_ * (0);
    hessian[5] = this->prefactor_ * (0);
    hessian[6] = this->prefactor_ * (0);
    hessian[7] = this->prefactor_ * (0);
    hessian[8] = this->prefactor_ * (0);
    hessian[9] = this->prefactor_ * (0);
    hessian[10] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta2, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en1, 2) + 2*std::pow(en3, 2) + 2*std::pow(en4, 2) + 2*std::pow(en5, 2) + 2*std::pow(en6, 2) + 2*std::pow(en7, 2) + 2*std::pow(en8, 2))));
    hessian[11] = this->prefactor_ * (0);
    hessian[12] = this->prefactor_ * (0);
    hessian[13] = this->prefactor_ * (0);
    hessian[14] = this->prefactor_ * (0);
    hessian[15] = this->prefactor_ * (0);
    hessian[16] = this->prefactor_ * (0);
    hessian[17] = this->prefactor_ * (0);
    hessian[18] = this->prefactor_ * (0);
    hessian[19] = this->prefactor_ * (0);
    hessian[20] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta3, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en1, 2) + 2*std::pow(en2, 2) + 2*std::pow(en4, 2) + 2*std::pow(en5, 2) + 2*std::pow(en6, 2) + 2*std::pow(en7, 2) + 2*std::pow(en8, 2))));
    hessian[21] = this->prefactor_ * (0);
    hessian[22] = this->prefactor_ * (0);
    hessian[23] = this->prefactor_ * (0);
    hessian[24] = this->prefactor_ * (0);
    hessian[25] = this->prefactor_ * (0);
    hessian[26] = this->prefactor_ * (0);
    hessian[27] = this->prefactor_ * (0);
    hessian[28] = this->prefactor_ * (0);
    hessian[29] = this->prefactor_ * (0);
    hessian[30] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta4, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en1, 2) + 2*std::pow(en2, 2) + 2*std::pow(en3, 2) + 2*std::pow(en5, 2) + 2*std::pow(en6, 2) + 2*std::pow(en7, 2) + 2*std::pow(en8, 2))));
    hessian[31] = this->prefactor_ * (0);
    hessian[32] = this->prefactor_ * (0);
    hessian[33] = this->prefactor_ * (0);
    hessian[34] = this->prefactor_ * (0);
    hessian[35] = this->prefactor_ * (0);
    hessian[36] = this->prefactor_ * (0);
    hessian[37] = this->prefactor_ * (0);
    hessian[38] = this->prefactor_ * (0);
    hessian[39] = this->prefactor_ * (0);
    hessian[40] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta5, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en1, 2) + 2*std::pow(en2, 2) + 2*std::pow(en3, 2) + 2*std::pow(en4, 2) + 2*std::pow(en6, 2) + 2*std::pow(en7, 2) + 2*std::pow(en8, 2))));
    hessian[41] = this->prefactor_ * (0);
    hessian[42] = this->prefactor_ * (0);
    hessian[43] = this->prefactor_ * (0);
    hessian[44] = this->prefactor_ * (0);
    hessian[45] = this->prefactor_ * (0);
    hessian[46] = this->prefactor_ * (0);
    hessian[47] = this->prefactor_ * (0);
    hessian[48] = this->prefactor_ * (0);
    hessian[49] = this->prefactor_ * (0);
    hessian[50] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta6, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en1, 2) + 2*std::pow(en2, 2) + 2*std::pow(en3, 2) + 2*std::pow(en4, 2) + 2*std::pow(en5, 2) + 2*std::pow(en7, 2) + 2*std::pow(en8, 2))));
    hessian[51] = this->prefactor_ * (0);
    hessian[52] = this->prefactor_ * (0);
    hessian[53] = this->prefactor_ * (0);
    hessian[54] = this->prefactor_ * (0);
    hessian[55] = this->prefactor_ * (0);
    hessian[56] = this->prefactor_ * (0);
    hessian[57] = this->prefactor_ * (0);
    hessian[58] = this->prefactor_ * (0);
    hessian[59] = this->prefactor_ * (0);
    hessian[60] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta7, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en1, 2) + 2*std::pow(en2, 2) + 2*std::pow(en3, 2) + 2*std::pow(en4, 2) + 2*std::pow(en5, 2) + 2*std::pow(en6, 2) + 2*std::pow(en8, 2))));
    hessian[61] = this->prefactor_ * (0);
    hessian[62] = this->prefactor_ * (0);
    hessian[63] = this->prefactor_ * (0);
    hessian[64] = this->prefactor_ * (0);
    hessian[65] = this->prefactor_ * (0);
    hessian[66] = this->prefactor_ * (0);
    hessian[67] = this->prefactor_ * (0);
    hessian[68] = this->prefactor_ * (0);
    hessian[69] = this->prefactor_ * (0);
    hessian[70] = this->prefactor_ * (2*std::pow(en9, 2)*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2) + mbar*(3*std::pow(eta8, 2) - 1) + mbar*(2*std::pow(en9, 2)*gmb + gmm*(2*std::pow(en1, 2) + 2*std::pow(en2, 2) + 2*std::pow(en3, 2) + 2*std::pow(en4, 2) + 2*std::pow(en5, 2) + 2*std::pow(en6, 2) + 2*std::pow(en7, 2))));
    hessian[71] = this->prefactor_ * (0);
    hessian[72] = this->prefactor_ * (0);
    hessian[73] = this->prefactor_ * (0);
    hessian[74] = this->prefactor_ * (0);
    hessian[75] = this->prefactor_ * (0);
    hessian[76] = this->prefactor_ * (0);
    hessian[77] = this->prefactor_ * (0);
    hessian[78] = this->prefactor_ * (0);
    hessian[79] = this->prefactor_ * (0);
    hessian[80] = this->prefactor_ * (2*gmb*mbar*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2)) + mbar*(3*std::pow(eta9, 2) - 1) + 2*(-std::pow(en1, 2) - std::pow(en2, 2) - std::pow(en3, 2) - std::pow(en4, 2) - std::pow(en5, 2) - std::pow(en6, 2) - std::pow(en7, 2) - std::pow(en8, 2))*(cgbeq*mug/Va - cgmeq*mug/Va + cvbeq*muv/Va - cvmeq*muv/Va - 1.0/2.0*std::pow(muv, 2)/(std::pow(Va, 2)*kvm) + (1.0/2.0)*std::pow(muv, 2)/(std::pow(Va, 2)*kvb) - 1.0/2.0*std::pow(mug, 2)/(std::pow(Va, 2)*kgm) + (1.0/2.0)*std::pow(mug, 2)/(std::pow(Va, 2)*kgb))/std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + std::pow(en4, 2) + std::pow(en5, 2) + std::pow(en6, 2) + std::pow(en7, 2) + std::pow(en8, 2) + std::pow(en9, 2) + eps, 2));
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = 135*dot(eta1, eta1)/512 + 135*dot(eta2, eta2)/512 + 135*dot(eta3, eta3)/512 + 135*dot(eta4, eta4)/512 + 135*dot(eta5, eta5)/512 + 135*dot(eta6, eta6)/512 + 135*dot(eta7, eta7)/512 + 135*dot(eta8, eta8)/512 + 135*dot(eta9, eta9)/512
 */
class FgGradEnergy : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgGradEnergy() : prefactor_(1.0) {}
 explicit FgGradEnergy(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgGradEnergy() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgGradEnergy::F() {
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    std::vector<double> eta1;
    for(unsigned int i=0;i<dimension;i++) eta1.push_back(input_vector[0*dimension+i]);
    std::vector<double> eta2;
    for(unsigned int i=0;i<dimension;i++) eta2.push_back(input_vector[1*dimension+i]);
    std::vector<double> eta3;
    for(unsigned int i=0;i<dimension;i++) eta3.push_back(input_vector[2*dimension+i]);
    std::vector<double> eta4;
    for(unsigned int i=0;i<dimension;i++) eta4.push_back(input_vector[3*dimension+i]);
    std::vector<double> eta5;
    for(unsigned int i=0;i<dimension;i++) eta5.push_back(input_vector[4*dimension+i]);
    std::vector<double> eta6;
    for(unsigned int i=0;i<dimension;i++) eta6.push_back(input_vector[5*dimension+i]);
    std::vector<double> eta7;
    for(unsigned int i=0;i<dimension;i++) eta7.push_back(input_vector[6*dimension+i]);
    std::vector<double> eta8;
    for(unsigned int i=0;i<dimension;i++) eta8.push_back(input_vector[7*dimension+i]);
    std::vector<double> eta9;
    for(unsigned int i=0;i<dimension;i++) eta9.push_back(input_vector[8*dimension+i]);
    double F = 0.263671875*std::inner_product(eta1.begin(), eta1.end(), eta1.begin(), 0.0) + 0.263671875*std::inner_product(eta2.begin(), eta2.end(), eta2.begin(), 0.0) + 0.263671875*std::inner_product(eta3.begin(), eta3.end(), eta3.begin(), 0.0) + 0.263671875*std::inner_product(eta4.begin(), eta4.end(), eta4.begin(), 0.0) + 0.263671875*std::inner_product(eta5.begin(), eta5.end(), eta5.begin(), 0.0) + 0.263671875*std::inner_product(eta6.begin(), eta6.end(), eta6.begin(), 0.0) + 0.263671875*std::inner_product(eta7.begin(), eta7.end(), eta7.begin(), 0.0) + 0.263671875*std::inner_product(eta8.begin(), eta8.end(), eta8.begin(), 0.0) + 0.263671875*std::inner_product(eta9.begin(), eta9.end(), eta9.begin(), 0.0);
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgGradEnergy::GradientF() {
  auto func = [&]([[maybe_unused]] const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    std::vector<double> gradient(9,0.0);
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgGradEnergy::HessianF() {
  auto func = [&]([[maybe_unused]] const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,  [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    std::vector<double> hessian(81,0.0);
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = eta9**2/(Va**2*kb*(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2)) + (eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)/(Va**2*km*(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2))
 */
class FgSusceptibility : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgSusceptibility() : prefactor_(1.0) {}
 explicit FgSusceptibility(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgSusceptibility() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgSusceptibility::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double eps = 1.0e-12;
 double   F = std::pow(eta9, 2)/(std::pow(Va, 2)*kb*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2))) + (std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(std::pow(Va, 2)*km*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)));
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgSusceptibility::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double eps = 1.0e-12;
    std::vector<double> gradient(1);
    gradient[0] = this->prefactor_ * (0);
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgSusceptibility::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double eps = 1.0e-12;
    std::vector<double> hessian(1);
    hessian[0] = this->prefactor_ * (0);
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = mu*(eta9**2/(Va**2*kb*(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2)) + (eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)/(Va**2*km*(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2))) + (cbeq*eta9**2/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2) + cmeq*(eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2))/Va
 */
class FgDensityVac : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgDensityVac() : prefactor_(1.0) {}
 explicit FgDensityVac(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgDensityVac() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgDensityVac::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.546;
    double eps = 1.0e-12;
 double   F = mu*(std::pow(eta9, 2)/(std::pow(Va, 2)*kb*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2))) + (std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(std::pow(Va, 2)*km*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)))) + (cbeq*std::pow(eta9, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + cmeq*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)))/Va;
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgDensityVac::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.546;
    double eps = 1.0e-12;
    std::vector<double> gradient(1);
    gradient[0] = this->prefactor_ * (std::pow(eta9, 2)/(std::pow(Va, 2)*kb*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2))) + (std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(std::pow(Va, 2)*km*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2))));
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgDensityVac::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.546;
    double eps = 1.0e-12;
    std::vector<double> hessian(1);
    hessian[0] = this->prefactor_ * (0);
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = mu*(eta9**2/(Va**2*kb*(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2)) + (eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)/(Va**2*km*(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2))) + (cbeq*eta9**2/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2) + cmeq*(eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2))/Va
 */
class FgDensityGas : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgDensityGas() : prefactor_(1.0) {}
 explicit FgDensityGas(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgDensityGas() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgDensityGas::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.454;
    double eps = 1.0e-12;
 double   F = mu*(std::pow(eta9, 2)/(std::pow(Va, 2)*kb*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2))) + (std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(std::pow(Va, 2)*km*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)))) + (cbeq*std::pow(eta9, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)) + cmeq*(std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2)))/Va;
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgDensityGas::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> gradient(1);
    gradient[0] = this->prefactor_ * (std::pow(eta9, 2)/(std::pow(Va, 2)*kb*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2))) + (std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(std::pow(Va, 2)*km*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2))));
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgDensityGas::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double Va = 0.0409;
    double km = 0.04015625;
    double kb = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> hessian(1);
    hessian[0] = this->prefactor_ * (0);
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = (eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2)/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2)
 */
class FgMatrixFraction : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgMatrixFraction() : prefactor_(1.0) {}
 explicit FgMatrixFraction(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgMatrixFraction() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMatrixFraction::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double eps = 1.0e-12;
 double   F = (std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2));
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMatrixFraction::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double eps = 1.0e-12;
    std::vector<double> gradient(1);
    gradient[0] = this->prefactor_ * (0);
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMatrixFraction::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double eps = 1.0e-12;
    std::vector<double> hessian(1);
    hessian[0] = this->prefactor_ * (0);
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = eta9**2/(eps + eta1**2 + eta2**2 + eta3**2 + eta4**2 + eta5**2 + eta6**2 + eta7**2 + eta8**2 + eta9**2)
 */
class SwitchingFunctionBubble : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  SwitchingFunctionBubble() : prefactor_(1.0) {}
 explicit SwitchingFunctionBubble(const double prefactor): prefactor_(prefactor) {}
  virtual ~SwitchingFunctionBubble() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionBubble::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double eps = 1.0e-12;
 double   F = std::pow(eta9, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2) + std::pow(eta4, 2) + std::pow(eta5, 2) + std::pow(eta6, 2) + std::pow(eta7, 2) + std::pow(eta8, 2) + std::pow(eta9, 2));
    return this->prefactor_ * F;
  };
  return func;
}

/**
 *
 * @brief Gradient
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionBubble::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double eps = 1.0e-12;
    std::vector<double> gradient(1);
    gradient[0] = this->prefactor_ * (0);
    return gradient;
  };
  return func;
}

/**
 *
 * @brief Hessian
 * @remark Hessian matrix stored in vector : H(i,j)->H(i*n+j)
 * 
 * @return std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> 
 */
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionBubble::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eta4 = auxiliary_vector[3];
    double eta5 = auxiliary_vector[4];
    double eta6 = auxiliary_vector[5];
    double eta7 = auxiliary_vector[6];
    double eta8 = auxiliary_vector[7];
    double eta9 = auxiliary_vector[8];
    double eps = 1.0e-12;
    std::vector<double> hessian(1);
    hessian[0] = this->prefactor_ * (0);
    return hessian;
  };
  return func;
}
