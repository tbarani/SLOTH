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
 *       F = mbar*(eta1**4/4 + eta1**2*eta2**2*gmm - eta1**2/2 + eta2**4/4 - eta2**2/2 + eta3**4/4 - eta3**2/2 + gmb*(eta1**2*eta3**2 + eta2**2*eta3**2) + 1/4) - eta3**2*(cgbeq*mug + cvbeq*muv)/(Va*(eps + eta1**2 + eta2**2 + eta3**2)) - (eta1**2 + eta2**2)*(cgmeq*mug + cvmeq*muv)/(Va*(eps + eta1**2 + eta2**2 + eta3**2)) - (mug**2/2 + muv**2/2)/(Va**2*k)
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
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double k = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
 double   F = mbar*((1.0/4.0)*std::pow(eta1, 4) + std::pow(eta1, 2)*std::pow(eta2, 2)*gmm - 1.0/2.0*std::pow(eta1, 2) + (1.0/4.0)*std::pow(eta2, 4) - 1.0/2.0*std::pow(eta2, 2) + (1.0/4.0)*std::pow(eta3, 4) - 1.0/2.0*std::pow(eta3, 2) + gmb*(std::pow(eta1, 2)*std::pow(eta3, 2) + std::pow(eta2, 2)*std::pow(eta3, 2)) + 1.0/4.0) - std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))) - (std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))) - ((1.0/2.0)*std::pow(mug, 2) + (1.0/2.0)*std::pow(muv, 2))/(std::pow(Va, 2)*k);
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
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double k = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> gradient(3);
    gradient[0] = this->prefactor_ * (mbar*(std::pow(eta1, 3) + 2*eta1*std::pow(eta2, 2)*gmm + 2*eta1*std::pow(eta3, 2)*gmb - eta1) + 2*eta1*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*eta1*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) - 2*eta1*(cgmeq*mug + cvmeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))));
    gradient[1] = this->prefactor_ * (mbar*(2*std::pow(eta1, 2)*eta2*gmm + std::pow(eta2, 3) + 2*eta2*std::pow(eta3, 2)*gmb - eta2) + 2*eta2*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*eta2*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) - 2*eta2*(cgmeq*mug + cvmeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))));
    gradient[2] = this->prefactor_ * (mbar*(std::pow(eta3, 3) - eta3 + gmb*(2*std::pow(eta1, 2)*eta3 + 2*std::pow(eta2, 2)*eta3)) + 2*std::pow(eta3, 3)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*eta3*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) - 2*eta3*(cgbeq*mug + cvbeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))));
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
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double k = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> hessian(9);
    hessian[0] = this->prefactor_ * (mbar*(3*std::pow(eta1, 2) + 2*std::pow(eta2, 2)*gmm + 2*std::pow(eta3, 2)*gmb - 1) - 8*std::pow(eta1, 2)*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*std::pow(eta1, 2)*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 8*std::pow(eta1, 2)*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) - 2*(cgmeq*mug + cvmeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))));
    hessian[1] = this->prefactor_ * (4*eta1*eta2*gmm*mbar - 8*eta1*eta2*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*eta1*eta2*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 8*eta1*eta2*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)));
    hessian[2] = this->prefactor_ * (4*eta1*eta3*gmb*mbar - 8*eta1*std::pow(eta3, 3)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*eta1*eta3*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 4*eta1*eta3*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 4*eta1*eta3*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)));
    hessian[3] = this->prefactor_ * (4*eta1*eta2*gmm*mbar - 8*eta1*eta2*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*eta1*eta2*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 8*eta1*eta2*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)));
    hessian[4] = this->prefactor_ * (mbar*(2*std::pow(eta1, 2)*gmm + 3*std::pow(eta2, 2) + 2*std::pow(eta3, 2)*gmb - 1) - 8*std::pow(eta2, 2)*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*std::pow(eta2, 2)*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 8*std::pow(eta2, 2)*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) - 2*(cgmeq*mug + cvmeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))));
    hessian[5] = this->prefactor_ * (4*eta2*eta3*gmb*mbar - 8*eta2*std::pow(eta3, 3)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*eta2*eta3*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 4*eta2*eta3*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 4*eta2*eta3*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)));
    hessian[6] = this->prefactor_ * (4*eta1*eta3*gmb*mbar - 8*eta1*std::pow(eta3, 3)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*eta1*eta3*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 4*eta1*eta3*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 4*eta1*eta3*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)));
    hessian[7] = this->prefactor_ * (4*eta2*eta3*gmb*mbar - 8*eta2*std::pow(eta3, 3)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*eta2*eta3*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 4*eta2*eta3*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 4*eta2*eta3*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)));
    hessian[8] = this->prefactor_ * (mbar*(3*std::pow(eta3, 2) + gmb*(2*std::pow(eta1, 2) + 2*std::pow(eta2, 2)) - 1) - 8*std::pow(eta3, 4)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) - 8*std::pow(eta3, 2)*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 3)) + 10*std::pow(eta3, 2)*(cgbeq*mug + cvbeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) + 2*(std::pow(eta1, 2) + std::pow(eta2, 2))*(cgmeq*mug + cvmeq*muv)/(Va*std::pow(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2), 2)) - 2*(cgbeq*mug + cvbeq*muv)/(Va*(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2))));
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = mbar*(gmb*(-en1**2*en3**2 + en1**2*eta3**2 + en3**2*eta1**2) + gmb*(-en2**2*en3**2 + en2**2*eta3**2 + en3**2*eta2**2) + gmm*(-en1**2*en2**2 + en1**2*eta2**2 + en2**2*eta1**2)) + mbar*(eta1**4/4 - eta1**2/2 + eta2**4/4 - eta2**2/2 + eta3**4/4 - eta3**2/2 + 1/4) + (en3**2*(eta1**2 + eta2**2) - eta3**2*(en1**2 + en2**2))*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*(en1**2 + en2**2 + en3**2 + eps)**2) - (en3**2*(cgbeq*mug + cvbeq*muv) + (en1**2 + en2**2)*(cgmeq*mug + cvmeq*muv))/(Va*(en1**2 + en2**2 + en3**2 + eps)) - (mug**2/2 + muv**2/2)/(Va**2*k)
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
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double en1 = exp_input_vector[0];
    double en2 = exp_input_vector[1];
    double en3 = exp_input_vector[2];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double k = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
 double   F = mbar*(gmb*(-std::pow(en1, 2)*std::pow(en3, 2) + std::pow(en1, 2)*std::pow(eta3, 2) + std::pow(en3, 2)*std::pow(eta1, 2)) + gmb*(-std::pow(en2, 2)*std::pow(en3, 2) + std::pow(en2, 2)*std::pow(eta3, 2) + std::pow(en3, 2)*std::pow(eta2, 2)) + gmm*(-std::pow(en1, 2)*std::pow(en2, 2) + std::pow(en1, 2)*std::pow(eta2, 2) + std::pow(en2, 2)*std::pow(eta1, 2))) + mbar*((1.0/4.0)*std::pow(eta1, 4) - 1.0/2.0*std::pow(eta1, 2) + (1.0/4.0)*std::pow(eta2, 4) - 1.0/2.0*std::pow(eta2, 2) + (1.0/4.0)*std::pow(eta3, 4) - 1.0/2.0*std::pow(eta3, 2) + 1.0/4.0) + (std::pow(en3, 2)*(std::pow(eta1, 2) + std::pow(eta2, 2)) - std::pow(eta3, 2)*(std::pow(en1, 2) + std::pow(en2, 2)))*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps, 2)) - (std::pow(en3, 2)*(cgbeq*mug + cvbeq*muv) + (std::pow(en1, 2) + std::pow(en2, 2))*(cgmeq*mug + cvmeq*muv))/(Va*(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps)) - ((1.0/2.0)*std::pow(mug, 2) + (1.0/2.0)*std::pow(muv, 2))/(std::pow(Va, 2)*k);
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
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double en1 = exp_input_vector[0];
    double en2 = exp_input_vector[1];
    double en3 = exp_input_vector[2];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double k = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> gradient(3);
    gradient[0] = this->prefactor_ * (mbar*(std::pow(eta1, 3) - eta1) + mbar*(2*std::pow(en2, 2)*eta1*gmm + 2*std::pow(en3, 2)*eta1*gmb) + 2*std::pow(en3, 2)*eta1*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps, 2)));
    gradient[1] = this->prefactor_ * (mbar*(std::pow(eta2, 3) - eta2) + mbar*(2*std::pow(en1, 2)*eta2*gmm + 2*std::pow(en3, 2)*eta2*gmb) + 2*std::pow(en3, 2)*eta2*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps, 2)));
    gradient[2] = this->prefactor_ * (mbar*(std::pow(eta3, 3) - eta3) + mbar*(2*std::pow(en1, 2)*eta3*gmb + 2*std::pow(en2, 2)*eta3*gmb) - 2*eta3*(std::pow(en1, 2) + std::pow(en2, 2))*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps, 2)));
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
    double muv = auxiliary_vector[0];
    double mug = auxiliary_vector[1];
    double en1 = exp_input_vector[0];
    double en2 = exp_input_vector[1];
    double en3 = exp_input_vector[2];
    double mbar = 0.0046875;
    double gmm = 1.5;
    double gmb = 0.922;
    double Va = 0.0409;
    double k = 0.04015625;
    double cvmeq = 2.507e-13;
    double cgmeq = 2.507e-13;
    double cvbeq = 0.546;
    double cgbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> hessian(9);
    hessian[0] = this->prefactor_ * (mbar*(3*std::pow(eta1, 2) - 1) + mbar*(2*std::pow(en2, 2)*gmm + 2*std::pow(en3, 2)*gmb) + 2*std::pow(en3, 2)*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps, 2)));
    hessian[1] = this->prefactor_ * (0);
    hessian[2] = this->prefactor_ * (0);
    hessian[3] = this->prefactor_ * (0);
    hessian[4] = this->prefactor_ * (mbar*(3*std::pow(eta2, 2) - 1) + mbar*(2*std::pow(en1, 2)*gmm + 2*std::pow(en3, 2)*gmb) + 2*std::pow(en3, 2)*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps, 2)));
    hessian[5] = this->prefactor_ * (0);
    hessian[6] = this->prefactor_ * (0);
    hessian[7] = this->prefactor_ * (0);
    hessian[8] = this->prefactor_ * (mbar*(3*std::pow(eta3, 2) - 1) + mbar*(2*std::pow(en1, 2)*gmb + 2*std::pow(en2, 2)*gmb) - 2*(std::pow(en1, 2) + std::pow(en2, 2))*(mug*(cgbeq - cgmeq) + muv*(cvbeq - cvmeq))/(Va*std::pow(std::pow(en1, 2) + std::pow(en2, 2) + std::pow(en3, 2) + eps, 2)));
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = 135*dot(eta1, eta1)/512 + 135*dot(eta2, eta2)/512 + 135*dot(eta3, eta3)/512
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
    double F = 0.263671875*std::inner_product(eta1.begin(), eta1.end(), eta1.begin(), 0.0) + 0.263671875*std::inner_product(eta2.begin(), eta2.end(), eta2.begin(), 0.0) + 0.263671875*std::inner_product(eta3.begin(), eta3.end(), eta3.begin(), 0.0);
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
    std::vector<double> gradient(3,0.0);
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
    std::vector<double> hessian(9,0.0);
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = 1/(Va**2*k)
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
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double Va = 0.0409;
    double k = 0.04015625;
 double   F = 1/(std::pow(Va, 2)*k);
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
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double Va = 0.0409;
    double k = 0.04015625;
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
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double Va = 0.0409;
    double k = 0.04015625;
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
 *       F = (cbeq*eta3**2/(eps + eta1**2 + eta2**2 + eta3**2) + cmeq*(eta1**2 + eta2**2)/(eps + eta1**2 + eta2**2 + eta3**2))/Va + mu/(Va**2*k)
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
    double Va = 0.0409;
    double k = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.546;
    double eps = 1.0e-12;
 double   F = (cbeq*std::pow(eta3, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2)) + cmeq*(std::pow(eta1, 2) + std::pow(eta2, 2))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2)))/Va + mu/(std::pow(Va, 2)*k);
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
    double Va = 0.0409;
    double k = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.546;
    double eps = 1.0e-12;
    std::vector<double> gradient(1);
    gradient[0] = this->prefactor_ * (1/(std::pow(Va, 2)*k));
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
    double Va = 0.0409;
    double k = 0.04015625;
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
 *       F = (cbeq*eta3**2/(eps + eta1**2 + eta2**2 + eta3**2) + cmeq*(eta1**2 + eta2**2)/(eps + eta1**2 + eta2**2 + eta3**2))/Va + mu/(Va**2*k)
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
    double Va = 0.0409;
    double k = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.454;
    double eps = 1.0e-12;
 double   F = (cbeq*std::pow(eta3, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2)) + cmeq*(std::pow(eta1, 2) + std::pow(eta2, 2))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2)))/Va + mu/(std::pow(Va, 2)*k);
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
    double Va = 0.0409;
    double k = 0.04015625;
    double cmeq = 2.507e-13;
    double cbeq = 0.454;
    double eps = 1.0e-12;
    std::vector<double> gradient(1);
    gradient[0] = this->prefactor_ * (1/(std::pow(Va, 2)*k));
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
    double Va = 0.0409;
    double k = 0.04015625;
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
 *       F = (eta1**2 + eta2**2)/(eps + eta1**2 + eta2**2 + eta3**2)
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
    double eps = 1.0e-12;
 double   F = (std::pow(eta1, 2) + std::pow(eta2, 2))/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2));
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
 *       F = eta1**2/(eps + eta1**2 + eta2**2 + eta3**2)
 */
class SwitchingFunctionGrain1 : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  SwitchingFunctionGrain1() : prefactor_(1.0) {}
 explicit SwitchingFunctionGrain1(const double prefactor): prefactor_(prefactor) {}
  virtual ~SwitchingFunctionGrain1() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionGrain1::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eps = 1.0e-12;
 double   F = std::pow(eta1, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2));
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionGrain1::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionGrain1::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
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
 *       F = eta2**2/(eps + eta1**2 + eta2**2 + eta3**2)
 */
class SwitchingFunctionGrain2 : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  SwitchingFunctionGrain2() : prefactor_(1.0) {}
 explicit SwitchingFunctionGrain2(const double prefactor): prefactor_(prefactor) {}
  virtual ~SwitchingFunctionGrain2() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionGrain2::F() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
    double eps = 1.0e-12;
 double   F = std::pow(eta2, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2));
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionGrain2::GradientF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> SwitchingFunctionGrain2::HessianF() {
  auto func = [&](const std::span<const double>& input_vector, [[maybe_unused]] const std::span<const double>&,const std::span<const double>& auxiliary_vector, [[maybe_unused]] const unsigned int dimension) {
    double mu = input_vector[0];
    double eta1 = auxiliary_vector[0];
    double eta2 = auxiliary_vector[1];
    double eta3 = auxiliary_vector[2];
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
 *       F = eta3**2/(eps + eta1**2 + eta2**2 + eta3**2)
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
    double eps = 1.0e-12;
 double   F = std::pow(eta3, 2)/(eps + std::pow(eta1, 2) + std::pow(eta2, 2) + std::pow(eta3, 2));
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
    double eps = 1.0e-12;
    std::vector<double> hessian(1);
    hessian[0] = this->prefactor_ * (0);
    return hessian;
  };
  return func;
}
