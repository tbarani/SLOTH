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
 *       F = mbar*(eta1**4/4 - eta1**2/2 + eta2**4/4 - eta2**2/2 + gmm*(-eta1**4 - eta2**4 + (eta1**2 + eta2**2)**2)/2 + 1/4)
 */
class FgMultiWell : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgMultiWell() : prefactor_(1.0) {}
 explicit FgMultiWell(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgMultiWell() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMultiWell::F() {
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
 double   F = mbar*((1.0/4.0)*std::pow(eta1, 4) - 1.0/2.0*std::pow(eta1, 2) + (1.0/4.0)*std::pow(eta2, 4) - 1.0/2.0*std::pow(eta2, 2) + (1.0/2.0)*gmm*(-std::pow(eta1, 4) - std::pow(eta2, 4) + std::pow(std::pow(eta1, 2) + std::pow(eta2, 2), 2)) + 1.0/4.0);
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMultiWell::GradientF() {
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    std::vector<double> gradient(2);
    gradient[0] = this->prefactor_ * (mbar*(std::pow(eta1, 3) - eta1 + (1.0/2.0)*gmm*(-4*std::pow(eta1, 3) + 4*eta1*(std::pow(eta1, 2) + std::pow(eta2, 2)))));
    gradient[1] = this->prefactor_ * (mbar*(std::pow(eta2, 3) - eta2 + (1.0/2.0)*gmm*(-4*std::pow(eta2, 3) + 4*eta2*(std::pow(eta1, 2) + std::pow(eta2, 2)))));
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMultiWell::HessianF() {
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double mbar = 0.0046875;
    double gmm = 1.5;
    std::vector<double> hessian(4);
    hessian[0] = this->prefactor_ * (mbar*(3*std::pow(eta1, 2) + 2*std::pow(eta2, 2)*gmm - 1));
    hessian[1] = this->prefactor_ * (4*eta1*eta2*gmm*mbar);
    hessian[2] = this->prefactor_ * (4*eta1*eta2*gmm*mbar);
    hessian[3] = this->prefactor_ * (mbar*(2*std::pow(eta1, 2)*gmm + 3*std::pow(eta2, 2) - 1));
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = mbar*(eta1**4/4 - eta1**2/2 + eta2**4/4 - eta2**2/2 + gmb*(-eta1**4 - eta2**4 + (eta1**2 + eta2**2)**2)/2 + 1/4)
 */
class FgMultiWellMB : public FunctionCoefficient {
 private:
  double prefactor_;
 protected:
  std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> F() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> GradientF() final;
  std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&,const std::span<const double>&, const unsigned int dimension)> HessianF() final;

 public:
  FgMultiWellMB() : prefactor_(1.0) {}
 explicit FgMultiWellMB(const double prefactor): prefactor_(prefactor) {}
  virtual ~FgMultiWellMB() = default;
};

/**
 *
 * @brief C++ function of the expression
 *
 *
 * @return std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)>
 */
 std::function<double(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMultiWellMB::F() {
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double mbar = 0.0046875;
    double gmb = 0.922;
 double   F = mbar*((1.0/4.0)*std::pow(eta1, 4) - 1.0/2.0*std::pow(eta1, 2) + (1.0/4.0)*std::pow(eta2, 4) - 1.0/2.0*std::pow(eta2, 2) + (1.0/2.0)*gmb*(-std::pow(eta1, 4) - std::pow(eta2, 4) + std::pow(std::pow(eta1, 2) + std::pow(eta2, 2), 2)) + 1.0/4.0);
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMultiWellMB::GradientF() {
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double mbar = 0.0046875;
    double gmb = 0.922;
    std::vector<double> gradient(2);
    gradient[0] = this->prefactor_ * (mbar*(std::pow(eta1, 3) - eta1 + (1.0/2.0)*gmb*(-4*std::pow(eta1, 3) + 4*eta1*(std::pow(eta1, 2) + std::pow(eta2, 2)))));
    gradient[1] = this->prefactor_ * (mbar*(std::pow(eta2, 3) - eta2 + (1.0/2.0)*gmb*(-4*std::pow(eta2, 3) + 4*eta2*(std::pow(eta1, 2) + std::pow(eta2, 2)))));
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
std::function<std::vector<double>(const std::span<const double>&,const std::span<const double>&, const std::span<const double>&, const unsigned int dimension)> FgMultiWellMB::HessianF() {
  auto func = [&](const std::span<const double>& input_vector,[[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const std::span<const double>&, [[maybe_unused]] const unsigned int dimension) {
    double eta1 = input_vector[0];
    double eta2 = input_vector[1];
    double mbar = 0.0046875;
    double gmb = 0.922;
    std::vector<double> hessian(4);
    hessian[0] = this->prefactor_ * (mbar*(3*std::pow(eta1, 2) + 2*std::pow(eta2, 2)*gmb - 1));
    hessian[1] = this->prefactor_ * (4*eta1*eta2*gmb*mbar);
    hessian[2] = this->prefactor_ * (4*eta1*eta2*gmb*mbar);
    hessian[3] = this->prefactor_ * (mbar*(2*std::pow(eta1, 2)*gmb + 3*std::pow(eta2, 2) - 1));
    return hessian;
  };
  return func;
}
/**
 *
 * @brief C++ function of the analytical expression
 *
 *       F = 135*dot(eta1, eta1)/512 + 135*dot(eta2, eta2)/512
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
    double F = 0.263671875*std::inner_product(eta1.begin(), eta1.end(), eta1.begin(), 0.0) + 0.263671875*std::inner_product(eta2.begin(), eta2.end(), eta2.begin(), 0.0);
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
    std::vector<double> gradient(2,0.0);
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
    std::vector<double> hessian(4,0.0);
    return hessian;
  };
  return func;
}
