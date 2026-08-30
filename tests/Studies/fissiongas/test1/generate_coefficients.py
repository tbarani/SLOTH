#!/usr/bin/env python3
"""Generate coefficients.json of the bicrystal test for any number of grains.

coefficients.json holds the thermodynamics of the test as sympy expressions.
All of those expressions depend on how many order parameters there are. The multi-well
and cross terms of the free energy are sums over the order parameters, and the switching
functions divide by the sum of their squares.

The order parameters follow the block order of the "OrderParameters" problem of main.cpp,

    eta1 .. etaN      one per grain,
    eta(N+1)          the single order parameter of the bubble phase,

and the switching functions are

    h_m = (eta1^2 + ... + etaN^2) / (sum_i eta_i^2 + eps),
    h_b = eta(N+1)^2 / (sum_i eta_i^2 + eps).

The seven classes are:

    FgFreeEnergy      Eq. 1 of the CMS paper without the gradient term, fully implicit
    FgFreeEnergySI    the same with the products of distinct order parameters split
                      semi-implicitly. The explicit variables en1 .. en(N+1) hold eta^n, and
                      gamma eta_i^2 eta_j^2 becomes
                      gamma (eta_i^2 en_j^2 + eta_j^2 en_i^2 - en_i^2 en_j^2).
                      The interpolation is expanded around eta^n, so the derivative with respect
                      to each block reduces to the fully implicit one at eta^{n+1} = eta^n.
    FgGradEnergy      (kappa / 2) sum_i |grad eta_i|^2
    FgSusceptibility  chi = h_m / (Va^2 k^m) + h_b / (Va^2 k^b)
    FgDensityVac      rho_v = chi mu + (h_m c_v^{m,eq} + h_b c_v^{b,eq}) / Va
    FgDensityGas      rho_g, idem
    FgMatrixFraction  h_m

The model parameters default to the non-dimensionalised values of Table 1 of L.K. Aagesen et al.,
Comput. Mater. Sci. 161 (2019) 35-45, the same values as the file coefficients.json. You can
override any of them on the command line. The vacancy and gas parabolas must have the same
curvatures because one FgSusceptibility class covers both species. maybe need to rework it.

Usage: generate_coefficients.py [-ng N] [-o coefficients.json] [--<parameter> value ...]

With two grains, the default, the output carries the same expressions as the shipped
coefficients.json, up to the way sympy prints them. main.cpp still declares the order parameters,
their initial conditions and the per-block coefficients one by one, so you have to extend it by
hand to the same number of grains.
"""

import argparse
import json
import sys

# Non-dimensionalised model parameters, Table 1 of the reference. They stay strings so that
# json.dumps writes them to the file exactly as typed.
DEFAULT_PARAMETERS = [
    # name, default, help
    ("mbar", "0.0046875", "well height m of the multi-well potential"),
    ("gmm", "1.5", "gamma between two grains (symmetric interface)"),
    ("gmb", "0.922", "gamma between a grain and the bubble phase (sets the dihedral angle)"),
    ("kappa", "0.52734375", "gradient energy coefficient kappa (the class carries kappa / 2)"),
    ("Va", "0.0409", "atomic volume, nm3"),
    ("kvm", "7.515625", "curvature of the vacancy free energy in the matrix"),
    ("kgm", "7.515625", "curvature of the gas free energy in the matrix"),
    ("kvb", "1.40625", "curvature of the vacancy free energy in the bubble"),
    ("kgb", "1.40625", "curvature of the gas free energy in the bubble"),
    ("cvmeq", "2.507e-13", "equilibrium vacancy concentration in the matrix"),
    ("cgmeq", "2.507e-13", "equilibrium gas concentration in the matrix"),
    ("cvbeq", "0.546", "equilibrium vacancy concentration in the bubble"),
    ("cgbeq", "0.454", "equilibrium gas concentration in the bubble"),
    ("eps", "1.0e-12", "regularisation of the denominator of the switching functions to avoid zero division"),
]

output_file = "FissionGasCoefficients"


def sq(v):
    return f"{v}*{v}"


def constants(names, values):
    return ",".join(f"({n}:{values[n]})" for n in names)


def omega(kv, kg, cveq, cgeq):
    """Grand potential density of one phase, -1/2 mu^2 / (Va^2 k) - mu c^eq / Va per species."""
    return (f"-0.5*muv*muv/(Va*Va*{kv}) - muv*{cveq}/Va"
            f" - 0.5*mug*mug/(Va*Va*{kg}) - mug*{cgeq}/Va")


def build(nb_grains, prm):
    if nb_grains < 1:
        raise ValueError("at least one grain is needed")
    if prm["kvm"] != prm["kgm"] or prm["kvb"] != prm["kgb"]:
        raise ValueError("kvm must equal kgm and kvb must equal kgb: one FgSusceptibility class "
                         "serves both species")

    nb_ops = nb_grains + 1
    grains = [f"eta{i}" for i in range(1, nb_grains + 1)]
    bubble = f"eta{nb_ops}"
    all_ops = grains + [bubble]
    en_grains = [f"en{i}" for i in range(1, nb_grains + 1)]
    en_bubble = f"en{nb_ops}"
    en_all = en_grains + [en_bubble]

    def ssum(names):
        return " + ".join(sq(v) for v in names)

    def sum_squares_plus_eps(names):
        return f"({ssum(names)}+eps)"

    # Multi-well term, sum_i (eta_i^4 / 4 - eta_i^2 / 2). The 0.25 added below lifts the minima
    # to zero.
    wells = " + ".join(f"({v}*{v}*{v}*{v}/4.0 - {v}*{v}/2.0)" for v in all_ops)
    # Cross terms, gamma_ij eta_i^2 eta_j^2 over unordered pairs. Eq. 1 writes gamma/2 over ordered
    # pairs, which is the same sum.
    cross_mm = " + ".join(f"{sq(grains[i])}*{sq(grains[j])}"
                          for i in range(nb_grains) for j in range(i + 1, nb_grains))
    cross_mb = " + ".join(f"{sq(g)}*{sq(bubble)}" for g in grains)
    cross = ""
    if cross_mm:
        cross += f" + gmm*({cross_mm})"
    cross += f" + gmb*({cross_mb})"

    h_m = f"(({ssum(grains)})/{sum_squares_plus_eps(all_ops)})"
    h_b = f"(({sq(bubble)})/{sum_squares_plus_eps(all_ops)})"
    omega_m = omega("kvm", "kgm", "cvmeq", "cgmeq")
    omega_b = omega("kvb", "kgb", "cvbeq", "cgbeq")

    fi = (f"mbar * ( {wells}{cross} + 0.25 )"
          f" + {h_m} * ( {omega_m} )"
          f" + {h_b} * ( {omega_b} )")

    # Semi-implicit form. Each product of two distinct order parameters keeps one factor at the new
    # time and the other at the old one, summed over both orderings. Subtracting the all-explicit
    # term makes the energy come out right, since at eta^{n+1} = eta^n the three pieces add up to
    # the physical eta_i^2 eta_j^2. The interpolated grand potential is expanded to first order
    # around eta^n. Only the numerators of the switching functions see the new time; the
    # denominators stay at eta^n.
    def split(a, b, an, bn):
        return f"{sq(a)}*{sq(bn)} + {sq(b)}*{sq(an)} - {sq(an)}*{sq(bn)}"

    si_mm = " + ".join(split(grains[i], grains[j], en_grains[i], en_grains[j])
                       for i in range(nb_grains) for j in range(i + 1, nb_grains))
    si_mb = " + ".join(split(g, bubble, gn, en_bubble) for g, gn in zip(grains, en_grains))
    si_cross = ""
    if si_mm:
        si_cross += f"gmm*({si_mm}) + "
    si_cross += f"gmb*({si_mb})"
    s_en = sum_squares_plus_eps(en_all)
    si = (f"mbar * ( {wells} + 0.25 )"
          f" + mbar * ( {si_cross} )"
          f" + ( ( {omega_m} ) - ( {omega_b} ) )"
          f" * ( ({ssum(grains)})*({sq(en_bubble)}) - ({sq(bubble)})*({ssum(en_grains)}) )"
          f" / ( {s_en}*{s_en} )"
          f" + ( ({ssum(en_grains)})*( {omega_m} ) + ({sq(en_bubble)})*( {omega_b} ) ) / {s_en}")

    chi = f"{h_m}/(Va*Va*km) + {h_b}/(Va*Va*kb)"

    def density(cmeq, cbeq):
        return f"( {chi} )*mu + ( {h_m}*{cmeq} + {h_b}*{cbeq} )/Va"

    thermo = ["mbar", "gmm", "gmb", "Va", "kvm", "kgm", "kvb", "kgb", "cvmeq", "cgmeq", "cvbeq",
              "cgbeq", "eps"]
    # FgSusceptibility and both densities read km and kb. The check at the top of build() makes
    # sure the vacancy and gas curvatures agree, so take the vacancy ones.
    mu_prm = dict(prm, km=prm["kvm"], kb=prm["kvb"])
    eta_range = f"eta(1..{nb_ops})"
    en_range = f"en(1..{nb_ops})"
    kappa_half = repr(float(prm["kappa"]) / 2.)

    return [
        {
            "expression": fi,
            "constants": constants(thermo, prm),
            "variables": eta_range,
            "auxiliary_variables": "muv,mug",
            "class_name": "FgFreeEnergy",
            "outputfile": output_file,
        },
        {
            "expression": si,
            "constants": constants(thermo, prm),
            "variables": eta_range,
            "explicit_variables": en_range,
            "auxiliary_variables": "muv,mug",
            "class_name": "FgFreeEnergySI",
            "outputfile": output_file,
        },
        {
            "expression": f"{kappa_half}*sdot({eta_range})",
            "variables": eta_range,
            "class_name": "FgGradEnergy",
            "outputfile": output_file,
            "gradient": True,
        },
        {
            "expression": chi,
            "constants": constants(["Va", "km", "kb", "eps"], mu_prm),
            "variables": "mu",
            "auxiliary_variables": eta_range,
            "class_name": "FgSusceptibility",
            "outputfile": output_file,
        },
        {
            "expression": density("cmeq", "cbeq"),
            "constants": constants(["Va", "km", "kb", "cmeq", "cbeq", "eps"],
                                   dict(mu_prm, cmeq=prm["cvmeq"], cbeq=prm["cvbeq"])),
            "variables": "mu",
            "auxiliary_variables": eta_range,
            "class_name": "FgDensityVac",
            "outputfile": output_file,
        },
        {
            "expression": density("cmeq", "cbeq"),
            "constants": constants(["Va", "km", "kb", "cmeq", "cbeq", "eps"],
                                   dict(mu_prm, cmeq=prm["cgmeq"], cbeq=prm["cgbeq"])),
            "variables": "mu",
            "auxiliary_variables": eta_range,
            "class_name": "FgDensityGas",
            "outputfile": output_file,
        },
        {
            "expression": h_m,
            "constants": constants(["eps"], prm),
            "variables": "mu",
            "auxiliary_variables": eta_range,
            "class_name": "FgMatrixFraction",
            "outputfile": output_file,
        },
    ]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate coefficients.json of the bicrystal test for N grains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ng", "--nb-grains", type=int, default=2,
                        help="number of grain order parameters (the bubble adds one)")
    parser.add_argument("-o", "--output", default=None,
                        help="output file (default: standard output)")
    group = parser.add_argument_group("model parameters (non-dimensionalised)")
    for name, default, doc in DEFAULT_PARAMETERS:
        group.add_argument(f"--{name}", default=default, help=doc)
    args = parser.parse_args(argv)
    for name, _, _ in DEFAULT_PARAMETERS:
        try:
            float(getattr(args, name))
        except ValueError:
            parser.error(f"--{name} expects a number, got {getattr(args, name)!r}")
    return args


def main(argv=None):
    args = parse_args(argv)
    prm = {name: getattr(args, name) for name, _, _ in DEFAULT_PARAMETERS}
    entries = build(args.nb_grains, prm)
    text = json.dumps(entries, indent=4) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"{args.output}: {args.nb_grains} grain(s), {args.nb_grains + 1} order parameters, "
              f"{len(entries)} coefficient classes", file=sys.stderr)


if __name__ == "__main__":
    main()
