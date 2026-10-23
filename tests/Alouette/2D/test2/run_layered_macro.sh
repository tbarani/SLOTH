#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026 Tommaso Barani
#
# @file run_layered_macro.sh
# @author Tommaso Barani (tommaso.barani@cea.fr)
# @brief The S1 offline chain gate: dump -> alouette batch -> table, for the
#        three layered cases, with the table-vs-constant agreement and the
#        exact analytic anchors asserted independently of the app.
# @version 0.1
# @date 2026-10-23
#
# @copyright CEA (C) 2026
#
# This file is part of SLOTH.
#
# SLOTH is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SLOTH is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# WHY THE SCRIPT ASSERTS THE NUMBERS THE APP ALREADY ASSERTS.
# The app checks its own anchors before writing a summary, so a failed anchor
# already fails the chain. These checks read the SUMMARY FILES instead, which is
# a different claim: that the numbers the two sides exchange through the dump
# and the table are the analytic ones. An app that computed the right answer but
# wrote a summary with a transposed row, a stale value or a missing key would
# pass its own checks and fail here. The constants below are the closed forms of
# docs/DESIGN_SLOTH_COUPLING.md 4.1, hard-coded on purpose: a script that
# re-derived them from the app's own helpers would only compare the app with
# itself.
#
# THE CHAIN, per case:
#   1. `--mode dump`     the macro loop + the dump (the state the batch reads)
#   2. the BATCH         a separate process, by path, over the frozen C ABI
#   3. `--mode table`    the macro loop with the lambda* the table reports
#   4. `--mode constant` the macro loop with the analytic layer lambda*
#   5. `--mode compare`  (4) and (3) must agree to 1e-12 relative
#
# Usage: run_layered_macro.sh <app> <alouette_offline_batch> <workdir>
set -u

app=$1
batch=$2
workdir=$3

# The exact series/parallel response of the layered strip (design 4.1). The
# layers are 0.5 thick on a unit square; lambda_below and lambda_above are the
# case's own values.
SERIES_LAMBDA='3.8095238095238093'
SERIES_CURRENT='380.9523809523809'
SERIES_T_QUARTER='702.3809523809524'
SERIES_T_CENTRE='704.7619047619048'
PARALLEL_LAMBDA='21'
PARALLEL_CURRENT='2100'
PARALLEL_T_QUARTER='725'
PARALLEL_T_CENTRE='750'
# The transient is HOMOGENEOUS (lambda = 2 everywhere; see the case comment in
# main.cpp: a layered transient has no closed form). Its sine decays at
# lambda (2 pi)^2 / (rho c_p), and t_end = 125 s is one decay time, so the
# amplitude is e^-1 and the probe sits where it is most sensitive to lambda:
# 3.68 K per unit lambda. A table carrying the harmonic 3.81 instead of 2 would
# read 728.05 (4.4 K off), the arithmetic 21 would read 725.00 (7.5 K off);
# both are far outside the 1 K anchor. The two probes share x = 0.25 and
# differ only in y, so they also pin that the field stayed y-independent.
TRANSIENT_T_QUARTER='732.4541567770688'
TRANSIENT_T_QUARTER_TOP='732.4541567770688'
TRANSIENT_T_CENTRE='750'

# The tolerances. The steady anchors are the exact discrete solution (the Q1
# space contains the piecewise-linear profile), so the bound is the linear
# solve's. The transient carries backward Euler's per-step gap against the
# exponential: 1% of the applied 100 K, which a wrong decay rate cannot pass.
STEADY_T_TOL='1e-6'
STEADY_FLUX_TOL='1e-6'
TRANSIENT_T_TOL='1.0'
TRANSIENT_FLUX_TOL='1e-6'
DRIVER_GAP_TOL='1e-9'

failures=0

fail() {
  printf 'FAILED %s\n' "$*"
  failures=$((failures + 1))
}

# check_value <label> <file> <key> <expected> <rel|abs> <tolerance>
check_value() {
  local label=$1 file=$2 key=$3 expected=$4 mode=$5 tol=$6
  if ! awk -v label="$label" -v file="$file" -v key="$key" -v expected="$expected" \
      -v mode="$mode" -v tol="$tol" '
    $1 == key { value = $2; found = 1 }
    END {
      if (!found) {
        printf("FAILED %-34s %s has no key %s\n", label, file, key)
        exit 1
      }
      if (mode == "rel") {
        scale = (expected < 0 ? -expected : expected)
        if (scale < 1) scale = 1
        gap = (value - expected) / scale
      } else {
        gap = value - expected
      }
      if (gap < 0) gap = -gap
      printf("%-6s %-34s %-22s computed=%-22.15g expected=%-22.15g gap=%.3e (tol %.1e)\n",
             (gap <= tol ? "GOLD" : "FAILED"), label, key, value, expected, gap, tol)
      exit (gap <= tol ? 0 : 1)
    }' "$file"; then
    failures=$((failures + 1))
  fi
}

# run <label> <command...>: run one chain step, reporting the command on failure.
run() {
  local label=$1
  shift
  if ! "$@"; then
    fail "$label: the command failed: $*"
    return 1
  fi
  return 0
}

# check_case <case> <axis> <lambda below> <lambda above> <lambda*> <current>
#            <T quarter> <T centre> <T quarter top> <T tolerance> <flux tolerance>
check_case() {
  local name=$1 axis=$2 below=$3 above=$4 lambda=$5 current=$6
  local t_quarter=$7 t_centre=$8 t_quarter_top=$9
  local t_tol=${10} flux_tol=${11}
  local dump="$workdir/$name.dump"
  local table="$workdir/$name.table"
  local summary_dump="$workdir/$name.dump.summary"
  local summary_table="$workdir/$name.table.summary"
  local summary_constant="$workdir/$name.constant.summary"

  printf '\n=== case %s (layers along %s: %s below, %s above) ===\n' \
         "$name" "$axis" "$below" "$above"
  run "$name dump" "$app" --case "$name" --mode dump \
      --dump "$dump" --summary "$summary_dump" || return 0
  run "$name batch" "$batch" --in "$dump" --out "$table" \
      --lambda "$below" "$above" --interface 0.5 --layer-axis "$axis" \
      --cell 16 16 --cell-size 1 1 || return 0
  run "$name table" "$app" --case "$name" --mode table \
      --table "$table" --summary "$summary_table" || return 0
  run "$name constant" "$app" --case "$name" --mode constant \
      --summary "$summary_constant" || return 0
  run "$name compare" "$app" --mode compare \
      --compare "$summary_constant" "$summary_table" || return 0

  # The offline-coupled answer, against the closed forms.
  check_value "$name/table" "$summary_table" mean_current_x "$current" rel "$flux_tol"
  check_value "$name/table" "$summary_table" lambda_eff "$lambda" rel "$flux_tol"
  check_value "$name/table" "$summary_table" T_probe_quarter "$t_quarter" abs "$t_tol"
  check_value "$name/table" "$summary_table" T_probe_centre "$t_centre" abs "$t_tol"
  check_value "$name/table" "$summary_table" T_probe_quarter_top "$t_quarter_top" abs "$t_tol"
  check_value "$name/table" "$summary_table" anchor_error 0 abs "$t_tol"
  check_value "$name/table" "$summary_table" mean_current_y 0 abs 1e-6
  check_value "$name/table" "$summary_table" T_left 700 abs 1e-12
  check_value "$name/table" "$summary_table" T_right 800 abs 1e-12
  # The per-step driver against a single long TimeDiscretization run: a
  # different claim from the anchors, and only the dump mode reports it.
  check_value "$name/dump" "$summary_dump" driver_gap 0 abs "$DRIVER_GAP_TOL"
}

if [ ! -x "$app" ]; then
  printf 'FAILED the macro app %s is not executable\n' "$app"
  exit 2
fi
if [ ! -x "$batch" ]; then
  printf 'FAILED the alouette batch %s is not executable (ALOUETTE_OFFLINE_BATCH)\n' "$batch"
  exit 2
fi

rm -rf "$workdir"
mkdir -p "$workdir" || exit 2

# The layer values are the case's own: series puts the CONDUCTING layer below
# and the gradient along x (the harmonic mean, 3.81); parallel puts the
# conducting layer above with the gradient along x (the arithmetic mean, 21);
# the transient is homogeneous and anchors the conductivity-sensitive decay.
check_case series x 40 2 "$SERIES_LAMBDA" "$SERIES_CURRENT" \
           "$SERIES_T_QUARTER" "$SERIES_T_CENTRE" "$SERIES_T_QUARTER" \
           "$STEADY_T_TOL" "$STEADY_FLUX_TOL"
check_case parallel y 2 40 "$PARALLEL_LAMBDA" "$PARALLEL_CURRENT" \
           "$PARALLEL_T_QUARTER" "$PARALLEL_T_CENTRE" "$PARALLEL_T_QUARTER" \
           "$STEADY_T_TOL" "$STEADY_FLUX_TOL"
check_case transient y 2 2 2 200 \
           "$TRANSIENT_T_QUARTER" "$TRANSIENT_T_CENTRE" "$TRANSIENT_T_QUARTER_TOP" \
           "$TRANSIENT_T_TOL" "$TRANSIENT_FLUX_TOL"

printf '\n'
if [ "$failures" -ne 0 ]; then
  printf 'FAILED t.layered_macro: %d check(s) failed (workdir %s)\n' "$failures" "$workdir"
  exit 1
fi
printf 'PASS t.layered_macro: dump -> batch -> table, the table-vs-constant agreement and the\n'
printf '     exact series/parallel/transient anchors are all green (workdir %s)\n' "$workdir"
exit 0
