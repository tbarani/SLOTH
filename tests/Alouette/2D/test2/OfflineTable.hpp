/**
 * @file OfflineTable.hpp
 * @author Tommaso Barani (tommaso.barani@cea.fr)
 * @brief Reader for the alouette S1 offline response table (the file the
 *        `alouette_offline_batch` program writes).
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
 * WHY THIS READER IS STRICT, AND WHY IT REFUSES MORE THAN IT PARSES.
 * The offline variant makes a text file the interface between two programs
 * that never share a runtime, so a misread row does not crash: it produces a
 * plausible conductivity and a plausible temperature field. Every refusal
 * below exists because the alternative is a wrong answer that looks right.
 *
 * Three of them are worth naming.
 *
 * 1. THE TABLE IS NOT ALLOWED TO VARY IN TIME. S1's microstructure is frozen,
 *    so the batch's lambda* columns are step-invariant by construction and the
 *    design note specifies that a time-varying table is S3 scope. Accepting a
 *    varying table here would silently implement "the microstructure evolves
 *    but nobody told the macro solver" -- the exact drift the note forbids.
 *    The check is a relative comparison against the first step, at 1e-10: the
 *    writer's %.17g round trip is exact, so any real variation is far above
 *    the threshold and any noise is far below it.
 *
 * 2. THE TENSOR MUST BE ISOTROPIC. The R2 provider
 *    (`MultiscaleCoefficient`) is a scalar per element: `double(int)`. It
 *    cannot express an anisotropic lambda*. A table whose lambda* is not
 *    lambda*I would be applied as its xx component and would silently drop the
 *    off-diagonal and the yy/xx difference -- so it is refused instead. This
 *    is the v1 scalar projection of the design note (4.1): for the layered
 *    gate's HOMOGENEOUS cells lambda* IS lambda*I exactly, and the day a
 *    genuinely anisotropic cell is wanted the tensor branch is promoted
 *    explicitly, with its own gate.
 *
 * 3. THE ROW ORDER IS ENFORCED, not merely the coverage. A file that repeats
 *    one material point and drops another is a short table that still parses;
 *    requiring `point e` in order is what makes the omission detectable.
 *
 * The parser reports `path:line` in every message. A refusal that cannot be
 * located in a 200 000-line table is a refusal nobody can act on.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alouette_offline {

/// The tag and version the batch writes (`kTableTag`, `kFormatVersion` in
/// `src/offline/batch.cxx`). Kept in sync by the format block in
/// `docs/DESIGN_SLOTH_COUPLING.md` 3.10, which both sides cite.
constexpr char kTableTag[] = "alouette-response-table";
/// The format version this reader understands.
constexpr int kFormatVersion = 1;
/// S1 is 2D; the table's `dim` must say so.
constexpr int kS1Dim = 2;
/// The relative bound on the step-to-step lambda* variation (refusal above).
constexpr double kTimeInvarianceTolerance = 1.0e-10;
/// The absolute bound on the off-diagonal and the xx/yy difference, relative to
/// the largest diagonal entry (refusal above).
constexpr double kIsotropyTolerance = 1.0e-10;

/**
 * @brief One material point's response at one step.
 */
struct Response {
  /// The effective conductivity, row-major (`[0]=xx`, `[1]=xy`, `[2]=yx`,
  /// `[3]=yy`).
  double lambda[4] = {0.0, 0.0, 0.0, 0.0};
  /// The mean current `<J> = lambda* . E` the batch computed from the dump.
  double current[2] = {0.0, 0.0};
};

/**
 * @brief One step of the table.
 */
struct Step {
  /// The macro time the batch echoed back from the dump.
  double time = 0.0;
  /// One response per material point, in file order.
  std::vector<Response> rows;
};

/**
 * @brief The parsed response table.
 */
struct Table {
  /// The declared dimension (must be 2).
  int dim = 0;
  /// The number of steps.
  int n_steps = 0;
  /// The number of material points.
  int n_material_points = 0;
  /// The steps, in file order.
  std::vector<Step> steps;

  /**
   * @brief The scalar lambda* of a material point (the xx diagonal).
   * @param element The material point.
   * @return The isotropic value; the reader has already refused an
   *         anisotropic table.
   */
  double lambda_star(int element) const {
    return steps.front().rows[static_cast<std::size_t>(element)].lambda[0];
  }
};

/**
 * @brief Abort the reader with a located message.
 * @param message The refusal.
 */
[[noreturn]] inline void refuse(const std::string& message) {
  throw std::runtime_error(message);
}

/**
 * @brief Read a finite number from a stream, refusing end-of-line.
 * @param stream The line's tokens.
 * @param where  `path:line`, for the message.
 * @param what   The field's name, for the message.
 * @return The value.
 */
inline double read_double(std::istringstream& stream, const std::string& where,
                          const char* what) {
  double value = 0.0;
  if (!(stream >> value)) {
    refuse(where + ": expected the number '" + std::string(what) +
           "', found end of line or a non-number");
  }
  if (!std::isfinite(value)) {
    refuse(where + ": '" + std::string(what) + "' must be finite");
  }
  return value;
}

/**
 * @brief Read a non-negative integer from a stream.
 * @param stream The line's tokens.
 * @param where  `path:line`, for the message.
 * @param what   The field's name, for the message.
 * @return The value.
 */
inline int read_int(std::istringstream& stream, const std::string& where,
                    const char* what) {
  long long value = 0;
  if (!(stream >> value)) {
    refuse(where + ": expected the integer '" + std::string(what) +
           "', found end of line or a non-integer");
  }
  if (value < 0 || value > 1000000000LL) {
    refuse(where + ": '" + std::string(what) + "' is out of range");
  }
  return static_cast<int>(value);
}

/**
 * @brief Read a required keyword.
 * @param stream  The line's tokens.
 * @param keyword The expected token.
 * @param where   `path:line`, for the message.
 */
inline void read_keyword(std::istringstream& stream, const char* keyword,
                         const std::string& where) {
  std::string token;
  if (!(stream >> token)) {
    refuse(where + ": expected '" + keyword + "', found end of line");
  }
  if (token != keyword) {
    refuse(where + ": expected '" + keyword + "', got '" + token + "'");
  }
}

/**
 * @brief Refuse trailing content on a fully parsed line.
 * @param stream The line's tokens.
 * @param where  `path:line`, for the message.
 */
inline void read_end(std::istringstream& stream, const std::string& where) {
  std::string extra;
  if (stream >> extra) {
    refuse(where + ": unexpected trailing token '" + extra + "'");
  }
}

/**
 * @brief Read and validate an S1 response table.
 *
 * @param path The table the batch wrote.
 * @return The table.
 *
 * @throws std::runtime_error On any format violation, a non-2D table, a
 *         time-varying lambda*, or an anisotropic lambda*. Every message names
 *         the offending line.
 */
inline Table read_table(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    refuse("cannot open the response table '" + path + "'");
  }
  Table table;
  std::string line;
  int line_no = 0;
  auto next = [&](std::istringstream& stream) -> bool {
    while (std::getline(in, line)) {
      ++line_no;
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.find_first_not_of(" \t") == std::string::npos) {
        continue;
      }
      stream.clear();
      stream.str(line);
      return true;
    }
    return false;
  };
  auto where = [&]() { return path + ":" + std::to_string(line_no); };

  std::istringstream stream;
  // The tag line: the format's identity and version, checked first so a
  // foreign file is refused by name instead of misparsed.
  if (!next(stream)) {
    refuse(path + ": empty response table");
  }
  read_keyword(stream, kTableTag, where());
  const int version = read_int(stream, where(), "version");
  read_end(stream, where());
  if (version != kFormatVersion) {
    refuse(where() + ": " + kTableTag + " version " +
           std::to_string(version) + " is not the version this reader reads (" +
           std::to_string(kFormatVersion) + ")");
  }

  if (!next(stream)) {
    refuse(path + ": truncated table (expected 'dim')");
  }
  read_keyword(stream, "dim", where());
  table.dim = read_int(stream, where(), "dim");
  read_end(stream, where());
  if (table.dim != kS1Dim) {
    refuse(where() + ": dim " + std::to_string(table.dim) +
           " is refused: the S1 offline chain is 2D");
  }

  if (!next(stream)) {
    refuse(path + ": truncated table (expected 'n_steps')");
  }
  read_keyword(stream, "n_steps", where());
  table.n_steps = read_int(stream, where(), "n_steps");
  read_end(stream, where());
  if (table.n_steps < 1) {
    refuse(where() + ": n_steps must be at least 1");
  }

  if (!next(stream)) {
    refuse(path + ": truncated table (expected 'n_material_points')");
  }
  read_keyword(stream, "n_material_points", where());
  table.n_material_points = read_int(stream, where(), "n_material_points");
  read_end(stream, where());
  if (table.n_material_points < 1) {
    refuse(where() + ": n_material_points must be at least 1");
  }

  table.steps.resize(static_cast<std::size_t>(table.n_steps));
  for (int s = 0; s < table.n_steps; ++s) {
    if (!next(stream)) {
      refuse(path + ": truncated table (expected the 'step " +
             std::to_string(s) + "' line)");
    }
    read_keyword(stream, "step", where());
    const int index = read_int(stream, where(), "step index");
    read_keyword(stream, "time", where());
    table.steps[static_cast<std::size_t>(s)].time =
        read_double(stream, where(), "time");
    read_end(stream, where());
    if (index != s) {
      refuse(where() + ": step index " + std::to_string(index) +
             " is out of order (expected " + std::to_string(s) + ")");
    }
    Step& step = table.steps[static_cast<std::size_t>(s)];
    step.rows.resize(static_cast<std::size_t>(table.n_material_points));
    for (int e = 0; e < table.n_material_points; ++e) {
      if (!next(stream)) {
        refuse(path + ": truncated table (expected 'point " +
               std::to_string(e) + "' at step " + std::to_string(s) + ")");
      }
      read_keyword(stream, "point", where());
      const int element = read_int(stream, where(), "element");
      if (element != e) {
        refuse(where() + ": point " + std::to_string(element) +
               " is out of order (expected " + std::to_string(e) +
               "; rows are ordered step, element)");
      }
      Response& response = step.rows[static_cast<std::size_t>(e)];
      for (double& value : response.lambda) {
        value = read_double(stream, where(), "lambda*");
      }
      response.current[0] = read_double(stream, where(), "current x");
      response.current[1] = read_double(stream, where(), "current y");
      read_end(stream, where());
    }
  }
  if (next(stream)) {
    refuse(where() + ": trailing content after the last step");
  }

  // ---- the two semantic refusals ---------------------------------------
  const Step& first = table.steps.front();
  for (int s = 1; s < table.n_steps; ++s) {
    const Step& step = table.steps[static_cast<std::size_t>(s)];
    for (int e = 0; e < table.n_material_points; ++e) {
      for (int i = 0; i < 4; ++i) {
        const double a = first.rows[static_cast<std::size_t>(e)].lambda[i];
        const double b = step.rows[static_cast<std::size_t>(e)].lambda[i];
        const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
        if (std::abs(a - b) > kTimeInvarianceTolerance * scale) {
          refuse(path + ": material point " + std::to_string(e) +
                 " has a time-varying lambda* (" + std::to_string(a) +
                 " at step 0 vs " + std::to_string(b) + " at step " +
                 std::to_string(s) +
                 "); S1's microstructure is frozen and an evolving table is "
                 "S3 scope");
        }
      }
    }
  }
  for (int e = 0; e < table.n_material_points; ++e) {
    const Response& response = first.rows[static_cast<std::size_t>(e)];
    const double scale =
        std::max(1.0, std::max(std::abs(response.lambda[0]),
                               std::abs(response.lambda[3])));
    if (std::abs(response.lambda[1]) > kIsotropyTolerance * scale ||
        std::abs(response.lambda[2]) > kIsotropyTolerance * scale ||
        std::abs(response.lambda[0] - response.lambda[3]) >
            kIsotropyTolerance * scale) {
      refuse(path + ": material point " + std::to_string(e) +
             " has an anisotropic lambda* ([" + std::to_string(response.lambda[0]) +
             ", " + std::to_string(response.lambda[1]) + ", " +
             std::to_string(response.lambda[2]) + ", " +
             std::to_string(response.lambda[3]) +
             "]); the S1 macro operator takes one SCALAR lambda* per element, "
             "so an anisotropic table would be silently projected");
    }
  }
  return table;
}

}  // namespace alouette_offline
