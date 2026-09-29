// Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Driver: runs every benchmark group, prints a human table to stdout and
// machine-readable JSON to stdout (--json <file> also writes it to a file).
//
// Usage:
//   ee_microbench [--quick] [--repeats K] [--size N] [--json out.json]
//                 [--group arith,pow,...]
//
// Exit status: 0 on success; non-zero (abort) if any checksum is non-finite —
// such a failure is fatal and never silently ignored.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../common/ee_bench.hpp"

std::vector<ee_bench::BenchResult> bench_arith(const ee_bench::BenchConfig &);
std::vector<ee_bench::BenchResult> bench_pow(const ee_bench::BenchConfig &);
std::vector<ee_bench::BenchResult> bench_transcendental(const ee_bench::BenchConfig &);
std::vector<ee_bench::BenchResult> bench_memory(const ee_bench::BenchConfig &);
std::vector<ee_bench::BenchResult> bench_branch(const ee_bench::BenchConfig &);

namespace {

bool want_group(const std::string &filter, const std::string &group) {
  if (filter.empty() || filter == "all")
    return true;
  // comma-separated match
  std::string f = "," + filter + ",";
  return f.find("," + group + ",") != std::string::npos;
}

} // namespace

int main(int argc, char **argv) {
  ee_bench::BenchConfig cfg;
  std::string json_path;
  std::string group_filter;
  bool emit_json_stdout = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      cfg = ee_bench::default_config_quick();
    } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      cfg.repeats = std::atoi(argv[++i]);
      if (cfg.repeats <= 0) {
        std::fprintf(stderr, "FATAL: --repeats must be positive\n");
        return 2;
      }
    } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      cfg.n_stream = static_cast<std::size_t>(std::atoll(argv[++i]));
      if (cfg.n_stream == 0) {
        std::fprintf(stderr, "FATAL: --size must be positive\n");
        return 2;
      }
    } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
      json_path = argv[++i];
      emit_json_stdout = true;
    } else if (std::strcmp(argv[i], "--group") == 0 && i + 1 < argc) {
      group_filter = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf("usage: ee_microbench [--quick] [--repeats K] [--size N] "
                  "[--json f] [--group g1,g2]\n");
      return 0;
    } else {
      std::fprintf(stderr, "FATAL: unknown argument '%s'\n", argv[i]);
      return 2;
    }
  }

  std::vector<ee_bench::BenchResult> all;
  auto append = [&](const std::vector<ee_bench::BenchResult> &v) {
    all.insert(all.end(), v.begin(), v.end());
  };
  if (want_group(group_filter, "arith"))
    append(bench_arith(cfg));
  if (want_group(group_filter, "pow"))
    append(bench_pow(cfg));
  if (want_group(group_filter, "transcendental"))
    append(bench_transcendental(cfg));
  if (want_group(group_filter, "memory"))
    append(bench_memory(cfg));
  if (want_group(group_filter, "branch"))
    append(bench_branch(cfg));

  if (all.empty()) {
    std::fprintf(stderr, "FATAL: no benchmarks selected\n");
    return 2;
  }

  // Human table goes to stderr when JSON is requested so that stdout carries
  // pure JSON (pipe-friendly: ee_microbench --json - | fit_weights.py -).
  const bool want_json = emit_json_stdout || !json_path.empty();
  ee_bench::print_table(all, want_json ? stderr : stdout);
  if (want_json) {
    ee_bench::print_json(all);
    if (!json_path.empty() && json_path != "-") {
      if (std::FILE *f = std::fopen(json_path.c_str(), "w")) {
        // Duplicate the JSON payload into the file (same content as stdout).
        std::fprintf(f, "{\"results\":[\n");
        for (std::size_t i = 0; i < all.size(); ++i) {
          const auto &r = all[i];
          std::fprintf(f,
                       "  {\"group\":\"%s\",\"name\":\"%s\",\"complexity_node\":\"%s\","
                       "\"ns_per_elem\":%.6f,\"ns_median\":%.6f,\"gops\":%.6f,"
                       "\"ops_per_elem\":%d,\"checksum\":%.6g}%s\n",
                       r.group.c_str(), r.name.c_str(), r.complexity_node.c_str(),
                       r.ns_per_elem, r.ns_median, r.gops, r.ops_per_elem, r.checksum,
                       i + 1 < all.size() ? "," : "");
        }
        std::fprintf(f, "]}\n");
        std::fclose(f);
      } else {
        std::fprintf(stderr, "FATAL: cannot open --json file '%s'\n", json_path.c_str());
        return 2;
      }
    }
  }
  return 0;
}
