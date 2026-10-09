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

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../common/ee_bench.hpp"
#if !defined(EE_HAVE_CUDA) && !defined(EE_HAVE_HIP)
#include <sched.h>
#endif
#if defined(EE_HAVE_CUDA)
#include <cuda_runtime.h>
#elif defined(EE_HAVE_HIP)
#include <hip/hip_runtime.h>
#endif

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

// Reject a leading sign. strtoull turns "-1" into a huge unsigned value.
bool parse_positive(const char *flag, const char *text, unsigned long long limit,
                    unsigned long long *out) {
  if (text == nullptr || text[0] < '0' || text[0] > '9') {
    std::fprintf(stderr, "FATAL: %s must be a positive integer, got '%s'\n", flag,
                 text == nullptr ? "" : text);
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || errno != 0 || value == 0 || value > limit) {
    std::fprintf(stderr, "FATAL: %s must be a positive integer, got '%s'\n", flag, text);
    return false;
  }
  *out = value;
  return true;
}

#if !defined(EE_HAVE_CUDA) && !defined(EE_HAVE_HIP)
int pin_current_cpu() {
  const int cpu = sched_getcpu();
  if (cpu < 0) {
    std::fprintf(stderr, "FATAL: sched_getcpu failed: %s\n", std::strerror(errno));
    return 2;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    std::fprintf(stderr, "FATAL: sched_setaffinity(cpu %d) failed: %s\n", cpu,
                 std::strerror(errno));
    return 2;
  }
  std::fprintf(stderr,
               "ee_host: pinned to cpu %d. The CPU governor is left alone; "
               "changing it needs privileges this process does not take.\n",
               cpu);
  return 0;
}
#endif

} // namespace

int main(int argc, char **argv) {
  std::string json_path;
  std::string group_filter;
  bool emit_json_stdout = false;
  bool want_quick = false;
  bool repeats_set = false;
  bool size_set = false;
  int repeats = 0;
  unsigned long long size_value = 0;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      want_quick = true;
    } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      unsigned long long value = 0;
      if (!parse_positive("--repeats", argv[++i], static_cast<unsigned long long>(INT_MAX),
                          &value))
        return 2;
      repeats = static_cast<int>(value);
      repeats_set = true;
    } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      if (!parse_positive("--size", argv[++i], std::numeric_limits<std::size_t>::max(),
                          &size_value))
        return 2;
      size_set = true;
    } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
      json_path = argv[++i];
      emit_json_stdout = true;
    } else if (std::strcmp(argv[i], "--group") == 0 && i + 1 < argc) {
      group_filter = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf("usage: ee_microbench [--quick] [--repeats K] [--size N] "
                  "[--json f] [--group g1,g2]\n"
                  "  --quick is applied first; --repeats and --size then override it.\n");
      return 0;
    } else {
      std::fprintf(stderr, "FATAL: unknown argument '%s'\n", argv[i]);
      return 2;
    }
  }

  // --quick replaces the whole config. Explicit --repeats / --size win
  // whether they appear before or after --quick.
  ee_bench::BenchConfig cfg =
      want_quick ? ee_bench::default_config_quick() : ee_bench::default_config();
  if (repeats_set)
    cfg.repeats = repeats;
  if (size_set)
    cfg.n_stream = static_cast<std::size_t>(size_value);

#if !defined(EE_HAVE_CUDA) && !defined(EE_HAVE_HIP)
  if (const int pin_status = pin_current_cpu())
    return pin_status;
#endif

#if defined(EE_HAVE_CUDA)
  {
    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err != cudaSuccess || ndev <= 0) {
      std::fprintf(stderr, "FATAL: no CUDA device (%s)\n",
                   err == cudaSuccess ? "count is 0" : cudaGetErrorString(err));
      return 2;
    }
    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "FATAL: cudaGetDeviceProperties: %s\n", cudaGetErrorString(err));
      return 2;
    }
    std::fprintf(stderr, "ee_cuda: device 0: %s (cc %d.%d, %.1f GiB)\n", prop.name, prop.major,
                 prop.minor, static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  }
#elif defined(EE_HAVE_HIP)
  {
    int ndev = 0;
    hipError_t err = hipGetDeviceCount(&ndev);
    if (err != hipSuccess || ndev <= 0) {
      std::fprintf(stderr, "FATAL: no HIP device (%s)\n",
                   err == hipSuccess ? "count is 0" : hipGetErrorString(err));
      return 2;
    }
    hipDeviceProp_t prop{};
    err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
      std::fprintf(stderr, "FATAL: hipGetDeviceProperties: %s\n", hipGetErrorString(err));
      return 2;
    }
    std::fprintf(stderr, "ee_hip: device 0: %s (%s, %.1f GiB)\n", prop.name, prop.gcnArchName,
                 static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  }
#endif

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
    ee_bench::print_json(all, stdout);
    if (!json_path.empty() && json_path != "-") {
      if (std::FILE *f = std::fopen(json_path.c_str(), "w")) {
        ee_bench::print_json(all, f);
        std::fclose(f);
      } else {
        std::fprintf(stderr, "FATAL: cannot open --json file '%s'\n", json_path.c_str());
        return 2;
      }
    }
  }
  return 0;
}
