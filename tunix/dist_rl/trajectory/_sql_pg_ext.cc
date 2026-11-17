// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#define TUNIX_HAS_POSIX_DLOPEN 1
#else
#define TUNIX_HAS_POSIX_DLOPEN 0
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace tunix::trajectory_ext {
namespace {

constexpr size_t kMaxPgQueryParams = 65534;
constexpr size_t kMaxInt32Bytes =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

// RAII wrapper around `Py_buffer` that releases the buffer with the GIL held.
class ScopedPyBuffer {
 public:
  ScopedPyBuffer() = default;
  ScopedPyBuffer(PyObject* obj, int flags) { Acquire(obj, flags); }
  ScopedPyBuffer(const ScopedPyBuffer&) = delete;
  ScopedPyBuffer& operator=(const ScopedPyBuffer&) = delete;
  ~ScopedPyBuffer() {
    if (valid_) {
      PyBuffer_Release(&view_);
    }
  }

  void Acquire(PyObject* obj, int flags) {
    if (PyObject_GetBuffer(obj, &view_, flags) != 0) {
      throw nb::python_error();
    }
    valid_ = true;
  }

  const Py_buffer& view() const noexcept { return view_; }
  bool valid() const noexcept { return valid_; }

 private:
  Py_buffer view_{};
  bool valid_ = false;
};

inline void WriteI16BE(uint8_t* p, int16_t v) noexcept {
  const auto u = static_cast<uint16_t>(v);
  p[0] = static_cast<uint8_t>((u >> 8) & 0xFFU);
  p[1] = static_cast<uint8_t>(u & 0xFFU);
}

inline void WriteI32BE(uint8_t* p, int32_t v) noexcept {
  const auto u = static_cast<uint32_t>(v);
  p[0] = static_cast<uint8_t>((u >> 24) & 0xFFU);
  p[1] = static_cast<uint8_t>((u >> 16) & 0xFFU);
  p[2] = static_cast<uint8_t>((u >> 8) & 0xFFU);
  p[3] = static_cast<uint8_t>(u & 0xFFU);
}

bool IsValidSqlIdentifier(std::string_view ident) noexcept {
  if (ident.empty() || ident.size() > 63) {
    return false;
  }
  const char first = ident[0];
  if (!((first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') ||
        first == '_')) {
    return false;
  }
  for (char ch : ident) {
    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_')) {
      return false;
    }
  }
  return true;
}

// Opaque representations of libpq structs.
struct PGconnOpaque;
struct PGresultOpaque;

// Move-only RAII wrapper for PGresultOpaque pointers.
using ScopedPgResult =
    std::unique_ptr<PGresultOpaque, void (*)(PGresultOpaque*)>;

inline ScopedPgResult MakeScopedPgResult(PGresultOpaque* res,
                                         void (*clear_fn)(PGresultOpaque*)) {
  return ScopedPgResult(
      res, clear_fn ? clear_fn : [](PGresultOpaque*) noexcept {});
}

enum PgConnStatus {
  kPgConnectionOk = 0,
};

enum PgExecStatus {
  kPgResEmptyQuery = 0,
  kPgResCommandOk = 1,
  kPgResTuplesOk = 2,
  kPgResCopyOut = 3,
  kPgResCopyIn = 4,
};

enum PgTransStatus {
  kPgTransIdle = 0,
  kPgTransActive = 1,
  kPgTransInTrans = 2,
  kPgTransInError = 3,
  kPgTransUnknown = 4,
};

struct LibpqApi {
  using PQexecParamsFn = PGresultOpaque* (*)(PGconnOpaque*, const char*, int,
                                             const unsigned int*,
                                             const char* const*, const int*,
                                             const int*, int);
  using PQexecFn = PGresultOpaque* (*)(PGconnOpaque*, const char*);
  using PQputCopyDataFn = int (*)(PGconnOpaque*, const char*, int);
  using PQputCopyEndFn = int (*)(PGconnOpaque*, const char*);
  using PQgetResultFn = PGresultOpaque* (*)(PGconnOpaque*);
  using PQresultStatusFn = int (*)(const PGresultOpaque*);
  using PQresultErrorFieldFn = char* (*)(const PGresultOpaque*, int);
  using PQerrorMessageFn = const char* (*)(const PGconnOpaque*);
  using PQclearFn = void (*)(PGresultOpaque*);
  using PQstatusFn = int (*)(const PGconnOpaque*);
  using PQtransactionStatusFn = int (*)(const PGconnOpaque*);
  using PQntuplesFn = int (*)(const PGresultOpaque*);
  using PQnfieldsFn = int (*)(const PGresultOpaque*);
  using PQgetisnullFn = int (*)(const PGresultOpaque*, int, int);
  using PQgetvalueFn = char* (*)(const PGresultOpaque*, int, int);
  using PQgetlengthFn = int (*)(const PGresultOpaque*, int, int);

  PQexecParamsFn exec_params = nullptr;
  PQexecFn exec = nullptr;
  PQputCopyDataFn put_copy_data = nullptr;
  PQputCopyEndFn put_copy_end = nullptr;
  PQgetResultFn get_result = nullptr;
  PQresultStatusFn result_status = nullptr;
  PQresultErrorFieldFn result_error_field = nullptr;
  PQerrorMessageFn error_message = nullptr;
  PQclearFn clear = nullptr;
  PQstatusFn conn_status = nullptr;
  PQtransactionStatusFn transaction_status = nullptr;
  PQntuplesFn ntuples = nullptr;
  PQnfieldsFn nfields = nullptr;
  PQgetisnullFn getisnull = nullptr;
  PQgetvalueFn getvalue = nullptr;
  PQgetlengthFn getlength = nullptr;

  bool has_exec_params() const noexcept {
    return exec_params != nullptr && result_status != nullptr &&
           error_message != nullptr && clear != nullptr &&
           conn_status != nullptr;
  }

  bool has_binary_copy() const noexcept {
    return exec != nullptr && put_copy_data != nullptr &&
           put_copy_end != nullptr && get_result != nullptr &&
           result_status != nullptr && error_message != nullptr &&
           clear != nullptr && conn_status != nullptr &&
           transaction_status != nullptr;
  }

  bool has_binary_fetch() const noexcept {
    return has_exec_params() && ntuples != nullptr && nfields != nullptr &&
           getisnull != nullptr && getvalue != nullptr && getlength != nullptr;
  }
};

LibpqApi g_libpq;
bool g_libpq_loaded = false;
std::mutex g_libpq_mutex;  // NOLINT(build/c++11)

#if TUNIX_HAS_POSIX_DLOPEN
void* LookupSymbol(void* handle, const char* name) noexcept {
  if (handle != nullptr) {
    return dlsym(handle, name);
  }
  return dlsym(RTLD_DEFAULT, name);
}
#endif

// Lazily and thread-safely resolves libpq dynamic symbols.
LibpqApi GetLibpqApi(std::string_view psycopg_lib_path) {
  std::lock_guard<std::mutex> lock(g_libpq_mutex);  // NOLINT(build/c++11)
  static auto& g_libpq_path = *new std::string();
  if (g_libpq_loaded && g_libpq_path == psycopg_lib_path &&
      (g_libpq.has_exec_params() || g_libpq.has_binary_copy())) {
    return g_libpq;
  }
  g_libpq = LibpqApi{};
  g_libpq_loaded = false;
  g_libpq_path = std::string(psycopg_lib_path);
#if TUNIX_HAS_POSIX_DLOPEN
  void* handle = nullptr;
  if (!psycopg_lib_path.empty()) {
    const std::string path_str(psycopg_lib_path);
#if defined(RTLD_NOLOAD)
    handle = dlopen(path_str.c_str(), RTLD_LAZY | RTLD_NOLOAD);
#endif
    if (handle == nullptr) {
      handle = dlopen(path_str.c_str(), RTLD_LAZY);
    }
    if (handle == nullptr) {
      g_libpq_loaded = true;
      return g_libpq;
    }
  }
  g_libpq.exec_params = reinterpret_cast<LibpqApi::PQexecParamsFn>(
      LookupSymbol(handle, "PQexecParams"));
  g_libpq.exec =
      reinterpret_cast<LibpqApi::PQexecFn>(LookupSymbol(handle, "PQexec"));
  g_libpq.put_copy_data = reinterpret_cast<LibpqApi::PQputCopyDataFn>(
      LookupSymbol(handle, "PQputCopyData"));
  g_libpq.put_copy_end = reinterpret_cast<LibpqApi::PQputCopyEndFn>(
      LookupSymbol(handle, "PQputCopyEnd"));
  g_libpq.get_result = reinterpret_cast<LibpqApi::PQgetResultFn>(
      LookupSymbol(handle, "PQgetResult"));
  g_libpq.result_status = reinterpret_cast<LibpqApi::PQresultStatusFn>(
      LookupSymbol(handle, "PQresultStatus"));
  g_libpq.result_error_field = reinterpret_cast<LibpqApi::PQresultErrorFieldFn>(
      LookupSymbol(handle, "PQresultErrorField"));
  g_libpq.error_message = reinterpret_cast<LibpqApi::PQerrorMessageFn>(
      LookupSymbol(handle, "PQerrorMessage"));
  g_libpq.clear =
      reinterpret_cast<LibpqApi::PQclearFn>(LookupSymbol(handle, "PQclear"));
  g_libpq.conn_status =
      reinterpret_cast<LibpqApi::PQstatusFn>(LookupSymbol(handle, "PQstatus"));
  g_libpq.transaction_status =
      reinterpret_cast<LibpqApi::PQtransactionStatusFn>(
          LookupSymbol(handle, "PQtransactionStatus"));
  g_libpq.ntuples = reinterpret_cast<LibpqApi::PQntuplesFn>(
      LookupSymbol(handle, "PQntuples"));
  g_libpq.nfields = reinterpret_cast<LibpqApi::PQnfieldsFn>(
      LookupSymbol(handle, "PQnfields"));
  g_libpq.getisnull = reinterpret_cast<LibpqApi::PQgetisnullFn>(
      LookupSymbol(handle, "PQgetisnull"));
  g_libpq.getvalue = reinterpret_cast<LibpqApi::PQgetvalueFn>(
      LookupSymbol(handle, "PQgetvalue"));
  g_libpq.getlength = reinterpret_cast<LibpqApi::PQgetlengthFn>(
      LookupSymbol(handle, "PQgetlength"));
#endif
  g_libpq_loaded = true;
  return g_libpq;
}

// Concatenates `std::string_view` pieces in a single allocation without
// intermediate `std::string` temporaries (standard C++17 equivalent of
// `absl::StrCat` for OSS nanobind builds).
std::string StrCat(std::initializer_list<std::string_view> pieces) {
  size_t total_size = 0;
  for (std::string_view piece : pieces) {
    total_size += piece.size();
  }
  std::string out;
  out.reserve(total_size);
  for (std::string_view piece : pieces) {
    out.append(piece.data(), piece.size());
  }
  return out;
}

std::string FormatPgErrorNoGil(const LibpqApi& pq, PGconnOpaque* pgconn,
                               PGresultOpaque* res) {
  std::string_view sqlstate = "08000";
  if (res != nullptr && pq.result_error_field != nullptr) {
    constexpr int kPgDiagSqlstate = 'C';
    const char* code = pq.result_error_field(res, kPgDiagSqlstate);
    if (code != nullptr && code[0] != '\0') {
      sqlstate = code;
    } else if (pq.conn_status(pgconn) == kPgConnectionOk) {
      sqlstate = "XX000";
    }
  }
  const char* msg = pq.error_message(pgconn);
  const std::string_view detail =
      (msg != nullptr)
          ? std::string_view(msg)
          : std::string_view("PostgreSQL binary execution failed.");
  return StrCat({"[SQLSTATE:", sqlstate, "] ", detail});
}

bool ExecCommandNoGil(const LibpqApi& pq, PGconnOpaque* pgconn, const char* sql,
                      int expected_status, std::string* err_out) {
  ScopedPgResult res = MakeScopedPgResult(pq.exec(pgconn, sql), pq.clear);
  if (!res || pq.result_status(res.get()) != expected_status) {
    *err_out = FormatPgErrorNoGil(pq, pgconn, res.get());
    return false;
  }
  return true;
}

// RAII guard to automatically issue ROLLBACK on failure if transaction was
// opened by the caller.
struct ScopedPgRollback {
  const LibpqApi& pq;
  PGconnOpaque* pgconn;
  bool armed = true;

  ~ScopedPgRollback() {
    if (armed) {
      std::string ignored;
      ExecCommandNoGil(pq, pgconn, "ROLLBACK", kPgResCommandOk, &ignored);
    }
  }

  void Disarm() noexcept { armed = false; }
};

void AppendBinaryCopyField(std::vector<uint8_t>* buf, std::string_view data) {
  const size_t offset = buf->size();
  buf->resize(offset + 4 + data.size());
  WriteI32BE(buf->data() + offset, static_cast<int32_t>(data.size()));
  if (!data.empty()) {
    std::memcpy(buf->data() + offset + 4, data.data(), data.size());
  }
}

bool PutAllCopyBytesNoGil(const LibpqApi& pq, PGconnOpaque* pgconn,
                          const uint8_t* ptr, size_t len,
                          std::string* err_out) {
  constexpr size_t kMaxCopyChunk = 64ULL * 1024ULL * 1024ULL;
  size_t sent = 0;
  while (sent < len) {
    const size_t chunk =
        (len - sent > kMaxCopyChunk) ? kMaxCopyChunk : (len - sent);
    if (pq.put_copy_data(pgconn, reinterpret_cast<const char*>(ptr + sent),
                         static_cast<int>(chunk)) != 1) {
      *err_out = FormatPgErrorNoGil(pq, pgconn, nullptr);
      return false;
    }
    sent += chunk;
  }
  return true;
}

bool UpsertStepViaCopyBinaryNoGil(
    const LibpqApi& pq, PGconnOpaque* pgconn, std::string_view run_id,
    std::string_view traj_id, int32_t step_id, std::string_view payload_json,
    std::string_view created_at_iso, const uint8_t* bin_ptr, size_t bin_len,
    bool has_binary, std::string_view binary_col, std::string* err_out) {
  const bool manage_tx = (pq.transaction_status(pgconn) == kPgTransIdle);
  if (manage_tx &&
      !ExecCommandNoGil(pq, pgconn, "BEGIN", kPgResCommandOk, err_out)) {
    return false;
  }
  ScopedPgRollback rollback_guard{pq, pgconn, manage_tx};

  const std::string temp_table =
      binary_col.empty() ? std::string("_tunix_step_stage_5")
                         : StrCat({"_tunix_step_stage_", binary_col});
  std::string stage_setup_sql;
  std::string copy_in_sql;
  if (binary_col.empty()) {
    stage_setup_sql =
        "CREATE TEMP TABLE IF NOT EXISTS _tunix_step_stage_5 ("
        "run_id text, trajectory_id text, step_id integer, "
        "payload text, created_at text"
        ") ON COMMIT DELETE ROWS; TRUNCATE _tunix_step_stage_5;";
    copy_in_sql =
        "COPY _tunix_step_stage_5 "
        "(run_id, trajectory_id, step_id, payload, created_at) "
        "FROM STDIN WITH (FORMAT binary)";
  } else {
    stage_setup_sql = StrCat({
        "CREATE TEMP TABLE IF NOT EXISTS ",
        temp_table,
        " (run_id text, trajectory_id text, step_id integer, payload text, ",
        binary_col,
        " bytea, created_at text) ON COMMIT DELETE ROWS; TRUNCATE ",
        temp_table,
        ";",
    });
    copy_in_sql = StrCat({
        "COPY ",
        temp_table,
        " (run_id, trajectory_id, step_id, payload, ",
        binary_col,
        ", created_at) FROM STDIN WITH (FORMAT binary)",
    });
  }

  if (!ExecCommandNoGil(pq, pgconn, stage_setup_sql.c_str(), kPgResCommandOk,
                        err_out)) {
    return false;
  }

  if (!ExecCommandNoGil(pq, pgconn, copy_in_sql.c_str(), kPgResCopyIn,
                        err_out)) {
    return false;
  }

  static constexpr uint8_t kPgCopySignature[19] = {
      'P',  'G',  'C',  'O',  'P',  'Y',  '\n', 0xFF, '\r', '\n',
      '\0', 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> prefix;
  prefix.reserve(64 + run_id.size() + traj_id.size() + payload_json.size());
  prefix.insert(prefix.end(), kPgCopySignature,
                kPgCopySignature + sizeof(kPgCopySignature));
  const size_t ncols_off = prefix.size();
  prefix.resize(ncols_off + 2);
  const int16_t num_cols = binary_col.empty() ? 5 : 6;
  WriteI16BE(prefix.data() + ncols_off, num_cols);

  AppendBinaryCopyField(&prefix, run_id);
  AppendBinaryCopyField(&prefix, traj_id);

  const size_t step_off = prefix.size();
  prefix.resize(step_off + 8);
  WriteI32BE(prefix.data() + step_off, 4);
  WriteI32BE(prefix.data() + step_off + 4, step_id);

  AppendBinaryCopyField(&prefix, payload_json);

  std::vector<uint8_t> suffix;
  if (!binary_col.empty()) {
    const size_t bytea_len_off = prefix.size();
    prefix.resize(bytea_len_off + 4);
    WriteI32BE(prefix.data() + bytea_len_off,
               has_binary ? static_cast<int32_t>(bin_len) : -1);
  }

  suffix.reserve(16 + created_at_iso.size());
  AppendBinaryCopyField(&suffix, created_at_iso);
  const size_t trailer_off = suffix.size();
  suffix.resize(trailer_off + 2);
  WriteI16BE(suffix.data() + trailer_off, -1);

  bool copy_ok =
      PutAllCopyBytesNoGil(pq, pgconn, prefix.data(), prefix.size(), err_out);
  if (copy_ok && !binary_col.empty() && has_binary && bin_len > 0) {
    copy_ok = PutAllCopyBytesNoGil(pq, pgconn, bin_ptr, bin_len, err_out);
  }
  if (copy_ok) {
    copy_ok =
        PutAllCopyBytesNoGil(pq, pgconn, suffix.data(), suffix.size(), err_out);
  }

  if (pq.put_copy_end(pgconn, copy_ok ? nullptr : "copy aborted") != 1 &&
      copy_ok) {
    *err_out = FormatPgErrorNoGil(pq, pgconn, nullptr);
    copy_ok = false;
  }

  PGresultOpaque* raw_copy_res = nullptr;
  while ((raw_copy_res = pq.get_result(pgconn)) != nullptr) {
    ScopedPgResult copy_res = MakeScopedPgResult(raw_copy_res, pq.clear);
    const int st = pq.result_status(copy_res.get());
    if (st != kPgResCommandOk && copy_ok) {
      *err_out = FormatPgErrorNoGil(pq, pgconn, copy_res.get());
      copy_ok = false;
    }
  }
  if (!copy_ok) {
    return false;
  }

  std::string merge_sql;
  if (binary_col.empty()) {
    merge_sql =
        "INSERT INTO steps (run_id, trajectory_id, step_id, payload, "
        "created_at) "
        "SELECT run_id, trajectory_id, step_id, payload::jsonb, "
        "created_at::timestamptz FROM _tunix_step_stage_5 "
        "ON CONFLICT (run_id, trajectory_id, step_id) DO UPDATE SET "
        "payload = EXCLUDED.payload, created_at = EXCLUDED.created_at";
  } else {
    merge_sql = StrCat({
        "INSERT INTO steps (run_id, trajectory_id, step_id, payload, ",
        binary_col,
        ", created_at) SELECT run_id, trajectory_id, step_id, payload::jsonb, ",
        binary_col,
        ", created_at::timestamptz FROM ",
        temp_table,
        " ON CONFLICT (run_id, trajectory_id, step_id) DO UPDATE SET "
        "payload = EXCLUDED.payload, ",
        binary_col,
        " = EXCLUDED.",
        binary_col,
        ", created_at = EXCLUDED.created_at",
    });
  }

  if (!ExecCommandNoGil(pq, pgconn, merge_sql.c_str(), kPgResCommandOk,
                        err_out)) {
    return false;
  }

  if (manage_tx) {
    if (!ExecCommandNoGil(pq, pgconn, "COMMIT", kPgResCommandOk, err_out)) {
      return false;
    }
    rollback_guard.Disarm();
  }
  return true;
}

bool CanExecutePgBinary(std::string_view psycopg_lib_path) {
  const LibpqApi pq = GetLibpqApi(psycopg_lib_path);
  return pq.has_exec_params() || pq.has_binary_copy();
}

bool CanFetchPgBinary(std::string_view psycopg_lib_path) {
  const LibpqApi pq = GetLibpqApi(psycopg_lib_path);
  return pq.has_binary_fetch();
}

void PgUpsertStep(uintptr_t pgconn_addr, std::string_view run_id,
                  std::string_view traj_id, int64_t step_id,
                  std::string_view payload_json,
                  std::string_view created_at_iso,
                  nb::handle binary_obj = nb::none(),
                  std::string_view binary_col = "",
                  std::string_view psycopg_lib_path = "") {
  if (pgconn_addr == 0) {
    throw nb::value_error("pgconn_ptr must be non-zero.");
  }
  if (step_id < 0 || step_id > static_cast<int64_t>(kMaxInt32Bytes)) {
    throw nb::value_error("step_id exceeds PostgreSQL integer range.");
  }
  const bool has_binary = !binary_obj.is_none();
  if (has_binary && binary_col.empty()) {
    throw nb::value_error(
        "binary_col must be non-empty when binary_data is provided.");
  }
  if (!binary_col.empty() && !IsValidSqlIdentifier(binary_col)) {
    throw nb::value_error("Invalid binary_col identifier.");
  }
  const LibpqApi pq = GetLibpqApi(psycopg_lib_path);
  if (!pq.has_exec_params() && !pq.has_binary_copy()) {
    throw std::runtime_error("libpq binary symbols are unavailable.");
  }

  ScopedPyBuffer bin_buf;
  const uint8_t* bin_ptr = nullptr;
  size_t bin_len = 0;
  if (has_binary && !binary_col.empty()) {
    bin_buf.Acquire(binary_obj.ptr(), PyBUF_SIMPLE);
    bin_ptr = static_cast<const uint8_t*>(bin_buf.view().buf);
    bin_len = static_cast<size_t>(bin_buf.view().len);
    if (bin_len > kMaxInt32Bytes) {
      throw nb::value_error(
          "Binary column byte length exceeds PostgreSQL 2 GiB parameter "
          "limit.");
    }
  }

  if (run_id.size() > kMaxInt32Bytes || traj_id.size() > kMaxInt32Bytes ||
      payload_json.size() > kMaxInt32Bytes ||
      created_at_iso.size() > kMaxInt32Bytes) {
    throw nb::value_error(
        "Step text field exceeds PostgreSQL parameter limit.");
  }

  auto* pgconn = reinterpret_cast<PGconnOpaque*>(pgconn_addr);
  std::string err_msg;
  bool ok = true;

  {
    nb::gil_scoped_release release;
    if (pq.has_exec_params()) {
      const std::string run_s(run_id);
      const std::string traj_s(traj_id);
      char step_buf[32];
      std::snprintf(step_buf, sizeof(step_buf), "%d",
                    static_cast<int32_t>(step_id));
      const std::string payload_s(payload_json);
      const std::string created_s(created_at_iso);

      ScopedPgResult res(nullptr, pq.clear);
      if (binary_col.empty()) {
        const char* param_values[5] = {
            run_s.c_str(),     traj_s.c_str(),    step_buf,
            payload_s.c_str(), created_s.c_str(),
        };
        const int param_lengths[5] = {
            static_cast<int>(run_s.size()),
            static_cast<int>(traj_s.size()),
            0,
            static_cast<int>(payload_s.size()),
            static_cast<int>(created_s.size()),
        };
        const int param_formats[5] = {0, 0, 0, 0, 0};

        static constexpr const char* kUpsertSql5 =
            "INSERT INTO steps "
            "(run_id, trajectory_id, step_id, payload, created_at) "
            "VALUES ($1, $2, $3::integer, $4::jsonb, $5::timestamptz) "
            "ON CONFLICT (run_id, trajectory_id, step_id) DO UPDATE SET "
            "payload = EXCLUDED.payload, "
            "created_at = EXCLUDED.created_at";

        res = MakeScopedPgResult(
            pq.exec_params(pgconn, kUpsertSql5, 5, nullptr, param_values,
                           param_lengths, param_formats, 0),
            pq.clear);
      } else {
        const char* param_values[6] = {
            run_s.c_str(),
            traj_s.c_str(),
            step_buf,
            payload_s.c_str(),
            has_binary ? reinterpret_cast<const char*>(bin_ptr) : nullptr,
            created_s.c_str(),
        };
        const int param_lengths[6] = {
            static_cast<int>(run_s.size()),
            static_cast<int>(traj_s.size()),
            0,
            static_cast<int>(payload_s.size()),
            has_binary ? static_cast<int>(bin_len) : 0,
            static_cast<int>(created_s.size()),
        };
        const int param_formats[6] = {0, 0, 0, 0, 1, 0};

        const std::string upsert_sql_6 = StrCat({
            "INSERT INTO steps (run_id, trajectory_id, step_id, payload, ",
            binary_col,
            ", created_at) VALUES ($1, $2, $3::integer, $4::jsonb, $5::bytea, "
            "$6::timestamptz) ON CONFLICT (run_id, trajectory_id, step_id) "
            "DO UPDATE SET payload = EXCLUDED.payload, ",
            binary_col,
            " = EXCLUDED.",
            binary_col,
            ", created_at = EXCLUDED.created_at",
        });

        res = MakeScopedPgResult(
            pq.exec_params(pgconn, upsert_sql_6.c_str(), 6, nullptr,
                           param_values, param_lengths, param_formats, 0),
            pq.clear);
      }

      if (!res || pq.result_status(res.get()) != kPgResCommandOk) {
        err_msg = FormatPgErrorNoGil(pq, pgconn, res.get());
        ok = false;
      }
    } else {
      ok = UpsertStepViaCopyBinaryNoGil(
          pq, pgconn, run_id, traj_id, static_cast<int32_t>(step_id),
          payload_json, created_at_iso, bin_ptr, bin_len, has_binary,
          binary_col, &err_msg);
    }
  }

  if (!ok) {
    throw std::runtime_error(err_msg);
  }
}

struct FreeDeleter {
  void operator()(void* p) const noexcept { std::free(p); }
};

struct FetchedStepRow {
  std::string trajectory_id;
  std::string payload_json;
  bool has_binary = false;
  size_t binary_len = 0;
  std::unique_ptr<uint8_t, FreeDeleter> binary_buf;
};

nb::list PgFetchSteps(uintptr_t pgconn_addr, std::string_view run_id,
                      const std::vector<std::string>& trajectory_ids,
                      std::string_view binary_col = "",
                      std::string_view psycopg_lib_path = "") {
  nb::list out;
  if (trajectory_ids.empty()) {
    return out;
  }
  if (trajectory_ids.size() > kMaxPgQueryParams) {
    throw nb::value_error(
        "trajectory_ids size exceeds PostgreSQL parameter limit (65534).");
  }
  if (pgconn_addr == 0) {
    throw nb::value_error("pgconn_ptr must be non-zero.");
  }
  if (run_id.size() > kMaxInt32Bytes) {
    throw nb::value_error("run_id size exceeds maximum allowed length.");
  }
  for (const auto& tid : trajectory_ids) {
    if (tid.size() > kMaxInt32Bytes) {
      throw nb::value_error(
          "trajectory_id size exceeds maximum allowed length.");
    }
  }
  if (!binary_col.empty() && !IsValidSqlIdentifier(binary_col)) {
    throw nb::value_error("Invalid binary_col identifier.");
  }
  const LibpqApi pq = GetLibpqApi(psycopg_lib_path);
  if (!pq.has_binary_fetch()) {
    throw std::runtime_error("libpq binary fetch symbols are unavailable.");
  }

  auto* pgconn = reinterpret_cast<PGconnOpaque*>(pgconn_addr);
  std::vector<FetchedStepRow> fetched_rows;
  std::string err_msg;
  bool ok = true;

  {
    nb::gil_scoped_release release;
    const std::string run_s(run_id);
    std::string sql;
    if (binary_col.empty()) {
      sql =
          "SELECT trajectory_id::text, payload::text "
          "FROM steps WHERE run_id = $1 AND trajectory_id IN (";
    } else {
      sql = StrCat({
          "SELECT trajectory_id::text, payload::text, ",
          binary_col,
          " FROM steps WHERE run_id = $1 AND trajectory_id IN (",
      });
    }

    std::vector<const char*> param_values;
    std::vector<int> param_lengths;
    std::vector<int> param_formats;
    const size_t nparams = 1 + trajectory_ids.size();
    param_values.reserve(nparams);
    param_lengths.reserve(nparams);
    param_formats.reserve(nparams);

    param_values.push_back(run_s.c_str());
    param_lengths.push_back(static_cast<int>(run_s.size()));
    param_formats.push_back(0);

    for (size_t i = 0; i < trajectory_ids.size(); ++i) {
      if (i > 0) {
        sql.append(", ");
      }
      sql.push_back('$');
      sql.append(std::to_string(i + 2));
      param_values.push_back(trajectory_ids[i].c_str());
      param_lengths.push_back(static_cast<int>(trajectory_ids[i].size()));
      param_formats.push_back(0);
    }
    sql.append(") ORDER BY trajectory_id, step_id");

    ScopedPgResult res = MakeScopedPgResult(
        pq.exec_params(pgconn, sql.c_str(), static_cast<int>(nparams), nullptr,
                       param_values.data(), param_lengths.data(),
                       param_formats.data(),
                       /*resultFormat=*/1),
        pq.clear);
    if (!res || pq.result_status(res.get()) != kPgResTuplesOk) {
      err_msg = FormatPgErrorNoGil(pq, pgconn, res.get());
      ok = false;
    } else {
      const int nrows = pq.ntuples(res.get());
      fetched_rows.resize(static_cast<size_t>(nrows));
      for (int r = 0; r < nrows; ++r) {
        FetchedStepRow& item = fetched_rows[static_cast<size_t>(r)];
        const char* tid_ptr = pq.getvalue(res.get(), r, 0);
        const int tid_len = pq.getlength(res.get(), r, 0);
        item.trajectory_id.assign(tid_ptr, static_cast<size_t>(tid_len));

        const char* pay_ptr = pq.getvalue(res.get(), r, 1);
        const int pay_len = pq.getlength(res.get(), r, 1);
        item.payload_json.assign(pay_ptr, static_cast<size_t>(pay_len));

        if (!binary_col.empty() && !pq.getisnull(res.get(), r, 2)) {
          const auto* bin_ptr =
              reinterpret_cast<const uint8_t*>(pq.getvalue(res.get(), r, 2));
          const int bin_len = pq.getlength(res.get(), r, 2);
          item.has_binary = true;
          if (bin_len > 0 && bin_ptr != nullptr) {
            item.binary_len = static_cast<size_t>(bin_len);
            auto* buf = static_cast<uint8_t*>(std::malloc(item.binary_len));
            if (buf == nullptr) {
              throw std::bad_alloc();
            }
            std::memcpy(buf, bin_ptr, item.binary_len);
            item.binary_buf.reset(buf);
          } else {
            item.binary_len = 0;
            auto* buf = static_cast<uint8_t*>(std::malloc(1));
            if (buf == nullptr) {
              throw std::bad_alloc();
            }
            item.binary_buf.reset(buf);
          }
        }
      }
    }
  }

  if (!ok) {
    throw std::runtime_error(err_msg);
  }

  for (FetchedStepRow& item : fetched_rows) {
    nb::object bin_obj = nb::none();
    if (item.has_binary && item.binary_buf != nullptr) {
      uint8_t* raw = item.binary_buf.release();
      nb::capsule owner(raw, [](void* p) noexcept { std::free(p); });
      size_t shape[1] = {item.binary_len};
      bin_obj = nb::cast(
          nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(raw, 1, shape, owner));
    }
    out.append(nb::make_tuple(item.trajectory_id, item.payload_json, bin_obj));
  }
  return out;
}

nb::object StripJsonNulBytes(nb::str json_obj) {
  const char* data = json_obj.c_str();
  Py_ssize_t py_len = 0;
#if !defined(Py_LIMITED_API)
  data = PyUnicode_AsUTF8AndSize(json_obj.ptr(), &py_len);
  if (data == nullptr) {
    throw nb::python_error();
  }
#else
  py_len = static_cast<Py_ssize_t>(std::strlen(data));
#endif
  const std::string_view sv(data, static_cast<size_t>(py_len));

  bool needs_sanitize = false;
  std::string cleaned;
  {
    nb::gil_scoped_release release;
    constexpr std::string_view kEscapedNul = "\\u0000";
    const bool has_raw_nul =
        (std::memchr(sv.data(), '\0', sv.size()) != nullptr);
    const bool has_esc_nul = (sv.find(kEscapedNul) != std::string_view::npos);
    if (has_raw_nul || has_esc_nul) {
      cleaned.reserve(sv.size());
      size_t i = 0;
      bool odd_backslashes = false;
      while (i < sv.size()) {
        const char ch = sv[i];
        if (ch == '\0') {
          odd_backslashes = false;
          ++i;
          continue;
        }
        if (!odd_backslashes && i + kEscapedNul.size() <= sv.size() &&
            std::memcmp(sv.data() + i, kEscapedNul.data(),
                        kEscapedNul.size()) == 0) {
          i += kEscapedNul.size();
          continue;
        }
        cleaned.push_back(ch);
        odd_backslashes = (ch == '\\') ? !odd_backslashes : false;
        ++i;
      }
      needs_sanitize = (cleaned.size() != sv.size());
    }
  }

  if (!needs_sanitize) {
    return json_obj;
  }
  return nb::cast(cleaned);
}

}  // namespace

NB_MODULE(_sql_pg_ext, m) {
  m.doc() =
      "C++ nanobind extension for domain-neutral PostgreSQL libpq wire "
      "execution and JSON sanitization.";
  m.def("can_execute_pg_binary", &CanExecutePgBinary,
        nb::arg("psycopg_lib_path") = "",
        "Returns True when libpq binary step upsert symbols are resolvable.");
  m.def("can_fetch_pg_binary", &CanFetchPgBinary,
        nb::arg("psycopg_lib_path") = "",
        "Returns True when libpq binary step fetch symbols are resolvable.");
  m.def("pg_upsert_step", &PgUpsertStep, nb::arg("pgconn_ptr"),
        nb::arg("run_id"), nb::arg("trajectory_id"), nb::arg("step_id"),
        nb::arg("payload_json"), nb::arg("created_at_iso"),
        nb::arg("binary_data") = nb::none(), nb::arg("binary_col") = "",
        nb::arg("psycopg_lib_path") = "",
        "Executes a binary STEPS_TABLE upsert directly over a psycopg2 PGconn "
        "pointer with the GIL released.");
  m.def("pg_fetch_steps", &PgFetchSteps, nb::arg("pgconn_ptr"),
        nb::arg("run_id"), nb::arg("trajectory_ids"),
        nb::arg("binary_col") = "", nb::arg("psycopg_lib_path") = "",
        "Fetches step JSON payloads and optional binary columns directly "
        "over a psycopg2 PGconn pointer with the GIL released.");
  m.def("strip_json_nul_bytes", &StripJsonNulBytes, nb::arg("json_str"),
        "Strips raw NUL bytes and \\u0000 escape sequences from a JSON string "
        "with the GIL released, returning the original str if none exist.");
}

}  // namespace tunix::trajectory_ext
