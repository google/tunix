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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 4
#define MAX_COLS 4

typedef struct {
  int status;
  char sqlstate[6];
  int ntuples;
  int nfields;
  char* values[MAX_ROWS][MAX_COLS];
  int is_null[MAX_ROWS][MAX_COLS];
  int lengths[MAX_ROWS][MAX_COLS];
} FakeResult;

static _Thread_local char g_last_error[512] = "";

static char* SafeStrdup(const char* s) {
  if (s == NULL) {
    return NULL;
  }
  size_t len = strlen(s);
  char* p = (char*)malloc(len + 1);
  if (p != NULL) {
    memcpy(p, s, len + 1);
  }
  return p;
}

int PQstatus(const void* conn) {
  (void)conn;
  return 0;  // CONNECTION_OK
}

int PQtransactionStatus(const void* conn) {
  (void)conn;
  return 0;  // PQTRANS_IDLE
}

const char* PQerrorMessage(const void* conn) {
  (void)conn;
  return g_last_error;
}

char* PQresultErrorField(const void* res, int fieldcode) {
  if (res == NULL) {
    return NULL;
  }
  FakeResult* r = (FakeResult*)res;
  if (fieldcode == 'C') {  // PG_DIAG_SQLSTATE
    return r->sqlstate;
  }
  return NULL;
}

int PQresultStatus(const void* res) {
  if (res == NULL) {
    return 7;  // PGRES_FATAL_ERROR
  }
  return ((FakeResult*)res)->status;
}

void PQclear(void* res) {
  if (res == NULL) {
    return;
  }
  FakeResult* r = (FakeResult*)res;
  for (int i = 0; i < r->ntuples; ++i) {
    for (int j = 0; j < r->nfields; ++j) {
      if (r->values[i][j] != NULL) {
        free(r->values[i][j]);
        r->values[i][j] = NULL;
      }
    }
  }
  free(r);
}

int PQntuples(const void* res) {
  if (res == NULL) {
    return 0;
  }
  return ((FakeResult*)res)->ntuples;
}

int PQnfields(const void* res) {
  if (res == NULL) {
    return 0;
  }
  return ((FakeResult*)res)->nfields;
}

int PQgetisnull(const void* res, int tup_num, int field_num) {
  if (res == NULL) {
    return 1;
  }
  FakeResult* r = (FakeResult*)res;
  if (tup_num < 0 || tup_num >= r->ntuples || field_num < 0 ||
      field_num >= r->nfields) {
    return 1;
  }
  return r->is_null[tup_num][field_num];
}

char* PQgetvalue(const void* res, int tup_num, int field_num) {
  static char kEmptyValue[] = "";
  if (res == NULL) {
    return kEmptyValue;
  }
  FakeResult* r = (FakeResult*)res;
  if (tup_num < 0 || tup_num >= r->ntuples || field_num < 0 ||
      field_num >= r->nfields) {
    return kEmptyValue;
  }
  if (r->values[tup_num][field_num] == NULL) {
    return kEmptyValue;
  }
  return r->values[tup_num][field_num];
}

int PQgetlength(const void* res, int tup_num, int field_num) {
  if (res == NULL) {
    return 0;
  }
  FakeResult* r = (FakeResult*)res;
  if (tup_num < 0 || tup_num >= r->ntuples || field_num < 0 ||
      field_num >= r->nfields) {
    return 0;
  }
  return r->lengths[tup_num][field_num];
}

void* PQexec(void* conn, const char* query) {
  (void)conn;
  FakeResult* res = (FakeResult*)calloc(1, sizeof(FakeResult));
  if (res == NULL) {
    return NULL;
  }
  res->status = 1;  // PGRES_COMMAND_OK
  snprintf(res->sqlstate, sizeof(res->sqlstate), "XX000");

  if (query != NULL && strstr(query, "fail_copy") != NULL) {
    res->status = 7;  // PGRES_FATAL_ERROR
    snprintf(g_last_error, sizeof(g_last_error), "Simulated copy failure");
    return res;
  }
  if (query != NULL && strstr(query, "COPY") != NULL) {
    res->status = 4;  // PGRES_COPY_IN
    return res;
  }
  return res;
}

#if !defined(FAKE_LIBPQ_OMIT_EXEC_PARAMS)
void* PQexecParams(void* conn, const char* command, int nParams,
                   const unsigned int* paramTypes,
                   const char* const* paramValues,
                   const int* paramLengths,
                   const int* paramFormats,
                   int resultFormat) {
  (void)conn;
  (void)paramTypes;
  (void)paramLengths;
  (void)paramFormats;
  (void)resultFormat;

  FakeResult* res = (FakeResult*)calloc(1, sizeof(FakeResult));
  if (res == NULL) {
    return NULL;
  }
  res->status = 1;  // PGRES_COMMAND_OK
  snprintf(res->sqlstate, sizeof(res->sqlstate), "XX000");

  if (command != NULL && strstr(command, "INSERT") != NULL) {
    if (nParams > 0 && paramValues != NULL && paramValues[0] != NULL &&
        strcmp(paramValues[0], "fail_upsert") == 0) {
      res->status = 7;  // PGRES_FATAL_ERROR
      snprintf(res->sqlstate, sizeof(res->sqlstate), "23505");
      snprintf(g_last_error, sizeof(g_last_error),
               "duplicate key value violates unique constraint");
      return res;
    }
    return res;
  }

  if (command != NULL && strstr(command, "SELECT") != NULL) {
    if (nParams > 0 && paramValues != NULL && paramValues[0] != NULL &&
        strcmp(paramValues[0], "fail_fetch") == 0) {
      res->status = 7;  // PGRES_FATAL_ERROR
      snprintf(res->sqlstate, sizeof(res->sqlstate), "08006");
      snprintf(g_last_error, sizeof(g_last_error),
               "connection failure during fetch");
      return res;
    }
    res->status = 2;  // PGRES_TUPLES_OK
    if (nParams > 1 && paramValues != NULL && paramValues[1] != NULL) {
      const char* tid = paramValues[1];
      const char* payload =
          "{\"step_id\": 1, \"source\": \"agent\", \"message\": \"test step\", "
          "\"timestamp\": \"2026-11-06T00:00:00Z\", \"observation\": null}";
      res->ntuples = 1;
      res->values[0][0] = SafeStrdup(tid);
      res->lengths[0][0] = (int)strlen(tid);
      res->values[0][1] = SafeStrdup(payload);
      res->lengths[0][1] = (int)strlen(payload);

      if (strstr(command, "routed_experts") != NULL ||
          strstr(command, "binary_data") != NULL) {
        res->nfields = 3;
        char* bin_val = (char*)malloc(4);
        if (bin_val != NULL) {
          bin_val[0] = 1;
          bin_val[1] = 2;
          bin_val[2] = 3;
          bin_val[3] = 4;
        }
        res->values[0][2] = bin_val;
        res->lengths[0][2] = 4;
      } else {
        res->nfields = 2;
      }
    }
    return res;
  }

  return res;
}
#endif

int PQputCopyData(void* conn, const char* buffer, int nbytes) {
  (void)conn;
  (void)buffer;
  (void)nbytes;
  return 1;
}

static _Thread_local int g_copy_result_pending = 0;

int PQputCopyEnd(void* conn, const char* errormsg) {
  (void)conn;
  if (errormsg != NULL) {
    snprintf(g_last_error, sizeof(g_last_error), "%s", errormsg);
    return -1;
  }
  g_copy_result_pending = 1;
  return 1;
}

void* PQgetResult(void* conn) {
  (void)conn;
  if (g_copy_result_pending) {
    g_copy_result_pending = 0;
    FakeResult* res = (FakeResult*)calloc(1, sizeof(FakeResult));
    if (res == NULL) {
      return NULL;
    }
    res->status = 1;  // PGRES_COMMAND_OK
    snprintf(res->sqlstate, sizeof(res->sqlstate), "XX000");
    return res;
  }
  return NULL;  // End of results
}
