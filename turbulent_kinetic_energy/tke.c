#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-label"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-label"
#endif
// Headers

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <float.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


// Initialisation

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s);
void
futhark_context_config_select_device_interactively(struct futhark_context_config *cfg);
void futhark_context_config_list_devices(struct futhark_context_config *cfg);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path);
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
struct futhark_context
*futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                        cl_command_queue queue);
void futhark_context_free(struct futhark_context *ctx);
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx);

// Arrays

struct futhark_f64_1d ;
struct futhark_f64_1d *futhark_new_f64_1d(struct futhark_context *ctx, const
                                          double *data, int64_t dim0);
struct futhark_f64_1d *futhark_new_raw_f64_1d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0);
int futhark_free_f64_1d(struct futhark_context *ctx,
                        struct futhark_f64_1d *arr);
int futhark_values_f64_1d(struct futhark_context *ctx,
                          struct futhark_f64_1d *arr, double *data);
cl_mem futhark_values_raw_f64_1d(struct futhark_context *ctx,
                                 struct futhark_f64_1d *arr);
const int64_t *futhark_shape_f64_1d(struct futhark_context *ctx,
                                    struct futhark_f64_1d *arr);
struct futhark_f64_2d ;
struct futhark_f64_2d *futhark_new_f64_2d(struct futhark_context *ctx, const
                                          double *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f64_2d *futhark_new_raw_f64_2d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_f64_2d(struct futhark_context *ctx,
                        struct futhark_f64_2d *arr);
int futhark_values_f64_2d(struct futhark_context *ctx,
                          struct futhark_f64_2d *arr, double *data);
cl_mem futhark_values_raw_f64_2d(struct futhark_context *ctx,
                                 struct futhark_f64_2d *arr);
const int64_t *futhark_shape_f64_2d(struct futhark_context *ctx,
                                    struct futhark_f64_2d *arr);
struct futhark_f64_3d ;
struct futhark_f64_3d *futhark_new_f64_3d(struct futhark_context *ctx, const
                                          double *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2);
struct futhark_f64_3d *futhark_new_raw_f64_3d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2);
int futhark_free_f64_3d(struct futhark_context *ctx,
                        struct futhark_f64_3d *arr);
int futhark_values_f64_3d(struct futhark_context *ctx,
                          struct futhark_f64_3d *arr, double *data);
cl_mem futhark_values_raw_f64_3d(struct futhark_context *ctx,
                                 struct futhark_f64_3d *arr);
const int64_t *futhark_shape_f64_3d(struct futhark_context *ctx,
                                    struct futhark_f64_3d *arr);
struct futhark_i32_2d ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx, const
                                          int32_t *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_i32_2d(struct futhark_context *ctx,
                        struct futhark_i32_2d *arr);
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data);
cl_mem futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                 struct futhark_i32_2d *arr);
const int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                                    struct futhark_i32_2d *arr);

// Opaque values


// Entry points

int futhark_entry_integrate_tke(struct futhark_context *ctx,
                                struct futhark_f64_3d **out0,
                                struct futhark_f64_3d **out1,
                                struct futhark_f64_3d **out2,
                                struct futhark_f64_3d **out3,
                                struct futhark_f64_3d **out4,
                                struct futhark_f64_3d **out5,
                                struct futhark_f64_2d **out6, const
                                struct futhark_f64_3d *in0, const
                                struct futhark_f64_3d *in1, const
                                struct futhark_f64_3d *in2, const
                                struct futhark_f64_3d *in3, const
                                struct futhark_f64_3d *in4, const
                                struct futhark_f64_3d *in5, const
                                struct futhark_f64_3d *in6, const
                                struct futhark_f64_3d *in7, const
                                struct futhark_f64_3d *in8, const
                                struct futhark_f64_3d *in9, const
                                struct futhark_f64_3d *in10, const
                                struct futhark_f64_3d *in11, const
                                struct futhark_f64_1d *in12, const
                                struct futhark_f64_1d *in13, const
                                struct futhark_f64_1d *in14, const
                                struct futhark_f64_1d *in15, const
                                struct futhark_f64_1d *in16, const
                                struct futhark_f64_1d *in17, const
                                struct futhark_f64_1d *in18, const
                                struct futhark_f64_1d *in19, const
                                struct futhark_i32_2d *in20, const
                                struct futhark_f64_3d *in21, const
                                struct futhark_f64_3d *in22, const
                                struct futhark_f64_3d *in23, const
                                struct futhark_f64_2d *in24);

// Miscellaneous

int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
#define FUTHARK_BACKEND_opencl
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
#include <stdarg.h>
// Start of util.h.
//
// Various helper functions that are useful in all generated C code.

#include <errno.h>
#include <string.h>

static const char *fut_progname = "(embedded Futhark)";

static void futhark_panic(int eval, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s: ", fut_progname);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(eval);
}

// For generating arbitrary-sized error messages.  It is the callers
// responsibility to free the buffer at some point.
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); // Must re-init.
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}


static inline void check_err(int errval, int sets_errno, const char *fun, int line,
                            const char *msg, ...) {
  if (errval) {
    char str[256];
    char errnum[10];

    va_list vl;
    va_start(vl, msg);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, msg, vl);
    fprintf(stderr, " in %s() at line %d with error code %s\n",
            fun, line,
            sets_errno ? strerror(errno) : errnum);
    exit(errval);
  }
}

#define CHECK_ERR(err, msg...) check_err(err, 0, __func__, __LINE__, msg)
#define CHECK_ERRNO(err, msg...) check_err(err, 1, __func__, __LINE__, msg)

// Read a file into a NUL-terminated string; returns NULL on error.
static void* slurp_file(const char *filename, size_t *size) {
  unsigned char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (unsigned char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

// Dump 'n' bytes from 'buf' into the file at the designated location.
// Returns 0 on success.
static int dump_file(const char *file, const void *buf, size_t n) {
  FILE *f = fopen(file, "w");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(buf, sizeof(char), n, f) != n) {
    return 1;
  }

  if (fclose(f) != 0) {
    return 1;
  }

  return 0;
}

struct str_builder {
  char *str;
  size_t capacity; // Size of buffer.
  size_t used; // Bytes used, *not* including final zero.
};

static void str_builder_init(struct str_builder *b) {
  b->capacity = 10;
  b->used = 0;
  b->str = malloc(b->capacity);
  b->str[0] = 0;
}

static void str_builder(struct str_builder *b, const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = (size_t)vsnprintf(NULL, 0, s, vl);

  while (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }

  va_start(vl, s); // Must re-init.
  vsnprintf(b->str+b->used, b->capacity-b->used, s, vl);
  b->used += needed;
}

// End of util.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
// Assuming POSIX

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

static int64_t get_wall_time_ns(void) {
  struct timespec time;
  assert(clock_gettime(CLOCK_REALTIME, &time) == 0);
  return time.tv_sec * 1000000000 + time.tv_nsec;
}


static inline uint64_t rdtsc() {
  unsigned int hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return  ((uint64_t) lo) | (((uint64_t) hi) << 32);
}

static inline void rdtsc_wait(uint64_t n) {
  const uint64_t start = rdtsc();
  while (rdtsc() < (start + n)) {
    __asm__("PAUSE");
  }
}
static inline void spin_for(uint64_t nb_cycles) {
  rdtsc_wait(nb_cycles);
}


#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, const void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    next_token(buf, bufsize);

    if (sscanf(buf, "%"SCNu64, (uint64_t*)&shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNi8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNu8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { futhark_panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      futhark_panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { futhark_panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  futhark_panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    futhark_panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    futhark_panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    futhark_panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    futhark_panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, stdin);
    if (ret != 1) {
      futhark_panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    futhark_panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, stdin);
  if (num_elems_read != elem_count) {
    futhark_panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int end_of_input() {
  skipspaces();
  char token[2];
  next_token(token, sizeof(token));
  if (strcmp(token, "") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 0; i < rank; i++) {
        printf("[%"PRIi64"]", shape[i]);
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      const unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type,
                       const void *data,
                       const int64_t *shape,
                       const int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    int64_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, (size_t)elem_size, 1, stdin);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

#define __private
static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"help", no_argument, NULL, 7},
                                           {"device", required_argument, NULL,
                                            8}, {"default-group-size",
                                                 required_argument, NULL, 9},
                                           {"default-num-groups",
                                            required_argument, NULL, 10},
                                           {"default-tile-size",
                                            required_argument, NULL, 11},
                                           {"default-threshold",
                                            required_argument, NULL, 12},
                                           {"print-sizes", no_argument, NULL,
                                            13}, {"size", required_argument,
                                                  NULL, 14}, {"tuning",
                                                              required_argument,
                                                              NULL, 15},
                                           {"platform", required_argument, NULL,
                                            16}, {"dump-opencl",
                                                  required_argument, NULL, 17},
                                           {"load-opencl", required_argument,
                                            NULL, 18}, {"dump-opencl-binary",
                                                        required_argument, NULL,
                                                        19},
                                           {"load-opencl-binary",
                                            required_argument, NULL, 20},
                                           {"build-option", required_argument,
                                            NULL, 21}, {"profile", no_argument,
                                                        NULL, 22},
                                           {"list-devices", no_argument, NULL,
                                            23}, {0, 0, 0, 0}};
    static char *option_descriptions =
                "  -t/--write-runtime-to FILE Print the time taken to execute the program to the indicated file, an integral number of microseconds.\n  -r/--runs INT              Perform NUM runs of the program.\n  -D/--debugging             Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                   Print various low-overhead logging information to stderr while running.\n  -e/--entry-point NAME      The entry point to run. Defaults to main.\n  -b/--binary-output         Print the program result in the binary output format.\n  -h/--help                  Print help information and exit.\n  -d/--device NAME           Use the first OpenCL device whose name contains the given string.\n  --default-group-size INT   The default size of OpenCL workgroups that are launched.\n  --default-num-groups INT   The default number of OpenCL workgroups that are launched.\n  --default-tile-size INT    The default tile size used when performing two-dimensional tiling.\n  --default-threshold INT    The default parallelism threshold.\n  --print-sizes              Print all sizes that can be set with -size or --tuning.\n  --size ASSIGNMENT          Set a configurable run-time parameter to the given value.\n  --tuning FILE              Read size=value assignments from the given file.\n  -p/--platform NAME         Use the first OpenCL platform whose name contains the given string.\n  --dump-opencl FILE         Dump the embedded OpenCL program to the indicated file.\n  --load-opencl FILE         Instead of using the embedded OpenCL program, load it from the indicated file.\n  --dump-opencl-binary FILE  Dump the compiled version of the embedded OpenCL program to the indicated file.\n  --load-opencl-binary FILE  Load an OpenCL binary from the indicated file.\n  --build-option OPT         Add an additional build option to the string passed to clBuildProgram().\n  -P/--profile               Gather profiling data while executing and print out a summary at the end.\n  --list-devices             List all OpenCL devices and platforms available on the system.\n";
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bhd:p:P", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                futhark_panic(1, "Cannot open %s: %s\n", optarg,
                              strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                futhark_panic(1, "Need a positive number of runs, not %s\n",
                              optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'h') {
            printf("Usage: %s [OPTION]...\nOptions:\n\n%s\nFor more information, consult the Futhark User's Guide or the man pages.\n",
                   fut_progname, option_descriptions);
            exit(0);
        }
        if (ch == 8 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 9)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 11)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 12)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 13) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 14) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    futhark_panic(1, "Unknown size: %s\n", name);
            } else
                futhark_panic(1, "Invalid argument for size option: %s\n",
                              optarg);
        }
        if (ch == 15) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const
                                                               char *,
                                                               size_t)) futhark_context_config_set_size);
            
            if (ret != NULL)
                futhark_panic(1, "When loading tuning from '%s': %s\n", optarg,
                              ret);
        }
        if (ch == 16 || ch == 'p')
            futhark_context_config_set_platform(cfg, optarg);
        if (ch == 17) {
            futhark_context_config_dump_program_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 18)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 19) {
            futhark_context_config_dump_binary_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 20)
            futhark_context_config_load_binary_from(cfg, optarg);
        if (ch == 21)
            futhark_context_config_add_build_option(cfg, optarg);
        if (ch == 22 || ch == 'P')
            futhark_context_config_set_profiling(cfg, 1);
        if (ch == 23) {
            futhark_context_config_list_devices(cfg);
            entry_point = NULL;
        }
        if (ch == ':')
            futhark_panic(-1, "Missing argument for option %s\n", argv[optind -
                                                                       1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "  -t/--write-runtime-to FILE Print the time taken to execute the program to the indicated file, an integral number of microseconds.\n  -r/--runs INT              Perform NUM runs of the program.\n  -D/--debugging             Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log                   Print various low-overhead logging information to stderr while running.\n  -e/--entry-point NAME      The entry point to run. Defaults to main.\n  -b/--binary-output         Print the program result in the binary output format.\n  -h/--help                  Print help information and exit.\n  -d/--device NAME           Use the first OpenCL device whose name contains the given string.\n  --default-group-size INT   The default size of OpenCL workgroups that are launched.\n  --default-num-groups INT   The default number of OpenCL workgroups that are launched.\n  --default-tile-size INT    The default tile size used when performing two-dimensional tiling.\n  --default-threshold INT    The default parallelism threshold.\n  --print-sizes              Print all sizes that can be set with -size or --tuning.\n  --size ASSIGNMENT          Set a configurable run-time parameter to the given value.\n  --tuning FILE              Read size=value assignments from the given file.\n  -p/--platform NAME         Use the first OpenCL platform whose name contains the given string.\n  --dump-opencl FILE         Dump the embedded OpenCL program to the indicated file.\n  --load-opencl FILE         Instead of using the embedded OpenCL program, load it from the indicated file.\n  --dump-opencl-binary FILE  Dump the compiled version of the embedded OpenCL program to the indicated file.\n  --load-opencl-binary FILE  Load an OpenCL binary from the indicated file.\n  --build-option OPT         Add an additional build option to the string passed to clBuildProgram().\n  -P/--profile               Gather profiling data while executing and print out a summary at the end.\n  --list-devices             List all OpenCL devices and platforms available on the system.\n");
            futhark_panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_integrate_tke(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    struct futhark_f64_3d *read_value_53115;
    int64_t read_shape_53116[3];
    double *read_arr_53117 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53117, read_shape_53116, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53118;
    int64_t read_shape_53119[3];
    double *read_arr_53120 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53120, read_shape_53119, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53121;
    int64_t read_shape_53122[3];
    double *read_arr_53123 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53123, read_shape_53122, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53124;
    int64_t read_shape_53125[3];
    double *read_arr_53126 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53126, read_shape_53125, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53127;
    int64_t read_shape_53128[3];
    double *read_arr_53129 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53129, read_shape_53128, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53130;
    int64_t read_shape_53131[3];
    double *read_arr_53132 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53132, read_shape_53131, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53133;
    int64_t read_shape_53134[3];
    double *read_arr_53135 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53135, read_shape_53134, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53136;
    int64_t read_shape_53137[3];
    double *read_arr_53138 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53138, read_shape_53137, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53139;
    int64_t read_shape_53140[3];
    double *read_arr_53141 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53141, read_shape_53140, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 8,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53142;
    int64_t read_shape_53143[3];
    double *read_arr_53144 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53144, read_shape_53143, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 9,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53145;
    int64_t read_shape_53146[3];
    double *read_arr_53147 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53147, read_shape_53146, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      10, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53148;
    int64_t read_shape_53149[3];
    double *read_arr_53150 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53150, read_shape_53149, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      11, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53151;
    int64_t read_shape_53152[1];
    double *read_arr_53153 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53153, read_shape_53152, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      12, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53154;
    int64_t read_shape_53155[1];
    double *read_arr_53156 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53156, read_shape_53155, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      13, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53157;
    int64_t read_shape_53158[1];
    double *read_arr_53159 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53159, read_shape_53158, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      14, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53160;
    int64_t read_shape_53161[1];
    double *read_arr_53162 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53162, read_shape_53161, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      15, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53163;
    int64_t read_shape_53164[1];
    double *read_arr_53165 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53165, read_shape_53164, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      16, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53166;
    int64_t read_shape_53167[1];
    double *read_arr_53168 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53168, read_shape_53167, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      17, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53169;
    int64_t read_shape_53170[1];
    double *read_arr_53171 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53171, read_shape_53170, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      18, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_53172;
    int64_t read_shape_53173[1];
    double *read_arr_53174 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53174, read_shape_53173, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      19, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_i32_2d *read_value_53175;
    int64_t read_shape_53176[2];
    int32_t *read_arr_53177 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_53177, read_shape_53176, 2) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      20, "[][]", i32_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53178;
    int64_t read_shape_53179[3];
    double *read_arr_53180 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53180, read_shape_53179, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      21, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53181;
    int64_t read_shape_53182[3];
    double *read_arr_53183 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53183, read_shape_53182, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      22, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_53184;
    int64_t read_shape_53185[3];
    double *read_arr_53186 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53186, read_shape_53185, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      23, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_2d *read_value_53187;
    int64_t read_shape_53188[2];
    double *read_arr_53189 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_53189, read_shape_53188, 2) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      24, "[][]", f64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"integrate_tke\"");
    
    struct futhark_f64_3d *result_53190;
    struct futhark_f64_3d *result_53191;
    struct futhark_f64_3d *result_53192;
    struct futhark_f64_3d *result_53193;
    struct futhark_f64_3d *result_53194;
    struct futhark_f64_3d *result_53195;
    struct futhark_f64_2d *result_53196;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_53115 = futhark_new_f64_3d(ctx, read_arr_53117,
                                                      read_shape_53116[0],
                                                      read_shape_53116[1],
                                                      read_shape_53116[2])) !=
            0);
        assert((read_value_53118 = futhark_new_f64_3d(ctx, read_arr_53120,
                                                      read_shape_53119[0],
                                                      read_shape_53119[1],
                                                      read_shape_53119[2])) !=
            0);
        assert((read_value_53121 = futhark_new_f64_3d(ctx, read_arr_53123,
                                                      read_shape_53122[0],
                                                      read_shape_53122[1],
                                                      read_shape_53122[2])) !=
            0);
        assert((read_value_53124 = futhark_new_f64_3d(ctx, read_arr_53126,
                                                      read_shape_53125[0],
                                                      read_shape_53125[1],
                                                      read_shape_53125[2])) !=
            0);
        assert((read_value_53127 = futhark_new_f64_3d(ctx, read_arr_53129,
                                                      read_shape_53128[0],
                                                      read_shape_53128[1],
                                                      read_shape_53128[2])) !=
            0);
        assert((read_value_53130 = futhark_new_f64_3d(ctx, read_arr_53132,
                                                      read_shape_53131[0],
                                                      read_shape_53131[1],
                                                      read_shape_53131[2])) !=
            0);
        assert((read_value_53133 = futhark_new_f64_3d(ctx, read_arr_53135,
                                                      read_shape_53134[0],
                                                      read_shape_53134[1],
                                                      read_shape_53134[2])) !=
            0);
        assert((read_value_53136 = futhark_new_f64_3d(ctx, read_arr_53138,
                                                      read_shape_53137[0],
                                                      read_shape_53137[1],
                                                      read_shape_53137[2])) !=
            0);
        assert((read_value_53139 = futhark_new_f64_3d(ctx, read_arr_53141,
                                                      read_shape_53140[0],
                                                      read_shape_53140[1],
                                                      read_shape_53140[2])) !=
            0);
        assert((read_value_53142 = futhark_new_f64_3d(ctx, read_arr_53144,
                                                      read_shape_53143[0],
                                                      read_shape_53143[1],
                                                      read_shape_53143[2])) !=
            0);
        assert((read_value_53145 = futhark_new_f64_3d(ctx, read_arr_53147,
                                                      read_shape_53146[0],
                                                      read_shape_53146[1],
                                                      read_shape_53146[2])) !=
            0);
        assert((read_value_53148 = futhark_new_f64_3d(ctx, read_arr_53150,
                                                      read_shape_53149[0],
                                                      read_shape_53149[1],
                                                      read_shape_53149[2])) !=
            0);
        assert((read_value_53151 = futhark_new_f64_1d(ctx, read_arr_53153,
                                                      read_shape_53152[0])) !=
            0);
        assert((read_value_53154 = futhark_new_f64_1d(ctx, read_arr_53156,
                                                      read_shape_53155[0])) !=
            0);
        assert((read_value_53157 = futhark_new_f64_1d(ctx, read_arr_53159,
                                                      read_shape_53158[0])) !=
            0);
        assert((read_value_53160 = futhark_new_f64_1d(ctx, read_arr_53162,
                                                      read_shape_53161[0])) !=
            0);
        assert((read_value_53163 = futhark_new_f64_1d(ctx, read_arr_53165,
                                                      read_shape_53164[0])) !=
            0);
        assert((read_value_53166 = futhark_new_f64_1d(ctx, read_arr_53168,
                                                      read_shape_53167[0])) !=
            0);
        assert((read_value_53169 = futhark_new_f64_1d(ctx, read_arr_53171,
                                                      read_shape_53170[0])) !=
            0);
        assert((read_value_53172 = futhark_new_f64_1d(ctx, read_arr_53174,
                                                      read_shape_53173[0])) !=
            0);
        assert((read_value_53175 = futhark_new_i32_2d(ctx, read_arr_53177,
                                                      read_shape_53176[0],
                                                      read_shape_53176[1])) !=
            0);
        assert((read_value_53178 = futhark_new_f64_3d(ctx, read_arr_53180,
                                                      read_shape_53179[0],
                                                      read_shape_53179[1],
                                                      read_shape_53179[2])) !=
            0);
        assert((read_value_53181 = futhark_new_f64_3d(ctx, read_arr_53183,
                                                      read_shape_53182[0],
                                                      read_shape_53182[1],
                                                      read_shape_53182[2])) !=
            0);
        assert((read_value_53184 = futhark_new_f64_3d(ctx, read_arr_53186,
                                                      read_shape_53185[0],
                                                      read_shape_53185[1],
                                                      read_shape_53185[2])) !=
            0);
        assert((read_value_53187 = futhark_new_f64_2d(ctx, read_arr_53189,
                                                      read_shape_53188[0],
                                                      read_shape_53188[1])) !=
            0);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_integrate_tke(ctx, &result_53190, &result_53191,
                                        &result_53192, &result_53193,
                                        &result_53194, &result_53195,
                                        &result_53196, read_value_53115,
                                        read_value_53118, read_value_53121,
                                        read_value_53124, read_value_53127,
                                        read_value_53130, read_value_53133,
                                        read_value_53136, read_value_53139,
                                        read_value_53142, read_value_53145,
                                        read_value_53148, read_value_53151,
                                        read_value_53154, read_value_53157,
                                        read_value_53160, read_value_53163,
                                        read_value_53166, read_value_53169,
                                        read_value_53172, read_value_53175,
                                        read_value_53178, read_value_53181,
                                        read_value_53184, read_value_53187);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        assert(futhark_free_f64_3d(ctx, read_value_53115) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53118) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53121) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53124) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53127) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53130) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53133) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53136) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53139) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53142) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53145) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53148) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53151) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53154) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53157) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53160) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53163) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53166) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53169) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53172) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_53175) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53178) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53181) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53184) == 0);
        assert(futhark_free_f64_2d(ctx, read_value_53187) == 0);
        assert(futhark_free_f64_3d(ctx, result_53190) == 0);
        assert(futhark_free_f64_3d(ctx, result_53191) == 0);
        assert(futhark_free_f64_3d(ctx, result_53192) == 0);
        assert(futhark_free_f64_3d(ctx, result_53193) == 0);
        assert(futhark_free_f64_3d(ctx, result_53194) == 0);
        assert(futhark_free_f64_3d(ctx, result_53195) == 0);
        assert(futhark_free_f64_2d(ctx, result_53196) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_53115 = futhark_new_f64_3d(ctx, read_arr_53117,
                                                      read_shape_53116[0],
                                                      read_shape_53116[1],
                                                      read_shape_53116[2])) !=
            0);
        assert((read_value_53118 = futhark_new_f64_3d(ctx, read_arr_53120,
                                                      read_shape_53119[0],
                                                      read_shape_53119[1],
                                                      read_shape_53119[2])) !=
            0);
        assert((read_value_53121 = futhark_new_f64_3d(ctx, read_arr_53123,
                                                      read_shape_53122[0],
                                                      read_shape_53122[1],
                                                      read_shape_53122[2])) !=
            0);
        assert((read_value_53124 = futhark_new_f64_3d(ctx, read_arr_53126,
                                                      read_shape_53125[0],
                                                      read_shape_53125[1],
                                                      read_shape_53125[2])) !=
            0);
        assert((read_value_53127 = futhark_new_f64_3d(ctx, read_arr_53129,
                                                      read_shape_53128[0],
                                                      read_shape_53128[1],
                                                      read_shape_53128[2])) !=
            0);
        assert((read_value_53130 = futhark_new_f64_3d(ctx, read_arr_53132,
                                                      read_shape_53131[0],
                                                      read_shape_53131[1],
                                                      read_shape_53131[2])) !=
            0);
        assert((read_value_53133 = futhark_new_f64_3d(ctx, read_arr_53135,
                                                      read_shape_53134[0],
                                                      read_shape_53134[1],
                                                      read_shape_53134[2])) !=
            0);
        assert((read_value_53136 = futhark_new_f64_3d(ctx, read_arr_53138,
                                                      read_shape_53137[0],
                                                      read_shape_53137[1],
                                                      read_shape_53137[2])) !=
            0);
        assert((read_value_53139 = futhark_new_f64_3d(ctx, read_arr_53141,
                                                      read_shape_53140[0],
                                                      read_shape_53140[1],
                                                      read_shape_53140[2])) !=
            0);
        assert((read_value_53142 = futhark_new_f64_3d(ctx, read_arr_53144,
                                                      read_shape_53143[0],
                                                      read_shape_53143[1],
                                                      read_shape_53143[2])) !=
            0);
        assert((read_value_53145 = futhark_new_f64_3d(ctx, read_arr_53147,
                                                      read_shape_53146[0],
                                                      read_shape_53146[1],
                                                      read_shape_53146[2])) !=
            0);
        assert((read_value_53148 = futhark_new_f64_3d(ctx, read_arr_53150,
                                                      read_shape_53149[0],
                                                      read_shape_53149[1],
                                                      read_shape_53149[2])) !=
            0);
        assert((read_value_53151 = futhark_new_f64_1d(ctx, read_arr_53153,
                                                      read_shape_53152[0])) !=
            0);
        assert((read_value_53154 = futhark_new_f64_1d(ctx, read_arr_53156,
                                                      read_shape_53155[0])) !=
            0);
        assert((read_value_53157 = futhark_new_f64_1d(ctx, read_arr_53159,
                                                      read_shape_53158[0])) !=
            0);
        assert((read_value_53160 = futhark_new_f64_1d(ctx, read_arr_53162,
                                                      read_shape_53161[0])) !=
            0);
        assert((read_value_53163 = futhark_new_f64_1d(ctx, read_arr_53165,
                                                      read_shape_53164[0])) !=
            0);
        assert((read_value_53166 = futhark_new_f64_1d(ctx, read_arr_53168,
                                                      read_shape_53167[0])) !=
            0);
        assert((read_value_53169 = futhark_new_f64_1d(ctx, read_arr_53171,
                                                      read_shape_53170[0])) !=
            0);
        assert((read_value_53172 = futhark_new_f64_1d(ctx, read_arr_53174,
                                                      read_shape_53173[0])) !=
            0);
        assert((read_value_53175 = futhark_new_i32_2d(ctx, read_arr_53177,
                                                      read_shape_53176[0],
                                                      read_shape_53176[1])) !=
            0);
        assert((read_value_53178 = futhark_new_f64_3d(ctx, read_arr_53180,
                                                      read_shape_53179[0],
                                                      read_shape_53179[1],
                                                      read_shape_53179[2])) !=
            0);
        assert((read_value_53181 = futhark_new_f64_3d(ctx, read_arr_53183,
                                                      read_shape_53182[0],
                                                      read_shape_53182[1],
                                                      read_shape_53182[2])) !=
            0);
        assert((read_value_53184 = futhark_new_f64_3d(ctx, read_arr_53186,
                                                      read_shape_53185[0],
                                                      read_shape_53185[1],
                                                      read_shape_53185[2])) !=
            0);
        assert((read_value_53187 = futhark_new_f64_2d(ctx, read_arr_53189,
                                                      read_shape_53188[0],
                                                      read_shape_53188[1])) !=
            0);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_integrate_tke(ctx, &result_53190, &result_53191,
                                        &result_53192, &result_53193,
                                        &result_53194, &result_53195,
                                        &result_53196, read_value_53115,
                                        read_value_53118, read_value_53121,
                                        read_value_53124, read_value_53127,
                                        read_value_53130, read_value_53133,
                                        read_value_53136, read_value_53139,
                                        read_value_53142, read_value_53145,
                                        read_value_53148, read_value_53151,
                                        read_value_53154, read_value_53157,
                                        read_value_53160, read_value_53163,
                                        read_value_53166, read_value_53169,
                                        read_value_53172, read_value_53175,
                                        read_value_53178, read_value_53181,
                                        read_value_53184, read_value_53187);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL) {
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
            fflush(runtime_file);
        }
        assert(futhark_free_f64_3d(ctx, read_value_53115) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53118) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53121) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53124) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53127) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53130) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53133) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53136) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53139) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53142) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53145) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53148) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53151) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53154) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53157) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53160) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53163) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53166) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53169) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_53172) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_53175) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53178) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53181) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_53184) == 0);
        assert(futhark_free_f64_2d(ctx, read_value_53187) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f64_3d(ctx, result_53190) == 0);
            assert(futhark_free_f64_3d(ctx, result_53191) == 0);
            assert(futhark_free_f64_3d(ctx, result_53192) == 0);
            assert(futhark_free_f64_3d(ctx, result_53193) == 0);
            assert(futhark_free_f64_3d(ctx, result_53194) == 0);
            assert(futhark_free_f64_3d(ctx, result_53195) == 0);
            assert(futhark_free_f64_2d(ctx, result_53196) == 0);
        }
    }
    free(read_arr_53117);
    free(read_arr_53120);
    free(read_arr_53123);
    free(read_arr_53126);
    free(read_arr_53129);
    free(read_arr_53132);
    free(read_arr_53135);
    free(read_arr_53138);
    free(read_arr_53141);
    free(read_arr_53144);
    free(read_arr_53147);
    free(read_arr_53150);
    free(read_arr_53153);
    free(read_arr_53156);
    free(read_arr_53159);
    free(read_arr_53162);
    free(read_arr_53165);
    free(read_arr_53168);
    free(read_arr_53171);
    free(read_arr_53174);
    free(read_arr_53177);
    free(read_arr_53180);
    free(read_arr_53183);
    free(read_arr_53186);
    free(read_arr_53189);
    if (binary_output)
        set_binary_mode(stdout);
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_53190)[0] *
                             futhark_shape_f64_3d(ctx, result_53190)[1] *
                             futhark_shape_f64_3d(ctx, result_53190)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_53190, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_53190), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_53191)[0] *
                             futhark_shape_f64_3d(ctx, result_53191)[1] *
                             futhark_shape_f64_3d(ctx, result_53191)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_53191, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_53191), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_53192)[0] *
                             futhark_shape_f64_3d(ctx, result_53192)[1] *
                             futhark_shape_f64_3d(ctx, result_53192)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_53192, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_53192), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_53193)[0] *
                             futhark_shape_f64_3d(ctx, result_53193)[1] *
                             futhark_shape_f64_3d(ctx, result_53193)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_53193, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_53193), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_53194)[0] *
                             futhark_shape_f64_3d(ctx, result_53194)[1] *
                             futhark_shape_f64_3d(ctx, result_53194)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_53194, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_53194), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_53195)[0] *
                             futhark_shape_f64_3d(ctx, result_53195)[1] *
                             futhark_shape_f64_3d(ctx, result_53195)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_53195, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_53195), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_2d(ctx,
                                                                  result_53196)[0] *
                             futhark_shape_f64_2d(ctx, result_53196)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_2d(ctx, result_53196, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_2d(ctx, result_53196), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f64_3d(ctx, result_53190) == 0);
    assert(futhark_free_f64_3d(ctx, result_53191) == 0);
    assert(futhark_free_f64_3d(ctx, result_53192) == 0);
    assert(futhark_free_f64_3d(ctx, result_53193) == 0);
    assert(futhark_free_f64_3d(ctx, result_53194) == 0);
    assert(futhark_free_f64_3d(ctx, result_53195) == 0);
    assert(futhark_free_f64_2d(ctx, result_53196) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="integrate_tke", .fun =
                                                futrts_cli_entry_integrate_tke}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        futhark_panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    
    char *error = futhark_context_get_error(ctx);
    
    if (error != NULL)
        futhark_panic(1, "%s", error);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        
        char *report = futhark_context_report(ctx);
        
        fputs(report, stderr);
        free(report);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Start of lock.h.

// A very simple cross-platform implementation of locks.  Uses
// pthreads on Unix and some Windows thing there.  Futhark's
// host-level code is not multithreaded, but user code may be, so we
// need some mechanism for ensuring atomic access to API functions.
// This is that mechanism.  It is not exposed to user code at all, so
// we do not have to worry about name collisions.

#ifdef _WIN32

typedef HANDLE lock_t;

static void create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  // Default security attributes.
                      FALSE, // Initially unlocked.
                      NULL); // Unnamed.
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
// Assuming POSIX

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  // Nothing to do for pthreads.
  (void)lock;
}

#endif

// End of lock.h.

static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t udiv_up8(uint8_t x, uint8_t y)
{
    return (x + y - 1) / y;
}
static inline uint16_t udiv_up16(uint16_t x, uint16_t y)
{
    return (x + y - 1) / y;
}
static inline uint32_t udiv_up32(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}
static inline uint64_t udiv_up64(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline uint8_t udiv_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint16_t udiv_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint32_t udiv_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint64_t udiv_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint8_t umod_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint16_t umod_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint32_t umod_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint64_t umod_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t sdiv_up8(int8_t x, int8_t y)
{
    return sdiv8(x + y - 1, y);
}
static inline int16_t sdiv_up16(int16_t x, int16_t y)
{
    return sdiv16(x + y - 1, y);
}
static inline int32_t sdiv_up32(int32_t x, int32_t y)
{
    return sdiv32(x + y - 1, y);
}
static inline int64_t sdiv_up64(int64_t x, int64_t y)
{
    return sdiv64(x + y - 1, y);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t sdiv_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : sdiv8(x, y);
}
static inline int16_t sdiv_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : sdiv16(x, y);
}
static inline int32_t sdiv_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : sdiv32(x, y);
}
static inline int64_t sdiv_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : sdiv64(x, y);
}
static inline int8_t sdiv_up_safe8(int8_t x, int8_t y)
{
    return sdiv_safe8(x + y - 1, y);
}
static inline int16_t sdiv_up_safe16(int16_t x, int16_t y)
{
    return sdiv_safe16(x + y - 1, y);
}
static inline int32_t sdiv_up_safe32(int32_t x, int32_t y)
{
    return sdiv_safe32(x + y - 1, y);
}
static inline int64_t sdiv_up_safe64(int64_t x, int64_t y)
{
    return sdiv_safe64(x + y - 1, y);
}
static inline int8_t smod_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : smod8(x, y);
}
static inline int16_t smod_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : smod16(x, y);
}
static inline int32_t smod_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : smod32(x, y);
}
static inline int64_t smod_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : smod64(x, y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t squot_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int16_t squot_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int32_t squot_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int64_t squot_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int8_t srem_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int16_t srem_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int32_t srem_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int64_t srem_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((int8_t) (uint8_t) x)
#define zext_i8_i16(x) ((int16_t) (uint8_t) x)
#define zext_i8_i32(x) ((int32_t) (uint8_t) x)
#define zext_i8_i64(x) ((int64_t) (uint8_t) x)
#define zext_i16_i8(x) ((int8_t) (uint16_t) x)
#define zext_i16_i16(x) ((int16_t) (uint16_t) x)
#define zext_i16_i32(x) ((int32_t) (uint16_t) x)
#define zext_i16_i64(x) ((int64_t) (uint16_t) x)
#define zext_i32_i8(x) ((int8_t) (uint32_t) x)
#define zext_i32_i16(x) ((int16_t) (uint32_t) x)
#define zext_i32_i32(x) ((int32_t) (uint32_t) x)
#define zext_i32_i64(x) ((int64_t) (uint32_t) x)
#define zext_i64_i8(x) ((int8_t) (uint64_t) x)
#define zext_i64_i16(x) ((int16_t) (uint64_t) x)
#define zext_i64_i32(x) ((int32_t) (uint64_t) x)
#define zext_i64_i64(x) ((int64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_ctzz8(int8_t x)
{
    int i = 0;
    
    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int i = 0;
    
    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int i = 0;
    
    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int i = 0;
    
    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_ctzz8(int8_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 8 : y - 1;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 16 : y - 1;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 32 : y - 1;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int y = __ffsll(x);
    
    return y == 0 ? 64 : y - 1;
}
#else
static int32_t futrts_ctzz8(int8_t x)
{
    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz16(int16_t x)
{
    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz32(int32_t x)
{
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int32_t futrts_ctzz64(int64_t x)
{
    return x == 0 ? 64 : __builtin_ctzl(x);
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
static int init_constants(struct futhark_context *);
static int free_constants(struct futhark_context *);
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
typedef cl_mem fl_mem_t;
// Start of free_list.h.

// An entry in the free list.  May be invalid, to avoid having to
// deallocate entries as soon as they are removed.  There is also a
// tag, to help with memory reuse.
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

// Remove invalid entries from the free list.
static void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }

  // Now p is the number of used elements.  We don't want it to go
  // less than the default capacity (although in practice it's OK as
  // long as it doesn't become 1).
  if (p < 30) {
    p = 30;
  }
  l->entries = realloc(l->entries, p * sizeof(struct free_list_entry));
  l->capacity = p;
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

// Find and remove a memory block of the indicated tag, or if that
// does not exist, another memory block with exactly the desired size.
// Returns 0 on success.
static int free_list_find(struct free_list *l, size_t size,
                          size_t *size_out, fl_mem_t *mem_out) {
  int size_match = -1;
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid &&
        size <= l->entries[i].size &&
        (size_match < 0 || l->entries[i].size < l->entries[size_match].size)) {
      // If this entry is valid, has sufficient size, and is smaller than the
      // best entry found so far, use this entry.
      size_match = i;
    }
  }

  if (size_match >= 0) {
    l->entries[size_match].valid = 0;
    *size_out = l->entries[size_match].size;
    *mem_out = l->entries[size_match].mem;
    l->used--;
    return 0;
  } else {
    return 1;
  }
}

// Remove the first block in the free list.  Returns 0 if a block was
// removed, and nonzero if the free list was already empty.
static int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

// End of free_list.h.

// Start of opencl.h.

#define OPENCL_SUCCEED_FATAL(e) opencl_succeed_fatal(e, #e, __FILE__, __LINE__)
#define OPENCL_SUCCEED_NONFATAL(e) opencl_succeed_nonfatal(e, #e, __FILE__, __LINE__)
// Take care not to override an existing error.
#define OPENCL_SUCCEED_OR_RETURN(e) {             \
    char *serror = OPENCL_SUCCEED_NONFATAL(e);    \
    if (serror) {                                 \
      if (!ctx->error) {                          \
        ctx->error = serror;                      \
        return bad;                               \
      } else {                                    \
        free(serror);                             \
      }                                           \
    }                                             \
  }

// OPENCL_SUCCEED_OR_RETURN returns the value of the variable 'bad' in
// scope.  By default, it will be this one.  Create a local variable
// of some other type if needed.  This is a bit of a hack, but it
// saves effort in the code generator.
static const int bad = 1;

struct opencl_config {
  int debugging;
  int profiling;
  int logging;
  int preferred_device_num;
  const char *preferred_platform;
  const char *preferred_device;
  int ignore_blacklist;

  const char* dump_program_to;
  const char* load_program_from;
  const char* dump_binary_to;
  const char* load_binary_from;

  size_t default_group_size;
  size_t default_num_groups;
  size_t default_tile_size;
  size_t default_threshold;

  int default_group_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  size_t *size_values;
  const char **size_classes;
};

static void opencl_config_init(struct opencl_config *cfg,
                               int num_sizes,
                               const char *size_names[],
                               const char *size_vars[],
                               size_t *size_values,
                               const char *size_classes[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->profiling = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_platform = "";
  cfg->preferred_device = "";
  cfg->ignore_blacklist = 0;
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;
  cfg->dump_binary_to = NULL;
  cfg->load_binary_from = NULL;

  // The following are dummy sizes that mean the concrete defaults
  // will be set during initialisation via hardware-inspection-based
  // heuristics.
  cfg->default_group_size = 0;
  cfg->default_num_groups = 0;
  cfg->default_tile_size = 0;
  cfg->default_threshold = 0;

  cfg->default_group_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

// A record of something that happened.
struct profiling_record {
  cl_event *event;
  int *runs;
  int64_t *runtime;
};

struct opencl_context {
  cl_device_id device;
  cl_context ctx;
  cl_command_queue queue;

  struct opencl_config cfg;

  struct free_list free_list;

  size_t max_group_size;
  size_t max_num_groups;
  size_t max_tile_size;
  size_t max_threshold;
  size_t max_local_memory;

  size_t lockstep_width;

  struct profiling_record *profiling_records;
  int profiling_records_capacity;
  int profiling_records_used;
};

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

// This function must be defined by the user.  It is invoked by
// setup_opencl() after the platform and device has been found, but
// before the program is loaded.  Its intended use is to tune
// constants based on the selected platform and device.
static void post_opencl_setup(struct opencl_context*, struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = (char*) malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

static const char* opencl_error_string(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed_fatal(unsigned int ret,
                                 const char *call,
                                 const char *file,
                                 int line) {
  if (ret != CL_SUCCESS) {
    futhark_panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_succeed_nonfatal(unsigned int ret,
                                     const char *call,
                                     const char *file,
                                     int line) {
  if (ret != CL_SUCCESS) {
    return msgprintf("%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
                     file, line, call, ret, opencl_error_string(ret));
  } else {
    return NULL;
  }
}

static void set_preferred_platform(struct opencl_config *cfg, const char *s) {
  cfg->preferred_platform = s;
  cfg->ignore_blacklist = 1;
}

static void set_preferred_device(struct opencl_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
  cfg->ignore_blacklist = 1;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED_FATAL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED_FATAL(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

// Returns 0 on success.
static int list_devices(void) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  const char *cur_platform = "";
  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strcmp(cur_platform, device.platform_name) != 0) {
      printf("Platform: %s\n", device.platform_name);
      cur_platform = device.platform_name;
    }
    printf("[%d]: %s\n", (int)i, device.device_name);
  }

  // Free all the platform and device names.
  for (size_t j = 0; j < num_devices; j++) {
    free(devices[j].platform_name);
    free(devices[j].device_name);
  }
  free(devices);

  return 0;
}

// Returns 0 on success.
static int select_device_interactively(struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;
  int ret = 1;

  opencl_all_device_options(&devices, &num_devices);

  printf("Choose OpenCL device:\n");
  const char *cur_platform = "";
  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strcmp(cur_platform, device.platform_name) != 0) {
      printf("Platform: %s\n", device.platform_name);
      cur_platform = device.platform_name;
    }
    printf("[%d] %s\n", (int)i, device.device_name);
  }

  int selection;
  printf("Choice: ");
  if (scanf("%d", &selection) == 1) {
    ret = 0;
    cfg->preferred_platform = "";
    cfg->preferred_device = "";
    cfg->preferred_device_num = selection;
    cfg->ignore_blacklist = 1;
  }

  // Free all the platform and device names.
  for (size_t j = 0; j < num_devices; j++) {
    free(devices[j].platform_name);
    free(devices[j].device_name);
  }
  free(devices);

  return ret;
}

static int is_blacklisted(const char *platform_name, const char *device_name,
                          const struct opencl_config *cfg) {
  if (strcmp(cfg->preferred_platform, "") != 0 ||
      strcmp(cfg->preferred_device, "") != 0) {
    return 0;
  } else if (strstr(platform_name, "Apple") != NULL &&
             strstr(device_name, "Intel(R) Core(TM)") != NULL) {
    return 1;
  } else {
    return 0;
  }
}

static struct opencl_device_option get_preferred_device(const struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strstr(device.platform_name, cfg->preferred_platform) != NULL &&
        strstr(device.device_name, cfg->preferred_device) != NULL &&
        (cfg->ignore_blacklist ||
         !is_blacklisted(device.platform_name, device.device_name, cfg)) &&
        num_device_matches++ == cfg->preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  futhark_panic(1, "Could not find acceptable OpenCL device.\n");
  exit(1); // Never reached
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int clBuildProgram_error = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (clBuildProgram_error != CL_SUCCESS &&
      clBuildProgram_error != CL_BUILD_PROGRAM_FAILURE) {
    OPENCL_SUCCEED_FATAL(clBuildProgram_error);
  }

  cl_build_status build_status;
  OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program,
                                             device,
                                             CL_PROGRAM_BUILD_STATUS,
                                             sizeof(cl_build_status),
                                             &build_status,
                                             NULL));

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));

    build_log = (char*) malloc(ret_val_size+1);
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

// Fields in a bitmask indicating which types we must be sure are
// available.
enum opencl_required_type { OPENCL_F64 = 1 };

// We take as input several strings representing the program, because
// C does not guarantee that the compiler supports particularly large
// literals.  Notably, Visual C has a limit of 2048 characters.  The
// array must be NULL-terminated.
static cl_program setup_opencl_with_command_queue(struct opencl_context *ctx,
                                                  cl_command_queue queue,
                                                  const char *srcs[],
                                                  int required_types,
                                                  const char *extra_build_opts[]) {
  int error;

  free_list_init(&ctx->free_list);
  ctx->queue = queue;

  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx->ctx, NULL));

  // Fill out the device info.  This is redundant work if we are
  // called from setup_opencl() (which is the common case), but I
  // doubt it matters much.
  struct opencl_device_option device_option;
  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_DEVICE,
                                       sizeof(cl_device_id),
                                       &device_option.device,
                                       NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id),
                                 &device_option.platform,
                                 NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type),
                                 &device_option.device_type,
                                 NULL));
  device_option.platform_name = opencl_platform_info(device_option.platform, CL_PLATFORM_NAME);
  device_option.device_name = opencl_device_info(device_option.device, CL_DEVICE_NAME);

  ctx->device = device_option.device;

  if (required_types & OPENCL_F64) {
    cl_uint supported;
    OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                   sizeof(cl_uint), &supported, NULL));
    if (!supported) {
      futhark_panic(1, "Program uses double-precision floats, but this is not supported on the chosen device: %s\n",
            device_option.device_name);
    }
  }

  size_t max_group_size;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  cl_ulong max_local_memory;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_LOCAL_MEM_SIZE,
                                       sizeof(size_t), &max_local_memory, NULL));

  // Futhark reserves 4 bytes for bookkeeping information.
  max_local_memory -= 4;

  // The OpenCL implementation may reserve some local memory bytes for
  // various purposes.  In principle, we should use
  // clGetKernelWorkGroupInfo() to figure out for each kernel how much
  // is actually available, but our current code generator design
  // makes this infeasible.  Instead, we have this nasty hack where we
  // arbitrarily subtract some bytes, based on empirical measurements
  // (but which might be arbitrarily wrong).  Fortunately, we rarely
  // try to really push the local memory usage.
  if (strstr(device_option.platform_name, "NVIDIA CUDA") != NULL) {
    max_local_memory -= 12;
  } else if (strstr(device_option.platform_name, "AMD") != NULL) {
    max_local_memory -= 16;
  }

  // Make sure this function is defined.
  post_opencl_setup(ctx, &device_option);

  if (max_group_size < ctx->cfg.default_group_size) {
    if (ctx->cfg.default_group_size_changed) {
      fprintf(stderr, "Note: Device limits default group size to %zu (down from %zu).\n",
              max_group_size, ctx->cfg.default_group_size);
    }
    ctx->cfg.default_group_size = max_group_size;
  }

  if (max_tile_size < ctx->cfg.default_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr, "Note: Device limits default tile size to %zu (down from %zu).\n",
              max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = max_tile_size;
  }

  ctx->max_group_size = max_group_size;
  ctx->max_tile_size = max_tile_size; // No limit.
  ctx->max_threshold = ctx->max_num_groups = 0; // No limit.
  ctx->max_local_memory = max_local_memory;

  // Now we go through all the sizes, clamp them to the valid range,
  // or set them to the default.
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    size_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    size_t max_value = 0, default_value = 0;
    if (strstr(size_class, "group_size") == size_class) {
      max_value = max_group_size;
      default_value = ctx->cfg.default_group_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = max_group_size; // Futhark assumes this constraint.
      default_value = ctx->cfg.default_num_groups;
      // XXX: as a quick and dirty hack, use twice as many threads for
      // histograms by default.  We really should just be smarter
      // about sizes somehow.
      if (strstr(size_name, ".seghist_") != NULL) {
        default_value *= 2;
      }
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = sqrt(max_group_size);
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      // Threshold can be as large as it takes.
      default_value = ctx->cfg.default_threshold;
    } else {
      // Bespoke sizes have no limit or default.
    }
    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %d (down from %d)\n",
              size_name, (int)max_value, (int)*size_value);
      *size_value = max_value;
    }
  }

  if (ctx->lockstep_width == 0) {
    ctx->lockstep_width = 1;
  }

  if (ctx->cfg.logging) {
    fprintf(stderr, "Lockstep width: %d\n", (int)ctx->lockstep_width);
    fprintf(stderr, "Default group size: %d\n", (int)ctx->cfg.default_group_size);
    fprintf(stderr, "Default number of groups: %d\n", (int)ctx->cfg.default_num_groups);
  }

  char *fut_opencl_src = NULL;
  cl_program prog;
  error = CL_SUCCESS;

  if (ctx->cfg.load_binary_from == NULL) {
    size_t src_size = 0;

    // Maybe we have to read OpenCL source from somewhere else (used for debugging).
    if (ctx->cfg.load_program_from != NULL) {
      fut_opencl_src = slurp_file(ctx->cfg.load_program_from, NULL);
      assert(fut_opencl_src != NULL);
    } else {
      // Construct the OpenCL source concatenating all the fragments.
      for (const char **src = srcs; src && *src; src++) {
        src_size += strlen(*src);
      }

      fut_opencl_src = (char*) malloc(src_size + 1);

      size_t n, i;
      for (i = 0, n = 0; srcs && srcs[i]; i++) {
        strncpy(fut_opencl_src+n, srcs[i], src_size-n);
        n += strlen(srcs[i]);
      }
      fut_opencl_src[src_size] = 0;
    }

    if (ctx->cfg.dump_program_to != NULL) {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "Dumping OpenCL source to %s...\n", ctx->cfg.dump_program_to);
      }

      dump_file(ctx->cfg.dump_program_to, fut_opencl_src, strlen(fut_opencl_src));
    }

    if (ctx->cfg.debugging) {
      fprintf(stderr, "Creating OpenCL program...\n");
    }

    const char* src_ptr[] = {fut_opencl_src};
    prog = clCreateProgramWithSource(ctx->ctx, 1, src_ptr, &src_size, &error);
    OPENCL_SUCCEED_FATAL(error);
  } else {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Loading OpenCL binary from %s...\n", ctx->cfg.load_binary_from);
    }
    size_t binary_size;
    unsigned char *fut_opencl_bin =
      (unsigned char*) slurp_file(ctx->cfg.load_binary_from, &binary_size);
    assert(fut_opencl_bin != NULL);
    const unsigned char *binaries[1] = { fut_opencl_bin };
    cl_int status = 0;

    prog = clCreateProgramWithBinary(ctx->ctx, 1, &device_option.device,
                                     &binary_size, binaries,
                                     &status, &error);

    OPENCL_SUCCEED_FATAL(status);
    OPENCL_SUCCEED_FATAL(error);
  }

  int compile_opts_size = 1024;

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    compile_opts_size += strlen(ctx->cfg.size_names[i]) + 20;
  }

  for (int i = 0; extra_build_opts[i] != NULL; i++) {
    compile_opts_size += strlen(extra_build_opts[i] + 1);
  }

  char *compile_opts = (char*) malloc(compile_opts_size);

  int w = snprintf(compile_opts, compile_opts_size,
                   "-DLOCKSTEP_WIDTH=%d ",
                   (int)ctx->lockstep_width);

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    w += snprintf(compile_opts+w, compile_opts_size-w,
                  "-D%s=%d ",
                  ctx->cfg.size_vars[i],
                  (int)ctx->cfg.size_values[i]);
  }

  for (int i = 0; extra_build_opts[i] != NULL; i++) {
    w += snprintf(compile_opts+w, compile_opts_size-w,
                  "%s ", extra_build_opts[i]);
  }

  if (ctx->cfg.debugging) {
    fprintf(stderr, "OpenCL compiler options: %s\n", compile_opts);
    fprintf(stderr, "Building OpenCL program...\n");
  }
  OPENCL_SUCCEED_FATAL(build_opencl_program(prog, device_option.device, compile_opts));

  free(compile_opts);
  free(fut_opencl_src);

  if (ctx->cfg.dump_binary_to != NULL) {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Dumping OpenCL binary to %s...\n", ctx->cfg.dump_binary_to);
    }

    size_t binary_size;
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                                          sizeof(size_t), &binary_size, NULL));
    unsigned char *binary = (unsigned char*) malloc(binary_size);
    unsigned char *binaries[1] = { binary };
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                                          sizeof(unsigned char*), binaries, NULL));

    dump_file(ctx->cfg.dump_binary_to, binary, binary_size);
  }

  return prog;
}

static cl_program setup_opencl(struct opencl_context *ctx,
                               const char *srcs[],
                               int required_types,
                               const char *extra_build_opts[]) {

  ctx->lockstep_width = 0; // Real value set later.

  struct opencl_device_option device_option = get_preferred_device(&ctx->cfg);

  if (ctx->cfg.logging) {
    describe_device_option(device_option);
  }

  // Note that NVIDIA's OpenCL requires the platform property
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)device_option.platform,
    0
  };

  cl_int clCreateContext_error;
  ctx->ctx = clCreateContext(properties, 1, &device_option.device, NULL, NULL, &clCreateContext_error);
  OPENCL_SUCCEED_FATAL(clCreateContext_error);

  cl_int clCreateCommandQueue_error;
  cl_command_queue queue =
    clCreateCommandQueue(ctx->ctx,
                         device_option.device,
                         ctx->cfg.profiling ? CL_QUEUE_PROFILING_ENABLE : 0,
                         &clCreateCommandQueue_error);
  OPENCL_SUCCEED_FATAL(clCreateCommandQueue_error);

  return setup_opencl_with_command_queue(ctx, queue, srcs, required_types, extra_build_opts);
}

// Count up the runtime all the profiling_records that occured during execution.
// Also clears the buffer of profiling_records.
static cl_int opencl_tally_profiling_records(struct opencl_context *ctx) {
  cl_int err;
  for (int i = 0; i < ctx->profiling_records_used; i++) {
    struct profiling_record record = ctx->profiling_records[i];

    cl_ulong start_t, end_t;

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_START,
                                       sizeof(start_t),
                                       &start_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_END,
                                       sizeof(end_t),
                                       &end_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    // OpenCL provides nanosecond resolution, but we want
    // microseconds.
    *record.runs += 1;
    *record.runtime += (end_t - start_t)/1000;

    if ((err = clReleaseEvent(*record.event)) != CL_SUCCESS) {
      return err;
    }
    free(record.event);
  }

  ctx->profiling_records_used = 0;

  return CL_SUCCESS;
}

// If profiling, produce an event associated with a profiling record.
static cl_event* opencl_get_event(struct opencl_context *ctx, int *runs, int64_t *runtime) {
    if (ctx->profiling_records_used == ctx->profiling_records_capacity) {
      ctx->profiling_records_capacity *= 2;
      ctx->profiling_records =
        realloc(ctx->profiling_records,
                ctx->profiling_records_capacity *
                sizeof(struct profiling_record));
    }
    cl_event *event = malloc(sizeof(cl_event));
    ctx->profiling_records[ctx->profiling_records_used].event = event;
    ctx->profiling_records[ctx->profiling_records_used].runs = runs;
    ctx->profiling_records[ctx->profiling_records_used].runtime = runtime;
    ctx->profiling_records_used++;
    return event;
}

// Allocate memory from driver. The problem is that OpenCL may perform
// lazy allocation, so we cannot know whether an allocation succeeded
// until the first time we try to use it.  Hence we immediately
// perform a write to see if the allocation succeeded.  This is slow,
// but the assumption is that this operation will be rare (most things
// will go through the free list).
static int opencl_alloc_actual(struct opencl_context *ctx, size_t size, cl_mem *mem_out) {
  int error;
  *mem_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size, NULL, &error);

  if (error != CL_SUCCESS) {
    return error;
  }

  int x = 2;
  error = clEnqueueWriteBuffer(ctx->queue, *mem_out, 1, 0, sizeof(x), &x, 0, NULL, NULL);

  // No need to wait for completion here. clWaitForEvents() cannot
  // return mem object allocation failures. This implies that the
  // buffer is faulted onto the device on enqueue. (Observation by
  // Andreas Kloeckner.)

  return error;
}

static int opencl_alloc(struct opencl_context *ctx, size_t min_size, const char *tag, cl_mem *mem_out) {
  (void)tag;
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;

  if (free_list_find(&ctx->free_list, min_size, &size, mem_out) == 0) {
    // Successfully found a free block.  Is it big enough?
    //
    // FIXME: we might also want to check whether the block is *too
    // big*, to avoid internal fragmentation.  However, this can
    // sharply impact performance on programs where arrays change size
    // frequently.  Fortunately, such allocations are usually fairly
    // short-lived, as they are necessarily within a loop, so the risk
    // of internal fragmentation resulting in an OOM situation is
    // limited.  However, it would be preferable if we could go back
    // and *shrink* oversize allocations when we encounter an OOM
    // condition.  That is technically feasible, since we do not
    // expose OpenCL pointer values directly to the application, but
    // instead rely on a level of indirection.
    if (size >= min_size) {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "No need to allocate: Found a block in the free list.\n");
      }

      return CL_SUCCESS;
    } else {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "Found a free block, but it was too small.\n");
      }

      // Not just right - free it.
      int error = clReleaseMemObject(*mem_out);
      if (error != CL_SUCCESS) {
        return error;
      }
    }
  }

  // We have to allocate a new block from the driver.  If the
  // allocation does not succeed, then we might be in an out-of-memory
  // situation.  We now start freeing things from the free list until
  // we think we have freed enough that the allocation will succeed.
  // Since we don't know how far the allocation is from fitting, we
  // have to check after every deallocation.  This might be pretty
  // expensive.  Let's hope that this case is hit rarely.

  if (ctx->cfg.debugging) {
    fprintf(stderr, "Actually allocating the desired block.\n");
  }

  int error = opencl_alloc_actual(ctx, min_size, mem_out);

  while (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Out of OpenCL memory: releasing entry from the free list...\n");
    }
    cl_mem mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      error = clReleaseMemObject(mem);
      if (error != CL_SUCCESS) {
        return error;
      }
    } else {
      break;
    }
    error = opencl_alloc_actual(ctx, min_size, mem_out);
  }

  return error;
}

static int opencl_free(struct opencl_context *ctx, cl_mem mem, const char *tag) {
  size_t size;
  cl_mem existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, -1, &size, &existing_mem) == 0) {
    int error = clReleaseMemObject(existing_mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  int error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

  if (error == CL_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return error;
}

static int opencl_free_all(struct opencl_context *ctx) {
  cl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    int error = clReleaseMemObject(mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  return CL_SUCCESS;
}

// Free everything that belongs to 'ctx', but do not free 'ctx'
// itself.
static void teardown_opencl(struct opencl_context *ctx) {
  (void)opencl_tally_profiling_records(ctx);
  free(ctx->profiling_records);
  (void)opencl_free_all(ctx);
  (void)clReleaseCommandQueue(ctx->queue);
  (void)clReleaseContext(ctx->ctx);
}

// End of opencl.h.

static const char *opencl_program[] =
                  {"#ifdef cl_clang_storage_class_specifiers\n#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable\n#endif\n#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#ifdef cl_nv_pragma_unroll\nstatic inline void mem_fence_global()\n{\n    asm(\"membar.gl;\");\n}\n#else\nstatic inline void mem_fence_global()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n}\n#endif\nstatic inline void mem_fence_local()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE);\n}\nstatic inline uint8_t add8(uint8_t x, uint8_t y)\n{\n    return x + y;\n}\nstatic inline uint16_t add16(uint16_t x, uint16_t y)\n{\n    return x + y;\n}\nstatic inline uint32_t add32(uint32_t x, uint32_t y)\n{\n    return x + y;\n}\nstatic inline uint64_t add64(uint64_t x, uint64_t y)\n{\n    return x + y;\n}\nstatic inline uint8_t sub8(uint8_t x, uint8_t y)\n{\n    return x - y;\n}\nstatic inline uint16_t sub16(uint16_t x, uint16_t y)\n{\n    return x - y;\n}\nstatic inline uint32_t sub32(uint32_t x, uint32_t y)\n{\n    return x - y;\n}\nstatic inline uint64_t sub64(uint64_t x, uint64_t y)\n{\n    return x - y;\n}\nstatic inline uint8_t mul8(uint8_t x, uint8_t y)\n{\n    return x * y;\n}\nstatic inline uint16_t mul16(uint16_t x, uint16_t y)\n{\n    return x * y;\n}\nstatic inline uint32_t mul32(uint32_t x, uint32_t y)\n{\n    return x * y;\n}\nstatic inline uint64_t mul64(uint64_t x, uint64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_",
                   "t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t udiv_up8(uint8_t x, uint8_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint16_t udiv_up16(uint16_t x, uint16_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint32_t udiv_up32(uint32_t x, uint32_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint64_t udiv_up64(uint64_t x, uint64_t y)\n{\n    return (x + y - 1) / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline uint8_t udiv_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint16_t udiv_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint32_t udiv_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint64_t udiv_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : (x + y - 1) / y;\n}\nstatic inline uint8_t umod_safe8(uint8_t x, uint8_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint16_t umod_safe16(uint16_t x, uint16_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint32_t umod_safe32(uint32_t x, uint32_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline uint64_t umod_safe64(uint64_t x, uint64_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ",
                   "? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t sdiv_up8(int8_t x, int8_t y)\n{\n    return sdiv8(x + y - 1, y);\n}\nstatic inline int16_t sdiv_up16(int16_t x, int16_t y)\n{\n    return sdiv16(x + y - 1, y);\n}\nstatic inline int32_t sdiv_up32(int32_t x, int32_t y)\n{\n    return sdiv32(x + y - 1, y);\n}\nstatic inline int64_t sdiv_up64(int64_t x, int64_t y)\n{\n    return sdiv64(x + y - 1, y);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t sdiv_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : sdiv8(x, y);\n}\nstatic inline int16_t sdiv_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : sdiv16(x, y);\n}\nstatic inline int32_t sdiv_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : sdiv32(x, y);\n}\nstatic inline int64_t sdiv_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : sdiv64(x, y);\n}\nstatic inline int8_t sdiv_up_safe8(int8_t x, int8_t y)\n{\n    return sdiv_safe8(x + y - 1, y);\n}\nstatic inline int16_t sdiv_up_safe16(int16_t x, in",
                   "t16_t y)\n{\n    return sdiv_safe16(x + y - 1, y);\n}\nstatic inline int32_t sdiv_up_safe32(int32_t x, int32_t y)\n{\n    return sdiv_safe32(x + y - 1, y);\n}\nstatic inline int64_t sdiv_up_safe64(int64_t x, int64_t y)\n{\n    return sdiv_safe64(x + y - 1, y);\n}\nstatic inline int8_t smod_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : smod8(x, y);\n}\nstatic inline int16_t smod_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : smod16(x, y);\n}\nstatic inline int32_t smod_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : smod32(x, y);\n}\nstatic inline int64_t smod_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : smod64(x, y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t squot_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int16_t squot_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int32_t squot_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int64_t squot_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : x / y;\n}\nstatic inline int8_t srem_safe8(int8_t x, int8_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int16_t srem_safe16(int16_t x, int16_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int32_t srem_safe32(int32_t x, int32_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int64_t srem_safe64(int64_t x, int64_t y)\n{\n    return y == 0 ? 0 : x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    ret",
                   "urn x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstat",
                   "ic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline bool ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline bool ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline bool ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline bool ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline bool ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline bool slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline bool slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline bool slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline bool slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline bool sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle16(int16_t x, int16_t y)\n{\n    return x <= y",
                   ";\n}\nstatic inline bool sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\n#define sext_i8_i8(x) ((int8_t) (int8_t) x)\n#define sext_i8_i16(x) ((int16_t) (int8_t) x)\n#define sext_i8_i32(x) ((int32_t) (int8_t) x)\n#define sext_i8_i64(x) ((int64_t) (int8_t) x)\n#define sext_i16_i8(x) ((int8_t) (int16_t) x)\n#define sext_i16_i16(x) ((int16_t) (int16_t) x)\n#define sext_i16_i32(x) ((int32_t) (int16_t) x)\n#define sext_i16_i64(x) ((int64_t) (int16_t) x)\n#define sext_i32_i8(x) ((int8_t) (int32_t) x)\n#define sext_i32_i16(x) ((int16_t) (int32_t) x)\n#define sext_i32_i32(x) ((int32_t) (int32_t) x)\n#define",
                   " sext_i32_i64(x) ((int64_t) (int32_t) x)\n#define sext_i64_i8(x) ((int8_t) (int64_t) x)\n#define sext_i64_i16(x) ((int16_t) (int64_t) x)\n#define sext_i64_i32(x) ((int32_t) (int64_t) x)\n#define sext_i64_i64(x) ((int64_t) (int64_t) x)\n#define zext_i8_i8(x) ((int8_t) (uint8_t) x)\n#define zext_i8_i16(x) ((int16_t) (uint8_t) x)\n#define zext_i8_i32(x) ((int32_t) (uint8_t) x)\n#define zext_i8_i64(x) ((int64_t) (uint8_t) x)\n#define zext_i16_i8(x) ((int8_t) (uint16_t) x)\n#define zext_i16_i16(x) ((int16_t) (uint16_t) x)\n#define zext_i16_i32(x) ((int32_t) (uint16_t) x)\n#define zext_i16_i64(x) ((int64_t) (uint16_t) x)\n#define zext_i32_i8(x) ((int8_t) (uint32_t) x)\n#define zext_i32_i16(x) ((int16_t) (uint32_t) x)\n#define zext_i32_i32(x) ((int32_t) (uint32_t) x)\n#define zext_i32_i64(x) ((int64_t) (uint32_t) x)\n#define zext_i64_i8(x) ((int8_t) (uint64_t) x)\n#define zext_i64_i16(x) ((int16_t) (uint64_t) x)\n#define zext_i64_i32(x) ((int32_t) (uint64_t) x)\n#define zext_i64_i64(x) ((int64_t) (uint64_t) x)\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return popcount(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return __popc(zext_i8_i32(x));\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return __popc(zext_i16_i32(x));\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    return __popc(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return __popcll(x);\n}\n#else\nstatic int32_t futrts_popc8(int8_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_po",
                   "pc64(int64_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul_hi(a, b);\n}\n#elif defined(__CUDA_ARCH__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mulhi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul64hi(a, b);\n}\n#else\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    uint64_t aa = a;\n    uint64_t bb = b;\n    \n    return aa * bb >> 32;\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    __uint128_t aa = a;\n    __uint128_t bb = b;\n    \n    return aa * bb >> 64;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return mad_hi(a, b, c);\n}\n#else\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return futrts_mul_hi8(a,",
                   " b) + c;\n}\nstatic uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)\n{\n    return futrts_mul_hi16(a, b) + c;\n}\nstatic uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)\n{\n    return futrts_mul_hi32(a, b) + c;\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return futrts_mul_hi64(a, b) + c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return clz(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return __clz(zext_i8_i32(x)) - 24;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return __clz(zext_i16_i32(x)) - 16;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return __clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return __clzll(x);\n}\n#else\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    int i = 0;\n    \n    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    ret",
                   "urn i;\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    int i = 0;\n    \n    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    int i = 0;\n    \n    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    int i = 0;\n    \n    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)\n        ;\n    return i;\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 8 : y - 1;\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 16 : y - 1;\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    int y = __ffs(x);\n    \n    return y == 0 ? 32 : y - 1;\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    int y = __ffsll(x);\n    \n    return y == 0 ? 64 : y - 1;\n}\n#else\nstatic int32_t futrts_ctzz8(int8_t x)\n{\n    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);\n}\nstatic int32_t futrts_ctzz16(int16_t x)\n{\n    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);\n}\nstatic int32_t futrts_ctzz32(int32_t x)\n{\n    return x == 0 ? 32 : __builtin_ctz(x);\n}\nstatic int32_t futrts_ctzz64(int64_t x)\n{\n    return x == 0 ? 64 : __builtin_ctzl(x);\n}\n#endif\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return fmin(x, y);\n}\nstatic inline float fmax32(float x, float y)\n{\n    return fmax(x, y);\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline bool cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline bool cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return (float) x;\n}",
                   "\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return (float) x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return (int8_t) x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return (int16_t) x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return (int32_t) x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return (int64_t) x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return (uint8_t) x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return (uint16_t) x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return (uint32_t) x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return (uint64_t) x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_cosh32(float x)\n{\n    return cosh(x);\n}\nstatic inline float futrts_sinh32(float x)\n{\n    return sinh(x);\n}\nstatic inline float futrts_tanh32(float x)\n{\n    return tanh(x);\n}\nstatic inline float futrts_acosh32(floa",
                   "t x)\n{\n    return acosh(x);\n}\nstatic inline float futrts_asinh32(float x)\n{\n    return asinh(x);\n}\nstatic inline float futrts_atanh32(float x)\n{\n    return atanh(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_gamma32(float x)\n{\n    return tgamma(x);\n}\nstatic inline float futrts_lgamma32(float x)\n{\n    return lgamma(x);\n}\nstatic inline bool futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline bool futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#ifdef __OPENCL_VERSION__\nstatic inline float fmod32(float x, float y)\n{\n    return fmod(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floor(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceil(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return mix(v0, v1, t);\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return mad(a, b, c);\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fma(a, b, c);\n}\n#else\nstatic inline float fmod32(float x, float y)\n{\n    return fmodf(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rintf(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floorf(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceilf(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return v0 + (v1 - v0) * t;\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return a * b + c;\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fmaf(a, b, c);\n}\n#endif\nstatic inline double fdiv64(double x, double",
                   " y)\n{\n    return x / y;\n}\nstatic inline double fadd64(double x, double y)\n{\n    return x + y;\n}\nstatic inline double fsub64(double x, double y)\n{\n    return x - y;\n}\nstatic inline double fmul64(double x, double y)\n{\n    return x * y;\n}\nstatic inline double fmin64(double x, double y)\n{\n    return fmin(x, y);\n}\nstatic inline double fmax64(double x, double y)\n{\n    return fmax(x, y);\n}\nstatic inline double fpow64(double x, double y)\n{\n    return pow(x, y);\n}\nstatic inline bool cmplt64(double x, double y)\n{\n    return x < y;\n}\nstatic inline bool cmple64(double x, double y)\n{\n    return x <= y;\n}\nstatic inline double sitofp_i8_f64(int8_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i16_f64(int16_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i32_f64(int32_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i64_f64(int64_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i8_f64(uint8_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i16_f64(uint16_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i32_f64(uint32_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i64_f64(uint64_t x)\n{\n    return (double) x;\n}\nstatic inline int8_t fptosi_f64_i8(double x)\n{\n    return (int8_t) x;\n}\nstatic inline int16_t fptosi_f64_i16(double x)\n{\n    return (int16_t) x;\n}\nstatic inline int32_t fptosi_f64_i32(double x)\n{\n    return (int32_t) x;\n}\nstatic inline int64_t fptosi_f64_i64(double x)\n{\n    return (int64_t) x;\n}\nstatic inline uint8_t fptoui_f64_i8(double x)\n{\n    return (uint8_t) x;\n}\nstatic inline uint16_t fptoui_f64_i16(double x)\n{\n    return (uint16_t) x;\n}\nstatic inline uint32_t fptoui_f64_i32(double x)\n{\n    return (uint32_t) x;\n}\nstatic inline uint64_t fptoui_f64_i64(double x)\n{\n    return (uint64_t) x;\n}\nstatic inline double futrts_log64(double x)\n{\n    return log(x);\n}\nstatic inline double futrts_log2_64(double x)\n{\n    return log2(x);\n}\nstatic inline double futrts_log10_64(double x)\n{\n    return log10(x);\n",
                   "}\nstatic inline double futrts_sqrt64(double x)\n{\n    return sqrt(x);\n}\nstatic inline double futrts_exp64(double x)\n{\n    return exp(x);\n}\nstatic inline double futrts_cos64(double x)\n{\n    return cos(x);\n}\nstatic inline double futrts_sin64(double x)\n{\n    return sin(x);\n}\nstatic inline double futrts_tan64(double x)\n{\n    return tan(x);\n}\nstatic inline double futrts_acos64(double x)\n{\n    return acos(x);\n}\nstatic inline double futrts_asin64(double x)\n{\n    return asin(x);\n}\nstatic inline double futrts_atan64(double x)\n{\n    return atan(x);\n}\nstatic inline double futrts_cosh64(double x)\n{\n    return cosh(x);\n}\nstatic inline double futrts_sinh64(double x)\n{\n    return sinh(x);\n}\nstatic inline double futrts_tanh64(double x)\n{\n    return tanh(x);\n}\nstatic inline double futrts_acosh64(double x)\n{\n    return acosh(x);\n}\nstatic inline double futrts_asinh64(double x)\n{\n    return asinh(x);\n}\nstatic inline double futrts_atanh64(double x)\n{\n    return atanh(x);\n}\nstatic inline double futrts_atan2_64(double x, double y)\n{\n    return atan2(x, y);\n}\nstatic inline double futrts_gamma64(double x)\n{\n    return tgamma(x);\n}\nstatic inline double futrts_lgamma64(double x)\n{\n    return lgamma(x);\n}\nstatic inline double futrts_fma64(double a, double b, double c)\n{\n    return fma(a, b, c);\n}\nstatic inline double futrts_round64(double x)\n{\n    return rint(x);\n}\nstatic inline double futrts_ceil64(double x)\n{\n    return ceil(x);\n}\nstatic inline double futrts_floor64(double x)\n{\n    return floor(x);\n}\nstatic inline bool futrts_isnan64(double x)\n{\n    return isnan(x);\n}\nstatic inline bool futrts_isinf64(double x)\n{\n    return isinf(x);\n}\nstatic inline int64_t futrts_to_bits64(double x)\n{\n    union {\n        double f;\n        int64_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double futrts_from_bits64(int64_t x)\n{\n    union {\n        int64_t f;\n        double t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double fmod64(double x, double y)\n{\n    return fmod(x, y)",
                   ";\n}\n#ifdef __OPENCL_VERSION__\nstatic inline double futrts_lerp64(double v0, double v1, double t)\n{\n    return mix(v0, v1, t);\n}\nstatic inline double futrts_mad64(double a, double b, double c)\n{\n    return mad(a, b, c);\n}\n#else\nstatic inline double futrts_lerp64(double v0, double v1, double t)\n{\n    return v0 + (v1 - v0) * t;\n}\nstatic inline double futrts_mad64(double a, double b, double c)\n{\n    return a * b + c;\n}\n#endif\nstatic inline float fpconv_f32_f32(float x)\n{\n    return (float) x;\n}\nstatic inline double fpconv_f32_f64(float x)\n{\n    return (double) x;\n}\nstatic inline float fpconv_f64_f32(double x)\n{\n    return (float) x;\n}\nstatic inline double fpconv_f64_f64(double x)\n{\n    return (double) x;\n}\n// Start of atomics.h\n\ninline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_add(p, x);\n#endif\n}\n\ninline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_add(p, x);\n#endif\n}\n\ninline float atomic_fadd_f32_global(volatile __global float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do {\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline float atomic_fadd_f32_local(volatile __local float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do {\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {\n#",
                   "ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  ret",
                   "urn atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\n// End of atomics.h\n\n\n\n\n__kernel void builtinzhreplicate_f64zireplicate_52879(__global\n                                                      unsigned char *mem_52875,\n                                                      int32_t num_elems_52876,\n                                                      double val_52877)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_52879;\n    int32_t replicate_ltid_52880;\n    int32_t replicate_gid_52881;\n    \n    replicate_gtid_52879 = get_global_id(0);\n    replicate_ltid_52880 = get_local_id(0);\n    replicate_gid_52881 = get_group_id(0);\n    if (slt64(replicate_gtid_52879, num_elems_52876)",
                   ") {\n        ((__global double *) mem_52875)[sext_i32_i64(replicate_gtid_52879)] =\n            val_52877;\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64(__local volatile\n                                    int64_t *block_9_backing_aligned_0,\n                                    int32_t destoffset_1, int32_t srcoffset_3,\n                                    int32_t num_arrays_4, int32_t x_elems_5,\n                                    int32_t y_elems_6, int32_t mulx_7,\n                                    int32_t muly_8, __global\n                                    unsigned char *destmem_0, __global\n                                    unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t x_index_31 = get_global_id_0_37;\n    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;\n    \n    if (slt32(x_index_31, x_elems_5)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_in_35 ",
                   "= (y_index_32 + j_43 * 8) * x_elems_5 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {\n                ((__local double *) block_9)[sext_i32_i64((get_local_id_1_39 +\n                                                           j_43 * 8) * 33 +\n                                             get_local_id_0_38)] = ((__global\n                                                                     double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                                         index_in_35)];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;\n    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;\n    if (slt32(x_index_31, y_elems_6)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {\n                ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                                index_out_36)] = ((__local\n                                                                   double *) block_9)[sext_i32_i64(get_local_id_0_38 *\n                                                                                      33 +\n                                                                                      get_local_id_1_39 +\n                                                                                      j_43 *\n                                                                                      8)];\n            }\n        }\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64_low_height(__local volatile\n                                               int64_t *block_9_backing_aligned_0,\n                                               int32_t destoffset_1,\n               ",
                   "                                int32_t srcoffset_3,\n                                               int32_t num_arrays_4,\n                                               int32_t x_elems_5,\n                                               int32_t y_elems_6,\n                                               int32_t mulx_7, int32_t muly_8,\n                                               __global\n                                               unsigned char *destmem_0,\n                                               __global unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +\n            srem32(get_local_id_1_39, mulx_7) * 16;\n    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,\n                                                          mulx_7);\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && sl",
                   "t32(y_index_32, y_elems_6)) {\n        ((__local double *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +\n                                     get_local_id_0_38)] = ((__global\n                                                             double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                                 index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);\n    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +\n        srem32(get_local_id_0_38, mulx_7) * 16;\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {\n        ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                        index_out_36)] = ((__local\n                                                           double *) block_9)[sext_i32_i64(get_local_id_0_38 *\n                                                                              17 +\n                                                                              get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64_low_width(__local volatile\n                                              int64_t *block_9_backing_aligned_0,\n                                              int32_t destoffset_1,\n                                              int32_t srcoffset_3,\n                                              int32_t num_arrays_4,\n                                              int32_t x_elems_5,\n                                              int32_t y_elems_6, int32_t mulx_7,\n                                              int32_t muly_8, __global\n                                              unsigned char *destmem_0, __global\n                                              unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int bloc",
                   "k_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,\n                                                          muly_8);\n    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +\n            srem32(get_local_id_0_38, muly_8) * 16;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {\n        ((__local double *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +\n                                     get_local_id_0_38)] = ((__global\n                                                             double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                                 index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +\n        srem32(get_local_id_1_39, muly_8) * 16;\n    y_index_32 = get_group_id_0_40 * 16 + squot32(g",
                   "et_local_id_1_39, muly_8);\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {\n        ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                        index_out_36)] = ((__local\n                                                           double *) block_9)[sext_i32_i64(get_local_id_0_38 *\n                                                                              17 +\n                                                                              get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64_small(__local volatile\n                                          int64_t *block_9_backing_aligned_0,\n                                          int32_t destoffset_1,\n                                          int32_t srcoffset_3,\n                                          int32_t num_arrays_4,\n                                          int32_t x_elems_5, int32_t y_elems_6,\n                                          int32_t mulx_7, int32_t muly_8,\n                                          __global unsigned char *destmem_0,\n                                          __global unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group",
                   "_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *\n                                          x_elems_5) * (y_elems_6 * x_elems_5);\n    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *\n                                        x_elems_5), y_elems_6);\n    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;\n    \n    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {\n        ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                        index_out_36)] = ((__global\n                                                           double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                               index_in_35)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void integrate_tkezisegmap_44495(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t distance_42130,\n                                          int64_t m_42138,\n                                          int64_t num_groups_45942, __global\n                                          unsigned char *mem_52706, __global\n                                          unsigned char *mem_52710, __global\n                          ",
                   "                unsigned char *mem_52719, __global\n                                          unsigned char *mem_52730, __global\n                                          unsigned char *mem_52742)\n{\n    #define segmap_group_sizze_45941 (integrate_tkezisegmap_group_sizze_44498)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_52945;\n    int32_t local_tid_52946;\n    int64_t group_sizze_52949;\n    int32_t wave_sizze_52948;\n    int32_t group_tid_52947;\n    \n    global_tid_52945 = get_global_id(0);\n    local_tid_52946 = get_local_id(0);\n    group_sizze_52949 = get_local_size(0);\n    wave_sizze_52948 = LOCKSTEP_WIDTH;\n    group_tid_52947 = get_group_id(0);\n    \n    int32_t phys_tid_44495;\n    \n    phys_tid_44495 = global_tid_52945;\n    \n    int32_t phys_group_id_52950;\n    \n    phys_group_id_52950 = get_group_id(0);\n    for (int32_t i_52951 = 0; i_52951 <\n         sdiv_up32(sext_i64_i32(sdiv_up64(xdim_41931 * ydim_41932,\n                                          segmap_group_sizze_45941)) -\n                   phys_group_id_52950, sext_i64_i32(num_groups_45942));\n         i_52951++) {\n        int32_t virt_group_id_52952 = phys_group_id_52950 + i_52951 *\n                sext_i64_i32(num_groups_45942);\n        int64_t gtid_44493 = squot64(sext_i32_i64(virt_group_id_52952) *\n                                     segmap_group_sizze_45941 +\n                                     sext_i32_i64(local_tid_52946), ydim_41932);\n        int64_t gtid_44494 = sext_i32_i64(virt_group_id_52952) *\n                segmap_group_sizze_45941 + sext_i32_i64(local_tid_52946) -\n                squot64(sext_i32_i64(virt_group_id_52952) *\n                        segmap_group_sizze_45941 +\n        ",
                   "                sext_i32_i64(local_tid_52946), ydim_41932) * ydim_41932;\n        \n        if (slt64(gtid_44493, xdim_41931) && slt64(gtid_44494, ydim_41932)) {\n            for (int64_t i_52953 = 0; i_52953 < zzdim_41933; i_52953++) {\n                ((__global double *) mem_52730)[phys_tid_44495 + i_52953 *\n                                                (num_groups_45942 *\n                                                 segmap_group_sizze_45941)] =\n                    ((__global double *) mem_52719)[gtid_44493 * ydim_41932 +\n                                                    gtid_44494 + i_52953 *\n                                                    (ydim_41932 * xdim_41931)];\n            }\n            for (int64_t i_45949 = 0; i_45949 < distance_42130; i_45949++) {\n                int64_t binop_y_45951 = -1 * i_45949;\n                int64_t binop_x_45952 = m_42138 + binop_y_45951;\n                bool x_45953 = sle64(0, binop_x_45952);\n                bool y_45954 = slt64(binop_x_45952, zzdim_41933);\n                bool bounds_check_45955 = x_45953 && y_45954;\n                bool index_certs_45956;\n                \n                if (!bounds_check_45955) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 25) ==\n                            -1) {\n                            global_failure_args[0] = binop_x_45952;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double x_45957 = ((__global double *) mem_52710)[binop_x_45952 *\n                                                                 (ydim_41932 *\n                                                                  xdim_41931) +\n                                                                 gtid_44493 *\n                                ",
                   "                                 ydim_41932 +\n                                                                 gtid_44494];\n                double x_45958 = ((__global double *) mem_52706)[binop_x_45952 *\n                                                                 (ydim_41932 *\n                                                                  xdim_41931) +\n                                                                 gtid_44493 *\n                                                                 ydim_41932 +\n                                                                 gtid_44494];\n                int64_t i_45959 = add64(1, binop_x_45952);\n                bool x_45960 = sle64(0, i_45959);\n                bool y_45961 = slt64(i_45959, zzdim_41933);\n                bool bounds_check_45962 = x_45960 && y_45961;\n                bool index_certs_45963;\n                \n                if (!bounds_check_45962) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 26) ==\n                            -1) {\n                            global_failure_args[0] = i_45959;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double y_45964 = ((__global\n                                   double *) mem_52730)[phys_tid_44495 +\n                                                        i_45959 *\n                                                        (num_groups_45942 *\n                                                         segmap_group_sizze_45941)];\n                double y_45965 = x_45958 * y_45964;\n                double lw_val_45966 = x_45957 - y_45965;\n                \n                ((__global double *) mem_52730)[phys_tid_44495 + binop_x_45952 *\n                                                (num_group",
                   "s_45942 *\n                                                 segmap_group_sizze_45941)] =\n                    lw_val_45966;\n            }\n            for (int64_t i_52955 = 0; i_52955 < zzdim_41933; i_52955++) {\n                ((__global double *) mem_52742)[i_52955 * (ydim_41932 *\n                                                           xdim_41931) +\n                                                gtid_44493 * ydim_41932 +\n                                                gtid_44494] = ((__global\n                                                                double *) mem_52730)[phys_tid_44495 +\n                                                                                     i_52955 *\n                                                                                     (num_groups_45942 *\n                                                                                      segmap_group_sizze_45941)];\n            }\n        }\n        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_45941\n}\n__kernel void integrate_tkezisegmap_44535(__global int *global_failure,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t distance_42130, __global\n                                          unsigned char *mem_52464, __global\n                                          unsigned char *mem_52714)\n{\n    #define segmap_group_sizze_45930 (integrate_tkezisegmap_group_sizze_44539)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52940;\n    int32_t local_tid_52941;\n    int64_t group_sizze_52944;\n    int32_t wave_sizze_52943;\n    int32_t group_tid_52942;\n    \n    global_tid_52940 = get_global_id(0);\n    local_tid_52941",
                   " = get_local_id(0);\n    group_sizze_52944 = get_local_size(0);\n    wave_sizze_52943 = LOCKSTEP_WIDTH;\n    group_tid_52942 = get_group_id(0);\n    \n    int32_t phys_tid_44535;\n    \n    phys_tid_44535 = global_tid_52940;\n    \n    int64_t gtid_44530;\n    \n    gtid_44530 = squot64(sext_i32_i64(group_tid_52942) *\n                         segmap_group_sizze_45930 +\n                         sext_i32_i64(local_tid_52941), ydim_41932);\n    \n    int64_t gtid_44531;\n    \n    gtid_44531 = sext_i32_i64(group_tid_52942) * segmap_group_sizze_45930 +\n        sext_i32_i64(local_tid_52941) - squot64(sext_i32_i64(group_tid_52942) *\n                                                segmap_group_sizze_45930 +\n                                                sext_i32_i64(local_tid_52941),\n                                                ydim_41932) * ydim_41932;\n    \n    int64_t gtid_slice_44532;\n    \n    gtid_slice_44532 = sext_i32_i64(group_tid_52942) *\n        segmap_group_sizze_45930 + sext_i32_i64(local_tid_52941) -\n        squot64(sext_i32_i64(group_tid_52942) * segmap_group_sizze_45930 +\n                sext_i32_i64(local_tid_52941), ydim_41932) * ydim_41932 -\n        (sext_i32_i64(group_tid_52942) * segmap_group_sizze_45930 +\n         sext_i32_i64(local_tid_52941) - squot64(sext_i32_i64(group_tid_52942) *\n                                                 segmap_group_sizze_45930 +\n                                                 sext_i32_i64(local_tid_52941),\n                                                 ydim_41932) * ydim_41932);\n    if ((slt64(gtid_44530, xdim_41931) && slt64(gtid_44531, ydim_41932)) &&\n        slt64(gtid_slice_44532, 1)) {\n        int64_t index_primexp_52201 = distance_42130 + gtid_slice_44532;\n        double v_45937 = ((__global double *) mem_52714)[gtid_44530 *\n                                                         (zzdim_41933 *\n                                                          ydim_41932) +\n                                                         gt",
                   "id_44531 *\n                                                         zzdim_41933 +\n                                                         index_primexp_52201];\n        \n        if (((sle64(0, gtid_44530) && slt64(gtid_44530, xdim_41931)) &&\n             (sle64(0, gtid_44531) && slt64(gtid_44531, ydim_41932))) &&\n            (sle64(0, index_primexp_52201) && slt64(index_primexp_52201,\n                                                    zzdim_41933))) {\n            ((__global double *) mem_52464)[gtid_44530 * (zzdim_41933 *\n                                                          ydim_41932) +\n                                            gtid_44531 * zzdim_41933 +\n                                            index_primexp_52201] = v_45937;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_45930\n}\n__kernel void integrate_tkezisegmap_44613(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t distance_42130,\n                                          int64_t num_groups_45863, __global\n                                          unsigned char *mem_52645, __global\n                                          unsigned char *mem_52649, __global\n                                          unsigned char *mem_52653, __global\n                                          unsigned char *mem_52657, __global\n                                          unsigned char *mem_52661, __global\n                                          unsigned char *mem_52665, __global\n                                          unsigned char *mem_52684, __global\n                                          unsigned char *mem_52689, __global\n   ",
                   "                                       unsigned char *mem_52706, __global\n                                          unsigned char *mem_52710)\n{\n    #define segmap_group_sizze_45862 (integrate_tkezisegmap_group_sizze_44616)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_52926;\n    int32_t local_tid_52927;\n    int64_t group_sizze_52930;\n    int32_t wave_sizze_52929;\n    int32_t group_tid_52928;\n    \n    global_tid_52926 = get_global_id(0);\n    local_tid_52927 = get_local_id(0);\n    group_sizze_52930 = get_local_size(0);\n    wave_sizze_52929 = LOCKSTEP_WIDTH;\n    group_tid_52928 = get_group_id(0);\n    \n    int32_t phys_tid_44613;\n    \n    phys_tid_44613 = global_tid_52926;\n    \n    int32_t phys_group_id_52931;\n    \n    phys_group_id_52931 = get_group_id(0);\n    for (int32_t i_52932 = 0; i_52932 <\n         sdiv_up32(sext_i64_i32(sdiv_up64(xdim_41931 * ydim_41932,\n                                          segmap_group_sizze_45862)) -\n                   phys_group_id_52931, sext_i64_i32(num_groups_45863));\n         i_52932++) {\n        int32_t virt_group_id_52933 = phys_group_id_52931 + i_52932 *\n                sext_i64_i32(num_groups_45863);\n        int64_t gtid_44611 = squot64(sext_i32_i64(virt_group_id_52933) *\n                                     segmap_group_sizze_45862 +\n                                     sext_i32_i64(local_tid_52927), ydim_41932);\n        int64_t gtid_44612 = sext_i32_i64(virt_group_id_52933) *\n                segmap_group_sizze_45862 + sext_i32_i64(local_tid_52927) -\n                squot64(sext_i32_i64(virt_group_id_52933) *\n                        segmap_group_sizze_45862 +\n                        sext_i32_i64(local_tid_52927), ydim_41",
                   "932) * ydim_41932;\n        \n        if (slt64(gtid_44611, xdim_41931) && slt64(gtid_44612, ydim_41932)) {\n            for (int64_t i_52934 = 0; i_52934 < zzdim_41933; i_52934++) {\n                ((__global double *) mem_52684)[phys_tid_44613 + i_52934 *\n                                                (num_groups_45863 *\n                                                 segmap_group_sizze_45862)] =\n                    ((__global double *) mem_52649)[gtid_44611 * ydim_41932 +\n                                                    gtid_44612 + i_52934 *\n                                                    (ydim_41932 * xdim_41931)];\n            }\n            for (int64_t i_52935 = 0; i_52935 < zzdim_41933; i_52935++) {\n                ((__global double *) mem_52689)[phys_tid_44613 + i_52935 *\n                                                (num_groups_45863 *\n                                                 segmap_group_sizze_45862)] =\n                    ((__global double *) mem_52645)[gtid_44611 * ydim_41932 +\n                                                    gtid_44612 + i_52935 *\n                                                    (ydim_41932 * xdim_41931)];\n            }\n            for (int64_t i_45875 = 0; i_45875 < distance_42130; i_45875++) {\n                int64_t index_primexp_45878 = add64(1, i_45875);\n                bool x_45879 = sle64(0, index_primexp_45878);\n                bool y_45880 = slt64(index_primexp_45878, zzdim_41933);\n                bool bounds_check_45881 = x_45879 && y_45880;\n                bool index_certs_45882;\n                \n                if (!bounds_check_45881) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 23) ==\n                            -1) {\n                            global_failure_args[0] = index_primexp_45878;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        local_failure =",
                   " true;\n                        goto error_0;\n                    }\n                }\n                \n                double x_45883 = ((__global\n                                   double *) mem_52653)[index_primexp_45878 *\n                                                        (ydim_41932 *\n                                                         xdim_41931) +\n                                                        gtid_44611 *\n                                                        ydim_41932 +\n                                                        gtid_44612];\n                double x_45884 = ((__global\n                                   double *) mem_52657)[index_primexp_45878 *\n                                                        (ydim_41932 *\n                                                         xdim_41931) +\n                                                        gtid_44611 *\n                                                        ydim_41932 +\n                                                        gtid_44612];\n                bool y_45885 = slt64(i_45875, zzdim_41933);\n                bool index_certs_45886;\n                \n                if (!y_45885) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 24) ==\n                            -1) {\n                            global_failure_args[0] = i_45875;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double y_45887 = ((__global\n                                   double *) mem_52684)[phys_tid_44613 +\n                                                        i_45875 *\n                                                        (num_groups_45863 *\n                                                         segmap_group_sizze_45862)];\n   ",
                   "             double y_45888 = x_45884 * y_45887;\n                double y_45889 = x_45883 - y_45888;\n                double norm_factor_45890 = 1.0 / y_45889;\n                double x_45891 = ((__global\n                                   double *) mem_52661)[index_primexp_45878 *\n                                                        (ydim_41932 *\n                                                         xdim_41931) +\n                                                        gtid_44611 *\n                                                        ydim_41932 +\n                                                        gtid_44612];\n                double lw_val_45892 = norm_factor_45890 * x_45891;\n                \n                ((__global double *) mem_52684)[phys_tid_44613 +\n                                                index_primexp_45878 *\n                                                (num_groups_45863 *\n                                                 segmap_group_sizze_45862)] =\n                    lw_val_45892;\n                \n                double x_45894 = ((__global\n                                   double *) mem_52665)[index_primexp_45878 *\n                                                        (ydim_41932 *\n                                                         xdim_41931) +\n                                                        gtid_44611 *\n                                                        ydim_41932 +\n                                                        gtid_44612];\n                double y_45895 = ((__global\n                                   double *) mem_52689)[phys_tid_44613 +\n                                                        i_45875 *\n                                                        (num_groups_45863 *\n                                                         segmap_group_sizze_45862)];\n                double y_45896 = x_45884 * y_45895;\n                double x_45897 = x_45894 - y_45896;\n                double lw_val_45898 =",
                   " norm_factor_45890 * x_45897;\n                \n                ((__global double *) mem_52689)[phys_tid_44613 +\n                                                index_primexp_45878 *\n                                                (num_groups_45863 *\n                                                 segmap_group_sizze_45862)] =\n                    lw_val_45898;\n            }\n            for (int64_t i_52938 = 0; i_52938 < zzdim_41933; i_52938++) {\n                ((__global double *) mem_52706)[i_52938 * (ydim_41932 *\n                                                           xdim_41931) +\n                                                gtid_44611 * ydim_41932 +\n                                                gtid_44612] = ((__global\n                                                                double *) mem_52684)[phys_tid_44613 +\n                                                                                     i_52938 *\n                                                                                     (num_groups_45863 *\n                                                                                      segmap_group_sizze_45862)];\n            }\n            for (int64_t i_52939 = 0; i_52939 < zzdim_41933; i_52939++) {\n                ((__global double *) mem_52710)[i_52939 * (ydim_41932 *\n                                                           xdim_41931) +\n                                                gtid_44611 * ydim_41932 +\n                                                gtid_44612] = ((__global\n                                                                double *) mem_52689)[phys_tid_44613 +\n                                                                                     i_52939 *\n                                                                                     (num_groups_45863 *\n                                                                                      segmap_group_sizze_45862)];\n            }\n        }\n        barrier(CLK_GL",
                   "OBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_45862\n}\n__kernel void integrate_tkezisegmap_44690(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933, __global\n                                          unsigned char *mem_52624, __global\n                                          unsigned char *mem_52628, __global\n                                          unsigned char *mem_52632, __global\n                                          unsigned char *mem_52637, __global\n                                          unsigned char *mem_52641)\n{\n    #define segmap_group_sizze_45837 (integrate_tkezisegmap_group_sizze_44694)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52921;\n    int32_t local_tid_52922;\n    int64_t group_sizze_52925;\n    int32_t wave_sizze_52924;\n    int32_t group_tid_52923;\n    \n    global_tid_52921 = get_global_id(0);\n    local_tid_52922 = get_local_id(0);\n    group_sizze_52925 = get_local_size(0);\n    wave_sizze_52924 = LOCKSTEP_WIDTH;\n    group_tid_52923 = get_group_id(0);\n    \n    int32_t phys_tid_44690;\n    \n    phys_tid_44690 = global_tid_52921;\n    \n    int64_t gtid_44687;\n    \n    gtid_44687 = squot64(sext_i32_i64(group_tid_52923) *\n                         segmap_group_sizze_45837 +\n                         sext_i32_i64(local_tid_52922), ydim_41932 *\n                         zzdim_41933);\n    \n    int64_t gtid_44688;\n    \n    gtid_44688 = squot64(sext_i32_i64(group_tid_52923) *\n                         segmap_group_sizze_45837 +\n                         sext_i32_i64(",
                   "local_tid_52922) -\n                         squot64(sext_i32_i64(group_tid_52923) *\n                                 segmap_group_sizze_45837 +\n                                 sext_i32_i64(local_tid_52922), ydim_41932 *\n                                 zzdim_41933) * (ydim_41932 * zzdim_41933),\n                         zzdim_41933);\n    \n    int64_t gtid_44689;\n    \n    gtid_44689 = sext_i32_i64(group_tid_52923) * segmap_group_sizze_45837 +\n        sext_i32_i64(local_tid_52922) - squot64(sext_i32_i64(group_tid_52923) *\n                                                segmap_group_sizze_45837 +\n                                                sext_i32_i64(local_tid_52922),\n                                                ydim_41932 * zzdim_41933) *\n        (ydim_41932 * zzdim_41933) - squot64(sext_i32_i64(group_tid_52923) *\n                                             segmap_group_sizze_45837 +\n                                             sext_i32_i64(local_tid_52922) -\n                                             squot64(sext_i32_i64(group_tid_52923) *\n                                                     segmap_group_sizze_45837 +\n                                                     sext_i32_i64(local_tid_52922),\n                                                     ydim_41932 * zzdim_41933) *\n                                             (ydim_41932 * zzdim_41933),\n                                             zzdim_41933) * zzdim_41933;\n    if ((slt64(gtid_44687, xdim_41931) && slt64(gtid_44688, ydim_41932)) &&\n        slt64(gtid_44689, zzdim_41933)) {\n        bool cond_45845 = gtid_44689 == 0;\n        double lifted_0_f_res_45846;\n        \n        if (cond_45845) {\n            bool y_45847 = slt64(0, zzdim_41933);\n            bool index_certs_45848;\n            \n            if (!y_45847) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 21) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n     ",
                   "                   global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_45849 = ((__global double *) mem_52628)[gtid_44687 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_44688 *\n                                                             zzdim_41933];\n            double y_45850 = ((__global double *) mem_52624)[gtid_44687 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_44688 *\n                                                             zzdim_41933];\n            double lifted_0_f_res_t_res_45851 = x_45849 / y_45850;\n            \n            lifted_0_f_res_45846 = lifted_0_f_res_t_res_45851;\n        } else {\n            lifted_0_f_res_45846 = 0.0;\n        }\n        \n        double lifted_0_f_res_45852;\n        \n        if (cond_45845) {\n            bool y_45853 = slt64(0, zzdim_41933);\n            bool index_certs_45854;\n            \n            if (!y_45853) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 22) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_45855 = ((__global double *) mem_52632)[gtid_44687 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_44",
                   "688 *\n                                                             zzdim_41933];\n            double y_45856 = ((__global double *) mem_52624)[gtid_44687 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_44688 *\n                                                             zzdim_41933];\n            double lifted_0_f_res_t_res_45857 = x_45855 / y_45856;\n            \n            lifted_0_f_res_45852 = lifted_0_f_res_t_res_45857;\n        } else {\n            lifted_0_f_res_45852 = 0.0;\n        }\n        ((__global double *) mem_52637)[gtid_44687 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_44688 *\n                                        zzdim_41933 + gtid_44689] =\n            lifted_0_f_res_45852;\n        ((__global double *) mem_52641)[gtid_44687 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_44688 *\n                                        zzdim_41933 + gtid_44689] =\n            lifted_0_f_res_45846;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_45837\n}\n__kernel void integrate_tkezisegmap_44880(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t ydim_41976,\n                                          int64_t ydim_41978,\n                                          int64_t zzdim_41979,\n                                          int64_t ydim_41981,\n                                          int64_t zzdim_41982,\n                                        ",
                   "  int64_t ydim_41984,\n                                          int64_t zzdim_41985,\n                                          int64_t ydim_41987, int64_t y_42127,\n                                          int64_t y_42128, __global\n                                          unsigned char *tketau_mem_52436,\n                                          __global\n                                          unsigned char *dzzt_mem_52452,\n                                          __global\n                                          unsigned char *dzzw_mem_52453,\n                                          __global\n                                          unsigned char *kbot_mem_52456,\n                                          __global\n                                          unsigned char *kappaM_mem_52457,\n                                          __global unsigned char *mxl_mem_52458,\n                                          __global\n                                          unsigned char *forc_mem_52459,\n                                          __global\n                                          unsigned char *forc_tke_surface_mem_52460,\n                                          __global unsigned char *mem_52620,\n                                          __global unsigned char *mem_52624,\n                                          __global unsigned char *mem_52628,\n                                          __global unsigned char *mem_52632)\n{\n    #define segmap_group_sizze_45696 (integrate_tkezisegmap_group_sizze_44884)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52916;\n    int32_t local_tid_52917;\n    int64_t group_sizze_52920;\n    int32_t wave_sizze_52919;\n    int32_t group_tid_52918;\n    \n    global_tid_52916 = get_global_id(0);\n    local_tid_52917 = get_local_id(0);\n    group_sizze_52920 = get_local_size(0);\n    wave_sizze_52919 = LOCKSTEP_WIDTH",
                   ";\n    group_tid_52918 = get_group_id(0);\n    \n    int32_t phys_tid_44880;\n    \n    phys_tid_44880 = global_tid_52916;\n    \n    int64_t gtid_44877;\n    \n    gtid_44877 = squot64(sext_i32_i64(group_tid_52918) *\n                         segmap_group_sizze_45696 +\n                         sext_i32_i64(local_tid_52917), ydim_41932 *\n                         zzdim_41933);\n    \n    int64_t gtid_44878;\n    \n    gtid_44878 = squot64(sext_i32_i64(group_tid_52918) *\n                         segmap_group_sizze_45696 +\n                         sext_i32_i64(local_tid_52917) -\n                         squot64(sext_i32_i64(group_tid_52918) *\n                                 segmap_group_sizze_45696 +\n                                 sext_i32_i64(local_tid_52917), ydim_41932 *\n                                 zzdim_41933) * (ydim_41932 * zzdim_41933),\n                         zzdim_41933);\n    \n    int64_t gtid_44879;\n    \n    gtid_44879 = sext_i32_i64(group_tid_52918) * segmap_group_sizze_45696 +\n        sext_i32_i64(local_tid_52917) - squot64(sext_i32_i64(group_tid_52918) *\n                                                segmap_group_sizze_45696 +\n                                                sext_i32_i64(local_tid_52917),\n                                                ydim_41932 * zzdim_41933) *\n        (ydim_41932 * zzdim_41933) - squot64(sext_i32_i64(group_tid_52918) *\n                                             segmap_group_sizze_45696 +\n                                             sext_i32_i64(local_tid_52917) -\n                                             squot64(sext_i32_i64(group_tid_52918) *\n                                                     segmap_group_sizze_45696 +\n                                                     sext_i32_i64(local_tid_52917),\n                                                     ydim_41932 * zzdim_41933) *\n                                             (ydim_41932 * zzdim_41933),\n                                             zzdim_41933) * zzdim",
                   "_41933;\n    if ((slt64(gtid_44877, xdim_41931) && slt64(gtid_44878, ydim_41932)) &&\n        slt64(gtid_44879, zzdim_41933)) {\n        bool binop_x_52364 = sle64(2, gtid_44878);\n        bool binop_x_52365 = sle64(2, gtid_44877);\n        bool binop_y_52366 = slt64(gtid_44877, y_42127);\n        bool binop_y_52367 = binop_x_52365 && binop_y_52366;\n        bool binop_x_52368 = binop_x_52364 && binop_y_52367;\n        bool binop_y_52369 = slt64(gtid_44878, y_42128);\n        bool index_primexp_52370 = binop_x_52368 && binop_y_52369;\n        double lifted_0_f_res_45706;\n        double lifted_0_f_res_45707;\n        double lifted_0_f_res_45708;\n        double lifted_0_f_res_45709;\n        \n        if (index_primexp_52370) {\n            double tke_45722 = ((__global\n                                 double *) tketau_mem_52436)[gtid_44877 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_44878 *\n                                                             zzdim_41933 +\n                                                             gtid_44879];\n            double max_res_45723 = fmax64(0.0, tke_45722);\n            double sqrt_res_45724;\n            \n            sqrt_res_45724 = futrts_sqrt64(max_res_45723);\n            \n            int32_t x_45727 = ((__global int32_t *) kbot_mem_52456)[gtid_44877 *\n                                                                    ydim_41976 +\n                                                                    gtid_44878];\n            int32_t ks_val_45728 = sub32(x_45727, 1);\n            bool land_mask_45729 = sle32(0, ks_val_45728);\n            int32_t i64_res_45730 = sext_i64_i32(gtid_44879);\n            bool edge_mask_t_res_45731 = i64_res_45730 == ks_val_45728;\n            bool x_45732 = land_mask_45729 && edge_mask_t_res_45731;\n            bool water_mask_t_res_45733 = sle32(",
                   "ks_val_45728, i64_res_45730);\n            bool x_45734 = land_mask_45729 && water_mask_t_res_45733;\n            double kappa_45735 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_44877 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_44878 *\n                                                               zzdim_41979 +\n                                                               gtid_44879];\n            bool cond_45736 = slt64(0, gtid_44879);\n            double deltam1_45737;\n            \n            if (cond_45736) {\n                double y_45739 = ((__global\n                                   double *) dzzt_mem_52452)[gtid_44879];\n                double x_45740 = 1.0 / y_45739;\n                double x_45741 = 0.5 * x_45740;\n                int64_t i_45742 = sub64(gtid_44879, 1);\n                bool x_45743 = sle64(0, i_45742);\n                bool y_45744 = slt64(i_45742, zzdim_41933);\n                bool bounds_check_45745 = x_45743 && y_45744;\n                bool index_certs_45748;\n                \n                if (!bounds_check_45745) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 18) ==\n                            -1) {\n                            global_failure_args[0] = gtid_44877;\n                            global_failure_args[1] = gtid_44878;\n                            global_failure_args[2] = i_45742;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double x_45749 = ((__global\n          ",
                   "                         double *) kappaM_mem_52457)[gtid_44877 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_44878 *\n                                                               zzdim_41979 +\n                                                               i_45742];\n                double y_45750 = kappa_45735 + x_45749;\n                double deltam1_t_res_45751 = x_45741 * y_45750;\n                \n                deltam1_45737 = deltam1_t_res_45751;\n            } else {\n                deltam1_45737 = 0.0;\n            }\n            \n            int64_t y_45752 = sub64(zzdim_41933, 1);\n            bool cond_45753 = slt64(gtid_44879, y_45752);\n            double delta_45754;\n            \n            if (cond_45753) {\n                int64_t i_45755 = add64(1, gtid_44879);\n                bool x_45756 = sle64(0, i_45755);\n                bool y_45757 = slt64(i_45755, zzdim_41933);\n                bool bounds_check_45758 = x_45756 && y_45757;\n                bool index_certs_45759;\n                \n                if (!bounds_check_45758) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 19) ==\n                            -1) {\n                            global_failure_args[0] = i_45755;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double y_45760 = ((__global double *) dzzt_mem_52452)[i_45755];\n                double x_45761 = 1.0 / y_45760;\n                double x_45762 = 0.5 * x_45761;\n                bool index_certs_45765;\n                \n                if (!bounds_check_45758) {\n                    {\n                        if (atomic_cmpxchg_i32_global(g",
                   "lobal_failure, -1, 20) ==\n                            -1) {\n                            global_failure_args[0] = gtid_44877;\n                            global_failure_args[1] = gtid_44878;\n                            global_failure_args[2] = i_45755;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double y_45766 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_44877 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_44878 *\n                                                               zzdim_41979 +\n                                                               i_45755];\n                double y_45767 = kappa_45735 + y_45766;\n                double delta_t_res_45768 = x_45762 * y_45767;\n                \n                delta_45754 = delta_t_res_45768;\n            } else {\n                delta_45754 = 0.0;\n            }\n            \n            double dzzwzz_45770 = ((__global\n                                    double *) dzzw_mem_52453)[gtid_44879];\n            bool cond_f_res_45772 = !x_45734;\n            bool x_45773 = !x_45732;\n            bool y_45774 = cond_f_res_45772 && x_45773;\n            bool cond_45775 = x_45732 || y_45774;\n            double a_45776;\n            \n            if (cond_45775) {\n                a_45776 = 0.0;\n            } else {\n                bool x_45777 = cond_45736 && cond_45753;\n                double a_f_res_45778;\n                \n                if (x_45777) {\n                    double negate_arg_45779 = deltam1_45737 / d",
                   "zzwzz_45770;\n                    double a_f_res_t_res_45780 = 0.0 - negate_arg_45779;\n                    \n                    a_f_res_45778 = a_f_res_t_res_45780;\n                } else {\n                    bool cond_45781 = gtid_44879 == y_45752;\n                    double a_f_res_f_res_45782;\n                    \n                    if (cond_45781) {\n                        double y_45783 = 0.5 * dzzwzz_45770;\n                        double negate_arg_45784 = deltam1_45737 / y_45783;\n                        double a_f_res_f_res_t_res_45785 = 0.0 -\n                               negate_arg_45784;\n                        \n                        a_f_res_f_res_45782 = a_f_res_f_res_t_res_45785;\n                    } else {\n                        a_f_res_f_res_45782 = 0.0;\n                    }\n                    a_f_res_45778 = a_f_res_f_res_45782;\n                }\n                a_45776 = a_f_res_45778;\n            }\n            \n            double b_45786;\n            \n            if (cond_f_res_45772) {\n                b_45786 = 1.0;\n            } else {\n                double mxls_45771 = ((__global\n                                      double *) mxl_mem_52458)[gtid_44877 *\n                                                               (zzdim_41982 *\n                                                                ydim_41981) +\n                                                               gtid_44878 *\n                                                               zzdim_41982 +\n                                                               gtid_44879];\n                double b_f_res_45787;\n                \n                if (x_45732) {\n                    double y_45788 = delta_45754 / dzzwzz_45770;\n                    double x_45789 = 1.0 + y_45788;\n                    double x_45790 = 0.7 / mxls_45771;\n                    double y_45791 = sqrt_res_45724 * x_45790;\n                    double b_f_res_t_res_45792 = x_45789 + y_45791;\n                    \n   ",
                   "                 b_f_res_45787 = b_f_res_t_res_45792;\n                } else {\n                    bool x_45793 = cond_45736 && cond_45753;\n                    double b_f_res_f_res_45794;\n                    \n                    if (x_45793) {\n                        double x_45795 = deltam1_45737 + delta_45754;\n                        double y_45796 = x_45795 / dzzwzz_45770;\n                        double x_45797 = 1.0 + y_45796;\n                        double x_45798 = 0.7 * sqrt_res_45724;\n                        double y_45799 = x_45798 / mxls_45771;\n                        double b_f_res_f_res_t_res_45800 = x_45797 + y_45799;\n                        \n                        b_f_res_f_res_45794 = b_f_res_f_res_t_res_45800;\n                    } else {\n                        bool cond_45801 = gtid_44879 == y_45752;\n                        double b_f_res_f_res_f_res_45802;\n                        \n                        if (cond_45801) {\n                            double y_45803 = 0.5 * dzzwzz_45770;\n                            double y_45804 = deltam1_45737 / y_45803;\n                            double x_45805 = 1.0 + y_45804;\n                            double x_45806 = 0.7 / mxls_45771;\n                            double y_45807 = sqrt_res_45724 * x_45806;\n                            double b_f_res_f_res_f_res_t_res_45808 = x_45805 +\n                                   y_45807;\n                            \n                            b_f_res_f_res_f_res_45802 =\n                                b_f_res_f_res_f_res_t_res_45808;\n                        } else {\n                            b_f_res_f_res_f_res_45802 = 0.0;\n                        }\n                        b_f_res_f_res_45794 = b_f_res_f_res_f_res_45802;\n                    }\n                    b_f_res_45787 = b_f_res_f_res_45794;\n                }\n                b_45786 = b_f_res_45787;\n            }\n            \n            double lifted_0_f_res_t_res_45809;\n            double lifted_0_f_res_t_r",
                   "es_45810;\n            \n            if (cond_f_res_45772) {\n                lifted_0_f_res_t_res_45809 = 0.0;\n                lifted_0_f_res_t_res_45810 = 0.0;\n            } else {\n                double negate_arg_45811 = delta_45754 / dzzwzz_45770;\n                double c_45812 = 0.0 - negate_arg_45811;\n                double y_45813 = ((__global\n                                   double *) forc_mem_52459)[gtid_44877 *\n                                                             (zzdim_41985 *\n                                                              ydim_41984) +\n                                                             gtid_44878 *\n                                                             zzdim_41985 +\n                                                             gtid_44879];\n                double tmp_45814 = tke_45722 + y_45813;\n                bool cond_45815 = gtid_44879 == y_45752;\n                double lifted_0_f_res_t_res_f_res_45816;\n                \n                if (cond_45815) {\n                    double y_45817 = ((__global\n                                       double *) forc_tke_surface_mem_52460)[gtid_44877 *\n                                                                             ydim_41987 +\n                                                                             gtid_44878];\n                    double y_45818 = 0.5 * dzzwzz_45770;\n                    double y_45819 = y_45817 / y_45818;\n                    double lifted_0_f_res_t_res_f_res_t_res_45820 = tmp_45814 +\n                           y_45819;\n                    \n                    lifted_0_f_res_t_res_f_res_45816 =\n                        lifted_0_f_res_t_res_f_res_t_res_45820;\n                } else {\n                    lifted_0_f_res_t_res_f_res_45816 = tmp_45814;\n                }\n                lifted_0_f_res_t_res_45809 = c_45812;\n                lifted_0_f_res_t_res_45810 = lifted_0_f_res_t_res_f_res_45816;\n            }\n            lifted_0_f_res_45706 = a",
                   "_45776;\n            lifted_0_f_res_45707 = b_45786;\n            lifted_0_f_res_45708 = lifted_0_f_res_t_res_45809;\n            lifted_0_f_res_45709 = lifted_0_f_res_t_res_45810;\n        } else {\n            lifted_0_f_res_45706 = 0.0;\n            lifted_0_f_res_45707 = 0.0;\n            lifted_0_f_res_45708 = 0.0;\n            lifted_0_f_res_45709 = 0.0;\n        }\n        ((__global double *) mem_52620)[gtid_44877 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_44878 *\n                                        zzdim_41933 + gtid_44879] =\n            lifted_0_f_res_45706;\n        ((__global double *) mem_52624)[gtid_44877 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_44878 *\n                                        zzdim_41933 + gtid_44879] =\n            lifted_0_f_res_45707;\n        ((__global double *) mem_52628)[gtid_44877 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_44878 *\n                                        zzdim_41933 + gtid_44879] =\n            lifted_0_f_res_45708;\n        ((__global double *) mem_52632)[gtid_44877 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_44878 *\n                                        zzdim_41933 + gtid_44879] =\n            lifted_0_f_res_45709;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_45696\n}\n__kernel void integrate_tkezisegmap_46175(__global int *global_failure,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t ydim_41935,\n                                          int64_t zzdim_41936, int64_t y_42127,\n                                          int64_t y_42128, __global\n                                          unsigned char *tketaup1_mem_52437,\n             ",
                   "                             __global\n                                          unsigned char *lifted_11_map_res_mem_52749,\n                                          __global unsigned char *mem_52753,\n                                          __global unsigned char *mem_52755,\n                                          __global unsigned char *mem_52762)\n{\n    #define segmap_group_sizze_46402 (integrate_tkezisegmap_group_sizze_46179)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52963;\n    int32_t local_tid_52964;\n    int64_t group_sizze_52967;\n    int32_t wave_sizze_52966;\n    int32_t group_tid_52965;\n    \n    global_tid_52963 = get_global_id(0);\n    local_tid_52964 = get_local_id(0);\n    group_sizze_52967 = get_local_size(0);\n    wave_sizze_52966 = LOCKSTEP_WIDTH;\n    group_tid_52965 = get_group_id(0);\n    \n    int32_t phys_tid_46175;\n    \n    phys_tid_46175 = global_tid_52963;\n    \n    int64_t gtid_46172;\n    \n    gtid_46172 = squot64(sext_i32_i64(group_tid_52965) *\n                         segmap_group_sizze_46402 +\n                         sext_i32_i64(local_tid_52964), ydim_41932 *\n                         zzdim_41933);\n    \n    int64_t gtid_46173;\n    \n    gtid_46173 = squot64(sext_i32_i64(group_tid_52965) *\n                         segmap_group_sizze_46402 +\n                         sext_i32_i64(local_tid_52964) -\n                         squot64(sext_i32_i64(group_tid_52965) *\n                                 segmap_group_sizze_46402 +\n                                 sext_i32_i64(local_tid_52964), ydim_41932 *\n                                 zzdim_41933) * (ydim_41932 * zzdim_41933),\n                         zzdim_41933);\n    \n    int64_t gtid_46174;\n    \n    gtid_46174 = sext_i32_i64(group_tid_52965) * segmap_group_sizze_46402 +\n        sext_i32_i64(local_tid_52964) - squot64(sext_i32_i64(group_tid_52965) *\n                       ",
                   "                         segmap_group_sizze_46402 +\n                                                sext_i32_i64(local_tid_52964),\n                                                ydim_41932 * zzdim_41933) *\n        (ydim_41932 * zzdim_41933) - squot64(sext_i32_i64(group_tid_52965) *\n                                             segmap_group_sizze_46402 +\n                                             sext_i32_i64(local_tid_52964) -\n                                             squot64(sext_i32_i64(group_tid_52965) *\n                                                     segmap_group_sizze_46402 +\n                                                     sext_i32_i64(local_tid_52964),\n                                                     ydim_41932 * zzdim_41933) *\n                                             (ydim_41932 * zzdim_41933),\n                                             zzdim_41933) * zzdim_41933;\n    if ((slt64(gtid_46172, xdim_41931) && slt64(gtid_46173, ydim_41932)) &&\n        slt64(gtid_46174, zzdim_41933)) {\n        int32_t ks_val_46409 = ((__global int32_t *) mem_52753)[gtid_46172 *\n                                                                ydim_41932 +\n                                                                gtid_46173];\n        bool cond_46410 = ((__global bool *) mem_52755)[gtid_46172 *\n                                                        ydim_41932 +\n                                                        gtid_46173];\n        bool binop_x_52371 = sle64(2, gtid_46173);\n        bool binop_x_52372 = sle64(2, gtid_46172);\n        bool binop_y_52373 = slt64(gtid_46172, y_42127);\n        bool binop_y_52374 = binop_x_52372 && binop_y_52373;\n        bool binop_x_52375 = binop_x_52371 && binop_y_52374;\n        bool binop_y_52376 = slt64(gtid_46173, y_42128);\n        bool index_primexp_52377 = binop_x_52375 && binop_y_52376;\n        int32_t i64_res_46413 = sext_i64_i32(gtid_46174);\n        bool water_mask_t_res_46414 = sle32(ks_val_46409, i64_res_46413)",
                   ";\n        bool x_46415 = cond_46410 && water_mask_t_res_46414;\n        bool x_46416 = x_46415 && index_primexp_52377;\n        double lifted_0_f_res_46417;\n        \n        if (x_46416) {\n            double lifted_0_f_res_t_res_46424 = ((__global\n                                                  double *) lifted_11_map_res_mem_52749)[gtid_46172 *\n                                                                                         (zzdim_41933 *\n                                                                                          ydim_41932) +\n                                                                                         gtid_46173 *\n                                                                                         zzdim_41933 +\n                                                                                         gtid_46174];\n            \n            lifted_0_f_res_46417 = lifted_0_f_res_t_res_46424;\n        } else {\n            double lifted_0_f_res_f_res_46431 = ((__global\n                                                  double *) tketaup1_mem_52437)[gtid_46172 *\n                                                                                (zzdim_41936 *\n                                                                                 ydim_41935) +\n                                                                                gtid_46173 *\n                                                                                zzdim_41936 +\n                                                                                gtid_46174];\n            \n            lifted_0_f_res_46417 = lifted_0_f_res_f_res_46431;\n        }\n        ((__global double *) mem_52762)[gtid_46172 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_46173 *\n                                        zzdim_41933 + gtid_46174] =\n            lifted_0_f_res_46417;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_46402\n}\n__kernel void",
                   " integrate_tkezisegmap_46244(__global int *global_failure,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t ydim_41976, int64_t y_42127,\n                                          int64_t y_42128, __global\n                                          unsigned char *kbot_mem_52456,\n                                          __global unsigned char *mem_52753,\n                                          __global unsigned char *mem_52755,\n                                          __global unsigned char *mem_52757)\n{\n    #define segmap_group_sizze_46368 (integrate_tkezisegmap_group_sizze_46247)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52958;\n    int32_t local_tid_52959;\n    int64_t group_sizze_52962;\n    int32_t wave_sizze_52961;\n    int32_t group_tid_52960;\n    \n    global_tid_52958 = get_global_id(0);\n    local_tid_52959 = get_local_id(0);\n    group_sizze_52962 = get_local_size(0);\n    wave_sizze_52961 = LOCKSTEP_WIDTH;\n    group_tid_52960 = get_group_id(0);\n    \n    int32_t phys_tid_46244;\n    \n    phys_tid_46244 = global_tid_52958;\n    \n    int64_t gtid_46242;\n    \n    gtid_46242 = squot64(sext_i32_i64(group_tid_52960) *\n                         segmap_group_sizze_46368 +\n                         sext_i32_i64(local_tid_52959), ydim_41932);\n    \n    int64_t gtid_46243;\n    \n    gtid_46243 = sext_i32_i64(group_tid_52960) * segmap_group_sizze_46368 +\n        sext_i32_i64(local_tid_52959) - squot64(sext_i32_i64(group_tid_52960) *\n                                                segmap_group_sizze_46368 +\n                                                sext_i32_i64(local_tid_52959),\n                                                ydim_41932) * ydim_41932;\n    if (slt64(gtid_46242, xdim_41931) && slt64(gtid_46243, ydim_41932)) ",
                   "{\n        bool binop_x_52208 = sle64(2, gtid_46242);\n        bool binop_y_52211 = slt64(gtid_46242, y_42127);\n        bool index_primexp_52212 = binop_x_52208 && binop_y_52211;\n        int32_t x_46383 = ((__global int32_t *) kbot_mem_52456)[gtid_46242 *\n                                                                ydim_41976 +\n                                                                gtid_46243];\n        int32_t ks_val_46384 = sub32(x_46383, 1);\n        bool cond_46385 = sle32(0, ks_val_46384);\n        bool cond_t_res_46386 = sle64(2, gtid_46243);\n        bool x_46387 = cond_t_res_46386 && index_primexp_52212;\n        bool cond_t_res_46388 = slt64(gtid_46243, y_42128);\n        bool x_46389 = x_46387 && cond_t_res_46388;\n        \n        ((__global int32_t *) mem_52753)[gtid_46242 * ydim_41932 + gtid_46243] =\n            ks_val_46384;\n        ((__global bool *) mem_52755)[gtid_46242 * ydim_41932 + gtid_46243] =\n            cond_46385;\n        ((__global bool *) mem_52757)[gtid_46242 * ydim_41932 + gtid_46243] =\n            x_46389;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_46368\n}\n__kernel void integrate_tkezisegmap_46990(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t ydim_41959,\n                                          int64_t zzdim_41960,\n                                          int64_t ydim_41962,\n                                          int64_t zzdim_41963, int64_t y_42127,\n                                          int64_t y_42128,\n                                          int64_t distance_42130,\n                                          int64_t y_42385, int64_t y_4238",
                   "6,\n                                          __global\n                                          unsigned char *tketau_mem_52436,\n                                          __global\n                                          unsigned char *maskU_mem_52445,\n                                          __global\n                                          unsigned char *maskV_mem_52446,\n                                          __global unsigned char *dxu_mem_52449,\n                                          __global unsigned char *dyu_mem_52451,\n                                          __global\n                                          unsigned char *cost_mem_52454,\n                                          __global\n                                          unsigned char *cosu_mem_52455,\n                                          __global unsigned char *mem_52762,\n                                          __global unsigned char *mem_52779,\n                                          __global unsigned char *mem_52783,\n                                          __global unsigned char *mem_52787)\n{\n    #define segmap_group_sizze_47475 (integrate_tkezisegmap_group_sizze_46994)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52973;\n    int32_t local_tid_52974;\n    int64_t group_sizze_52977;\n    int32_t wave_sizze_52976;\n    int32_t group_tid_52975;\n    \n    global_tid_52973 = get_global_id(0);\n    local_tid_52974 = get_local_id(0);\n    group_sizze_52977 = get_local_size(0);\n    wave_sizze_52976 = LOCKSTEP_WIDTH;\n    group_tid_52975 = get_group_id(0);\n    \n    int32_t phys_tid_46990;\n    \n    phys_tid_46990 = global_tid_52973;\n    \n    int64_t gtid_46987;\n    \n    gtid_46987 = squot64(sext_i32_i64(group_tid_52975) *\n                         segmap_group_sizze_47475 +\n                         sext_i32_i64(local_tid_52974), ydim_41932 *\n                         zz",
                   "dim_41933);\n    \n    int64_t gtid_46988;\n    \n    gtid_46988 = squot64(sext_i32_i64(group_tid_52975) *\n                         segmap_group_sizze_47475 +\n                         sext_i32_i64(local_tid_52974) -\n                         squot64(sext_i32_i64(group_tid_52975) *\n                                 segmap_group_sizze_47475 +\n                                 sext_i32_i64(local_tid_52974), ydim_41932 *\n                                 zzdim_41933) * (ydim_41932 * zzdim_41933),\n                         zzdim_41933);\n    \n    int64_t gtid_46989;\n    \n    gtid_46989 = sext_i32_i64(group_tid_52975) * segmap_group_sizze_47475 +\n        sext_i32_i64(local_tid_52974) - squot64(sext_i32_i64(group_tid_52975) *\n                                                segmap_group_sizze_47475 +\n                                                sext_i32_i64(local_tid_52974),\n                                                ydim_41932 * zzdim_41933) *\n        (ydim_41932 * zzdim_41933) - squot64(sext_i32_i64(group_tid_52975) *\n                                             segmap_group_sizze_47475 +\n                                             sext_i32_i64(local_tid_52974) -\n                                             squot64(sext_i32_i64(group_tid_52975) *\n                                                     segmap_group_sizze_47475 +\n                                                     sext_i32_i64(local_tid_52974),\n                                                     ydim_41932 * zzdim_41933) *\n                                             (ydim_41932 * zzdim_41933),\n                                             zzdim_41933) * zzdim_41933;\n    if ((slt64(gtid_46987, xdim_41931) && slt64(gtid_46988, ydim_41932)) &&\n        slt64(gtid_46989, zzdim_41933)) {\n        bool index_primexp_52264 = slt64(gtid_46987, y_42385);\n        bool binop_x_52378 = sle64(2, gtid_46988);\n        bool binop_x_52379 = sle64(2, gtid_46987);\n        bool binop_y_52380 = slt64(gtid_46987, y_42127);\n        b",
                   "ool binop_y_52381 = binop_x_52379 && binop_y_52380;\n        bool binop_x_52382 = binop_x_52378 && binop_y_52381;\n        bool binop_y_52383 = slt64(gtid_46988, y_42128);\n        bool index_primexp_52384 = binop_x_52382 && binop_y_52383;\n        bool index_primexp_52259 = slt64(gtid_46988, y_42386);\n        bool cond_t_res_47486 = gtid_46989 == distance_42130;\n        bool x_47487 = cond_t_res_47486 && index_primexp_52384;\n        double lifted_0_f_res_47488;\n        \n        if (x_47487) {\n            double tke_val_47501 = ((__global double *) mem_52762)[gtid_46987 *\n                                                                   (zzdim_41933 *\n                                                                    ydim_41932) +\n                                                                   gtid_46988 *\n                                                                   zzdim_41933 +\n                                                                   gtid_46989];\n            bool cond_47502 = tke_val_47501 < 0.0;\n            double lifted_0_f_res_t_res_47503;\n            \n            if (cond_47502) {\n                lifted_0_f_res_t_res_47503 = 0.0;\n            } else {\n                lifted_0_f_res_t_res_47503 = tke_val_47501;\n            }\n            lifted_0_f_res_47488 = lifted_0_f_res_t_res_47503;\n        } else {\n            double lifted_0_f_res_f_res_47516 = ((__global\n                                                  double *) mem_52762)[gtid_46987 *\n                                                                       (zzdim_41933 *\n                                                                        ydim_41932) +\n                                                                       gtid_46988 *\n                                                                       zzdim_41933 +\n                                                                       gtid_46989];\n            \n            lifted_0_f_res_47488 = lifted_0_f_res_f_res_47516;\n        }\n",
                   "        \n        double lifted_0_f_res_47517;\n        \n        if (index_primexp_52264) {\n            int64_t i_47518 = add64(1, gtid_46987);\n            bool x_47519 = sle64(0, i_47518);\n            bool y_47520 = slt64(i_47518, xdim_41931);\n            bool bounds_check_47521 = x_47519 && y_47520;\n            bool index_certs_47530;\n            \n            if (!bounds_check_47521) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 29) ==\n                        -1) {\n                        global_failure_args[0] = i_47518;\n                        global_failure_args[1] = gtid_46988;\n                        global_failure_args[2] = gtid_46989;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_47531 = ((__global double *) tketau_mem_52436)[i_47518 *\n                                                                    (zzdim_41933 *\n                                                                     ydim_41932) +\n                                                                    gtid_46988 *\n                                                                    zzdim_41933 +\n                                                                    gtid_46989];\n            double y_47538 = ((__global double *) tketau_mem_52436)[gtid_46987 *\n                                                                    (zzdim_41933 *\n                                                                     ydim_41932) +\n                                                                    gtid_46988 *\n                                                                    zzdim_41933 +\n                                                                    gtid_46989];\n            double y_4",
                   "7539 = x_47531 - y_47538;\n            double x_47540 = 2000.0 * y_47539;\n            double x_47542 = ((__global double *) cost_mem_52454)[gtid_46988];\n            double y_47544 = ((__global double *) dxu_mem_52449)[gtid_46987];\n            double y_47545 = x_47542 * y_47544;\n            double x_47546 = x_47540 / y_47545;\n            double y_47547 = ((__global double *) maskU_mem_52445)[gtid_46987 *\n                                                                   (zzdim_41960 *\n                                                                    ydim_41959) +\n                                                                   gtid_46988 *\n                                                                   zzdim_41960 +\n                                                                   gtid_46989];\n            double lifted_0_f_res_t_res_47548 = x_47546 * y_47547;\n            \n            lifted_0_f_res_47517 = lifted_0_f_res_t_res_47548;\n        } else {\n            lifted_0_f_res_47517 = 0.0;\n        }\n        \n        double lifted_0_f_res_47549;\n        \n        if (index_primexp_52259) {\n            int64_t i_47553 = add64(1, gtid_46988);\n            bool x_47554 = sle64(0, i_47553);\n            bool y_47555 = slt64(i_47553, ydim_41932);\n            bool bounds_check_47556 = x_47554 && y_47555;\n            bool index_certs_47562;\n            \n            if (!bounds_check_47556) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 30) ==\n                        -1) {\n                        global_failure_args[0] = gtid_46987;\n                        global_failure_args[1] = i_47553;\n                        global_failure_args[2] = gtid_46989;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n       ",
                   "     }\n            \n            double x_47563 = ((__global double *) tketau_mem_52436)[gtid_46987 *\n                                                                    (zzdim_41933 *\n                                                                     ydim_41932) +\n                                                                    i_47553 *\n                                                                    zzdim_41933 +\n                                                                    gtid_46989];\n            double y_47569 = ((__global double *) tketau_mem_52436)[gtid_46987 *\n                                                                    (zzdim_41933 *\n                                                                     ydim_41932) +\n                                                                    gtid_46988 *\n                                                                    zzdim_41933 +\n                                                                    gtid_46989];\n            double y_47570 = x_47563 - y_47569;\n            double x_47571 = 2000.0 * y_47570;\n            double y_47573 = ((__global double *) dyu_mem_52451)[gtid_46988];\n            double x_47574 = x_47571 / y_47573;\n            double y_47575 = ((__global double *) maskV_mem_52446)[gtid_46987 *\n                                                                   (zzdim_41963 *\n                                                                    ydim_41962) +\n                                                                   gtid_46988 *\n                                                                   zzdim_41963 +\n                                                                   gtid_46989];\n            double x_47576 = x_47574 * y_47575;\n            double y_47577 = ((__global double *) cosu_mem_52455)[gtid_46988];\n            double lifted_0_f_res_t_res_47578 = x_47576 * y_47577;\n            \n            lifted_0_f_res_47549 = lifted_0_f_res_t_res_47578;\n        } else {\n        ",
                   "    lifted_0_f_res_47549 = 0.0;\n        }\n        ((__global double *) mem_52779)[gtid_46987 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_46988 *\n                                        zzdim_41933 + gtid_46989] =\n            lifted_0_f_res_47549;\n        ((__global double *) mem_52783)[gtid_46987 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_46988 *\n                                        zzdim_41933 + gtid_46989] =\n            lifted_0_f_res_47517;\n        ((__global double *) mem_52787)[gtid_46987 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_46988 *\n                                        zzdim_41933 + gtid_46989] =\n            lifted_0_f_res_47488;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_47475\n}\n__kernel void integrate_tkezisegmap_47205(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933, int64_t y_42127,\n                                          int64_t y_42128, int64_t y_42386,\n                                          __global\n                                          unsigned char *dzzw_mem_52453,\n                                          __global unsigned char *mem_52766,\n                                          __global unsigned char *mem_52769,\n                                          __global unsigned char *mem_52771,\n                                          __global unsigned char *mem_52774)\n{\n    #define segmap_group_sizze_47428 (integrate_tkezisegmap_group_sizze_47208)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_fai",
                   "lure >= 0)\n        return;\n    \n    int32_t global_tid_52968;\n    int32_t local_tid_52969;\n    int64_t group_sizze_52972;\n    int32_t wave_sizze_52971;\n    int32_t group_tid_52970;\n    \n    global_tid_52968 = get_global_id(0);\n    local_tid_52969 = get_local_id(0);\n    group_sizze_52972 = get_local_size(0);\n    wave_sizze_52971 = LOCKSTEP_WIDTH;\n    group_tid_52970 = get_group_id(0);\n    \n    int32_t phys_tid_47205;\n    \n    phys_tid_47205 = global_tid_52968;\n    \n    int64_t gtid_47203;\n    \n    gtid_47203 = squot64(sext_i32_i64(group_tid_52970) *\n                         segmap_group_sizze_47428 +\n                         sext_i32_i64(local_tid_52969), ydim_41932);\n    \n    int64_t gtid_47204;\n    \n    gtid_47204 = sext_i32_i64(group_tid_52970) * segmap_group_sizze_47428 +\n        sext_i32_i64(local_tid_52969) - squot64(sext_i32_i64(group_tid_52970) *\n                                                segmap_group_sizze_47428 +\n                                                sext_i32_i64(local_tid_52969),\n                                                ydim_41932) * ydim_41932;\n    if (slt64(gtid_47203, xdim_41931) && slt64(gtid_47204, ydim_41932)) {\n        bool binop_x_52248 = sle64(2, gtid_47203);\n        bool binop_y_52251 = slt64(gtid_47203, y_42127);\n        bool index_primexp_52252 = binop_x_52248 && binop_y_52251;\n        bool cond_t_res_47436 = sle64(2, gtid_47204);\n        bool x_47437 = cond_t_res_47436 && index_primexp_52252;\n        bool cond_t_res_47438 = slt64(gtid_47204, y_42128);\n        bool x_47439 = x_47437 && cond_t_res_47438;\n        double lifted_0_f_res_47440;\n        \n        if (x_47439) {\n            int64_t i_47447 = sub64(zzdim_41933, 1);\n            bool x_47448 = sle64(0, i_47447);\n            bool y_47449 = slt64(i_47447, zzdim_41933);\n            bool bounds_check_47450 = x_47448 && y_47449;\n            bool index_certs_47453;\n            \n            if (!bounds_check_47450) {\n                {\n                    if (atomic_cmpxchg_",
                   "i32_global(global_failure, -1, 27) ==\n                        -1) {\n                        global_failure_args[0] = gtid_47203;\n                        global_failure_args[1] = gtid_47204;\n                        global_failure_args[2] = i_47447;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double tke_val_47454 = ((__global double *) mem_52766)[i_47447 *\n                                                                   (ydim_41932 *\n                                                                    xdim_41931) +\n                                                                   gtid_47203 *\n                                                                   ydim_41932 +\n                                                                   gtid_47204];\n            bool cond_47455 = tke_val_47454 < 0.0;\n            double lifted_0_f_res_t_res_47456;\n            \n            if (cond_47455) {\n                double x_47457 = 0.5 * tke_val_47454;\n                bool index_certs_47458;\n                \n                if (!bounds_check_47450) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 28) ==\n                            -1) {\n                            global_failure_args[0] = i_47447;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double y_47459 = ((__global double *) dzzw_mem_52453)[i_47447];\n                double x_47460 = x_47457 * y_47459;\n                double lifted_0_f_res_t_res_t_res_47461 = 0.0 - x_47460;\n                \n                lifted_0_f_res_t_res_474",
                   "56 = lifted_0_f_res_t_res_t_res_47461;\n            } else {\n                lifted_0_f_res_t_res_47456 = 0.0;\n            }\n            lifted_0_f_res_47440 = lifted_0_f_res_t_res_47456;\n        } else {\n            lifted_0_f_res_47440 = 0.0;\n        }\n        \n        bool cond_47462 = slt64(gtid_47204, y_42386);\n        \n        ((__global bool *) mem_52769)[gtid_47203 * ydim_41932 + gtid_47204] =\n            x_47439;\n        ((__global bool *) mem_52771)[gtid_47203 * ydim_41932 + gtid_47204] =\n            cond_47462;\n        ((__global double *) mem_52774)[gtid_47203 * ydim_41932 + gtid_47204] =\n            lifted_0_f_res_47440;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_47428\n}\n__kernel void integrate_tkezisegmap_49040(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t ydim_41950,\n                                          int64_t zzdim_41951,\n                                          int64_t ydim_41953,\n                                          int64_t zzdim_41954,\n                                          int64_t ydim_41956,\n                                          int64_t zzdim_41957,\n                                          int64_t ydim_41965,\n                                          int64_t zzdim_41966, int64_t y_42127,\n                                          int64_t y_42128,\n                                          int64_t distance_42130, __global\n                                          unsigned char *tketau_mem_52436,\n                                          __global\n                                          unsigned char *utau_mem_52442,\n                             ",
                   "             __global\n                                          unsigned char *vtau_mem_52443,\n                                          __global\n                                          unsigned char *wtau_mem_52444,\n                                          __global\n                                          unsigned char *maskW_mem_52447,\n                                          __global unsigned char *dxt_mem_52448,\n                                          __global unsigned char *dyt_mem_52450,\n                                          __global\n                                          unsigned char *dzzw_mem_52453,\n                                          __global\n                                          unsigned char *cost_mem_52454,\n                                          __global\n                                          unsigned char *cosu_mem_52455,\n                                          __global unsigned char *mem_52779,\n                                          __global unsigned char *mem_52783,\n                                          __global unsigned char *mem_52787,\n                                          __global unsigned char *mem_52792,\n                                          __global unsigned char *mem_52796,\n                                          __global unsigned char *mem_52800,\n                                          __global unsigned char *mem_52804)\n{\n    #define segmap_group_sizze_50208 (integrate_tkezisegmap_group_sizze_49044)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52978;\n    int32_t local_tid_52979;\n    int64_t group_sizze_52982;\n    int32_t wave_sizze_52981;\n    int32_t group_tid_52980;\n    \n    global_tid_52978 = get_global_id(0);\n    local_tid_52979 = get_local_id(0);\n    group_sizze_52982 = get_local_size(0);\n    wave_sizze_52981 = LOCKSTEP_WIDTH;\n    group_tid_52980 = get_group_id(",
                   "0);\n    \n    int32_t phys_tid_49040;\n    \n    phys_tid_49040 = global_tid_52978;\n    \n    int64_t gtid_49037;\n    \n    gtid_49037 = squot64(sext_i32_i64(group_tid_52980) *\n                         segmap_group_sizze_50208 +\n                         sext_i32_i64(local_tid_52979), ydim_41932 *\n                         zzdim_41933);\n    \n    int64_t gtid_49038;\n    \n    gtid_49038 = squot64(sext_i32_i64(group_tid_52980) *\n                         segmap_group_sizze_50208 +\n                         sext_i32_i64(local_tid_52979) -\n                         squot64(sext_i32_i64(group_tid_52980) *\n                                 segmap_group_sizze_50208 +\n                                 sext_i32_i64(local_tid_52979), ydim_41932 *\n                                 zzdim_41933) * (ydim_41932 * zzdim_41933),\n                         zzdim_41933);\n    \n    int64_t gtid_49039;\n    \n    gtid_49039 = sext_i32_i64(group_tid_52980) * segmap_group_sizze_50208 +\n        sext_i32_i64(local_tid_52979) - squot64(sext_i32_i64(group_tid_52980) *\n                                                segmap_group_sizze_50208 +\n                                                sext_i32_i64(local_tid_52979),\n                                                ydim_41932 * zzdim_41933) *\n        (ydim_41932 * zzdim_41933) - squot64(sext_i32_i64(group_tid_52980) *\n                                             segmap_group_sizze_50208 +\n                                             sext_i32_i64(local_tid_52979) -\n                                             squot64(sext_i32_i64(group_tid_52980) *\n                                                     segmap_group_sizze_50208 +\n                                                     sext_i32_i64(local_tid_52979),\n                                                     ydim_41932 * zzdim_41933) *\n                                             (ydim_41932 * zzdim_41933),\n                                             zzdim_41933) * zzdim_41933;\n    if ((slt64(gtid_49037, xd",
                   "im_41931) && slt64(gtid_49038, ydim_41932)) &&\n        slt64(gtid_49039, zzdim_41933)) {\n        bool index_primexp_52313 = sle64(2, gtid_49037);\n        bool index_primexp_52310 = slt64(gtid_49037, y_42127);\n        bool index_primexp_52298 = sle64(2, gtid_49038);\n        bool index_primexp_52295 = slt64(gtid_49038, y_42128);\n        bool binop_y_52402 = index_primexp_52310 && index_primexp_52313;\n        bool binop_x_52403 = index_primexp_52298 && binop_y_52402;\n        bool index_primexp_52405 = index_primexp_52295 && binop_x_52403;\n        bool binop_y_52395 = sle64(1, gtid_49037);\n        bool binop_y_52396 = index_primexp_52310 && binop_y_52395;\n        bool binop_y_52397 = index_primexp_52298 && binop_y_52396;\n        bool index_primexp_52398 = index_primexp_52295 && binop_y_52397;\n        bool binop_y_52386 = sle64(1, gtid_49038);\n        bool binop_x_52387 = index_primexp_52295 && binop_y_52386;\n        bool binop_x_52389 = index_primexp_52313 && binop_x_52387;\n        bool index_primexp_52391 = index_primexp_52310 && binop_x_52389;\n        double previous_50232 = ((__global double *) mem_52787)[gtid_49037 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                gtid_49038 *\n                                                                zzdim_41933 +\n                                                                gtid_49039];\n        bool cond_t_res_50233 = gtid_49039 == distance_42130;\n        bool x_50234 = cond_t_res_50233 && index_primexp_52405;\n        double lifted_0_f_res_50235;\n        \n        if (x_50234) {\n            double y_50245 = ((__global double *) maskW_mem_52447)[gtid_49037 *\n                                                                   (zzdim_41966 *\n                                                                    ydim_41965) +\n                                    ",
                   "                               gtid_49038 *\n                                                                   zzdim_41966 +\n                                                                   gtid_49039];\n            double x_50246 = ((__global double *) mem_52783)[gtid_49037 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_49038 *\n                                                             zzdim_41933 +\n                                                             gtid_49039];\n            int64_t i_50247 = sub64(gtid_49037, 1);\n            bool x_50248 = sle64(0, i_50247);\n            bool y_50249 = slt64(i_50247, xdim_41931);\n            bool bounds_check_50250 = x_50248 && y_50249;\n            bool index_certs_50253;\n            \n            if (!bounds_check_50250) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 31) ==\n                        -1) {\n                        global_failure_args[0] = i_50247;\n                        global_failure_args[1] = gtid_49038;\n                        global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_50254 = ((__global double *) mem_52783)[i_50247 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_49038 *\n                                                             zzdim_41933 +\n                                                             gtid",
                   "_49039];\n            double x_50255 = x_50246 - y_50254;\n            double x_50257 = ((__global double *) cost_mem_52454)[gtid_49038];\n            double y_50259 = ((__global double *) dxt_mem_52448)[gtid_49037];\n            double y_50260 = x_50257 * y_50259;\n            double x_50261 = x_50255 / y_50260;\n            double x_50262 = ((__global double *) mem_52779)[gtid_49037 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_49038 *\n                                                             zzdim_41933 +\n                                                             gtid_49039];\n            int64_t i_50263 = sub64(gtid_49038, 1);\n            bool x_50264 = sle64(0, i_50263);\n            bool y_50265 = slt64(i_50263, ydim_41932);\n            bool bounds_check_50266 = x_50264 && y_50265;\n            bool index_certs_50268;\n            \n            if (!bounds_check_50266) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 32) ==\n                        -1) {\n                        global_failure_args[0] = gtid_49037;\n                        global_failure_args[1] = i_50263;\n                        global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_50269 = ((__global double *) mem_52779)[gtid_49037 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             i_50263 *\n                                   ",
                   "                          zzdim_41933 +\n                                                             gtid_49039];\n            double x_50270 = x_50262 - y_50269;\n            double y_50271 = ((__global double *) dyt_mem_52450)[gtid_49038];\n            double y_50272 = x_50257 * y_50271;\n            double y_50273 = x_50270 / y_50272;\n            double y_50274 = x_50261 + y_50273;\n            double y_50275 = y_50245 * y_50274;\n            double lifted_0_f_res_t_res_50276 = previous_50232 + y_50275;\n            \n            lifted_0_f_res_50235 = lifted_0_f_res_t_res_50276;\n        } else {\n            lifted_0_f_res_50235 = previous_50232;\n        }\n        \n        double lifted_0_f_res_50277;\n        \n        if (index_primexp_52398) {\n            double x_50282 = ((__global double *) cost_mem_52454)[gtid_49038];\n            double y_50287 = ((__global double *) dxt_mem_52448)[gtid_49037];\n            double dx_50288 = x_50282 * y_50287;\n            double velS_50292 = ((__global\n                                  double *) utau_mem_52442)[gtid_49037 *\n                                                            (zzdim_41951 *\n                                                             ydim_41950) +\n                                                            gtid_49038 *\n                                                            zzdim_41951 +\n                                                            gtid_49039];\n            int64_t i_50293 = sub64(gtid_49037, 1);\n            bool x_50294 = sle64(0, i_50293);\n            bool y_50295 = slt64(i_50293, xdim_41931);\n            bool bounds_check_50296 = x_50294 && y_50295;\n            bool index_certs_50299;\n            \n            if (!bounds_check_50296) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 33) ==\n                        -1) {\n                        global_failure_args[0] = i_50293;\n                        global_failure_args[1] = gtid_49038;\n               ",
                   "         global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double maskWs_50301 = ((__global\n                                    double *) maskW_mem_52447)[gtid_49037 *\n                                                               (zzdim_41966 *\n                                                                ydim_41965) +\n                                                               gtid_49038 *\n                                                               zzdim_41966 +\n                                                               gtid_49039];\n            int64_t i_50302 = add64(1, gtid_49037);\n            bool x_50303 = sle64(0, i_50302);\n            bool y_50304 = slt64(i_50302, xdim_41931);\n            bool bounds_check_50305 = x_50303 && y_50304;\n            bool index_certs_50308;\n            \n            if (!bounds_check_50305) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 34) ==\n                        -1) {\n                        global_failure_args[0] = i_50302;\n                        global_failure_args[1] = gtid_49038;\n                        global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double maskWp1_50309 = ((__global\n                                     double *) maskW_mem_52447)[i_50302 *\n                                                                (zzdim_41966 *\n                                   ",
                   "                              ydim_41965) +\n                                                                gtid_49038 *\n                                                                zzdim_41966 +\n                                                                gtid_49039];\n            int64_t i_50310 = add64(2, gtid_49037);\n            bool x_50311 = sle64(0, i_50310);\n            bool y_50312 = slt64(i_50310, xdim_41931);\n            bool bounds_check_50313 = x_50311 && y_50312;\n            bool index_certs_50316;\n            \n            if (!bounds_check_50313) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 35) ==\n                        -1) {\n                        global_failure_args[0] = i_50310;\n                        global_failure_args[1] = gtid_49038;\n                        global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double varSM1_50318 = ((__global\n                                    double *) tketau_mem_52436)[i_50293 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                gtid_49038 *\n                                                                zzdim_41933 +\n                                                                gtid_49039];\n            double varS_50319 = ((__global\n                                  double *) tketau_mem_52436)[gtid_49037 *\n                                                              (zzdim_41933 *\n                                                               ydim_41932) +\n                                  ",
                   "                            gtid_49038 *\n                                                              zzdim_41933 +\n                                                              gtid_49039];\n            double varSP1_50320 = ((__global\n                                    double *) tketau_mem_52436)[i_50302 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                gtid_49038 *\n                                                                zzdim_41933 +\n                                                                gtid_49039];\n            double varSP2_50321 = ((__global\n                                    double *) tketau_mem_52436)[i_50310 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                gtid_49038 *\n                                                                zzdim_41933 +\n                                                                gtid_49039];\n            int64_t y_50322 = sub64(xdim_41931, 1);\n            bool cond_50323 = slt64(gtid_49037, y_50322);\n            double maskUtr_50324;\n            \n            if (cond_50323) {\n                double maskUtr_t_res_50325 = maskWs_50301 * maskWp1_50309;\n                \n                maskUtr_50324 = maskUtr_t_res_50325;\n            } else {\n                maskUtr_50324 = 0.0;\n            }\n            \n            double maskUtrP1_50326;\n            \n            if (cond_50323) {\n                double maskwp2_50317 = ((__global\n                                         double *) maskW_mem_52447)[i_50310 *\n                                                                    (zzdim_41966 *\n                                                                     ydim_41965) +\n",
                   "                                                                    gtid_49038 *\n                                                                    zzdim_41966 +\n                                                                    gtid_49039];\n                double maskUtrP1_t_res_50327 = maskWp1_50309 * maskwp2_50317;\n                \n                maskUtrP1_50326 = maskUtrP1_t_res_50327;\n            } else {\n                maskUtrP1_50326 = 0.0;\n            }\n            \n            double maskUtrM1_50328;\n            \n            if (cond_50323) {\n                double maskWm1_50300 = ((__global\n                                         double *) maskW_mem_52447)[i_50293 *\n                                                                    (zzdim_41966 *\n                                                                     ydim_41965) +\n                                                                    gtid_49038 *\n                                                                    zzdim_41966 +\n                                                                    gtid_49039];\n                double maskUtrM1_t_res_50329 = maskWm1_50300 * maskWs_50301;\n                \n                maskUtrM1_50328 = maskUtrM1_t_res_50329;\n            } else {\n                maskUtrM1_50328 = 0.0;\n            }\n            \n            double abs_arg_50330 = velS_50292 / dx_50288;\n            double abs_res_50331 = fabs(abs_arg_50330);\n            double x_50332 = varSP2_50321 - varSP1_50320;\n            double rjp_50333 = maskUtrP1_50326 * x_50332;\n            double x_50334 = varSP1_50320 - varS_50319;\n            double rj_50335 = maskUtr_50324 * x_50334;\n            double x_50336 = varS_50319 - varSM1_50318;\n            double rjm_50337 = maskUtrM1_50328 * x_50336;\n            double abs_res_50338 = fabs(rj_50335);\n            bool cond_50339 = abs_res_50338 < 1.0e-20;\n            double divisor_50340;\n            \n            if (cond_50339) {\n                divisor_",
                   "50340 = 1.0e-20;\n            } else {\n                divisor_50340 = rj_50335;\n            }\n            \n            bool cond_50341 = 0.0 < velS_50292;\n            double cr_50342;\n            \n            if (cond_50341) {\n                double cr_t_res_50343 = rjm_50337 / divisor_50340;\n                \n                cr_50342 = cr_t_res_50343;\n            } else {\n                double cr_f_res_50344 = rjp_50333 / divisor_50340;\n                \n                cr_50342 = cr_f_res_50344;\n            }\n            \n            double min_res_50345 = fmin64(2.0, cr_50342);\n            double min_arg_50346 = 2.0 * cr_50342;\n            double min_res_50347 = fmin64(1.0, min_arg_50346);\n            double max_res_50348 = fmax64(min_res_50345, min_res_50347);\n            double max_res_50349 = fmax64(0.0, max_res_50348);\n            double y_50350 = varS_50319 + varSP1_50320;\n            double x_50351 = velS_50292 * y_50350;\n            double x_50352 = 0.5 * x_50351;\n            double abs_res_50353 = fabs(velS_50292);\n            double x_50354 = 1.0 - max_res_50349;\n            double y_50355 = abs_res_50331 * max_res_50349;\n            double y_50356 = x_50354 + y_50355;\n            double x_50357 = abs_res_50353 * y_50356;\n            double x_50358 = rj_50335 * x_50357;\n            double y_50359 = 0.5 * x_50358;\n            double calcflux_res_50360 = x_50352 - y_50359;\n            \n            lifted_0_f_res_50277 = calcflux_res_50360;\n        } else {\n            lifted_0_f_res_50277 = 0.0;\n        }\n        \n        double lifted_0_f_res_50361;\n        \n        if (index_primexp_52391) {\n            double x_50366 = ((__global double *) cost_mem_52454)[gtid_49038];\n            double y_50367 = ((__global double *) dyt_mem_52450)[gtid_49038];\n            double dx_50368 = x_50366 * y_50367;\n            double velS_50375 = ((__global\n                                  double *) vtau_mem_52443)[gtid_49037 *\n                                                ",
                   "            (zzdim_41954 *\n                                                             ydim_41953) +\n                                                            gtid_49038 *\n                                                            zzdim_41954 +\n                                                            gtid_49039];\n            int64_t i_50376 = sub64(gtid_49038, 1);\n            bool x_50377 = sle64(0, i_50376);\n            bool y_50378 = slt64(i_50376, ydim_41932);\n            bool bounds_check_50379 = x_50377 && y_50378;\n            bool index_certs_50381;\n            \n            if (!bounds_check_50379) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 36) ==\n                        -1) {\n                        global_failure_args[0] = gtid_49037;\n                        global_failure_args[1] = i_50376;\n                        global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double maskWs_50383 = ((__global\n                                    double *) maskW_mem_52447)[gtid_49037 *\n                                                               (zzdim_41966 *\n                                                                ydim_41965) +\n                                                               gtid_49038 *\n                                                               zzdim_41966 +\n                                                               gtid_49039];\n            int64_t i_50384 = add64(1, gtid_49038);\n            bool x_50385 = sle64(0, i_50384);\n            bool y_50386 = slt64(i_50384, ydim_41932);\n            bool bounds_check_50387 = x_50385 && y_50386;\n            bool index_certs_50389;\n            \n   ",
                   "         if (!bounds_check_50387) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 37) ==\n                        -1) {\n                        global_failure_args[0] = gtid_49037;\n                        global_failure_args[1] = i_50384;\n                        global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double maskWp1_50390 = ((__global\n                                     double *) maskW_mem_52447)[gtid_49037 *\n                                                                (zzdim_41966 *\n                                                                 ydim_41965) +\n                                                                i_50384 *\n                                                                zzdim_41966 +\n                                                                gtid_49039];\n            int64_t i_50391 = add64(2, gtid_49038);\n            bool x_50392 = sle64(0, i_50391);\n            bool y_50393 = slt64(i_50391, ydim_41932);\n            bool bounds_check_50394 = x_50392 && y_50393;\n            bool index_certs_50396;\n            \n            if (!bounds_check_50394) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 38) ==\n                        -1) {\n                        global_failure_args[0] = gtid_49037;\n                        global_failure_args[1] = i_50391;\n                        global_failure_args[2] = gtid_49039;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n        ",
                   "            return;\n                }\n            }\n            \n            double varSM1_50398 = ((__global\n                                    double *) tketau_mem_52436)[gtid_49037 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                i_50376 *\n                                                                zzdim_41933 +\n                                                                gtid_49039];\n            double varS_50399 = ((__global\n                                  double *) tketau_mem_52436)[gtid_49037 *\n                                                              (zzdim_41933 *\n                                                               ydim_41932) +\n                                                              gtid_49038 *\n                                                              zzdim_41933 +\n                                                              gtid_49039];\n            double varSP1_50400 = ((__global\n                                    double *) tketau_mem_52436)[gtid_49037 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                i_50384 *\n                                                                zzdim_41933 +\n                                                                gtid_49039];\n            double varSP2_50401 = ((__global\n                                    double *) tketau_mem_52436)[gtid_49037 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                i_50391 *\n                                                                z",
                   "zdim_41933 +\n                                                                gtid_49039];\n            int64_t y_50402 = sub64(ydim_41932, 1);\n            bool cond_50403 = slt64(gtid_49038, y_50402);\n            double maskVtr_50404;\n            \n            if (cond_50403) {\n                double maskVtr_t_res_50405 = maskWs_50383 * maskWp1_50390;\n                \n                maskVtr_50404 = maskVtr_t_res_50405;\n            } else {\n                maskVtr_50404 = 0.0;\n            }\n            \n            double maskVtrP1_50406;\n            \n            if (cond_50403) {\n                double maskwp2_50397 = ((__global\n                                         double *) maskW_mem_52447)[gtid_49037 *\n                                                                    (zzdim_41966 *\n                                                                     ydim_41965) +\n                                                                    i_50391 *\n                                                                    zzdim_41966 +\n                                                                    gtid_49039];\n                double maskVtrP1_t_res_50407 = maskWp1_50390 * maskwp2_50397;\n                \n                maskVtrP1_50406 = maskVtrP1_t_res_50407;\n            } else {\n                maskVtrP1_50406 = 0.0;\n            }\n            \n            double maskVtrM1_50408;\n            \n            if (cond_50403) {\n                double maskWm1_50382 = ((__global\n                                         double *) maskW_mem_52447)[gtid_49037 *\n                                                                    (zzdim_41966 *\n                                                                     ydim_41965) +\n                                                                    i_50376 *\n                                                                    zzdim_41966 +\n                                                                    gtid_49039];\n                doub",
                   "le maskVtrM1_t_res_50409 = maskWm1_50382 * maskWs_50383;\n                \n                maskVtrM1_50408 = maskVtrM1_t_res_50409;\n            } else {\n                maskVtrM1_50408 = 0.0;\n            }\n            \n            double calcflux_arg_50410 = ((__global\n                                          double *) cosu_mem_52455)[gtid_49038];\n            double scaledVel_50411 = velS_50375 * calcflux_arg_50410;\n            double abs_arg_50412 = scaledVel_50411 / dx_50368;\n            double abs_res_50413 = fabs(abs_arg_50412);\n            double x_50414 = varSP2_50401 - varSP1_50400;\n            double rjp_50415 = maskVtrP1_50406 * x_50414;\n            double x_50416 = varSP1_50400 - varS_50399;\n            double rj_50417 = maskVtr_50404 * x_50416;\n            double x_50418 = varS_50399 - varSM1_50398;\n            double rjm_50419 = maskVtrM1_50408 * x_50418;\n            double abs_res_50420 = fabs(rj_50417);\n            bool cond_50421 = abs_res_50420 < 1.0e-20;\n            double divisor_50422;\n            \n            if (cond_50421) {\n                divisor_50422 = 1.0e-20;\n            } else {\n                divisor_50422 = rj_50417;\n            }\n            \n            bool cond_50423 = 0.0 < velS_50375;\n            double cr_50424;\n            \n            if (cond_50423) {\n                double cr_t_res_50425 = rjm_50419 / divisor_50422;\n                \n                cr_50424 = cr_t_res_50425;\n            } else {\n                double cr_f_res_50426 = rjp_50415 / divisor_50422;\n                \n                cr_50424 = cr_f_res_50426;\n            }\n            \n            double min_res_50427 = fmin64(2.0, cr_50424);\n            double min_arg_50428 = 2.0 * cr_50424;\n            double min_res_50429 = fmin64(1.0, min_arg_50428);\n            double max_res_50430 = fmax64(min_res_50427, min_res_50429);\n            double max_res_50431 = fmax64(0.0, max_res_50430);\n            double y_50432 = varS_50399 + varSP1_50400;\n            double x",
                   "_50433 = scaledVel_50411 * y_50432;\n            double x_50434 = 0.5 * x_50433;\n            double abs_res_50435 = fabs(scaledVel_50411);\n            double x_50436 = 1.0 - max_res_50431;\n            double y_50437 = abs_res_50413 * max_res_50431;\n            double y_50438 = x_50436 + y_50437;\n            double x_50439 = abs_res_50435 * y_50438;\n            double x_50440 = rj_50417 * x_50439;\n            double y_50441 = 0.5 * x_50440;\n            double calcflux_res_50442 = x_50434 - y_50441;\n            \n            lifted_0_f_res_50361 = calcflux_res_50442;\n        } else {\n            lifted_0_f_res_50361 = 0.0;\n        }\n        \n        bool cond_50443 = slt64(gtid_49039, distance_42130);\n        bool x_50444 = cond_50443 && index_primexp_52313;\n        bool x_50445 = x_50444 && index_primexp_52310;\n        bool x_50446 = x_50445 && index_primexp_52298;\n        bool x_50447 = x_50446 && index_primexp_52295;\n        double lifted_0_f_res_50448;\n        \n        if (x_50447) {\n            double velS_50458 = ((__global\n                                  double *) wtau_mem_52444)[gtid_49037 *\n                                                            (zzdim_41957 *\n                                                             ydim_41956) +\n                                                            gtid_49038 *\n                                                            zzdim_41957 +\n                                                            gtid_49039];\n            bool cond_50459 = gtid_49039 == 0;\n            bool cond_50460 = !cond_50459;\n            double varSM1_50461;\n            \n            if (cond_50460) {\n                int64_t i_50462 = sub64(gtid_49039, 1);\n                bool x_50463 = sle64(0, i_50462);\n                bool y_50464 = slt64(i_50462, zzdim_41933);\n                bool bounds_check_50465 = x_50463 && y_50464;\n                bool index_certs_50468;\n                \n                if (!bounds_check_50465) {\n                    {",
                   "\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 39) ==\n                            -1) {\n                            global_failure_args[0] = gtid_49037;\n                            global_failure_args[1] = gtid_49038;\n                            global_failure_args[2] = i_50462;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double varSM1_t_res_50469 = ((__global\n                                              double *) tketau_mem_52436)[gtid_49037 *\n                                                                          (zzdim_41933 *\n                                                                           ydim_41932) +\n                                                                          gtid_49038 *\n                                                                          zzdim_41933 +\n                                                                          i_50462];\n                \n                varSM1_50461 = varSM1_t_res_50469;\n            } else {\n                varSM1_50461 = 0.0;\n            }\n            \n            double varS_50470 = ((__global\n                                  double *) tketau_mem_52436)[gtid_49037 *\n                                                              (zzdim_41933 *\n                                                               ydim_41932) +\n                                                              gtid_49038 *\n                                                              zzdim_41933 +\n                                                              gtid_49039];\n            int64_t y_50471 = sub64(zzdim_41933, 2);\n            bool cond_50472 = slt64(gtid_49039, y_50471);\n           ",
                   " double varSP2_50473;\n            \n            if (cond_50472) {\n                int64_t i_50474 = add64(2, gtid_49039);\n                bool x_50475 = sle64(0, i_50474);\n                bool y_50476 = slt64(i_50474, zzdim_41933);\n                bool bounds_check_50477 = x_50475 && y_50476;\n                bool index_certs_50480;\n                \n                if (!bounds_check_50477) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 40) ==\n                            -1) {\n                            global_failure_args[0] = gtid_49037;\n                            global_failure_args[1] = gtid_49038;\n                            global_failure_args[2] = i_50474;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double varSP2_t_res_50481 = ((__global\n                                              double *) tketau_mem_52436)[gtid_49037 *\n                                                                          (zzdim_41933 *\n                                                                           ydim_41932) +\n                                                                          gtid_49038 *\n                                                                          zzdim_41933 +\n                                                                          i_50474];\n                \n                varSP2_50473 = varSP2_t_res_50481;\n            } else {\n                varSP2_50473 = 0.0;\n            }\n            \n            int64_t i_50482 = add64(1, gtid_49039);\n            bool x_50483 = sle64(0, i_50482);\n            bool y_50484 = slt64(i_50482, zzdim_41933);\n            bool bounds_check_50485 = x_50483 && y_5",
                   "0484;\n            bool index_certs_50488;\n            \n            if (!bounds_check_50485) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 41) ==\n                        -1) {\n                        global_failure_args[0] = gtid_49037;\n                        global_failure_args[1] = gtid_49038;\n                        global_failure_args[2] = i_50482;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double varSP1_50489 = ((__global\n                                    double *) tketau_mem_52436)[gtid_49037 *\n                                                                (zzdim_41933 *\n                                                                 ydim_41932) +\n                                                                gtid_49038 *\n                                                                zzdim_41933 +\n                                                                i_50482];\n            double maskWm1_50490;\n            \n            if (cond_50460) {\n                int64_t i_50491 = sub64(gtid_49039, 1);\n                bool x_50492 = sle64(0, i_50491);\n                bool y_50493 = slt64(i_50491, zzdim_41933);\n                bool bounds_check_50494 = x_50492 && y_50493;\n                bool index_certs_50497;\n                \n                if (!bounds_check_50494) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 42) ==\n                            -1) {\n                            global_failure_args[0] = gtid_49037;\n                            global_failure_args[1] = gtid_49038;\n                            global_failure_args[2] = i_50491;\n                            global_failure_args[3] = ",
                   "xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double maskWm1_t_res_50498 = ((__global\n                                               double *) maskW_mem_52447)[gtid_49037 *\n                                                                          (zzdim_41966 *\n                                                                           ydim_41965) +\n                                                                          gtid_49038 *\n                                                                          zzdim_41966 +\n                                                                          i_50491];\n                \n                maskWm1_50490 = maskWm1_t_res_50498;\n            } else {\n                maskWm1_50490 = 0.0;\n            }\n            \n            double maskWs_50499 = ((__global\n                                    double *) maskW_mem_52447)[gtid_49037 *\n                                                               (zzdim_41966 *\n                                                                ydim_41965) +\n                                                               gtid_49038 *\n                                                               zzdim_41966 +\n                                                               gtid_49039];\n            double maskWp1_50500 = ((__global\n                                     double *) maskW_mem_52447)[gtid_49037 *\n                                                                (zzdim_41966 *\n                                                                 ydim_41965) +\n                                                                gtid_49038 *\n                                                                zzdim_41966 +\n                                   ",
                   "                             i_50482];\n            double maskwp2_50501;\n            \n            if (cond_50472) {\n                int64_t i_50502 = add64(2, gtid_49039);\n                bool x_50503 = sle64(0, i_50502);\n                bool y_50504 = slt64(i_50502, zzdim_41933);\n                bool bounds_check_50505 = x_50503 && y_50504;\n                bool index_certs_50508;\n                \n                if (!bounds_check_50505) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 43) ==\n                            -1) {\n                            global_failure_args[0] = gtid_49037;\n                            global_failure_args[1] = gtid_49038;\n                            global_failure_args[2] = i_50502;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double maskwp2_t_res_50509 = ((__global\n                                               double *) maskW_mem_52447)[gtid_49037 *\n                                                                          (zzdim_41966 *\n                                                                           ydim_41965) +\n                                                                          gtid_49038 *\n                                                                          zzdim_41966 +\n                                                                          i_50502];\n                \n                maskwp2_50501 = maskwp2_t_res_50509;\n            } else {\n                maskwp2_50501 = 0.0;\n            }\n            \n            double maskWtr_50510 = maskWs_50499 * maskWp1_50500;\n            double maskWtrP1_50511 = maskWp1_50500 * maskwp2_50501;\n            double",
                   " maskWtrM1_50512 = maskWm1_50490 * maskWs_50499;\n            double dx_50514 = ((__global double *) dzzw_mem_52453)[gtid_49039];\n            double abs_arg_50515 = velS_50458 / dx_50514;\n            double abs_res_50516 = fabs(abs_arg_50515);\n            double x_50517 = varSP2_50473 - varSP1_50489;\n            double rjp_50518 = maskWtrP1_50511 * x_50517;\n            double x_50519 = varSP1_50489 - varS_50470;\n            double rj_50520 = maskWtr_50510 * x_50519;\n            double x_50521 = varS_50470 - varSM1_50461;\n            double rjm_50522 = maskWtrM1_50512 * x_50521;\n            double abs_res_50523 = fabs(rj_50520);\n            bool cond_50524 = abs_res_50523 < 1.0e-20;\n            double divisor_50525;\n            \n            if (cond_50524) {\n                divisor_50525 = 1.0e-20;\n            } else {\n                divisor_50525 = rj_50520;\n            }\n            \n            bool cond_50526 = 0.0 < velS_50458;\n            double cr_50527;\n            \n            if (cond_50526) {\n                double cr_t_res_50528 = rjm_50522 / divisor_50525;\n                \n                cr_50527 = cr_t_res_50528;\n            } else {\n                double cr_f_res_50529 = rjp_50518 / divisor_50525;\n                \n                cr_50527 = cr_f_res_50529;\n            }\n            \n            double min_res_50530 = fmin64(2.0, cr_50527);\n            double min_arg_50531 = 2.0 * cr_50527;\n            double min_res_50532 = fmin64(1.0, min_arg_50531);\n            double max_res_50533 = fmax64(min_res_50530, min_res_50532);\n            double max_res_50534 = fmax64(0.0, max_res_50533);\n            double y_50535 = varS_50470 + varSP1_50489;\n            double x_50536 = velS_50458 * y_50535;\n            double x_50537 = 0.5 * x_50536;\n            double abs_res_50538 = fabs(velS_50458);\n            double x_50539 = 1.0 - max_res_50534;\n            double y_50540 = abs_res_50516 * max_res_50534;\n            double y_50541 = x_50539 + y_50540;\n          ",
                   "  double x_50542 = abs_res_50538 * y_50541;\n            double x_50543 = rj_50520 * x_50542;\n            double y_50544 = 0.5 * x_50543;\n            double calcflux_res_50545 = x_50537 - y_50544;\n            \n            lifted_0_f_res_50448 = calcflux_res_50545;\n        } else {\n            lifted_0_f_res_50448 = 0.0;\n        }\n        ((__global double *) mem_52792)[gtid_49037 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_49038 *\n                                        zzdim_41933 + gtid_49039] =\n            lifted_0_f_res_50448;\n        ((__global double *) mem_52796)[gtid_49037 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_49038 *\n                                        zzdim_41933 + gtid_49039] =\n            lifted_0_f_res_50361;\n        ((__global double *) mem_52800)[gtid_49037 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_49038 *\n                                        zzdim_41933 + gtid_49039] =\n            lifted_0_f_res_50277;\n        ((__global double *) mem_52804)[gtid_49037 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_49038 *\n                                        zzdim_41933 + gtid_49039] =\n            lifted_0_f_res_50235;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_50208\n}\n__kernel void integrate_tkezisegmap_51177(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t ydim_41941,\n                                          int64_t zzdim_41942,\n                                          int64_t ydim_41965",
                   ",\n                                          int64_t zzdim_41966, int64_t y_42127,\n                                          int64_t y_42128,\n                                          int64_t distance_42130, __global\n                                          unsigned char *dtketau_mem_52439,\n                                          __global\n                                          unsigned char *maskW_mem_52447,\n                                          __global unsigned char *dxt_mem_52448,\n                                          __global unsigned char *dyt_mem_52450,\n                                          __global\n                                          unsigned char *dzzw_mem_52453,\n                                          __global\n                                          unsigned char *cost_mem_52454,\n                                          __global unsigned char *mem_52792,\n                                          __global unsigned char *mem_52796,\n                                          __global unsigned char *mem_52800,\n                                          __global unsigned char *mem_52809)\n{\n    #define segmap_group_sizze_51694 (integrate_tkezisegmap_group_sizze_51181)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52983;\n    int32_t local_tid_52984;\n    int64_t group_sizze_52987;\n    int32_t wave_sizze_52986;\n    int32_t group_tid_52985;\n    \n    global_tid_52983 = get_global_id(0);\n    local_tid_52984 = get_local_id(0);\n    group_sizze_52987 = get_local_size(0);\n    wave_sizze_52986 = LOCKSTEP_WIDTH;\n    group_tid_52985 = get_group_id(0);\n    \n    int32_t phys_tid_51177;\n    \n    phys_tid_51177 = global_tid_52983;\n    \n    int64_t gtid_51174;\n    \n    gtid_51174 = squot64(sext_i32_i64(group_tid_52985) *\n                         segmap_group_sizze_51694 +\n                         sext_i32_i64(local_tid_52984), yd",
                   "im_41932 *\n                         zzdim_41933);\n    \n    int64_t gtid_51175;\n    \n    gtid_51175 = squot64(sext_i32_i64(group_tid_52985) *\n                         segmap_group_sizze_51694 +\n                         sext_i32_i64(local_tid_52984) -\n                         squot64(sext_i32_i64(group_tid_52985) *\n                                 segmap_group_sizze_51694 +\n                                 sext_i32_i64(local_tid_52984), ydim_41932 *\n                                 zzdim_41933) * (ydim_41932 * zzdim_41933),\n                         zzdim_41933);\n    \n    int64_t gtid_51176;\n    \n    gtid_51176 = sext_i32_i64(group_tid_52985) * segmap_group_sizze_51694 +\n        sext_i32_i64(local_tid_52984) - squot64(sext_i32_i64(group_tid_52985) *\n                                                segmap_group_sizze_51694 +\n                                                sext_i32_i64(local_tid_52984),\n                                                ydim_41932 * zzdim_41933) *\n        (ydim_41932 * zzdim_41933) - squot64(sext_i32_i64(group_tid_52985) *\n                                             segmap_group_sizze_51694 +\n                                             sext_i32_i64(local_tid_52984) -\n                                             squot64(sext_i32_i64(group_tid_52985) *\n                                                     segmap_group_sizze_51694 +\n                                                     sext_i32_i64(local_tid_52984),\n                                                     ydim_41932 * zzdim_41933) *\n                                             (ydim_41932 * zzdim_41933),\n                                             zzdim_41933) * zzdim_41933;\n    if ((slt64(gtid_51174, xdim_41931) && slt64(gtid_51175, ydim_41932)) &&\n        slt64(gtid_51176, zzdim_41933)) {\n        bool binop_x_52406 = sle64(2, gtid_51175);\n        bool binop_x_52407 = sle64(2, gtid_51174);\n        bool binop_y_52408 = slt64(gtid_51174, y_42127);\n        bool binop_y_52409 = binop",
                   "_x_52407 && binop_y_52408;\n        bool binop_x_52410 = binop_x_52406 && binop_y_52409;\n        bool binop_y_52411 = slt64(gtid_51175, y_42128);\n        bool index_primexp_52412 = binop_x_52410 && binop_y_52411;\n        double tmp_51701;\n        \n        if (index_primexp_52412) {\n            double x_51714 = ((__global double *) maskW_mem_52447)[gtid_51174 *\n                                                                   (zzdim_41966 *\n                                                                    ydim_41965) +\n                                                                   gtid_51175 *\n                                                                   zzdim_41966 +\n                                                                   gtid_51176];\n            double x_51715 = ((__global double *) mem_52800)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n                                                             zzdim_41933 +\n                                                             gtid_51176];\n            int64_t i_51716 = sub64(gtid_51174, 1);\n            bool x_51717 = sle64(0, i_51716);\n            bool y_51718 = slt64(i_51716, xdim_41931);\n            bool bounds_check_51719 = x_51717 && y_51718;\n            bool index_certs_51722;\n            \n            if (!bounds_check_51719) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 44) ==\n                        -1) {\n                        global_failure_args[0] = i_51716;\n                        global_failure_args[1] = gtid_51175;\n                        global_failure_args[2] = gtid_51176;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_419",
                   "33;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_51723 = ((__global double *) mem_52800)[i_51716 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n                                                             zzdim_41933 +\n                                                             gtid_51176];\n            double x_51724 = x_51715 - y_51723;\n            double x_51726 = ((__global double *) cost_mem_52454)[gtid_51175];\n            double y_51728 = ((__global double *) dxt_mem_52448)[gtid_51174];\n            double y_51729 = x_51726 * y_51728;\n            double negate_arg_51730 = x_51724 / y_51729;\n            double x_51731 = 0.0 - negate_arg_51730;\n            double x_51732 = ((__global double *) mem_52796)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n                                                             zzdim_41933 +\n                                                             gtid_51176];\n            int64_t i_51733 = sub64(gtid_51175, 1);\n            bool x_51734 = sle64(0, i_51733);\n            bool y_51735 = slt64(i_51733, ydim_41932);\n            bool bounds_check_51736 = x_51734 && y_51735;\n            bool index_certs_51738;\n            \n            if (!bounds_check_51736) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 45) ==\n                        -1) {\n                        global_failure_args[0] = gtid_51174;\n                        global_failure_args[1] = i_51733;\n                        global_failure_args[2] = gtid_51176;\n             ",
                   "           global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_51739 = ((__global double *) mem_52796)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             i_51733 *\n                                                             zzdim_41933 +\n                                                             gtid_51176];\n            double x_51740 = x_51732 - y_51739;\n            double y_51741 = ((__global double *) dyt_mem_52450)[gtid_51175];\n            double y_51742 = x_51726 * y_51741;\n            double y_51743 = x_51740 / y_51742;\n            double y_51744 = x_51731 - y_51743;\n            double tmp_t_res_51745 = x_51714 * y_51744;\n            \n            tmp_51701 = tmp_t_res_51745;\n        } else {\n            double tmp_f_res_51758 = ((__global\n                                       double *) dtketau_mem_52439)[gtid_51174 *\n                                                                    (zzdim_41942 *\n                                                                     ydim_41941) +\n                                                                    gtid_51175 *\n                                                                    zzdim_41942 +\n                                                                    gtid_51176];\n            \n            tmp_51701 = tmp_f_res_51758;\n        }\n        \n        bool cond_51759 = gtid_51176 == 0;\n        double zz0_update_51760;\n        \n        if (cond_51759) {\n            bool y_51767 = slt64(0, zzdim_41933);\n            bool index_certs_51770;\n            \n            if (!y_51767) {\n               ",
                   " {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 46) ==\n                        -1) {\n                        global_failure_args[0] = gtid_51174;\n                        global_failure_args[1] = gtid_51175;\n                        global_failure_args[2] = 0;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_51771 = ((__global double *) mem_52792)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n                                                             zzdim_41933];\n            bool index_certs_51772;\n            \n            if (!y_51767) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 47) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_51773 = ((__global double *) dzzw_mem_52453)[0];\n            double y_51774 = x_51771 / y_51773;\n            double zz0_update_t_res_51775 = tmp_51701 - y_51774;\n            \n            zz0_update_51760 = zz0_update_t_res_51775;\n        } else {\n            zz0_update_51760 = tmp_51701;\n        }\n        \n        bool cond_51776 = sle64(1, gtid_51176);\n        bool cond_t_res_51777 = slt64(gtid_51176, distance_42130);\n        bool x_51778 = cond_51776 && cond_t_res_51777;\n        double zz_middle_update_51779;\n        \n        if (x_51778) {\n            double x_51792 = ",
                   "((__global double *) mem_52792)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n                                                             zzdim_41933 +\n                                                             gtid_51176];\n            int64_t i_51793 = sub64(gtid_51176, 1);\n            bool x_51794 = sle64(0, i_51793);\n            bool y_51795 = slt64(i_51793, zzdim_41933);\n            bool bounds_check_51796 = x_51794 && y_51795;\n            bool index_certs_51799;\n            \n            if (!bounds_check_51796) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 48) ==\n                        -1) {\n                        global_failure_args[0] = gtid_51174;\n                        global_failure_args[1] = gtid_51175;\n                        global_failure_args[2] = i_51793;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_51800 = ((__global double *) mem_52792)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n                                                             zzdim_41933 +\n                                                             i_51793];\n            double x_51801 = x_51792 - y_51800;\n            double y_51803 = ((__global double *) dzzw_mem_52453)[gtid_51176];\n            double y_51804 = x_51801 / y_51803;\n            double zz_middle_update_t_res_51805 =",
                   " zz0_update_51760 - y_51804;\n            \n            zz_middle_update_51779 = zz_middle_update_t_res_51805;\n        } else {\n            zz_middle_update_51779 = zz0_update_51760;\n        }\n        \n        bool cond_51806 = gtid_51176 == distance_42130;\n        double lifted_0_f_res_51807;\n        \n        if (cond_51806) {\n            double x_51820 = ((__global double *) mem_52792)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n                                                             zzdim_41933 +\n                                                             gtid_51176];\n            int64_t i_51821 = sub64(gtid_51176, 1);\n            bool x_51822 = sle64(0, i_51821);\n            bool y_51823 = slt64(i_51821, zzdim_41933);\n            bool bounds_check_51824 = x_51822 && y_51823;\n            bool index_certs_51827;\n            \n            if (!bounds_check_51824) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 49) ==\n                        -1) {\n                        global_failure_args[0] = gtid_51174;\n                        global_failure_args[1] = gtid_51175;\n                        global_failure_args[2] = i_51821;\n                        global_failure_args[3] = xdim_41931;\n                        global_failure_args[4] = ydim_41932;\n                        global_failure_args[5] = zzdim_41933;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_51828 = ((__global double *) mem_52792)[gtid_51174 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_51175 *\n             ",
                   "                                                zzdim_41933 +\n                                                             i_51821];\n            double x_51829 = x_51820 - y_51828;\n            double y_51831 = ((__global double *) dzzw_mem_52453)[gtid_51176];\n            double y_51832 = 0.5 * y_51831;\n            double y_51833 = x_51829 / y_51832;\n            double lifted_0_f_res_t_res_51834 = zz_middle_update_51779 -\n                   y_51833;\n            \n            lifted_0_f_res_51807 = lifted_0_f_res_t_res_51834;\n        } else {\n            lifted_0_f_res_51807 = zz_middle_update_51779;\n        }\n        ((__global double *) mem_52809)[gtid_51174 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_51175 *\n                                        zzdim_41933 + gtid_51176] =\n            lifted_0_f_res_51807;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_51694\n}\n__kernel void integrate_tkezisegmap_51974(__global int *global_failure,\n                                          int64_t xdim_41931,\n                                          int64_t ydim_41932,\n                                          int64_t zzdim_41933,\n                                          int64_t ydim_41947,\n                                          int64_t zzdim_41948, __global\n                                          unsigned char *dtketaum1_mem_52441,\n                                          __global unsigned char *mem_52804,\n                                          __global unsigned char *mem_52809,\n                                          __global unsigned char *mem_52814)\n{\n    #define segmap_group_sizze_52125 (integrate_tkezisegmap_group_sizze_51978)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_52988;\n    int32_t local_tid_52989;\n    int64_t group_sizze_52992;\n    int32_t wave_sizze_52991;\n    int3",
                   "2_t group_tid_52990;\n    \n    global_tid_52988 = get_global_id(0);\n    local_tid_52989 = get_local_id(0);\n    group_sizze_52992 = get_local_size(0);\n    wave_sizze_52991 = LOCKSTEP_WIDTH;\n    group_tid_52990 = get_group_id(0);\n    \n    int32_t phys_tid_51974;\n    \n    phys_tid_51974 = global_tid_52988;\n    \n    int64_t gtid_51971;\n    \n    gtid_51971 = squot64(sext_i32_i64(group_tid_52990) *\n                         segmap_group_sizze_52125 +\n                         sext_i32_i64(local_tid_52989), ydim_41932 *\n                         zzdim_41933);\n    \n    int64_t gtid_51972;\n    \n    gtid_51972 = squot64(sext_i32_i64(group_tid_52990) *\n                         segmap_group_sizze_52125 +\n                         sext_i32_i64(local_tid_52989) -\n                         squot64(sext_i32_i64(group_tid_52990) *\n                                 segmap_group_sizze_52125 +\n                                 sext_i32_i64(local_tid_52989), ydim_41932 *\n                                 zzdim_41933) * (ydim_41932 * zzdim_41933),\n                         zzdim_41933);\n    \n    int64_t gtid_51973;\n    \n    gtid_51973 = sext_i32_i64(group_tid_52990) * segmap_group_sizze_52125 +\n        sext_i32_i64(local_tid_52989) - squot64(sext_i32_i64(group_tid_52990) *\n                                                segmap_group_sizze_52125 +\n                                                sext_i32_i64(local_tid_52989),\n                                                ydim_41932 * zzdim_41933) *\n        (ydim_41932 * zzdim_41933) - squot64(sext_i32_i64(group_tid_52990) *\n                                             segmap_group_sizze_52125 +\n                                             sext_i32_i64(local_tid_52989) -\n                                             squot64(sext_i32_i64(group_tid_52990) *\n                                                     segmap_group_sizze_52125 +\n                                                     sext_i32_i64(local_tid_52989),\n                                 ",
                   "                    ydim_41932 * zzdim_41933) *\n                                             (ydim_41932 * zzdim_41933),\n                                             zzdim_41933) * zzdim_41933;\n    if ((slt64(gtid_51971, xdim_41931) && slt64(gtid_51972, ydim_41932)) &&\n        slt64(gtid_51973, zzdim_41933)) {\n        double x_52139 = ((__global double *) mem_52804)[gtid_51971 *\n                                                         (zzdim_41933 *\n                                                          ydim_41932) +\n                                                         gtid_51972 *\n                                                         zzdim_41933 +\n                                                         gtid_51973];\n        double y_52140 = ((__global double *) mem_52809)[gtid_51971 *\n                                                         (zzdim_41933 *\n                                                          ydim_41932) +\n                                                         gtid_51972 *\n                                                         zzdim_41933 +\n                                                         gtid_51973];\n        double x_52141 = 1.6 * y_52140;\n        double y_52142 = ((__global double *) dtketaum1_mem_52441)[gtid_51971 *\n                                                                   (zzdim_41948 *\n                                                                    ydim_41947) +\n                                                                   gtid_51972 *\n                                                                   zzdim_41948 +\n                                                                   gtid_51973];\n        double y_52143 = 0.6 * y_52142;\n        double y_52144 = x_52141 - y_52143;\n        double lifted_0_f_res_52145 = x_52139 + y_52144;\n        \n        ((__global double *) mem_52814)[gtid_51971 * (zzdim_41933 *\n                                                      ydim_41932) + gtid_51972 *\n              ",
                   "                          zzdim_41933 + gtid_51973] =\n            lifted_0_f_res_52145;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_52125\n}\n__kernel void integrate_tkezisegmap_intragroup_43307(__global\n                                                     int *global_failure,\n                                                     int failure_is_an_option,\n                                                     __global\n                                                     int64_t *global_failure_args,\n                                                     __local volatile\n                                                     int64_t *mem_52543_backing_aligned_0,\n                                                     __local volatile\n                                                     int64_t *mem_52554_backing_aligned_1,\n                                                     __local volatile\n                                                     int64_t *mem_52531_backing_aligned_2,\n                                                     __local volatile\n                                                     int64_t *mem_52509_backing_aligned_3,\n                                                     __local volatile\n                                                     int64_t *mem_52504_backing_aligned_4,\n                                                     __local volatile\n                                                     int64_t *mem_52528_backing_aligned_5,\n                                                     __local volatile\n                                                     int64_t *mem_52525_backing_aligned_6,\n                                                     __local volatile\n                                                     int64_t *mem_52485_backing_aligned_7,\n                                                     __local volatile\n                                                     int64_t *mem_52482_backing_aligned_8,\n                               ",
                   "                      __local volatile\n                                                     int64_t *mem_52478_backing_aligned_9,\n                                                     __local volatile\n                                                     int64_t *mem_52475_backing_aligned_10,\n                                                     __local volatile\n                                                     int64_t *mem_52472_backing_aligned_11,\n                                                     __local volatile\n                                                     int64_t *mem_52469_backing_aligned_12,\n                                                     int64_t xdim_41931,\n                                                     int64_t ydim_41932,\n                                                     int64_t zzdim_41933,\n                                                     int64_t ydim_41976,\n                                                     int64_t ydim_41978,\n                                                     int64_t zzdim_41979,\n                                                     int64_t ydim_41981,\n                                                     int64_t zzdim_41982,\n                                                     int64_t ydim_41984,\n                                                     int64_t zzdim_41985,\n                                                     int64_t ydim_41987,\n                                                     int64_t y_42127,\n                                                     int64_t y_42128,\n                                                     int64_t distance_42130,\n                                                     int64_t m_42138,\n                                                     int64_t computed_group_sizze_43057,\n                                                     __global\n                                                     unsigned char *tketau_mem_52436,\n                                                     ",
                   "__global\n                                                     unsigned char *dzzt_mem_52452,\n                                                     __global\n                                                     unsigned char *dzzw_mem_52453,\n                                                     __global\n                                                     unsigned char *kbot_mem_52456,\n                                                     __global\n                                                     unsigned char *kappaM_mem_52457,\n                                                     __global\n                                                     unsigned char *mxl_mem_52458,\n                                                     __global\n                                                     unsigned char *forc_mem_52459,\n                                                     __global\n                                                     unsigned char *forc_tke_surface_mem_52460,\n                                                     __global\n                                                     unsigned char *mem_52558)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_52543_backing_12 = (__local volatile\n                                                            char *) mem_52543_backing_aligned_0;\n    __local volatile char *restrict mem_52554_backing_11 = (__local volatile\n                                                            char *) mem_52554_backing_aligned_1;\n    __local volatile char *restrict mem_52531_backing_10 = (__local volatile\n                                                            char *) mem_52531_backing_aligned_2;\n    __local volatile char *restrict mem_52509_backing_9 = (__local volatile\n                                                           char *) mem_52509_backing_aligned_3;\n    __local volatile char *restrict mem_52504_backing_8 = (__local volatile\n               ",
                   "                                            char *) mem_52504_backing_aligned_4;\n    __local volatile char *restrict mem_52528_backing_7 = (__local volatile\n                                                           char *) mem_52528_backing_aligned_5;\n    __local volatile char *restrict mem_52525_backing_6 = (__local volatile\n                                                           char *) mem_52525_backing_aligned_6;\n    __local volatile char *restrict mem_52485_backing_5 = (__local volatile\n                                                           char *) mem_52485_backing_aligned_7;\n    __local volatile char *restrict mem_52482_backing_4 = (__local volatile\n                                                           char *) mem_52482_backing_aligned_8;\n    __local volatile char *restrict mem_52478_backing_3 = (__local volatile\n                                                           char *) mem_52478_backing_aligned_9;\n    __local volatile char *restrict mem_52475_backing_2 = (__local volatile\n                                                           char *) mem_52475_backing_aligned_10;\n    __local volatile char *restrict mem_52472_backing_1 = (__local volatile\n                                                           char *) mem_52472_backing_aligned_11;\n    __local volatile char *restrict mem_52469_backing_0 = (__local volatile\n                                                           char *) mem_52469_backing_aligned_12;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_52885;\n    int32_t local_tid_52886;\n    int64_t group_sizze_52889;\n    int32_t wave_sizze_52888;\n    int32_t group_tid_52887;\n    \n    global_tid_52885 = get_global_id(0);\n    local_tid_52886 = get_local_id(0);\n    group_sizze_52889 = get_local_size(0);\n    wave_sizze_52888 ",
                   "= LOCKSTEP_WIDTH;\n    group_tid_52887 = get_group_id(0);\n    \n    int32_t phys_tid_43307;\n    \n    phys_tid_43307 = group_tid_52887;\n    \n    int32_t ltid_pre_52890;\n    \n    ltid_pre_52890 = local_tid_52886;\n    \n    int32_t ltid_pre_52891;\n    \n    ltid_pre_52891 = local_tid_52886;\n    \n    int32_t ltid_pre_52892;\n    \n    ltid_pre_52892 = local_tid_52886 - local_tid_52886;\n    \n    int32_t ltid_pre_52893;\n    \n    ltid_pre_52893 = squot32(local_tid_52886, sext_i64_i32(zzdim_41933));\n    \n    int32_t ltid_pre_52894;\n    \n    ltid_pre_52894 = local_tid_52886 - squot32(local_tid_52886,\n                                               sext_i64_i32(zzdim_41933)) *\n        sext_i64_i32(zzdim_41933);\n    \n    int64_t gtid_43055;\n    \n    gtid_43055 = sext_i32_i64(group_tid_52887);\n    \n    bool cond_43720;\n    \n    cond_43720 = sle64(2, gtid_43055);\n    \n    bool cond_t_res_43721 = slt64(gtid_43055, y_42127);\n    bool x_43722 = cond_43720 && cond_t_res_43721;\n    __local char *mem_52469;\n    \n    mem_52469 = (__local char *) mem_52469_backing_0;\n    \n    __local char *mem_52472;\n    \n    mem_52472 = (__local char *) mem_52472_backing_1;\n    \n    __local char *mem_52475;\n    \n    mem_52475 = (__local char *) mem_52475_backing_2;\n    \n    __local char *mem_52478;\n    \n    mem_52478 = (__local char *) mem_52478_backing_3;\n    \n    int64_t gtid_43172 = sext_i32_i64(ltid_pre_52893);\n    int64_t gtid_43173 = sext_i32_i64(ltid_pre_52894);\n    int32_t phys_tid_43174 = local_tid_52886;\n    \n    if (slt64(gtid_43172, ydim_41932) && slt64(gtid_43173, zzdim_41933)) {\n        bool binop_y_52154 = sle64(2, gtid_43172);\n        bool binop_x_52155 = x_43722 && binop_y_52154;\n        bool binop_y_52158 = slt64(gtid_43172, y_42128);\n        bool index_primexp_52159 = binop_x_52155 && binop_y_52158;\n        double lifted_0_f_res_43736;\n        double lifted_0_f_res_43737;\n        double lifted_0_f_res_43738;\n        double lifted_0_f_res_43739;\n        \n        if (index_primexp_52159) {\n  ",
                   "          double tke_43752 = ((__global\n                                 double *) tketau_mem_52436)[gtid_43055 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_43172 *\n                                                             zzdim_41933 +\n                                                             gtid_43173];\n            double max_res_43753 = fmax64(0.0, tke_43752);\n            double sqrt_res_43754;\n            \n            sqrt_res_43754 = futrts_sqrt64(max_res_43753);\n            \n            int32_t x_43757 = ((__global int32_t *) kbot_mem_52456)[gtid_43055 *\n                                                                    ydim_41976 +\n                                                                    gtid_43172];\n            int32_t ks_val_43758 = sub32(x_43757, 1);\n            bool land_mask_43759 = sle32(0, ks_val_43758);\n            int32_t i64_res_43760 = sext_i64_i32(gtid_43173);\n            bool edge_mask_t_res_43761 = i64_res_43760 == ks_val_43758;\n            bool x_43762 = land_mask_43759 && edge_mask_t_res_43761;\n            bool water_mask_t_res_43763 = sle32(ks_val_43758, i64_res_43760);\n            bool x_43764 = land_mask_43759 && water_mask_t_res_43763;\n            double kappa_43765 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_43055 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_43172 *\n                                                               zzdim_41979 +\n                                                               gtid_43173];\n            bool cond_43766 = slt64(0, gtid_43173);\n            double deltam1_43767;\n            \n            if (cond_",
                   "43766) {\n                double y_43769 = ((__global\n                                   double *) dzzt_mem_52452)[gtid_43173];\n                double x_43770 = 1.0 / y_43769;\n                double x_43771 = 0.5 * x_43770;\n                int64_t i_43772 = sub64(gtid_43173, 1);\n                bool x_43773 = sle64(0, i_43772);\n                bool y_43774 = slt64(i_43772, zzdim_41933);\n                bool bounds_check_43775 = x_43773 && y_43774;\n                bool index_certs_43778;\n                \n                if (!bounds_check_43775) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 0) ==\n                            -1) {\n                            global_failure_args[0] = gtid_43055;\n                            global_failure_args[1] = gtid_43172;\n                            global_failure_args[2] = i_43772;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double x_43779 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_43055 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_43172 *\n                                                               zzdim_41979 +\n                                                               i_43772];\n                double y_43780 = kappa_43765 + x_43779;\n                double deltam1_t_res_43781 = x_43771 * y_43780;\n                \n                deltam1_43767 = deltam1_t_res_43781;\n            } else {\n               ",
                   " deltam1_43767 = 0.0;\n            }\n            \n            int64_t y_43782 = sub64(zzdim_41933, 1);\n            bool cond_43783 = slt64(gtid_43173, y_43782);\n            double delta_43784;\n            \n            if (cond_43783) {\n                int64_t i_43785 = add64(1, gtid_43173);\n                bool x_43786 = sle64(0, i_43785);\n                bool y_43787 = slt64(i_43785, zzdim_41933);\n                bool bounds_check_43788 = x_43786 && y_43787;\n                bool index_certs_43789;\n                \n                if (!bounds_check_43788) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 1) ==\n                            -1) {\n                            global_failure_args[0] = i_43785;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double y_43790 = ((__global double *) dzzt_mem_52452)[i_43785];\n                double x_43791 = 1.0 / y_43790;\n                double x_43792 = 0.5 * x_43791;\n                bool index_certs_43795;\n                \n                if (!bounds_check_43788) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 2) ==\n                            -1) {\n                            global_failure_args[0] = gtid_43055;\n                            global_failure_args[1] = gtid_43172;\n                            global_failure_args[2] = i_43785;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n              ",
                   "  }\n                \n                double y_43796 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_43055 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_43172 *\n                                                               zzdim_41979 +\n                                                               i_43785];\n                double y_43797 = kappa_43765 + y_43796;\n                double delta_t_res_43798 = x_43792 * y_43797;\n                \n                delta_43784 = delta_t_res_43798;\n            } else {\n                delta_43784 = 0.0;\n            }\n            \n            double dzzwzz_43800 = ((__global\n                                    double *) dzzw_mem_52453)[gtid_43173];\n            bool cond_f_res_43802 = !x_43764;\n            bool x_43803 = !x_43762;\n            bool y_43804 = cond_f_res_43802 && x_43803;\n            bool cond_43805 = x_43762 || y_43804;\n            double a_43806;\n            \n            if (cond_43805) {\n                a_43806 = 0.0;\n            } else {\n                bool x_43807 = cond_43766 && cond_43783;\n                double a_f_res_43808;\n                \n                if (x_43807) {\n                    double negate_arg_43809 = deltam1_43767 / dzzwzz_43800;\n                    double a_f_res_t_res_43810 = 0.0 - negate_arg_43809;\n                    \n                    a_f_res_43808 = a_f_res_t_res_43810;\n                } else {\n                    bool cond_43811 = gtid_43173 == y_43782;\n                    double a_f_res_f_res_43812;\n                    \n                    if (cond_43811) {\n                        double y_43813 = 0.5 * dzzwzz_43800;\n                        double negate_arg_43814 = deltam1_43767 / y_43813;\n                        double a_f_res_f_res_t_res_43815 = 0.0 -\n              ",
                   "                 negate_arg_43814;\n                        \n                        a_f_res_f_res_43812 = a_f_res_f_res_t_res_43815;\n                    } else {\n                        a_f_res_f_res_43812 = 0.0;\n                    }\n                    a_f_res_43808 = a_f_res_f_res_43812;\n                }\n                a_43806 = a_f_res_43808;\n            }\n            \n            double b_43816;\n            \n            if (cond_f_res_43802) {\n                b_43816 = 1.0;\n            } else {\n                double mxls_43801 = ((__global\n                                      double *) mxl_mem_52458)[gtid_43055 *\n                                                               (zzdim_41982 *\n                                                                ydim_41981) +\n                                                               gtid_43172 *\n                                                               zzdim_41982 +\n                                                               gtid_43173];\n                double b_f_res_43817;\n                \n                if (x_43762) {\n                    double y_43818 = delta_43784 / dzzwzz_43800;\n                    double x_43819 = 1.0 + y_43818;\n                    double x_43820 = 0.7 / mxls_43801;\n                    double y_43821 = sqrt_res_43754 * x_43820;\n                    double b_f_res_t_res_43822 = x_43819 + y_43821;\n                    \n                    b_f_res_43817 = b_f_res_t_res_43822;\n                } else {\n                    bool x_43823 = cond_43766 && cond_43783;\n                    double b_f_res_f_res_43824;\n                    \n                    if (x_43823) {\n                        double x_43825 = deltam1_43767 + delta_43784;\n                        double y_43826 = x_43825 / dzzwzz_43800;\n                        double x_43827 = 1.0 + y_43826;\n                        double x_43828 = 0.7 * sqrt_res_43754;\n                        double y_43829 = x_43828 / mxls_43801;\n           ",
                   "             double b_f_res_f_res_t_res_43830 = x_43827 + y_43829;\n                        \n                        b_f_res_f_res_43824 = b_f_res_f_res_t_res_43830;\n                    } else {\n                        bool cond_43831 = gtid_43173 == y_43782;\n                        double b_f_res_f_res_f_res_43832;\n                        \n                        if (cond_43831) {\n                            double y_43833 = 0.5 * dzzwzz_43800;\n                            double y_43834 = deltam1_43767 / y_43833;\n                            double x_43835 = 1.0 + y_43834;\n                            double x_43836 = 0.7 / mxls_43801;\n                            double y_43837 = sqrt_res_43754 * x_43836;\n                            double b_f_res_f_res_f_res_t_res_43838 = x_43835 +\n                                   y_43837;\n                            \n                            b_f_res_f_res_f_res_43832 =\n                                b_f_res_f_res_f_res_t_res_43838;\n                        } else {\n                            b_f_res_f_res_f_res_43832 = 0.0;\n                        }\n                        b_f_res_f_res_43824 = b_f_res_f_res_f_res_43832;\n                    }\n                    b_f_res_43817 = b_f_res_f_res_43824;\n                }\n                b_43816 = b_f_res_43817;\n            }\n            \n            double lifted_0_f_res_t_res_43839;\n            double lifted_0_f_res_t_res_43840;\n            \n            if (cond_f_res_43802) {\n                lifted_0_f_res_t_res_43839 = 0.0;\n                lifted_0_f_res_t_res_43840 = 0.0;\n            } else {\n                double negate_arg_43841 = delta_43784 / dzzwzz_43800;\n                double c_43842 = 0.0 - negate_arg_43841;\n                double y_43843 = ((__global\n                                   double *) forc_mem_52459)[gtid_43055 *\n                                                             (zzdim_41985 *\n                                                              ydim_4198",
                   "4) +\n                                                             gtid_43172 *\n                                                             zzdim_41985 +\n                                                             gtid_43173];\n                double tmp_43844 = tke_43752 + y_43843;\n                bool cond_43845 = gtid_43173 == y_43782;\n                double lifted_0_f_res_t_res_f_res_43846;\n                \n                if (cond_43845) {\n                    double y_43847 = ((__global\n                                       double *) forc_tke_surface_mem_52460)[gtid_43055 *\n                                                                             ydim_41987 +\n                                                                             gtid_43172];\n                    double y_43848 = 0.5 * dzzwzz_43800;\n                    double y_43849 = y_43847 / y_43848;\n                    double lifted_0_f_res_t_res_f_res_t_res_43850 = tmp_43844 +\n                           y_43849;\n                    \n                    lifted_0_f_res_t_res_f_res_43846 =\n                        lifted_0_f_res_t_res_f_res_t_res_43850;\n                } else {\n                    lifted_0_f_res_t_res_f_res_43846 = tmp_43844;\n                }\n                lifted_0_f_res_t_res_43839 = c_43842;\n                lifted_0_f_res_t_res_43840 = lifted_0_f_res_t_res_f_res_43846;\n            }\n            lifted_0_f_res_43736 = a_43806;\n            lifted_0_f_res_43737 = b_43816;\n            lifted_0_f_res_43738 = lifted_0_f_res_t_res_43839;\n            lifted_0_f_res_43739 = lifted_0_f_res_t_res_43840;\n        } else {\n            lifted_0_f_res_43736 = 0.0;\n            lifted_0_f_res_43737 = 0.0;\n            lifted_0_f_res_43738 = 0.0;\n            lifted_0_f_res_43739 = 0.0;\n        }\n        ((__local double *) mem_52469)[gtid_43172 * zzdim_41933 + gtid_43173] =\n            lifted_0_f_res_43736;\n        ((__local double *) mem_52472)[gtid_43172 * zzdim_41933 + gtid_43173] =\n            l",
                   "ifted_0_f_res_43737;\n        ((__local double *) mem_52475)[gtid_43172 * zzdim_41933 + gtid_43173] =\n            lifted_0_f_res_43738;\n        ((__local double *) mem_52478)[gtid_43172 * zzdim_41933 + gtid_43173] =\n            lifted_0_f_res_43739;\n    }\n    \n  error_0:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_52482;\n    \n    mem_52482 = (__local char *) mem_52482_backing_4;\n    \n    __local char *mem_52485;\n    \n    mem_52485 = (__local char *) mem_52485_backing_5;\n    \n    int64_t gtid_43151 = sext_i32_i64(ltid_pre_52893);\n    int64_t gtid_43152 = sext_i32_i64(ltid_pre_52894);\n    int32_t phys_tid_43153 = local_tid_52886;\n    \n    if (slt64(gtid_43151, ydim_41932) && slt64(gtid_43152, zzdim_41933)) {\n        bool cond_43857 = gtid_43152 == 0;\n        double lifted_0_f_res_43858;\n        \n        if (cond_43857) {\n            bool y_43859 = slt64(0, zzdim_41933);\n            bool index_certs_43860;\n            \n            if (!y_43859) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 3) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_43861 = ((__local double *) mem_52475)[gtid_43151 *\n                                                            zzdim_41933];\n            double y_43862 = ((__local double *) mem_52472)[gtid_43151 *\n                                                            zzdim_41933];\n            double lifted_0_f_res_t_res_43863 = x_43861 / y_43862;\n            \n            lifted_0_f_res_43858 = lifted_0_f_res_t_res_43863;\n        } else {\n            lifted_0_f_res_43858 = 0.0;\n        }\n        \n        double lifted_0_f_res_438",
                   "64;\n        \n        if (cond_43857) {\n            bool y_43865 = slt64(0, zzdim_41933);\n            bool index_certs_43866;\n            \n            if (!y_43865) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 4) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_43867 = ((__local double *) mem_52478)[gtid_43151 *\n                                                            zzdim_41933];\n            double y_43868 = ((__local double *) mem_52472)[gtid_43151 *\n                                                            zzdim_41933];\n            double lifted_0_f_res_t_res_43869 = x_43867 / y_43868;\n            \n            lifted_0_f_res_43864 = lifted_0_f_res_t_res_43869;\n        } else {\n            lifted_0_f_res_43864 = 0.0;\n        }\n        ((__local double *) mem_52482)[gtid_43151 * zzdim_41933 + gtid_43152] =\n            lifted_0_f_res_43864;\n        ((__local double *) mem_52485)[gtid_43151 * zzdim_41933 + gtid_43152] =\n            lifted_0_f_res_43858;\n    }\n    \n  error_1:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_52525;\n    \n    mem_52525 = (__local char *) mem_52525_backing_6;\n    \n    __local char *mem_52528;\n    \n    mem_52528 = (__local char *) mem_52528_backing_7;\n    \n    int64_t gtid_43116 = sext_i32_i64(ltid_pre_52890);\n    int32_t phys_tid_43117 = local_tid_52886;\n    \n    if (slt64(gtid_43116, ydim_41932)) {\n        __local char *mem_52504;\n        \n        mem_52504 = (__local char *) mem_52504_backing_8;\n        for (int64_t i_52895 = 0; i_52895 < zzdim_41933; i_52895++) {\n            ((__local double *) mem_52504)[i_52895] = ((__",
                   "local\n                                                        double *) mem_52485)[gtid_43116 *\n                                                                             zzdim_41933 +\n                                                                             i_52895];\n        }\n        \n        __local char *mem_52509;\n        \n        mem_52509 = (__local char *) mem_52509_backing_9;\n        for (int64_t i_52896 = 0; i_52896 < zzdim_41933; i_52896++) {\n            ((__local double *) mem_52509)[i_52896] = ((__local\n                                                        double *) mem_52482)[gtid_43116 *\n                                                                             zzdim_41933 +\n                                                                             i_52896];\n        }\n        for (int64_t i_43880 = 0; i_43880 < distance_42130; i_43880++) {\n            int64_t index_primexp_43883 = add64(1, i_43880);\n            bool x_43884 = sle64(0, index_primexp_43883);\n            bool y_43885 = slt64(index_primexp_43883, zzdim_41933);\n            bool bounds_check_43886 = x_43884 && y_43885;\n            bool index_certs_43887;\n            \n            if (!bounds_check_43886) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 5) ==\n                        -1) {\n                        global_failure_args[0] = index_primexp_43883;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_2;\n                }\n            }\n            \n            double x_43888 = ((__local double *) mem_52472)[gtid_43116 *\n                                                            zzdim_41933 +\n                                                            index_primexp_43883];\n            double x_43889 = ((__local double *) mem_52469)[gtid_43116 *\n                                                            zz",
                   "dim_41933 +\n                                                            index_primexp_43883];\n            bool y_43890 = slt64(i_43880, zzdim_41933);\n            bool index_certs_43891;\n            \n            if (!y_43890) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 6) ==\n                        -1) {\n                        global_failure_args[0] = i_43880;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_2;\n                }\n            }\n            \n            double y_43892 = ((__local double *) mem_52504)[i_43880];\n            double y_43893 = x_43889 * y_43892;\n            double y_43894 = x_43888 - y_43893;\n            double norm_factor_43895 = 1.0 / y_43894;\n            double x_43896 = ((__local double *) mem_52475)[gtid_43116 *\n                                                            zzdim_41933 +\n                                                            index_primexp_43883];\n            double lw_val_43897 = norm_factor_43895 * x_43896;\n            \n            ((__local double *) mem_52504)[index_primexp_43883] = lw_val_43897;\n            \n            double x_43899 = ((__local double *) mem_52478)[gtid_43116 *\n                                                            zzdim_41933 +\n                                                            index_primexp_43883];\n            double y_43900 = ((__local double *) mem_52509)[i_43880];\n            double y_43901 = x_43889 * y_43900;\n            double x_43902 = x_43899 - y_43901;\n            double lw_val_43903 = norm_factor_43895 * x_43902;\n            \n            ((__local double *) mem_52509)[index_primexp_43883] = lw_val_43903;\n        }\n        for (int64_t i_52899 = 0; i_52899 < zzdim_41933; i_52899++) {\n            ((__local double *) mem_52525)[gtid_43116 * zzdim_41933 + i_52899] =\n                ((__local doub",
                   "le *) mem_52504)[i_52899];\n        }\n        for (int64_t i_52900 = 0; i_52900 < zzdim_41933; i_52900++) {\n            ((__local double *) mem_52528)[gtid_43116 * zzdim_41933 + i_52900] =\n                ((__local double *) mem_52509)[i_52900];\n        }\n    }\n    \n  error_2:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_52531;\n    \n    mem_52531 = (__local char *) mem_52531_backing_10;\n    for (int64_t i_52901 = 0; i_52901 < sdiv_up64(ydim_41932 * zzdim_41933 -\n                                                  sext_i32_i64(local_tid_52886),\n                                                  computed_group_sizze_43057);\n         i_52901++) {\n        ((__local double *) mem_52531)[squot64(i_52901 *\n                                               computed_group_sizze_43057 +\n                                               sext_i32_i64(local_tid_52886),\n                                               zzdim_41933) * zzdim_41933 +\n                                       (i_52901 * computed_group_sizze_43057 +\n                                        sext_i32_i64(local_tid_52886) -\n                                        squot64(i_52901 *\n                                                computed_group_sizze_43057 +\n                                                sext_i32_i64(local_tid_52886),\n                                                zzdim_41933) * zzdim_41933)] =\n            0.0;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int64_t gtid_43088 = sext_i32_i64(ltid_pre_52891);\n    int64_t gtid_slice_43089 = sext_i32_i64(ltid_pre_52892);\n    int32_t phys_tid_43092 = local_tid_52886;\n    \n    if (slt64(gtid_43088, ydim_41932) && slt64(gtid_slice_43089, 1)) {\n        int64_t index_primexp_52164 = distance_42130 + gtid_slice_43089;\n        double v_43914 = ((__local double *) mem_52528)[gtid_43088 *\n                                                        zzdim_41933 +\n                    ",
                   "                                    index_primexp_52164];\n        \n        if ((sle64(0, gtid_43088) && slt64(gtid_43088, ydim_41932)) && (sle64(0,\n                                                                              index_primexp_52164) &&\n                                                                        slt64(index_primexp_52164,\n                                                                              zzdim_41933))) {\n            ((__local double *) mem_52531)[gtid_43088 * zzdim_41933 +\n                                           index_primexp_52164] = v_43914;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_52554;\n    \n    mem_52554 = (__local char *) mem_52554_backing_11;\n    \n    int64_t gtid_43061 = sext_i32_i64(ltid_pre_52890);\n    int32_t phys_tid_43062 = local_tid_52886;\n    \n    if (slt64(gtid_43061, ydim_41932)) {\n        __local char *mem_52543;\n        \n        mem_52543 = (__local char *) mem_52543_backing_12;\n        for (int64_t i_52902 = 0; i_52902 < zzdim_41933; i_52902++) {\n            ((__local double *) mem_52543)[i_52902] = ((__local\n                                                        double *) mem_52531)[gtid_43061 *\n                                                                             zzdim_41933 +\n                                                                             i_52902];\n        }\n        for (int64_t i_43921 = 0; i_43921 < distance_42130; i_43921++) {\n            int64_t binop_y_43923 = -1 * i_43921;\n            int64_t binop_x_43924 = m_42138 + binop_y_43923;\n            bool x_43925 = sle64(0, binop_x_43924);\n            bool y_43926 = slt64(binop_x_43924, zzdim_41933);\n            bool bounds_check_43927 = x_43925 && y_43926;\n            bool index_certs_43928;\n            \n            if (!bounds_check_43927) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 7) ==\n                        -1) {\n                        global_fai",
                   "lure_args[0] = binop_x_43924;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_4;\n                }\n            }\n            \n            double x_43929 = ((__local double *) mem_52528)[gtid_43061 *\n                                                            zzdim_41933 +\n                                                            binop_x_43924];\n            double x_43930 = ((__local double *) mem_52525)[gtid_43061 *\n                                                            zzdim_41933 +\n                                                            binop_x_43924];\n            int64_t i_43931 = add64(1, binop_x_43924);\n            bool x_43932 = sle64(0, i_43931);\n            bool y_43933 = slt64(i_43931, zzdim_41933);\n            bool bounds_check_43934 = x_43932 && y_43933;\n            bool index_certs_43935;\n            \n            if (!bounds_check_43934) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 8) ==\n                        -1) {\n                        global_failure_args[0] = i_43931;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_4;\n                }\n            }\n            \n            double y_43936 = ((__local double *) mem_52543)[i_43931];\n            double y_43937 = x_43930 * y_43936;\n            double lw_val_43938 = x_43929 - y_43937;\n            \n            ((__local double *) mem_52543)[binop_x_43924] = lw_val_43938;\n        }\n        for (int64_t i_52904 = 0; i_52904 < zzdim_41933; i_52904++) {\n            ((__local double *) mem_52554)[gtid_43061 * zzdim_41933 + i_52904] =\n                ((__local double *) mem_52543)[i_52904];\n        }\n    }\n    \n  error_4:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        retu",
                   "rn;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_52905 = 0; i_52905 < sdiv_up64(ydim_41932 * zzdim_41933 -\n                                                  sext_i32_i64(local_tid_52886),\n                                                  computed_group_sizze_43057);\n         i_52905++) {\n        ((__global double *) mem_52558)[gtid_43055 * (zzdim_41933 *\n                                                      ydim_41932) +\n                                        squot64(i_52905 *\n                                                computed_group_sizze_43057 +\n                                                sext_i32_i64(local_tid_52886),\n                                                zzdim_41933) * zzdim_41933 +\n                                        (i_52905 * computed_group_sizze_43057 +\n                                         sext_i32_i64(local_tid_52886) -\n                                         squot64(i_52905 *\n                                                 computed_group_sizze_43057 +\n                                                 sext_i32_i64(local_tid_52886),\n                                                 zzdim_41933) * zzdim_41933)] =\n            ((__local double *) mem_52554)[squot64(i_52905 *\n                                                   computed_group_sizze_43057 +\n                                                   sext_i32_i64(local_tid_52886),\n                                                   zzdim_41933) * zzdim_41933 +\n                                           (i_52905 *\n                                            computed_group_sizze_43057 +\n                                            sext_i32_i64(local_tid_52886) -\n                                            squot64(i_52905 *\n                                                    computed_group_sizze_43057 +\n                                                    sext_i32_i64(local_tid_52886),\n                                                    zzdim_41933) *\n                           ",
                   "                 zzdim_41933)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n  error_5:\n    return;\n}\n__kernel void integrate_tkezisegmap_intragroup_44093(__global\n                                                     int *global_failure,\n                                                     int failure_is_an_option,\n                                                     __global\n                                                     int64_t *global_failure_args,\n                                                     __local volatile\n                                                     int64_t *mem_52599_backing_aligned_0,\n                                                     __local volatile\n                                                     int64_t *mem_52573_backing_aligned_1,\n                                                     __local volatile\n                                                     int64_t *mem_52571_backing_aligned_2,\n                                                     __local volatile\n                                                     int64_t *mem_52568_backing_aligned_3,\n                                                     __local volatile\n                                                     int64_t *mem_52566_backing_aligned_4,\n                                                     __local volatile\n                                                     int64_t *mem_52564_backing_aligned_5,\n                                                     __local volatile\n                                                     int64_t *mem_52562_backing_aligned_6,\n                                                     int64_t xdim_41931,\n                                                     int64_t ydim_41932,\n                                                     int64_t zzdim_41933,\n                                                     int64_t ydim_41976,\n                                                     int64_t ydim_41978,\n                                             ",
                   "        int64_t zzdim_41979,\n                                                     int64_t ydim_41981,\n                                                     int64_t zzdim_41982,\n                                                     int64_t ydim_41984,\n                                                     int64_t zzdim_41985,\n                                                     int64_t ydim_41987,\n                                                     int64_t y_42127,\n                                                     int64_t y_42128,\n                                                     int64_t distance_42130,\n                                                     int64_t m_42138, __global\n                                                     unsigned char *tketau_mem_52436,\n                                                     __global\n                                                     unsigned char *dzzt_mem_52452,\n                                                     __global\n                                                     unsigned char *dzzw_mem_52453,\n                                                     __global\n                                                     unsigned char *kbot_mem_52456,\n                                                     __global\n                                                     unsigned char *kappaM_mem_52457,\n                                                     __global\n                                                     unsigned char *mxl_mem_52458,\n                                                     __global\n                                                     unsigned char *forc_mem_52459,\n                                                     __global\n                                                     unsigned char *forc_tke_surface_mem_52460,\n                                                     __global\n                                                     unsigned char *mem_52615)\n{\n    const int block_dim0 = 0;\n    const int ",
                   "block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_52599_backing_6 = (__local volatile\n                                                           char *) mem_52599_backing_aligned_0;\n    __local volatile char *restrict mem_52573_backing_5 = (__local volatile\n                                                           char *) mem_52573_backing_aligned_1;\n    __local volatile char *restrict mem_52571_backing_4 = (__local volatile\n                                                           char *) mem_52571_backing_aligned_2;\n    __local volatile char *restrict mem_52568_backing_3 = (__local volatile\n                                                           char *) mem_52568_backing_aligned_3;\n    __local volatile char *restrict mem_52566_backing_2 = (__local volatile\n                                                           char *) mem_52566_backing_aligned_4;\n    __local volatile char *restrict mem_52564_backing_1 = (__local volatile\n                                                           char *) mem_52564_backing_aligned_5;\n    __local volatile char *restrict mem_52562_backing_0 = (__local volatile\n                                                           char *) mem_52562_backing_aligned_6;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_52906;\n    int32_t local_tid_52907;\n    int64_t group_sizze_52910;\n    int32_t wave_sizze_52909;\n    int32_t group_tid_52908;\n    \n    global_tid_52906 = get_global_id(0);\n    local_tid_52907 = get_local_id(0);\n    group_sizze_52910 = get_local_size(0);\n    wave_sizze_52909 = LOCKSTEP_WIDTH;\n    group_tid_52908 = get_group_id(0);\n    \n    int32_t phys_tid_44093;\n    \n    phys_tid_44093 = group_tid_52908;\n    \n    int32_t ltid_pre_52911;\n    \n    ltid_pre_52911 = local_tid_52907;\n ",
                   "   \n    int64_t gtid_43954;\n    \n    gtid_43954 = squot64(sext_i32_i64(group_tid_52908), ydim_41932);\n    \n    int64_t gtid_43955;\n    \n    gtid_43955 = sext_i32_i64(group_tid_52908) -\n        squot64(sext_i32_i64(group_tid_52908), ydim_41932) * ydim_41932;\n    \n    bool binop_x_52177;\n    \n    binop_x_52177 = sle64(2, gtid_43954);\n    \n    bool binop_y_52180 = slt64(gtid_43954, y_42127);\n    bool index_primexp_52181 = binop_x_52177 && binop_y_52180;\n    bool cond_t_res_45483 = sle64(2, gtid_43955);\n    bool x_45484 = cond_t_res_45483 && index_primexp_52181;\n    bool cond_t_res_45485 = slt64(gtid_43955, y_42128);\n    bool x_45486 = x_45484 && cond_t_res_45485;\n    __local char *mem_52562;\n    \n    mem_52562 = (__local char *) mem_52562_backing_0;\n    \n    __local char *mem_52564;\n    \n    mem_52564 = (__local char *) mem_52564_backing_1;\n    \n    __local char *mem_52566;\n    \n    mem_52566 = (__local char *) mem_52566_backing_2;\n    \n    __local char *mem_52568;\n    \n    mem_52568 = (__local char *) mem_52568_backing_3;\n    \n    int64_t gtid_43959 = sext_i32_i64(ltid_pre_52911);\n    int32_t phys_tid_43960 = local_tid_52907;\n    \n    if (slt64(gtid_43959, zzdim_41933)) {\n        double lifted_0_f_res_45492;\n        double lifted_0_f_res_45493;\n        double lifted_0_f_res_45494;\n        double lifted_0_f_res_45495;\n        \n        if (x_45486) {\n            double tke_45508 = ((__global\n                                 double *) tketau_mem_52436)[gtid_43954 *\n                                                             (zzdim_41933 *\n                                                              ydim_41932) +\n                                                             gtid_43955 *\n                                                             zzdim_41933 +\n                                                             gtid_43959];\n            double max_res_45509 = fmax64(0.0, tke_45508);\n            double sqrt_res_45510;\n            \n            sqrt_res_45510 = futr",
                   "ts_sqrt64(max_res_45509);\n            \n            int32_t x_45513 = ((__global int32_t *) kbot_mem_52456)[gtid_43954 *\n                                                                    ydim_41976 +\n                                                                    gtid_43955];\n            int32_t ks_val_45514 = sub32(x_45513, 1);\n            bool land_mask_45515 = sle32(0, ks_val_45514);\n            int32_t i64_res_45516 = sext_i64_i32(gtid_43959);\n            bool edge_mask_t_res_45517 = i64_res_45516 == ks_val_45514;\n            bool x_45518 = land_mask_45515 && edge_mask_t_res_45517;\n            bool water_mask_t_res_45519 = sle32(ks_val_45514, i64_res_45516);\n            bool x_45520 = land_mask_45515 && water_mask_t_res_45519;\n            double kappa_45521 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_43954 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_43955 *\n                                                               zzdim_41979 +\n                                                               gtid_43959];\n            bool cond_45522 = slt64(0, gtid_43959);\n            double deltam1_45523;\n            \n            if (cond_45522) {\n                double y_45525 = ((__global\n                                   double *) dzzt_mem_52452)[gtid_43959];\n                double x_45526 = 1.0 / y_45525;\n                double x_45527 = 0.5 * x_45526;\n                int64_t i_45528 = sub64(gtid_43959, 1);\n                bool x_45529 = sle64(0, i_45528);\n                bool y_45530 = slt64(i_45528, zzdim_41933);\n                bool bounds_check_45531 = x_45529 && y_45530;\n                bool index_certs_45534;\n                \n                if (!bounds_check_45531) {\n                    {\n                        if (atomic_cmpxchg_i32_global(gl",
                   "obal_failure, -1, 9) ==\n                            -1) {\n                            global_failure_args[0] = gtid_43954;\n                            global_failure_args[1] = gtid_43955;\n                            global_failure_args[2] = i_45528;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double x_45535 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_43954 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_43955 *\n                                                               zzdim_41979 +\n                                                               i_45528];\n                double y_45536 = kappa_45521 + x_45535;\n                double deltam1_t_res_45537 = x_45527 * y_45536;\n                \n                deltam1_45523 = deltam1_t_res_45537;\n            } else {\n                deltam1_45523 = 0.0;\n            }\n            \n            int64_t y_45538 = sub64(zzdim_41933, 1);\n            bool cond_45539 = slt64(gtid_43959, y_45538);\n            double delta_45540;\n            \n            if (cond_45539) {\n                int64_t i_45541 = add64(1, gtid_43959);\n                bool x_45542 = sle64(0, i_45541);\n                bool y_45543 = slt64(i_45541, zzdim_41933);\n                bool bounds_check_45544 = x_45542 && y_45543;\n                bool index_certs_45545;\n                \n                if (!bounds_check_45544) {\n                    {\n                        if (atomic_cmpxchg_i",
                   "32_global(global_failure, -1, 10) ==\n                            -1) {\n                            global_failure_args[0] = i_45541;\n                            global_failure_args[1] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double y_45546 = ((__global double *) dzzt_mem_52452)[i_45541];\n                double x_45547 = 1.0 / y_45546;\n                double x_45548 = 0.5 * x_45547;\n                bool index_certs_45551;\n                \n                if (!bounds_check_45544) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 11) ==\n                            -1) {\n                            global_failure_args[0] = gtid_43954;\n                            global_failure_args[1] = gtid_43955;\n                            global_failure_args[2] = i_45541;\n                            global_failure_args[3] = xdim_41931;\n                            global_failure_args[4] = ydim_41932;\n                            global_failure_args[5] = zzdim_41933;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double y_45552 = ((__global\n                                   double *) kappaM_mem_52457)[gtid_43954 *\n                                                               (zzdim_41979 *\n                                                                ydim_41978) +\n                                                               gtid_43955 *\n                                                               zzdim_41979 +\n                                                               i_45541];\n                double y_45553 = kappa_45521 + y_45552;\n                double delta_t_res_45554 = x_455",
                   "48 * y_45553;\n                \n                delta_45540 = delta_t_res_45554;\n            } else {\n                delta_45540 = 0.0;\n            }\n            \n            double dzzwzz_45556 = ((__global\n                                    double *) dzzw_mem_52453)[gtid_43959];\n            bool cond_f_res_45558 = !x_45520;\n            bool x_45559 = !x_45518;\n            bool y_45560 = cond_f_res_45558 && x_45559;\n            bool cond_45561 = x_45518 || y_45560;\n            double a_45562;\n            \n            if (cond_45561) {\n                a_45562 = 0.0;\n            } else {\n                bool x_45563 = cond_45522 && cond_45539;\n                double a_f_res_45564;\n                \n                if (x_45563) {\n                    double negate_arg_45565 = deltam1_45523 / dzzwzz_45556;\n                    double a_f_res_t_res_45566 = 0.0 - negate_arg_45565;\n                    \n                    a_f_res_45564 = a_f_res_t_res_45566;\n                } else {\n                    bool cond_45567 = gtid_43959 == y_45538;\n                    double a_f_res_f_res_45568;\n                    \n                    if (cond_45567) {\n                        double y_45569 = 0.5 * dzzwzz_45556;\n                        double negate_arg_45570 = deltam1_45523 / y_45569;\n                        double a_f_res_f_res_t_res_45571 = 0.0 -\n                               negate_arg_45570;\n                        \n                        a_f_res_f_res_45568 = a_f_res_f_res_t_res_45571;\n                    } else {\n                        a_f_res_f_res_45568 = 0.0;\n                    }\n                    a_f_res_45564 = a_f_res_f_res_45568;\n                }\n                a_45562 = a_f_res_45564;\n            }\n            \n            double b_45572;\n            \n            if (cond_f_res_45558) {\n                b_45572 = 1.0;\n            } else {\n                double mxls_45557 = ((__global\n                                      double *) mxl_mem_52458)[gtid_43954",
                   " *\n                                                               (zzdim_41982 *\n                                                                ydim_41981) +\n                                                               gtid_43955 *\n                                                               zzdim_41982 +\n                                                               gtid_43959];\n                double b_f_res_45573;\n                \n                if (x_45518) {\n                    double y_45574 = delta_45540 / dzzwzz_45556;\n                    double x_45575 = 1.0 + y_45574;\n                    double x_45576 = 0.7 / mxls_45557;\n                    double y_45577 = sqrt_res_45510 * x_45576;\n                    double b_f_res_t_res_45578 = x_45575 + y_45577;\n                    \n                    b_f_res_45573 = b_f_res_t_res_45578;\n                } else {\n                    bool x_45579 = cond_45522 && cond_45539;\n                    double b_f_res_f_res_45580;\n                    \n                    if (x_45579) {\n                        double x_45581 = deltam1_45523 + delta_45540;\n                        double y_45582 = x_45581 / dzzwzz_45556;\n                        double x_45583 = 1.0 + y_45582;\n                        double x_45584 = 0.7 * sqrt_res_45510;\n                        double y_45585 = x_45584 / mxls_45557;\n                        double b_f_res_f_res_t_res_45586 = x_45583 + y_45585;\n                        \n                        b_f_res_f_res_45580 = b_f_res_f_res_t_res_45586;\n                    } else {\n                        bool cond_45587 = gtid_43959 == y_45538;\n                        double b_f_res_f_res_f_res_45588;\n                        \n                        if (cond_45587) {\n                            double y_45589 = 0.5 * dzzwzz_45556;\n                            double y_45590 = deltam1_45523 / y_45589;\n                            double x_45591 = 1.0 + y_45590;\n                            double x_45592 = 0.7",
                   " / mxls_45557;\n                            double y_45593 = sqrt_res_45510 * x_45592;\n                            double b_f_res_f_res_f_res_t_res_45594 = x_45591 +\n                                   y_45593;\n                            \n                            b_f_res_f_res_f_res_45588 =\n                                b_f_res_f_res_f_res_t_res_45594;\n                        } else {\n                            b_f_res_f_res_f_res_45588 = 0.0;\n                        }\n                        b_f_res_f_res_45580 = b_f_res_f_res_f_res_45588;\n                    }\n                    b_f_res_45573 = b_f_res_f_res_45580;\n                }\n                b_45572 = b_f_res_45573;\n            }\n            \n            double lifted_0_f_res_t_res_45595;\n            double lifted_0_f_res_t_res_45596;\n            \n            if (cond_f_res_45558) {\n                lifted_0_f_res_t_res_45595 = 0.0;\n                lifted_0_f_res_t_res_45596 = 0.0;\n            } else {\n                double negate_arg_45597 = delta_45540 / dzzwzz_45556;\n                double c_45598 = 0.0 - negate_arg_45597;\n                double y_45599 = ((__global\n                                   double *) forc_mem_52459)[gtid_43954 *\n                                                             (zzdim_41985 *\n                                                              ydim_41984) +\n                                                             gtid_43955 *\n                                                             zzdim_41985 +\n                                                             gtid_43959];\n                double tmp_45600 = tke_45508 + y_45599;\n                bool cond_45601 = gtid_43959 == y_45538;\n                double lifted_0_f_res_t_res_f_res_45602;\n                \n                if (cond_45601) {\n                    double y_45603 = ((__global\n                                       double *) forc_tke_surface_mem_52460)[gtid_43954 *\n                                        ",
                   "                                     ydim_41987 +\n                                                                             gtid_43955];\n                    double y_45604 = 0.5 * dzzwzz_45556;\n                    double y_45605 = y_45603 / y_45604;\n                    double lifted_0_f_res_t_res_f_res_t_res_45606 = tmp_45600 +\n                           y_45605;\n                    \n                    lifted_0_f_res_t_res_f_res_45602 =\n                        lifted_0_f_res_t_res_f_res_t_res_45606;\n                } else {\n                    lifted_0_f_res_t_res_f_res_45602 = tmp_45600;\n                }\n                lifted_0_f_res_t_res_45595 = c_45598;\n                lifted_0_f_res_t_res_45596 = lifted_0_f_res_t_res_f_res_45602;\n            }\n            lifted_0_f_res_45492 = a_45562;\n            lifted_0_f_res_45493 = b_45572;\n            lifted_0_f_res_45494 = lifted_0_f_res_t_res_45595;\n            lifted_0_f_res_45495 = lifted_0_f_res_t_res_45596;\n        } else {\n            lifted_0_f_res_45492 = 0.0;\n            lifted_0_f_res_45493 = 0.0;\n            lifted_0_f_res_45494 = 0.0;\n            lifted_0_f_res_45495 = 0.0;\n        }\n        ((__local double *) mem_52562)[gtid_43959] = lifted_0_f_res_45492;\n        ((__local double *) mem_52564)[gtid_43959] = lifted_0_f_res_45493;\n        ((__local double *) mem_52566)[gtid_43959] = lifted_0_f_res_45494;\n        ((__local double *) mem_52568)[gtid_43959] = lifted_0_f_res_45495;\n    }\n    \n  error_0:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_52571;\n    \n    mem_52571 = (__local char *) mem_52571_backing_4;\n    \n    __local char *mem_52573;\n    \n    mem_52573 = (__local char *) mem_52573_backing_5;\n    \n    int64_t gtid_44077 = sext_i32_i64(ltid_pre_52911);\n    int32_t phys_tid_44078 = local_tid_52907;\n    \n    if (slt64(gtid_44077, zzdim_41933)) {\n        bool cond_45610 = gtid_44077 == 0;\n        double lifted_0",
                   "_f_res_45611;\n        \n        if (cond_45610) {\n            bool y_45612 = slt64(0, zzdim_41933);\n            bool index_certs_45613;\n            \n            if (!y_45612) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 12) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_45614 = ((__local double *) mem_52566)[0];\n            double y_45615 = ((__local double *) mem_52564)[0];\n            double lifted_0_f_res_t_res_45616 = x_45614 / y_45615;\n            \n            lifted_0_f_res_45611 = lifted_0_f_res_t_res_45616;\n        } else {\n            lifted_0_f_res_45611 = 0.0;\n        }\n        \n        double lifted_0_f_res_45617;\n        \n        if (cond_45610) {\n            bool y_45618 = slt64(0, zzdim_41933);\n            bool index_certs_45619;\n            \n            if (!y_45618) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 13) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_41933;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_45620 = ((__local double *) mem_52568)[0];\n            double y_45621 = ((__local double *) mem_52564)[0];\n            double lifted_0_f_res_t_res_45622 = x_45620 / y_45621;\n            \n            lifted_0_f_res_45617 = lifted_0_f_res_t_res_45622;\n        } else {\n            lifted_0_f_res_45617 = 0.0;\n        }\n        ((__local double *) mem_52571)[gtid_44077] = lifted_0_f_res_45617;\n        ((__local double *) mem_52573)[gtid_44077",
                   "] = lifted_0_f_res_45611;\n    }\n    \n  error_1:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_45625 = 0; i_45625 < distance_42130; i_45625++) {\n        int64_t index_primexp_45628 = add64(1, i_45625);\n        bool x_45629 = sle64(0, index_primexp_45628);\n        bool y_45630 = slt64(index_primexp_45628, zzdim_41933);\n        bool bounds_check_45631 = x_45629 && y_45630;\n        bool index_certs_45632;\n        \n        if (!bounds_check_45631) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 14) == -1) {\n                    global_failure_args[0] = index_primexp_45628;\n                    global_failure_args[1] = zzdim_41933;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double x_45633 = ((__local double *) mem_52564)[index_primexp_45628];\n        double x_45634 = ((__local double *) mem_52562)[index_primexp_45628];\n        bool y_45635 = slt64(i_45625, zzdim_41933);\n        bool index_certs_45636;\n        \n        if (!y_45635) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 15) == -1) {\n                    global_failure_args[0] = i_45625;\n                    global_failure_args[1] = zzdim_41933;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double y_45637 = ((__local double *) mem_52573)[i_45625];\n        double y_45638 = x_45634 * y_45637;\n        double y_45639 = x_45633 - y_45638;\n        double norm_factor_45640 = 1.0 / y_45639;\n        double x_45641 = ((__local double *) mem_52566)[index_primexp_45628];\n        double lw_val_45642 = norm_factor_45640 * x_45641;\n        \n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_52907 == 0) {\n            ((__local double *) mem_52573)[index_primexp_45628] = lw_v",
                   "al_45642;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        double x_45644 = ((__local double *) mem_52568)[index_primexp_45628];\n        double y_45645 = ((__local double *) mem_52571)[i_45625];\n        double y_45646 = x_45634 * y_45645;\n        double x_45647 = x_45644 - y_45646;\n        double lw_val_45648 = norm_factor_45640 * x_45647;\n        \n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_52907 == 0) {\n            ((__local double *) mem_52571)[index_primexp_45628] = lw_val_45648;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    \n    __local char *mem_52599;\n    \n    mem_52599 = (__local char *) mem_52599_backing_6;\n    ((__local double *) mem_52599)[sext_i32_i64(local_tid_52907)] = 0.0;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_52914 = 0; i_52914 < sdiv_up64(1 -\n                                                  sext_i32_i64(local_tid_52907),\n                                                  zzdim_41933); i_52914++) {\n        ((__local double *) mem_52599)[distance_42130 + (i_52914 * zzdim_41933 +\n                                                         sext_i32_i64(local_tid_52907))] =\n            ((__local double *) mem_52571)[distance_42130 + (i_52914 *\n                                                             zzdim_41933 +\n                                                             sext_i32_i64(local_tid_52907))];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_45654 = 0; i_45654 < distance_42130; i_45654++) {\n        int64_t binop_y_45656 = -1 * i_45654;\n        int64_t binop_x_45657 = m_42138 + binop_y_45656;\n        bool x_45658 = sle64(0, binop_x_45657);\n        bool y_45659 = slt64(binop_x_45657, zzdim_41933);\n        bool bounds_check_45660 = x_45658 && y_45659;\n        bool index_certs_45661;\n        \n        if (!bounds_check_45660) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 16) == -1) {\n                    global_failure_args[0] = binop_x_45657;\n",
                   "                    global_failure_args[1] = zzdim_41933;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double x_45662 = ((__local double *) mem_52571)[binop_x_45657];\n        double x_45663 = ((__local double *) mem_52573)[binop_x_45657];\n        int64_t i_45664 = add64(1, binop_x_45657);\n        bool x_45665 = sle64(0, i_45664);\n        bool y_45666 = slt64(i_45664, zzdim_41933);\n        bool bounds_check_45667 = x_45665 && y_45666;\n        bool index_certs_45668;\n        \n        if (!bounds_check_45667) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 17) == -1) {\n                    global_failure_args[0] = i_45664;\n                    global_failure_args[1] = zzdim_41933;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double y_45669 = ((__local double *) mem_52599)[i_45664];\n        double y_45670 = x_45663 * y_45669;\n        double lw_val_45671 = x_45662 - y_45670;\n        \n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_52907 == 0) {\n            ((__local double *) mem_52599)[binop_x_45657] = lw_val_45671;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    ((__global double *) mem_52615)[gtid_43954 * (zzdim_41933 * ydim_41932) +\n                                    gtid_43955 * zzdim_41933 +\n                                    sext_i32_i64(local_tid_52907)] = ((__local\n                                                                       double *) mem_52599)[sext_i32_i64(local_tid_52907)];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n  error_2:\n    return;\n}\n",
                   NULL};
static const char *size_names[] = {"builtin#replicate_f64.group_size_52882",
                                   "integrate_tke.segmap_group_size_44498",
                                   "integrate_tke.segmap_group_size_44539",
                                   "integrate_tke.segmap_group_size_44616",
                                   "integrate_tke.segmap_group_size_44694",
                                   "integrate_tke.segmap_group_size_44884",
                                   "integrate_tke.segmap_group_size_46179",
                                   "integrate_tke.segmap_group_size_46247",
                                   "integrate_tke.segmap_group_size_46994",
                                   "integrate_tke.segmap_group_size_47208",
                                   "integrate_tke.segmap_group_size_49044",
                                   "integrate_tke.segmap_group_size_51181",
                                   "integrate_tke.segmap_group_size_51978",
                                   "integrate_tke.segmap_num_groups_44500",
                                   "integrate_tke.segmap_num_groups_44618",
                                   "integrate_tke.suff_intra_par_1",
                                   "integrate_tke.suff_intra_par_3"};
static const char *size_vars[] = {"builtinzhreplicate_f64zigroup_sizze_52882",
                                  "integrate_tkezisegmap_group_sizze_44498",
                                  "integrate_tkezisegmap_group_sizze_44539",
                                  "integrate_tkezisegmap_group_sizze_44616",
                                  "integrate_tkezisegmap_group_sizze_44694",
                                  "integrate_tkezisegmap_group_sizze_44884",
                                  "integrate_tkezisegmap_group_sizze_46179",
                                  "integrate_tkezisegmap_group_sizze_46247",
                                  "integrate_tkezisegmap_group_sizze_46994",
                                  "integrate_tkezisegmap_group_sizze_47208",
                                  "integrate_tkezisegmap_group_sizze_49044",
                                  "integrate_tkezisegmap_group_sizze_51181",
                                  "integrate_tkezisegmap_group_sizze_51978",
                                  "integrate_tkezisegmap_num_groups_44500",
                                  "integrate_tkezisegmap_num_groups_44618",
                                  "integrate_tkezisuff_intra_par_1",
                                  "integrate_tkezisuff_intra_par_3"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "num_groups", "num_groups",
                                     "threshold ()",
                                     "threshold (!integrate_tke.suff_intra_par_1)"};
int futhark_get_num_sizes(void)
{
    return 17;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    size_t builtinzhreplicate_f64zigroup_sizze_52882;
    size_t integrate_tkezisegmap_group_sizze_44498;
    size_t integrate_tkezisegmap_group_sizze_44539;
    size_t integrate_tkezisegmap_group_sizze_44616;
    size_t integrate_tkezisegmap_group_sizze_44694;
    size_t integrate_tkezisegmap_group_sizze_44884;
    size_t integrate_tkezisegmap_group_sizze_46179;
    size_t integrate_tkezisegmap_group_sizze_46247;
    size_t integrate_tkezisegmap_group_sizze_46994;
    size_t integrate_tkezisegmap_group_sizze_47208;
    size_t integrate_tkezisegmap_group_sizze_49044;
    size_t integrate_tkezisegmap_group_sizze_51181;
    size_t integrate_tkezisegmap_group_sizze_51978;
    size_t integrate_tkezisegmap_num_groups_44500;
    size_t integrate_tkezisegmap_num_groups_44618;
    size_t integrate_tkezisuff_intra_par_1;
    size_t integrate_tkezisuff_intra_par_3;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[17];
    int num_build_opts;
    const char **build_opts;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->num_build_opts = 0;
    cfg->build_opts = (const char **) malloc(sizeof(const char *));
    cfg->build_opts[0] = NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cfg->sizes[5] = 0;
    cfg->sizes[6] = 0;
    cfg->sizes[7] = 0;
    cfg->sizes[8] = 0;
    cfg->sizes[9] = 0;
    cfg->sizes[10] = 0;
    cfg->sizes[11] = 0;
    cfg->sizes[12] = 0;
    cfg->sizes[13] = 0;
    cfg->sizes[14] = 0;
    cfg->sizes[15] = 32;
    cfg->sizes[16] = 32;
    opencl_config_init(&cfg->opencl, 17, size_names, size_vars, cfg->sizes,
                       size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg->build_opts);
    free(cfg);
}
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt)
{
    cfg->build_opts[cfg->num_build_opts] = opt;
    cfg->num_build_opts++;
    cfg->build_opts = (const char **) realloc(cfg->build_opts,
                                              (cfg->num_build_opts + 1) *
                                              sizeof(const char *));
    cfg->build_opts[cfg->num_build_opts] = NULL;
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = cfg->opencl.logging = cfg->opencl.debugging = flag;
}
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->opencl.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->opencl, s);
}
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s)
{
    set_preferred_platform(&cfg->opencl, s);
}
void futhark_context_config_select_device_interactively(struct futhark_context_config *cfg)
{
    select_device_interactively(&cfg->opencl);
}
void futhark_context_config_list_devices(struct futhark_context_config *cfg)
{
    (void) cfg;
    list_devices();
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->opencl.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->opencl.load_program_from = path;
}
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path)
{
    cfg->opencl.dump_binary_to = path;
}
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path)
{
    cfg->opencl.load_binary_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->opencl.default_group_size = size;
    cfg->opencl.default_group_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->opencl.default_num_groups = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_tile_size = size;
    cfg->opencl.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 17; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    if (strcmp(size_name, "default_group_size") == 0) {
        cfg->opencl.default_group_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_num_groups") == 0) {
        cfg->opencl.default_num_groups = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_threshold") == 0) {
        cfg->opencl.default_threshold = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_tile_size") == 0) {
        cfg->opencl.default_tile_size = size_value;
        return 0;
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    int profiling_paused;
    int logging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct {
        int dummy;
    } constants;
    int total_runs;
    long total_runtime;
    cl_kernel builtinzhreplicate_f64zireplicate_52879;
    cl_kernel gpu_map_transpose_f64;
    cl_kernel gpu_map_transpose_f64_low_height;
    cl_kernel gpu_map_transpose_f64_low_width;
    cl_kernel gpu_map_transpose_f64_small;
    cl_kernel integrate_tkezisegmap_44495;
    cl_kernel integrate_tkezisegmap_44535;
    cl_kernel integrate_tkezisegmap_44613;
    cl_kernel integrate_tkezisegmap_44690;
    cl_kernel integrate_tkezisegmap_44880;
    cl_kernel integrate_tkezisegmap_46175;
    cl_kernel integrate_tkezisegmap_46244;
    cl_kernel integrate_tkezisegmap_46990;
    cl_kernel integrate_tkezisegmap_47205;
    cl_kernel integrate_tkezisegmap_49040;
    cl_kernel integrate_tkezisegmap_51177;
    cl_kernel integrate_tkezisegmap_51974;
    cl_kernel integrate_tkezisegmap_intragroup_43307;
    cl_kernel integrate_tkezisegmap_intragroup_44093;
    int64_t copy_dev_to_dev_total_runtime;
    int copy_dev_to_dev_runs;
    int64_t copy_dev_to_host_total_runtime;
    int copy_dev_to_host_runs;
    int64_t copy_host_to_dev_total_runtime;
    int copy_host_to_dev_runs;
    int64_t copy_scalar_to_dev_total_runtime;
    int copy_scalar_to_dev_runs;
    int64_t copy_scalar_from_dev_total_runtime;
    int copy_scalar_from_dev_runs;
    int64_t builtinzhreplicate_f64zireplicate_52879_total_runtime;
    int builtinzhreplicate_f64zireplicate_52879_runs;
    int64_t gpu_map_transpose_f64_total_runtime;
    int gpu_map_transpose_f64_runs;
    int64_t gpu_map_transpose_f64_low_height_total_runtime;
    int gpu_map_transpose_f64_low_height_runs;
    int64_t gpu_map_transpose_f64_low_width_total_runtime;
    int gpu_map_transpose_f64_low_width_runs;
    int64_t gpu_map_transpose_f64_small_total_runtime;
    int gpu_map_transpose_f64_small_runs;
    int64_t integrate_tkezisegmap_44495_total_runtime;
    int integrate_tkezisegmap_44495_runs;
    int64_t integrate_tkezisegmap_44535_total_runtime;
    int integrate_tkezisegmap_44535_runs;
    int64_t integrate_tkezisegmap_44613_total_runtime;
    int integrate_tkezisegmap_44613_runs;
    int64_t integrate_tkezisegmap_44690_total_runtime;
    int integrate_tkezisegmap_44690_runs;
    int64_t integrate_tkezisegmap_44880_total_runtime;
    int integrate_tkezisegmap_44880_runs;
    int64_t integrate_tkezisegmap_46175_total_runtime;
    int integrate_tkezisegmap_46175_runs;
    int64_t integrate_tkezisegmap_46244_total_runtime;
    int integrate_tkezisegmap_46244_runs;
    int64_t integrate_tkezisegmap_46990_total_runtime;
    int integrate_tkezisegmap_46990_runs;
    int64_t integrate_tkezisegmap_47205_total_runtime;
    int integrate_tkezisegmap_47205_runs;
    int64_t integrate_tkezisegmap_49040_total_runtime;
    int integrate_tkezisegmap_49040_runs;
    int64_t integrate_tkezisegmap_51177_total_runtime;
    int integrate_tkezisegmap_51177_runs;
    int64_t integrate_tkezisegmap_51974_total_runtime;
    int integrate_tkezisegmap_51974_runs;
    int64_t integrate_tkezisegmap_intragroup_43307_total_runtime;
    int integrate_tkezisegmap_intragroup_43307_runs;
    int64_t integrate_tkezisegmap_intragroup_44093_total_runtime;
    int integrate_tkezisegmap_intragroup_44093_runs;
    cl_mem global_failure;
    cl_mem global_failure_args;
    struct opencl_context opencl;
    struct sizes sizes;
    cl_int failure_is_an_option;
} ;
void post_opencl_setup(struct opencl_context *ctx,
                       struct opencl_device_option *option)
{
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "NVIDIA CUDA") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
        ctx->lockstep_width = 32;
    }
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "AMD Accelerated Parallel Processing") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->lockstep_width = 32;
    }
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->lockstep_width = 1;
    }
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
        size_t MAX_COMPUTE_UNITS_val = 0;
        
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(MAX_COMPUTE_UNITS_val), &MAX_COMPUTE_UNITS_val,
                        NULL);
        ctx->cfg.default_num_groups = 4 * MAX_COMPUTE_UNITS_val;
    }
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
        ctx->cfg.default_group_size = 256;
    }
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->cfg.default_tile_size = 32;
    }
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU) {
        ctx->cfg.default_threshold = 32768;
    }
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU) {
        ctx->lockstep_width = 1;
    }
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
        size_t MAX_COMPUTE_UNITS_val = 0;
        
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(MAX_COMPUTE_UNITS_val), &MAX_COMPUTE_UNITS_val,
                        NULL);
        ctx->cfg.default_num_groups = MAX_COMPUTE_UNITS_val;
    }
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
        ctx->cfg.default_group_size = 32;
    }
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU) {
        ctx->cfg.default_tile_size = 4;
    }
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU) {
        size_t MAX_COMPUTE_UNITS_val = 0;
        
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(MAX_COMPUTE_UNITS_val), &MAX_COMPUTE_UNITS_val,
                        NULL);
        ctx->cfg.default_threshold = MAX_COMPUTE_UNITS_val;
    }
}
static void init_context_early(struct futhark_context_config *cfg,
                               struct futhark_context *ctx)
{
    ctx->opencl.cfg = cfg->opencl;
    ctx->detail_memory = cfg->opencl.debugging;
    ctx->debugging = cfg->opencl.debugging;
    ctx->profiling = cfg->opencl.profiling;
    ctx->profiling_paused = 0;
    ctx->logging = cfg->opencl.logging;
    ctx->error = NULL;
    ctx->opencl.profiling_records_capacity = 200;
    ctx->opencl.profiling_records_used = 0;
    ctx->opencl.profiling_records =
        malloc(ctx->opencl.profiling_records_capacity *
        sizeof(struct profiling_record));
    create_lock(&ctx->lock);
    ctx->failure_is_an_option = 0;
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->copy_dev_to_dev_total_runtime = 0;
    ctx->copy_dev_to_dev_runs = 0;
    ctx->copy_dev_to_host_total_runtime = 0;
    ctx->copy_dev_to_host_runs = 0;
    ctx->copy_host_to_dev_total_runtime = 0;
    ctx->copy_host_to_dev_runs = 0;
    ctx->copy_scalar_to_dev_total_runtime = 0;
    ctx->copy_scalar_to_dev_runs = 0;
    ctx->copy_scalar_from_dev_total_runtime = 0;
    ctx->copy_scalar_from_dev_runs = 0;
    ctx->builtinzhreplicate_f64zireplicate_52879_total_runtime = 0;
    ctx->builtinzhreplicate_f64zireplicate_52879_runs = 0;
    ctx->gpu_map_transpose_f64_total_runtime = 0;
    ctx->gpu_map_transpose_f64_runs = 0;
    ctx->gpu_map_transpose_f64_low_height_total_runtime = 0;
    ctx->gpu_map_transpose_f64_low_height_runs = 0;
    ctx->gpu_map_transpose_f64_low_width_total_runtime = 0;
    ctx->gpu_map_transpose_f64_low_width_runs = 0;
    ctx->gpu_map_transpose_f64_small_total_runtime = 0;
    ctx->gpu_map_transpose_f64_small_runs = 0;
    ctx->integrate_tkezisegmap_44495_total_runtime = 0;
    ctx->integrate_tkezisegmap_44495_runs = 0;
    ctx->integrate_tkezisegmap_44535_total_runtime = 0;
    ctx->integrate_tkezisegmap_44535_runs = 0;
    ctx->integrate_tkezisegmap_44613_total_runtime = 0;
    ctx->integrate_tkezisegmap_44613_runs = 0;
    ctx->integrate_tkezisegmap_44690_total_runtime = 0;
    ctx->integrate_tkezisegmap_44690_runs = 0;
    ctx->integrate_tkezisegmap_44880_total_runtime = 0;
    ctx->integrate_tkezisegmap_44880_runs = 0;
    ctx->integrate_tkezisegmap_46175_total_runtime = 0;
    ctx->integrate_tkezisegmap_46175_runs = 0;
    ctx->integrate_tkezisegmap_46244_total_runtime = 0;
    ctx->integrate_tkezisegmap_46244_runs = 0;
    ctx->integrate_tkezisegmap_46990_total_runtime = 0;
    ctx->integrate_tkezisegmap_46990_runs = 0;
    ctx->integrate_tkezisegmap_47205_total_runtime = 0;
    ctx->integrate_tkezisegmap_47205_runs = 0;
    ctx->integrate_tkezisegmap_49040_total_runtime = 0;
    ctx->integrate_tkezisegmap_49040_runs = 0;
    ctx->integrate_tkezisegmap_51177_total_runtime = 0;
    ctx->integrate_tkezisegmap_51177_runs = 0;
    ctx->integrate_tkezisegmap_51974_total_runtime = 0;
    ctx->integrate_tkezisegmap_51974_runs = 0;
    ctx->integrate_tkezisegmap_intragroup_43307_total_runtime = 0;
    ctx->integrate_tkezisegmap_intragroup_43307_runs = 0;
    ctx->integrate_tkezisegmap_intragroup_44093_total_runtime = 0;
    ctx->integrate_tkezisegmap_intragroup_44093_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    cl_int no_error = -1;
    
    ctx->global_failure = clCreateBuffer(ctx->opencl.ctx, CL_MEM_READ_WRITE |
                                         CL_MEM_COPY_HOST_PTR, sizeof(cl_int),
                                         &no_error, &error);
    OPENCL_SUCCEED_OR_RETURN(error);
    // The +1 is to avoid zero-byte allocations.
    ctx->global_failure_args = clCreateBuffer(ctx->opencl.ctx,
                                              CL_MEM_READ_WRITE,
                                              sizeof(int64_t) * (6 + 1), NULL,
                                              &error);
    OPENCL_SUCCEED_OR_RETURN(error);
    {
        ctx->builtinzhreplicate_f64zireplicate_52879 = clCreateKernel(prog,
                                                                      "builtinzhreplicate_f64zireplicate_52879",
                                                                      &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "builtin#replicate_f64.replicate_52879");
    }
    {
        ctx->gpu_map_transpose_f64 = clCreateKernel(prog,
                                                    "gpu_map_transpose_f64",
                                                    &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "gpu_map_transpose_f64");
    }
    {
        ctx->gpu_map_transpose_f64_low_height = clCreateKernel(prog,
                                                               "gpu_map_transpose_f64_low_height",
                                                               &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "gpu_map_transpose_f64_low_height");
    }
    {
        ctx->gpu_map_transpose_f64_low_width = clCreateKernel(prog,
                                                              "gpu_map_transpose_f64_low_width",
                                                              &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "gpu_map_transpose_f64_low_width");
    }
    {
        ctx->gpu_map_transpose_f64_small = clCreateKernel(prog,
                                                          "gpu_map_transpose_f64_small",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "gpu_map_transpose_f64_small");
    }
    {
        ctx->integrate_tkezisegmap_44495 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_44495",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44495, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44495, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_44495");
    }
    {
        ctx->integrate_tkezisegmap_44535 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_44535",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44535, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_44535");
    }
    {
        ctx->integrate_tkezisegmap_44613 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_44613",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44613, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44613, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_44613");
    }
    {
        ctx->integrate_tkezisegmap_44690 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_44690",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44690, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44690, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_44690");
    }
    {
        ctx->integrate_tkezisegmap_44880 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_44880",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44880, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_44880, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_44880");
    }
    {
        ctx->integrate_tkezisegmap_46175 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_46175",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_46175");
    }
    {
        ctx->integrate_tkezisegmap_46244 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_46244",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_46244");
    }
    {
        ctx->integrate_tkezisegmap_46990 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_46990",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_46990");
    }
    {
        ctx->integrate_tkezisegmap_47205 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_47205",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_47205");
    }
    {
        ctx->integrate_tkezisegmap_49040 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_49040",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_49040");
    }
    {
        ctx->integrate_tkezisegmap_51177 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_51177",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_51177");
    }
    {
        ctx->integrate_tkezisegmap_51974 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_51974",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_51974");
    }
    {
        ctx->integrate_tkezisegmap_intragroup_43307 = clCreateKernel(prog,
                                                                     "integrate_tkezisegmap_intragroup_43307",
                                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                            0, sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                            2, sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_intragroup_43307");
    }
    {
        ctx->integrate_tkezisegmap_intragroup_44093 = clCreateKernel(prog,
                                                                     "integrate_tkezisegmap_intragroup_44093",
                                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                            0, sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                            2, sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_intragroup_44093");
    }
    ctx->sizes.builtinzhreplicate_f64zigroup_sizze_52882 = cfg->sizes[0];
    ctx->sizes.integrate_tkezisegmap_group_sizze_44498 = cfg->sizes[1];
    ctx->sizes.integrate_tkezisegmap_group_sizze_44539 = cfg->sizes[2];
    ctx->sizes.integrate_tkezisegmap_group_sizze_44616 = cfg->sizes[3];
    ctx->sizes.integrate_tkezisegmap_group_sizze_44694 = cfg->sizes[4];
    ctx->sizes.integrate_tkezisegmap_group_sizze_44884 = cfg->sizes[5];
    ctx->sizes.integrate_tkezisegmap_group_sizze_46179 = cfg->sizes[6];
    ctx->sizes.integrate_tkezisegmap_group_sizze_46247 = cfg->sizes[7];
    ctx->sizes.integrate_tkezisegmap_group_sizze_46994 = cfg->sizes[8];
    ctx->sizes.integrate_tkezisegmap_group_sizze_47208 = cfg->sizes[9];
    ctx->sizes.integrate_tkezisegmap_group_sizze_49044 = cfg->sizes[10];
    ctx->sizes.integrate_tkezisegmap_group_sizze_51181 = cfg->sizes[11];
    ctx->sizes.integrate_tkezisegmap_group_sizze_51978 = cfg->sizes[12];
    ctx->sizes.integrate_tkezisegmap_num_groups_44500 = cfg->sizes[13];
    ctx->sizes.integrate_tkezisegmap_num_groups_44618 = cfg->sizes[14];
    ctx->sizes.integrate_tkezisuff_intra_par_1 = cfg->sizes[15];
    ctx->sizes.integrate_tkezisuff_intra_par_3 = cfg->sizes[16];
    init_constants(ctx);
    // Clear the free list of any deallocations that occurred while initialising constants.
    OPENCL_SUCCEED_OR_RETURN(opencl_free_all(&ctx->opencl));
    // The program will be properly freed after all the kernels have also been freed.
    OPENCL_SUCCEED_OR_RETURN(clReleaseProgram(prog));
    return futhark_context_sync(ctx);
}
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    required_types |= OPENCL_F64;
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program, required_types,
                                   cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
struct futhark_context *futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                                               cl_command_queue queue)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    required_types |= OPENCL_F64;
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl_with_command_queue(&ctx->opencl, queue,
                                                      opencl_program,
                                                      required_types,
                                                      cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_constants(ctx);
    free_lock(&ctx->lock);
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->builtinzhreplicate_f64zireplicate_52879));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64_low_height));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64_low_width));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64_small));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_44495));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_44535));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_44613));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_44690));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_44880));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_46175));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_46244));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_46990));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_47205));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_49040));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_51177));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_51974));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_intragroup_43307));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_intragroup_44093));
    teardown_opencl(&ctx->opencl);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    cl_int failure_idx = -1;
    
    if (ctx->failure_is_an_option) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     ctx->global_failure,
                                                     CL_FALSE, 0,
                                                     sizeof(cl_int),
                                                     &failure_idx, 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_scalar_from_dev_runs,
                                                                                               &ctx->copy_scalar_from_dev_total_runtime)));
        ctx->failure_is_an_option = 0;
    }
    OPENCL_SUCCEED_OR_RETURN(clFinish(ctx->opencl.queue));
    if (failure_idx >= 0) {
        cl_int no_failure = -1;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      ctx->global_failure,
                                                      CL_TRUE, 0,
                                                      sizeof(cl_int),
                                                      &no_failure, 0, NULL,
                                                      NULL));
        
        int64_t args[6 + 1];
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     ctx->global_failure_args,
                                                     CL_TRUE, 0, sizeof(args),
                                                     &args, 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_host_runs,
                                                                                               &ctx->copy_dev_to_host_total_runtime)));
        switch (failure_idx) {
            
          case 0:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:123:36-52\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 1:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:124:69-76\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 2:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:125:44-60\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 3:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:4:37-40\n   #1  tke.fut:4:13-65\n   #2  tke.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke.fut:20:4-24:12\n   #9  tke.fut:171:15-29\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 4:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:5:37-40\n   #1  tke.fut:5:13-65\n   #2  tke.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke.fut:20:4-24:12\n   #9  tke.fut:171:15-29\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 5:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:8:35-38\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 6:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:8:49-55\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 7:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:16:25-34\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 8:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:16:51-63\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 9:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:123:36-52\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 10:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:124:69-76\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 11:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:125:44-60\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 12:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:4:37-40\n   #1  tke.fut:4:13-65\n   #2  tke.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke.fut:20:4-24:12\n   #9  tke.fut:171:15-29\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 13:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:5:37-40\n   #1  tke.fut:5:13-65\n   #2  tke.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke.fut:20:4-24:12\n   #9  tke.fut:171:15-29\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 14:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:8:35-38\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 15:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:8:49-55\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 16:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:16:25-34\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 17:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:16:51-63\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 18:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:123:36-52\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 19:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:124:69-76\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 20:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:125:44-60\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:111:20-164:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 21:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:4:37-40\n   #1  tke.fut:4:13-65\n   #2  tke.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke.fut:20:4-24:12\n   #9  tke.fut:171:15-29\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 22:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:5:37-40\n   #1  tke.fut:5:13-65\n   #2  tke.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke.fut:20:4-24:12\n   #9  tke.fut:171:15-29\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 23:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:8:35-38\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 24:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:8:49-55\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 25:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:16:25-34\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 26:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:16:51-63\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 27:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:186:39-60\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  tke.fut:183:25-191:21\n   #7  tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 28:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:188:51-61\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  tke.fut:183:25-191:21\n   #7  tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 29:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:209:40-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:206:29-213:5\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 30:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:218:40-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:215:30-221:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 31:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:227:62-89\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:223:20-232:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 32:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:229:64-92\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:223:20-232:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 33:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:241:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:235:21-254:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 34:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:243:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:235:21-254:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 35:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:244:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:235:21-254:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 36:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:260:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:255:22-273:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 37:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:262:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:255:22-273:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 38:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:263:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:255:22-273:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 39:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:278:57-71\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:274:20-292:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 40:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:280:61-75\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:274:20-292:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 41:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:281:42-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:274:20-292:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 42:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:282:58-71\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:274:20-292:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 43:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:285:62-75\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:274:20-292:21\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 44:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:297:66-85\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:293:19-310:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 45:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:299:56-76\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:293:19-310:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 46:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:303:62-78\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:293:19-310:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 47:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke.fut:303:82-87\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:293:19-310:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 48:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:305:87-105\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:293:19-310:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 49:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke.fut:307:87-105\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke.fut:293:19-310:17\n   #10 tke.fut:62:1-316:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
        }
        return 1;
    }
    return 0;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    lock_lock(&ctx->lock);
    ctx->error = OPENCL_SUCCEED_NONFATAL(opencl_free_all(&ctx->opencl));
    lock_unlock(&ctx->lock);
    return ctx->error != NULL;
}
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx)
{
    return ctx->opencl.queue;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED_OR_RETURN(opencl_free(&ctx->opencl, block->mem,
                                                 desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "space 'device'",
                      ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    OPENCL_SUCCEED_OR_RETURN(opencl_alloc(&ctx->opencl, size, desc,
                                          &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "default space",
                      ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
char *futhark_context_report(struct futhark_context *ctx)
{
    struct str_builder builder;
    
    str_builder_init(&builder);
    if (ctx->detail_memory || ctx->profiling) {
        str_builder(&builder,
                    "Peak memory usage for space 'device': %lld bytes.\n",
                    (long long) ctx->peak_mem_usage_device);
        { }
    }
    if (ctx->profiling) {
        OPENCL_SUCCEED_FATAL(opencl_tally_profiling_records(&ctx->opencl));
        str_builder(&builder,
                    "copy_dev_to_dev                       ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_dev_to_dev_runs,
                    (long) ctx->copy_dev_to_dev_total_runtime /
                    (ctx->copy_dev_to_dev_runs !=
                     0 ? ctx->copy_dev_to_dev_runs : 1),
                    (long) ctx->copy_dev_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_dev_runs;
        str_builder(&builder,
                    "copy_dev_to_host                      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_dev_to_host_runs,
                    (long) ctx->copy_dev_to_host_total_runtime /
                    (ctx->copy_dev_to_host_runs !=
                     0 ? ctx->copy_dev_to_host_runs : 1),
                    (long) ctx->copy_dev_to_host_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_host_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_host_runs;
        str_builder(&builder,
                    "copy_host_to_dev                      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_host_to_dev_runs,
                    (long) ctx->copy_host_to_dev_total_runtime /
                    (ctx->copy_host_to_dev_runs !=
                     0 ? ctx->copy_host_to_dev_runs : 1),
                    (long) ctx->copy_host_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_host_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_host_to_dev_runs;
        str_builder(&builder,
                    "copy_scalar_to_dev                    ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_scalar_to_dev_runs,
                    (long) ctx->copy_scalar_to_dev_total_runtime /
                    (ctx->copy_scalar_to_dev_runs !=
                     0 ? ctx->copy_scalar_to_dev_runs : 1),
                    (long) ctx->copy_scalar_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_to_dev_runs;
        str_builder(&builder,
                    "copy_scalar_from_dev                  ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->copy_scalar_from_dev_runs,
                    (long) ctx->copy_scalar_from_dev_total_runtime /
                    (ctx->copy_scalar_from_dev_runs !=
                     0 ? ctx->copy_scalar_from_dev_runs : 1),
                    (long) ctx->copy_scalar_from_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_from_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_from_dev_runs;
        str_builder(&builder,
                    "builtin#replicate_f64.replicate_52879 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->builtinzhreplicate_f64zireplicate_52879_runs,
                    (long) ctx->builtinzhreplicate_f64zireplicate_52879_total_runtime /
                    (ctx->builtinzhreplicate_f64zireplicate_52879_runs !=
                     0 ? ctx->builtinzhreplicate_f64zireplicate_52879_runs : 1),
                    (long) ctx->builtinzhreplicate_f64zireplicate_52879_total_runtime);
        ctx->total_runtime +=
            ctx->builtinzhreplicate_f64zireplicate_52879_total_runtime;
        ctx->total_runs += ctx->builtinzhreplicate_f64zireplicate_52879_runs;
        str_builder(&builder,
                    "gpu_map_transpose_f64                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->gpu_map_transpose_f64_runs,
                    (long) ctx->gpu_map_transpose_f64_total_runtime /
                    (ctx->gpu_map_transpose_f64_runs !=
                     0 ? ctx->gpu_map_transpose_f64_runs : 1),
                    (long) ctx->gpu_map_transpose_f64_total_runtime);
        ctx->total_runtime += ctx->gpu_map_transpose_f64_total_runtime;
        ctx->total_runs += ctx->gpu_map_transpose_f64_runs;
        str_builder(&builder,
                    "gpu_map_transpose_f64_low_height      ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->gpu_map_transpose_f64_low_height_runs,
                    (long) ctx->gpu_map_transpose_f64_low_height_total_runtime /
                    (ctx->gpu_map_transpose_f64_low_height_runs !=
                     0 ? ctx->gpu_map_transpose_f64_low_height_runs : 1),
                    (long) ctx->gpu_map_transpose_f64_low_height_total_runtime);
        ctx->total_runtime +=
            ctx->gpu_map_transpose_f64_low_height_total_runtime;
        ctx->total_runs += ctx->gpu_map_transpose_f64_low_height_runs;
        str_builder(&builder,
                    "gpu_map_transpose_f64_low_width       ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->gpu_map_transpose_f64_low_width_runs,
                    (long) ctx->gpu_map_transpose_f64_low_width_total_runtime /
                    (ctx->gpu_map_transpose_f64_low_width_runs !=
                     0 ? ctx->gpu_map_transpose_f64_low_width_runs : 1),
                    (long) ctx->gpu_map_transpose_f64_low_width_total_runtime);
        ctx->total_runtime +=
            ctx->gpu_map_transpose_f64_low_width_total_runtime;
        ctx->total_runs += ctx->gpu_map_transpose_f64_low_width_runs;
        str_builder(&builder,
                    "gpu_map_transpose_f64_small           ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->gpu_map_transpose_f64_small_runs,
                    (long) ctx->gpu_map_transpose_f64_small_total_runtime /
                    (ctx->gpu_map_transpose_f64_small_runs !=
                     0 ? ctx->gpu_map_transpose_f64_small_runs : 1),
                    (long) ctx->gpu_map_transpose_f64_small_total_runtime);
        ctx->total_runtime += ctx->gpu_map_transpose_f64_small_total_runtime;
        ctx->total_runs += ctx->gpu_map_transpose_f64_small_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_44495            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_44495_runs,
                    (long) ctx->integrate_tkezisegmap_44495_total_runtime /
                    (ctx->integrate_tkezisegmap_44495_runs !=
                     0 ? ctx->integrate_tkezisegmap_44495_runs : 1),
                    (long) ctx->integrate_tkezisegmap_44495_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_44495_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_44495_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_44535            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_44535_runs,
                    (long) ctx->integrate_tkezisegmap_44535_total_runtime /
                    (ctx->integrate_tkezisegmap_44535_runs !=
                     0 ? ctx->integrate_tkezisegmap_44535_runs : 1),
                    (long) ctx->integrate_tkezisegmap_44535_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_44535_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_44535_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_44613            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_44613_runs,
                    (long) ctx->integrate_tkezisegmap_44613_total_runtime /
                    (ctx->integrate_tkezisegmap_44613_runs !=
                     0 ? ctx->integrate_tkezisegmap_44613_runs : 1),
                    (long) ctx->integrate_tkezisegmap_44613_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_44613_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_44613_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_44690            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_44690_runs,
                    (long) ctx->integrate_tkezisegmap_44690_total_runtime /
                    (ctx->integrate_tkezisegmap_44690_runs !=
                     0 ? ctx->integrate_tkezisegmap_44690_runs : 1),
                    (long) ctx->integrate_tkezisegmap_44690_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_44690_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_44690_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_44880            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_44880_runs,
                    (long) ctx->integrate_tkezisegmap_44880_total_runtime /
                    (ctx->integrate_tkezisegmap_44880_runs !=
                     0 ? ctx->integrate_tkezisegmap_44880_runs : 1),
                    (long) ctx->integrate_tkezisegmap_44880_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_44880_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_44880_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_46175            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_46175_runs,
                    (long) ctx->integrate_tkezisegmap_46175_total_runtime /
                    (ctx->integrate_tkezisegmap_46175_runs !=
                     0 ? ctx->integrate_tkezisegmap_46175_runs : 1),
                    (long) ctx->integrate_tkezisegmap_46175_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_46175_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_46175_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_46244            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_46244_runs,
                    (long) ctx->integrate_tkezisegmap_46244_total_runtime /
                    (ctx->integrate_tkezisegmap_46244_runs !=
                     0 ? ctx->integrate_tkezisegmap_46244_runs : 1),
                    (long) ctx->integrate_tkezisegmap_46244_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_46244_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_46244_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_46990            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_46990_runs,
                    (long) ctx->integrate_tkezisegmap_46990_total_runtime /
                    (ctx->integrate_tkezisegmap_46990_runs !=
                     0 ? ctx->integrate_tkezisegmap_46990_runs : 1),
                    (long) ctx->integrate_tkezisegmap_46990_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_46990_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_46990_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_47205            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_47205_runs,
                    (long) ctx->integrate_tkezisegmap_47205_total_runtime /
                    (ctx->integrate_tkezisegmap_47205_runs !=
                     0 ? ctx->integrate_tkezisegmap_47205_runs : 1),
                    (long) ctx->integrate_tkezisegmap_47205_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_47205_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_47205_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_49040            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_49040_runs,
                    (long) ctx->integrate_tkezisegmap_49040_total_runtime /
                    (ctx->integrate_tkezisegmap_49040_runs !=
                     0 ? ctx->integrate_tkezisegmap_49040_runs : 1),
                    (long) ctx->integrate_tkezisegmap_49040_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_49040_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_49040_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_51177            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_51177_runs,
                    (long) ctx->integrate_tkezisegmap_51177_total_runtime /
                    (ctx->integrate_tkezisegmap_51177_runs !=
                     0 ? ctx->integrate_tkezisegmap_51177_runs : 1),
                    (long) ctx->integrate_tkezisegmap_51177_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_51177_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_51177_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_51974            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_51974_runs,
                    (long) ctx->integrate_tkezisegmap_51974_total_runtime /
                    (ctx->integrate_tkezisegmap_51974_runs !=
                     0 ? ctx->integrate_tkezisegmap_51974_runs : 1),
                    (long) ctx->integrate_tkezisegmap_51974_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_51974_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_51974_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_intragroup_43307 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_intragroup_43307_runs,
                    (long) ctx->integrate_tkezisegmap_intragroup_43307_total_runtime /
                    (ctx->integrate_tkezisegmap_intragroup_43307_runs !=
                     0 ? ctx->integrate_tkezisegmap_intragroup_43307_runs : 1),
                    (long) ctx->integrate_tkezisegmap_intragroup_43307_total_runtime);
        ctx->total_runtime +=
            ctx->integrate_tkezisegmap_intragroup_43307_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_intragroup_43307_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_intragroup_44093 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_intragroup_44093_runs,
                    (long) ctx->integrate_tkezisegmap_intragroup_44093_total_runtime /
                    (ctx->integrate_tkezisegmap_intragroup_44093_runs !=
                     0 ? ctx->integrate_tkezisegmap_intragroup_44093_runs : 1),
                    (long) ctx->integrate_tkezisegmap_intragroup_44093_total_runtime);
        ctx->total_runtime +=
            ctx->integrate_tkezisegmap_intragroup_44093_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_intragroup_44093_runs;
        str_builder(&builder, "%d operations with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
    return builder.str;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 1;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 0;
}
static int futrts_builtinzhgpu_map_transpose_f64(struct futhark_context *ctx,
                                                 struct memblock_device destmem_0,
                                                 int32_t destoffset_1,
                                                 struct memblock_device srcmem_2,
                                                 int32_t srcoffset_3,
                                                 int32_t num_arrays_4,
                                                 int32_t x_elems_5,
                                                 int32_t y_elems_6);
static int futrts_builtinzhreplicate_f64(struct futhark_context *ctx,
                                         struct memblock_device mem_52875,
                                         int32_t num_elems_52876,
                                         double val_52877);
static int futrts_integrate_tke(struct futhark_context *ctx,
                                struct memblock_device *out_mem_p_53018,
                                int64_t *out_out_arrsizze_53019,
                                int64_t *out_out_arrsizze_53020,
                                int64_t *out_out_arrsizze_53021,
                                struct memblock_device *out_mem_p_53022,
                                int64_t *out_out_arrsizze_53023,
                                int64_t *out_out_arrsizze_53024,
                                int64_t *out_out_arrsizze_53025,
                                struct memblock_device *out_mem_p_53026,
                                int64_t *out_out_arrsizze_53027,
                                int64_t *out_out_arrsizze_53028,
                                int64_t *out_out_arrsizze_53029,
                                struct memblock_device *out_mem_p_53030,
                                int64_t *out_out_arrsizze_53031,
                                int64_t *out_out_arrsizze_53032,
                                int64_t *out_out_arrsizze_53033,
                                struct memblock_device *out_mem_p_53034,
                                int64_t *out_out_arrsizze_53035,
                                int64_t *out_out_arrsizze_53036,
                                int64_t *out_out_arrsizze_53037,
                                struct memblock_device *out_mem_p_53038,
                                int64_t *out_out_arrsizze_53039,
                                int64_t *out_out_arrsizze_53040,
                                int64_t *out_out_arrsizze_53041,
                                struct memblock_device *out_mem_p_53042,
                                int64_t *out_out_arrsizze_53043,
                                int64_t *out_out_arrsizze_53044,
                                struct memblock_device tketau_mem_52436,
                                struct memblock_device tketaup1_mem_52437,
                                struct memblock_device tketaum1_mem_52438,
                                struct memblock_device dtketau_mem_52439,
                                struct memblock_device dtketaup1_mem_52440,
                                struct memblock_device dtketaum1_mem_52441,
                                struct memblock_device utau_mem_52442,
                                struct memblock_device vtau_mem_52443,
                                struct memblock_device wtau_mem_52444,
                                struct memblock_device maskU_mem_52445,
                                struct memblock_device maskV_mem_52446,
                                struct memblock_device maskW_mem_52447,
                                struct memblock_device dxt_mem_52448,
                                struct memblock_device dxu_mem_52449,
                                struct memblock_device dyt_mem_52450,
                                struct memblock_device dyu_mem_52451,
                                struct memblock_device dzzt_mem_52452,
                                struct memblock_device dzzw_mem_52453,
                                struct memblock_device cost_mem_52454,
                                struct memblock_device cosu_mem_52455,
                                struct memblock_device kbot_mem_52456,
                                struct memblock_device kappaM_mem_52457,
                                struct memblock_device mxl_mem_52458,
                                struct memblock_device forc_mem_52459,
                                struct memblock_device forc_tke_surface_mem_52460,
                                int64_t xdim_41931, int64_t ydim_41932,
                                int64_t zzdim_41933, int64_t xdim_41934,
                                int64_t ydim_41935, int64_t zzdim_41936,
                                int64_t xdim_41937, int64_t ydim_41938,
                                int64_t zzdim_41939, int64_t xdim_41940,
                                int64_t ydim_41941, int64_t zzdim_41942,
                                int64_t xdim_41943, int64_t ydim_41944,
                                int64_t zzdim_41945, int64_t xdim_41946,
                                int64_t ydim_41947, int64_t zzdim_41948,
                                int64_t xdim_41949, int64_t ydim_41950,
                                int64_t zzdim_41951, int64_t xdim_41952,
                                int64_t ydim_41953, int64_t zzdim_41954,
                                int64_t xdim_41955, int64_t ydim_41956,
                                int64_t zzdim_41957, int64_t xdim_41958,
                                int64_t ydim_41959, int64_t zzdim_41960,
                                int64_t xdim_41961, int64_t ydim_41962,
                                int64_t zzdim_41963, int64_t xdim_41964,
                                int64_t ydim_41965, int64_t zzdim_41966,
                                int64_t xdim_41967, int64_t xdim_41968,
                                int64_t ydim_41969, int64_t ydim_41970,
                                int64_t zzdim_41971, int64_t zzdim_41972,
                                int64_t ydim_41973, int64_t ydim_41974,
                                int64_t xdim_41975, int64_t ydim_41976,
                                int64_t xdim_41977, int64_t ydim_41978,
                                int64_t zzdim_41979, int64_t xdim_41980,
                                int64_t ydim_41981, int64_t zzdim_41982,
                                int64_t xdim_41983, int64_t ydim_41984,
                                int64_t zzdim_41985, int64_t xdim_41986,
                                int64_t ydim_41987);
static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
static int futrts_builtinzhgpu_map_transpose_f64(struct futhark_context *ctx,
                                                 struct memblock_device destmem_0,
                                                 int32_t destoffset_1,
                                                 struct memblock_device srcmem_2,
                                                 int32_t srcoffset_3,
                                                 int32_t num_arrays_4,
                                                 int32_t x_elems_5,
                                                 int32_t y_elems_6)
{
    (void) ctx;
    
    int err = 0;
    
    if (!(num_arrays_4 == 0 || (x_elems_5 == 0 || y_elems_6 == 0))) {
        int32_t muly_8 = squot32(16, x_elems_5);
        int32_t mulx_7 = squot32(16, y_elems_6);
        
        if (num_arrays_4 == 1 && (x_elems_5 == 1 || y_elems_6 == 1)) {
            if (sext_i32_i64(x_elems_5 * y_elems_6 * (int64_t) sizeof(double)) >
                0) {
                OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                             srcmem_2.mem,
                                                             destmem_0.mem,
                                                             sext_i32_i64(srcoffset_3),
                                                             sext_i32_i64(destoffset_1),
                                                             sext_i32_i64(x_elems_5 *
                                                             y_elems_6 *
                                                             (int64_t) sizeof(double)),
                                                             0, NULL,
                                                             ctx->profiling_paused ||
                                                             !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                       &ctx->copy_dev_to_dev_runs,
                                                                                                       &ctx->copy_dev_to_dev_total_runtime)));
                if (ctx->debugging)
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            }
        } else {
            if (sle32(x_elems_5, 8) && slt32(16, y_elems_6)) {
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        0, 2176, NULL));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        1, sizeof(destoffset_1),
                                                        &destoffset_1));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        2, sizeof(srcoffset_3),
                                                        &srcoffset_3));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        3, sizeof(num_arrays_4),
                                                        &num_arrays_4));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        4, sizeof(x_elems_5),
                                                        &x_elems_5));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        5, sizeof(y_elems_6),
                                                        &y_elems_6));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        6, sizeof(mulx_7),
                                                        &mulx_7));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        7, sizeof(muly_8),
                                                        &muly_8));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        8,
                                                        sizeof(destmem_0.mem),
                                                        &destmem_0.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_width,
                                                        9, sizeof(srcmem_2.mem),
                                                        &srcmem_2.mem));
                if (1 * ((size_t) sdiv_up32(x_elems_5, 16) * (size_t) 16) *
                    ((size_t) sdiv_up32(sdiv_up32(y_elems_6, muly_8), 16) *
                     (size_t) 16) * ((size_t) num_arrays_4 * (size_t) 1) != 0) {
                    const size_t global_work_sizze_52993[3] =
                                 {(size_t) sdiv_up32(x_elems_5, 16) *
                                  (size_t) 16,
                                  (size_t) sdiv_up32(sdiv_up32(y_elems_6,
                                                               muly_8), 16) *
                                  (size_t) 16, (size_t) num_arrays_4 *
                                  (size_t) 1};
                    const size_t local_work_sizze_52997[3] = {16, 16, 1};
                    int64_t time_start_52994 = 0, time_end_52995 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "gpu_map_transpose_f64_low_width");
                        fprintf(stderr, "%zu", global_work_sizze_52993[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_52993[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_52993[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_52997[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_52997[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_52997[2]);
                        fprintf(stderr,
                                "]; local memory parameters sum to %d bytes.\n",
                                (int) (0 + 2176));
                        time_start_52994 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->gpu_map_transpose_f64_low_width,
                                                                    3, NULL,
                                                                    global_work_sizze_52993,
                                                                    local_work_sizze_52997,
                                                                    0, NULL,
                                                                    ctx->profiling_paused ||
                                                                    !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                              &ctx->gpu_map_transpose_f64_low_width_runs,
                                                                                                              &ctx->gpu_map_transpose_f64_low_width_total_runtime)));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_52995 = get_wall_time();
                        
                        long time_diff_52996 = time_end_52995 -
                             time_start_52994;
                        
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "gpu_map_transpose_f64_low_width",
                                time_diff_52996);
                    }
                }
            } else {
                if (sle32(y_elems_6, 8) && slt32(16, x_elems_5)) {
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            0, 2176, NULL));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            1,
                                                            sizeof(destoffset_1),
                                                            &destoffset_1));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            2,
                                                            sizeof(srcoffset_3),
                                                            &srcoffset_3));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            3,
                                                            sizeof(num_arrays_4),
                                                            &num_arrays_4));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            4,
                                                            sizeof(x_elems_5),
                                                            &x_elems_5));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            5,
                                                            sizeof(y_elems_6),
                                                            &y_elems_6));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            6, sizeof(mulx_7),
                                                            &mulx_7));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            7, sizeof(muly_8),
                                                            &muly_8));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            8,
                                                            sizeof(destmem_0.mem),
                                                            &destmem_0.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_low_height,
                                                            9,
                                                            sizeof(srcmem_2.mem),
                                                            &srcmem_2.mem));
                    if (1 * ((size_t) sdiv_up32(sdiv_up32(x_elems_5, mulx_7),
                                                16) * (size_t) 16) *
                        ((size_t) sdiv_up32(y_elems_6, 16) * (size_t) 16) *
                        ((size_t) num_arrays_4 * (size_t) 1) != 0) {
                        const size_t global_work_sizze_52998[3] =
                                     {(size_t) sdiv_up32(sdiv_up32(x_elems_5,
                                                                   mulx_7),
                                                         16) * (size_t) 16,
                                      (size_t) sdiv_up32(y_elems_6, 16) *
                                      (size_t) 16, (size_t) num_arrays_4 *
                                      (size_t) 1};
                        const size_t local_work_sizze_53002[3] = {16, 16, 1};
                        int64_t time_start_52999 = 0, time_end_53000 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "gpu_map_transpose_f64_low_height");
                            fprintf(stderr, "%zu", global_work_sizze_52998[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_52998[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_52998[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_53002[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_53002[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_53002[2]);
                            fprintf(stderr,
                                    "]; local memory parameters sum to %d bytes.\n",
                                    (int) (0 + 2176));
                            time_start_52999 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->gpu_map_transpose_f64_low_height,
                                                                        3, NULL,
                                                                        global_work_sizze_52998,
                                                                        local_work_sizze_53002,
                                                                        0, NULL,
                                                                        ctx->profiling_paused ||
                                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                  &ctx->gpu_map_transpose_f64_low_height_runs,
                                                                                                                  &ctx->gpu_map_transpose_f64_low_height_total_runtime)));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_53000 = get_wall_time();
                            
                            long time_diff_53001 = time_end_53000 -
                                 time_start_52999;
                            
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "gpu_map_transpose_f64_low_height",
                                    time_diff_53001);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, 8) && sle32(y_elems_6, 8)) {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                0, 1, NULL));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                2,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                3,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                6,
                                                                sizeof(mulx_7),
                                                                &mulx_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                7,
                                                                sizeof(muly_8),
                                                                &muly_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                8,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64_small,
                                                                9,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * ((size_t) sdiv_up32(num_arrays_4 * x_elems_5 *
                                                    y_elems_6, 256) *
                                 (size_t) 256) != 0) {
                            const size_t global_work_sizze_53003[1] =
                                         {(size_t) sdiv_up32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256) *
                                         (size_t) 256};
                            const size_t local_work_sizze_53007[1] = {256};
                            int64_t time_start_53004 = 0, time_end_53005 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "gpu_map_transpose_f64_small");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_53003[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_53007[0]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 1));
                                time_start_53004 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->gpu_map_transpose_f64_small,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_53003,
                                                                            local_work_sizze_53007,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ||
                                                                            !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                      &ctx->gpu_map_transpose_f64_small_runs,
                                                                                                                      &ctx->gpu_map_transpose_f64_small_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_53005 = get_wall_time();
                                
                                long time_diff_53006 = time_end_53005 -
                                     time_start_53004;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "gpu_map_transpose_f64_small",
                                        time_diff_53006);
                            }
                        }
                    } else {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                0, 8448, NULL));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                2,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                3,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                6,
                                                                sizeof(mulx_7),
                                                                &mulx_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                7,
                                                                sizeof(muly_8),
                                                                &muly_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                8,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->gpu_map_transpose_f64,
                                                                9,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * ((size_t) sdiv_up32(x_elems_5, 32) *
                                 (size_t) 32) * ((size_t) sdiv_up32(y_elems_6,
                                                                    32) *
                                                 (size_t) 8) *
                            ((size_t) num_arrays_4 * (size_t) 1) != 0) {
                            const size_t global_work_sizze_53008[3] =
                                         {(size_t) sdiv_up32(x_elems_5, 32) *
                                          (size_t) 32,
                                          (size_t) sdiv_up32(y_elems_6, 32) *
                                          (size_t) 8, (size_t) num_arrays_4 *
                                          (size_t) 1};
                            const size_t local_work_sizze_53012[3] = {32, 8, 1};
                            int64_t time_start_53009 = 0, time_end_53010 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "gpu_map_transpose_f64");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_53008[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_53008[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_53008[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_53012[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_53012[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_53012[2]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 8448));
                                time_start_53009 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->gpu_map_transpose_f64,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_53008,
                                                                            local_work_sizze_53012,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ||
                                                                            !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                      &ctx->gpu_map_transpose_f64_runs,
                                                                                                                      &ctx->gpu_map_transpose_f64_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_53010 = get_wall_time();
                                
                                long time_diff_53011 = time_end_53010 -
                                     time_start_53009;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "gpu_map_transpose_f64",
                                        time_diff_53011);
                            }
                        }
                    }
                }
            }
        }
    }
    
  cleanup:
    { }
    return err;
}
static int futrts_builtinzhreplicate_f64(struct futhark_context *ctx,
                                         struct memblock_device mem_52875,
                                         int32_t num_elems_52876,
                                         double val_52877)
{
    (void) ctx;
    
    int err = 0;
    int64_t group_sizze_52882;
    
    group_sizze_52882 = ctx->sizes.builtinzhreplicate_f64zigroup_sizze_52882;
    
    int64_t num_groups_52883;
    
    num_groups_52883 = sdiv_up64(num_elems_52876, group_sizze_52882);
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->builtinzhreplicate_f64zireplicate_52879,
                                            0, sizeof(mem_52875.mem),
                                            &mem_52875.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->builtinzhreplicate_f64zireplicate_52879,
                                            1, sizeof(num_elems_52876),
                                            &num_elems_52876));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->builtinzhreplicate_f64zireplicate_52879,
                                            2, sizeof(val_52877), &val_52877));
    if (1 * ((size_t) num_groups_52883 * (size_t) group_sizze_52882) != 0) {
        const size_t global_work_sizze_53013[1] = {(size_t) num_groups_52883 *
                     (size_t) group_sizze_52882};
        const size_t local_work_sizze_53017[1] = {group_sizze_52882};
        int64_t time_start_53014 = 0, time_end_53015 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "builtin#replicate_f64.replicate_52879");
            fprintf(stderr, "%zu", global_work_sizze_53013[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53017[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53014 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->builtinzhreplicate_f64zireplicate_52879,
                                                        1, NULL,
                                                        global_work_sizze_53013,
                                                        local_work_sizze_53017,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->builtinzhreplicate_f64zireplicate_52879_runs,
                                                                                                  &ctx->builtinzhreplicate_f64zireplicate_52879_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53015 = get_wall_time();
            
            long time_diff_53016 = time_end_53015 - time_start_53014;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "builtin#replicate_f64.replicate_52879", time_diff_53016);
        }
    }
    
  cleanup:
    { }
    return err;
}
static int futrts_integrate_tke(struct futhark_context *ctx,
                                struct memblock_device *out_mem_p_53018,
                                int64_t *out_out_arrsizze_53019,
                                int64_t *out_out_arrsizze_53020,
                                int64_t *out_out_arrsizze_53021,
                                struct memblock_device *out_mem_p_53022,
                                int64_t *out_out_arrsizze_53023,
                                int64_t *out_out_arrsizze_53024,
                                int64_t *out_out_arrsizze_53025,
                                struct memblock_device *out_mem_p_53026,
                                int64_t *out_out_arrsizze_53027,
                                int64_t *out_out_arrsizze_53028,
                                int64_t *out_out_arrsizze_53029,
                                struct memblock_device *out_mem_p_53030,
                                int64_t *out_out_arrsizze_53031,
                                int64_t *out_out_arrsizze_53032,
                                int64_t *out_out_arrsizze_53033,
                                struct memblock_device *out_mem_p_53034,
                                int64_t *out_out_arrsizze_53035,
                                int64_t *out_out_arrsizze_53036,
                                int64_t *out_out_arrsizze_53037,
                                struct memblock_device *out_mem_p_53038,
                                int64_t *out_out_arrsizze_53039,
                                int64_t *out_out_arrsizze_53040,
                                int64_t *out_out_arrsizze_53041,
                                struct memblock_device *out_mem_p_53042,
                                int64_t *out_out_arrsizze_53043,
                                int64_t *out_out_arrsizze_53044,
                                struct memblock_device tketau_mem_52436,
                                struct memblock_device tketaup1_mem_52437,
                                struct memblock_device tketaum1_mem_52438,
                                struct memblock_device dtketau_mem_52439,
                                struct memblock_device dtketaup1_mem_52440,
                                struct memblock_device dtketaum1_mem_52441,
                                struct memblock_device utau_mem_52442,
                                struct memblock_device vtau_mem_52443,
                                struct memblock_device wtau_mem_52444,
                                struct memblock_device maskU_mem_52445,
                                struct memblock_device maskV_mem_52446,
                                struct memblock_device maskW_mem_52447,
                                struct memblock_device dxt_mem_52448,
                                struct memblock_device dxu_mem_52449,
                                struct memblock_device dyt_mem_52450,
                                struct memblock_device dyu_mem_52451,
                                struct memblock_device dzzt_mem_52452,
                                struct memblock_device dzzw_mem_52453,
                                struct memblock_device cost_mem_52454,
                                struct memblock_device cosu_mem_52455,
                                struct memblock_device kbot_mem_52456,
                                struct memblock_device kappaM_mem_52457,
                                struct memblock_device mxl_mem_52458,
                                struct memblock_device forc_mem_52459,
                                struct memblock_device forc_tke_surface_mem_52460,
                                int64_t xdim_41931, int64_t ydim_41932,
                                int64_t zzdim_41933, int64_t xdim_41934,
                                int64_t ydim_41935, int64_t zzdim_41936,
                                int64_t xdim_41937, int64_t ydim_41938,
                                int64_t zzdim_41939, int64_t xdim_41940,
                                int64_t ydim_41941, int64_t zzdim_41942,
                                int64_t xdim_41943, int64_t ydim_41944,
                                int64_t zzdim_41945, int64_t xdim_41946,
                                int64_t ydim_41947, int64_t zzdim_41948,
                                int64_t xdim_41949, int64_t ydim_41950,
                                int64_t zzdim_41951, int64_t xdim_41952,
                                int64_t ydim_41953, int64_t zzdim_41954,
                                int64_t xdim_41955, int64_t ydim_41956,
                                int64_t zzdim_41957, int64_t xdim_41958,
                                int64_t ydim_41959, int64_t zzdim_41960,
                                int64_t xdim_41961, int64_t ydim_41962,
                                int64_t zzdim_41963, int64_t xdim_41964,
                                int64_t ydim_41965, int64_t zzdim_41966,
                                int64_t xdim_41967, int64_t xdim_41968,
                                int64_t ydim_41969, int64_t ydim_41970,
                                int64_t zzdim_41971, int64_t zzdim_41972,
                                int64_t ydim_41973, int64_t ydim_41974,
                                int64_t xdim_41975, int64_t ydim_41976,
                                int64_t xdim_41977, int64_t ydim_41978,
                                int64_t zzdim_41979, int64_t xdim_41980,
                                int64_t ydim_41981, int64_t zzdim_41982,
                                int64_t xdim_41983, int64_t ydim_41984,
                                int64_t zzdim_41985, int64_t xdim_41986,
                                int64_t ydim_41987)
{
    (void) ctx;
    
    int err = 0;
    struct memblock_device out_mem_52847;
    
    out_mem_52847.references = NULL;
    
    int64_t out_arrsizze_52848;
    int64_t out_arrsizze_52849;
    int64_t out_arrsizze_52850;
    struct memblock_device out_mem_52851;
    
    out_mem_52851.references = NULL;
    
    int64_t out_arrsizze_52852;
    int64_t out_arrsizze_52853;
    int64_t out_arrsizze_52854;
    struct memblock_device out_mem_52855;
    
    out_mem_52855.references = NULL;
    
    int64_t out_arrsizze_52856;
    int64_t out_arrsizze_52857;
    int64_t out_arrsizze_52858;
    struct memblock_device out_mem_52859;
    
    out_mem_52859.references = NULL;
    
    int64_t out_arrsizze_52860;
    int64_t out_arrsizze_52861;
    int64_t out_arrsizze_52862;
    struct memblock_device out_mem_52863;
    
    out_mem_52863.references = NULL;
    
    int64_t out_arrsizze_52864;
    int64_t out_arrsizze_52865;
    int64_t out_arrsizze_52866;
    struct memblock_device out_mem_52867;
    
    out_mem_52867.references = NULL;
    
    int64_t out_arrsizze_52868;
    int64_t out_arrsizze_52869;
    int64_t out_arrsizze_52870;
    struct memblock_device out_mem_52871;
    
    out_mem_52871.references = NULL;
    
    int64_t out_arrsizze_52872;
    int64_t out_arrsizze_52873;
    bool dim_match_42013 = xdim_41931 == xdim_41934;
    bool dim_match_42014 = ydim_41932 == ydim_41935;
    bool dim_match_42015 = zzdim_41933 == zzdim_41936;
    bool y_42016 = dim_match_42013 && dim_match_42015;
    bool match_42017 = dim_match_42014 && y_42016;
    bool empty_or_match_cert_42018;
    
    if (!match_42017) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42019 = xdim_41931 == xdim_41937;
    bool dim_match_42020 = ydim_41932 == ydim_41938;
    bool dim_match_42021 = zzdim_41933 == zzdim_41939;
    bool y_42022 = dim_match_42019 && dim_match_42021;
    bool match_42023 = dim_match_42020 && y_42022;
    bool empty_or_match_cert_42024;
    
    if (!match_42023) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42026 = xdim_41931 == xdim_41940;
    bool dim_match_42027 = ydim_41932 == ydim_41941;
    bool dim_match_42028 = zzdim_41933 == zzdim_41942;
    bool y_42029 = dim_match_42026 && dim_match_42028;
    bool match_42030 = dim_match_42027 && y_42029;
    bool empty_or_match_cert_42031;
    
    if (!match_42030) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42032 = xdim_41931 == xdim_41943;
    bool dim_match_42033 = ydim_41932 == ydim_41944;
    bool dim_match_42034 = zzdim_41933 == zzdim_41945;
    bool y_42035 = dim_match_42032 && dim_match_42034;
    bool match_42036 = dim_match_42033 && y_42035;
    bool empty_or_match_cert_42037;
    
    if (!match_42036) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42039 = xdim_41931 == xdim_41946;
    bool dim_match_42040 = ydim_41932 == ydim_41947;
    bool dim_match_42041 = zzdim_41933 == zzdim_41948;
    bool y_42042 = dim_match_42039 && dim_match_42041;
    bool match_42043 = dim_match_42040 && y_42042;
    bool empty_or_match_cert_42044;
    
    if (!match_42043) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42046 = xdim_41931 == xdim_41949;
    bool dim_match_42047 = ydim_41932 == ydim_41950;
    bool dim_match_42048 = zzdim_41933 == zzdim_41951;
    bool y_42049 = dim_match_42046 && dim_match_42048;
    bool match_42050 = dim_match_42047 && y_42049;
    bool empty_or_match_cert_42051;
    
    if (!match_42050) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42052 = xdim_41931 == xdim_41952;
    bool dim_match_42053 = ydim_41932 == ydim_41953;
    bool dim_match_42054 = zzdim_41933 == zzdim_41954;
    bool y_42055 = dim_match_42052 && dim_match_42054;
    bool match_42056 = dim_match_42053 && y_42055;
    bool empty_or_match_cert_42057;
    
    if (!match_42056) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42058 = xdim_41931 == xdim_41955;
    bool dim_match_42059 = ydim_41932 == ydim_41956;
    bool dim_match_42060 = zzdim_41933 == zzdim_41957;
    bool y_42061 = dim_match_42058 && dim_match_42060;
    bool match_42062 = dim_match_42059 && y_42061;
    bool empty_or_match_cert_42063;
    
    if (!match_42062) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42064 = xdim_41931 == xdim_41958;
    bool dim_match_42065 = ydim_41932 == ydim_41959;
    bool dim_match_42066 = zzdim_41933 == zzdim_41960;
    bool y_42067 = dim_match_42064 && dim_match_42066;
    bool match_42068 = dim_match_42065 && y_42067;
    bool empty_or_match_cert_42069;
    
    if (!match_42068) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42070 = xdim_41931 == xdim_41961;
    bool dim_match_42071 = ydim_41932 == ydim_41962;
    bool dim_match_42072 = zzdim_41933 == zzdim_41963;
    bool y_42073 = dim_match_42070 && dim_match_42072;
    bool match_42074 = dim_match_42071 && y_42073;
    bool empty_or_match_cert_42075;
    
    if (!match_42074) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42076 = xdim_41931 == xdim_41964;
    bool dim_match_42077 = ydim_41932 == ydim_41965;
    bool dim_match_42078 = zzdim_41933 == zzdim_41966;
    bool y_42079 = dim_match_42076 && dim_match_42078;
    bool match_42080 = dim_match_42077 && y_42079;
    bool empty_or_match_cert_42081;
    
    if (!match_42080) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42082 = xdim_41931 == xdim_41967;
    bool empty_or_match_cert_42083;
    
    if (!dim_match_42082) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42084 = xdim_41931 == xdim_41968;
    bool empty_or_match_cert_42085;
    
    if (!dim_match_42084) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42086 = ydim_41932 == ydim_41969;
    bool empty_or_match_cert_42087;
    
    if (!dim_match_42086) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42088 = ydim_41932 == ydim_41970;
    bool empty_or_match_cert_42089;
    
    if (!dim_match_42088) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42090 = zzdim_41933 == zzdim_41971;
    bool empty_or_match_cert_42091;
    
    if (!dim_match_42090) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42092 = zzdim_41933 == zzdim_41972;
    bool empty_or_match_cert_42093;
    
    if (!dim_match_42092) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42094 = ydim_41932 == ydim_41973;
    bool empty_or_match_cert_42095;
    
    if (!dim_match_42094) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42096 = ydim_41932 == ydim_41974;
    bool empty_or_match_cert_42097;
    
    if (!dim_match_42096) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42098 = xdim_41931 == xdim_41975;
    bool dim_match_42099 = ydim_41932 == ydim_41976;
    bool match_42100 = dim_match_42098 && dim_match_42099;
    bool empty_or_match_cert_42101;
    
    if (!match_42100) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42102 = xdim_41931 == xdim_41977;
    bool dim_match_42103 = ydim_41932 == ydim_41978;
    bool dim_match_42104 = zzdim_41933 == zzdim_41979;
    bool y_42105 = dim_match_42102 && dim_match_42104;
    bool match_42106 = dim_match_42103 && y_42105;
    bool empty_or_match_cert_42107;
    
    if (!match_42106) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42108 = xdim_41931 == xdim_41980;
    bool dim_match_42109 = ydim_41932 == ydim_41981;
    bool dim_match_42110 = zzdim_41933 == zzdim_41982;
    bool y_42111 = dim_match_42108 && dim_match_42110;
    bool match_42112 = dim_match_42109 && y_42111;
    bool empty_or_match_cert_42113;
    
    if (!match_42112) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42114 = xdim_41931 == xdim_41983;
    bool dim_match_42115 = ydim_41932 == ydim_41984;
    bool dim_match_42116 = zzdim_41933 == zzdim_41985;
    bool y_42117 = dim_match_42114 && dim_match_42116;
    bool match_42118 = dim_match_42115 && y_42117;
    bool empty_or_match_cert_42119;
    
    if (!match_42118) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_42120 = xdim_41931 == xdim_41986;
    bool dim_match_42121 = ydim_41932 == ydim_41987;
    bool match_42122 = dim_match_42120 && dim_match_42121;
    bool empty_or_match_cert_42123;
    
    if (!match_42122) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    int64_t y_42127 = sub64(xdim_41931, 2);
    int64_t y_42128 = sub64(ydim_41932, 2);
    bool bounds_invalid_upwards_42129 = slt64(zzdim_41933, 1);
    int64_t distance_42130 = sub64(zzdim_41933, 1);
    bool valid_42131 = !bounds_invalid_upwards_42129;
    bool range_valid_c_42132;
    
    if (!valid_42131) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..<", zzdim_41933, " is invalid.",
                               "-> #0  tke.fut:7:30-34\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool x_42133 = sle64(0, distance_42130);
    bool y_42134 = slt64(distance_42130, zzdim_41933);
    bool bounds_check_42135 = x_42133 && y_42134;
    bool index_certs_42136;
    
    if (!bounds_check_42135) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Index [", distance_42130,
                               "] out of bounds for array of shape [",
                               zzdim_41933, "].",
                               "-> #0  tke.fut:13:24-35\n   #1  tke.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke.fut:20:4-24:12\n   #8  tke.fut:171:15-29\n   #9  tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    bool empty_slice_42137 = distance_42130 == 0;
    int64_t m_42138 = sub64(distance_42130, 1);
    bool zzero_leq_i_p_m_t_s_42139 = sle64(0, m_42138);
    bool i_p_m_t_s_leq_w_42140 = slt64(m_42138, zzdim_41933);
    bool y_42141 = zzero_leq_i_p_m_t_s_42139 && i_p_m_t_s_leq_w_42140;
    bool y_42142 = x_42133 && y_42141;
    bool ok_or_empty_42143 = empty_slice_42137 || y_42142;
    bool index_certs_42144;
    
    if (!ok_or_empty_42143) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Index [", 0, ":", distance_42130,
                               "] out of bounds for array of shape [",
                               zzdim_41933, "].",
                               "-> #0  /prelude/array.fut:24:29-36\n   #1  tke.fut:14:24-39\n   #2  tke.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke.fut:20:4-24:12\n   #9  tke.fut:171:15-29\n   #10 tke.fut:62:1-316:81\n");
        if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
            return 1;
        return 1;
    }
    
    int64_t one_intra_par_min_43301 = ydim_41932 * zzdim_41933;
    int64_t y_43304 = smin64(ydim_41932, one_intra_par_min_43301);
    int64_t intra_avail_par_43305 = smin64(ydim_41932, y_43304);
    int64_t computed_group_sizze_43057 = smax64(ydim_41932,
                                                one_intra_par_min_43301);
    int64_t max_group_sizze_43715;
    
    max_group_sizze_43715 = ctx->opencl.max_group_size;
    
    bool fits_43716 = sle64(computed_group_sizze_43057, max_group_sizze_43715);
    bool suff_intra_par_43714;
    
    suff_intra_par_43714 = ctx->sizes.integrate_tkezisuff_intra_par_1 <=
        intra_avail_par_43305;
    if (ctx->logging)
        fprintf(stderr, "Compared %s <= %d.\n",
                "integrate_tke.suff_intra_par_1", intra_avail_par_43305);
    
    bool intra_suff_and_fits_43717 = suff_intra_par_43714 && fits_43716;
    int64_t intra_num_groups_45472 = xdim_41931 * ydim_41932;
    bool fits_45474 = sle64(zzdim_41933, max_group_sizze_43715);
    bool suff_intra_par_45476;
    
    suff_intra_par_45476 = ctx->sizes.integrate_tkezisuff_intra_par_3 <=
        zzdim_41933;
    if (ctx->logging)
        fprintf(stderr, "Compared %s <= %d.\n",
                "integrate_tke.suff_intra_par_3", zzdim_41933);
    
    bool intra_suff_and_fits_45477 = fits_45474 && suff_intra_par_45476;
    int64_t nest_sizze_45695 = xdim_41931 * one_intra_par_min_43301;
    int64_t segmap_group_sizze_45696;
    
    segmap_group_sizze_45696 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_44884;
    
    int64_t segmap_group_sizze_45837;
    
    segmap_group_sizze_45837 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_44694;
    
    int64_t segmap_group_sizze_45862;
    
    segmap_group_sizze_45862 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_44616;
    
    int64_t num_groups_45863;
    int32_t max_num_groups_52874;
    
    max_num_groups_52874 = ctx->sizes.integrate_tkezisegmap_num_groups_44618;
    num_groups_45863 = sext_i64_i32(smax64(1,
                                           smin64(sdiv_up64(intra_num_groups_45472,
                                                            segmap_group_sizze_45862),
                                                  sext_i32_i64(max_num_groups_52874))));
    
    int64_t binop_x_52463 = zzdim_41933 * intra_num_groups_45472;
    int64_t bytes_52461 = 8 * binop_x_52463;
    struct memblock_device mem_52464;
    
    mem_52464.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52464, bytes_52461, "mem_52464")) {
        err = 1;
        goto cleanup;
    }
    if (futrts_builtinzhreplicate_f64(ctx, mem_52464, xdim_41931 * ydim_41932 *
                                      zzdim_41933, 0.0) != 0) {
        err = 1;
        goto cleanup;
    }
    
    int64_t segmap_group_sizze_45930;
    
    segmap_group_sizze_45930 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_44539;
    
    int64_t segmap_group_sizze_45941;
    
    segmap_group_sizze_45941 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_44498;
    
    int64_t num_groups_45942;
    int32_t max_num_groups_52884;
    
    max_num_groups_52884 = ctx->sizes.integrate_tkezisegmap_num_groups_44500;
    num_groups_45942 = sext_i64_i32(smax64(1,
                                           smin64(sdiv_up64(intra_num_groups_45472,
                                                            segmap_group_sizze_45941),
                                                  sext_i32_i64(max_num_groups_52884))));
    
    int64_t bytes_52467 = 8 * one_intra_par_min_43301;
    int64_t bytes_52503 = 8 * zzdim_41933;
    int64_t binop_x_52643 = xdim_41931 * zzdim_41933;
    int64_t binop_x_52644 = ydim_41932 * binop_x_52643;
    int64_t bytes_52642 = 8 * binop_x_52644;
    int64_t num_threads_52834 = segmap_group_sizze_45862 * num_groups_45863;
    int64_t total_sizze_52835 = bytes_52503 * num_threads_52834;
    int64_t total_sizze_52836 = bytes_52503 * num_threads_52834;
    int64_t num_threads_52838 = segmap_group_sizze_45941 * num_groups_45942;
    int64_t total_sizze_52839 = bytes_52503 * num_threads_52838;
    struct memblock_device lifted_11_map_res_mem_52749;
    
    lifted_11_map_res_mem_52749.references = NULL;
    
    int32_t local_memory_capacity_52957;
    
    local_memory_capacity_52957 = ctx->opencl.max_local_memory;
    if (sle64(bytes_52467 + bytes_52467 + bytes_52467 + bytes_52467 +
              bytes_52467 + bytes_52467 + bytes_52467 + bytes_52467 +
              bytes_52503 + bytes_52503 + bytes_52467 + bytes_52467 +
              bytes_52503, sext_i32_i64(local_memory_capacity_52957)) &&
        intra_suff_and_fits_43717) {
        struct memblock_device mem_52558;
        
        mem_52558.references = NULL;
        if (memblock_alloc_device(ctx, &mem_52558, bytes_52461, "mem_52558")) {
            err = 1;
            goto cleanup;
        }
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                1,
                                                sizeof(ctx->failure_is_an_option),
                                                &ctx->failure_is_an_option));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                3, bytes_52503, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                4, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                5, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                6, bytes_52503, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                7, bytes_52503, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                8, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                9, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                10, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                11, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                12, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                13, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                14, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                15, bytes_52467, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                16, sizeof(xdim_41931),
                                                &xdim_41931));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                17, sizeof(ydim_41932),
                                                &ydim_41932));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                18, sizeof(zzdim_41933),
                                                &zzdim_41933));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                19, sizeof(ydim_41976),
                                                &ydim_41976));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                20, sizeof(ydim_41978),
                                                &ydim_41978));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                21, sizeof(zzdim_41979),
                                                &zzdim_41979));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                22, sizeof(ydim_41981),
                                                &ydim_41981));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                23, sizeof(zzdim_41982),
                                                &zzdim_41982));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                24, sizeof(ydim_41984),
                                                &ydim_41984));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                25, sizeof(zzdim_41985),
                                                &zzdim_41985));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                26, sizeof(ydim_41987),
                                                &ydim_41987));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                27, sizeof(y_42127), &y_42127));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                28, sizeof(y_42128), &y_42128));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                29, sizeof(distance_42130),
                                                &distance_42130));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                30, sizeof(m_42138), &m_42138));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                31,
                                                sizeof(computed_group_sizze_43057),
                                                &computed_group_sizze_43057));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                32,
                                                sizeof(tketau_mem_52436.mem),
                                                &tketau_mem_52436.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                33, sizeof(dzzt_mem_52452.mem),
                                                &dzzt_mem_52452.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                34, sizeof(dzzw_mem_52453.mem),
                                                &dzzw_mem_52453.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                35, sizeof(kbot_mem_52456.mem),
                                                &kbot_mem_52456.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                36,
                                                sizeof(kappaM_mem_52457.mem),
                                                &kappaM_mem_52457.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                37, sizeof(mxl_mem_52458.mem),
                                                &mxl_mem_52458.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                38, sizeof(forc_mem_52459.mem),
                                                &forc_mem_52459.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                39,
                                                sizeof(forc_tke_surface_mem_52460.mem),
                                                &forc_tke_surface_mem_52460.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_43307,
                                                40, sizeof(mem_52558.mem),
                                                &mem_52558.mem));
        if (1 * ((size_t) xdim_41931 * (size_t) computed_group_sizze_43057) !=
            0) {
            const size_t global_work_sizze_53045[1] = {(size_t) xdim_41931 *
                         (size_t) computed_group_sizze_43057};
            const size_t local_work_sizze_53049[1] =
                         {computed_group_sizze_43057};
            int64_t time_start_53046 = 0, time_end_53047 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "integrate_tke.segmap_intragroup_43307");
                fprintf(stderr, "%zu", global_work_sizze_53045[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_53049[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + bytes_52503 + bytes_52467 + bytes_52467 +
                               bytes_52503 + bytes_52503 + bytes_52467 +
                               bytes_52467 + bytes_52467 + bytes_52467 +
                               bytes_52467 + bytes_52467 + bytes_52467 +
                               bytes_52467));
                time_start_53046 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->integrate_tkezisegmap_intragroup_43307,
                                                            1, NULL,
                                                            global_work_sizze_53045,
                                                            local_work_sizze_53049,
                                                            0, NULL,
                                                            ctx->profiling_paused ||
                                                            !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                      &ctx->integrate_tkezisegmap_intragroup_43307_runs,
                                                                                                      &ctx->integrate_tkezisegmap_intragroup_43307_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_53047 = get_wall_time();
                
                long time_diff_53048 = time_end_53047 - time_start_53046;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "integrate_tke.segmap_intragroup_43307",
                        time_diff_53048);
            }
        }
        ctx->failure_is_an_option = 1;
        if (memblock_set_device(ctx, &lifted_11_map_res_mem_52749, &mem_52558,
                                "mem_52558") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_52558, "mem_52558") != 0)
            return 1;
    } else {
        struct memblock_device lifted_11_map_res_mem_52748;
        
        lifted_11_map_res_mem_52748.references = NULL;
        
        int32_t local_memory_capacity_52956;
        
        local_memory_capacity_52956 = ctx->opencl.max_local_memory;
        if (sle64(bytes_52503 + bytes_52503 + bytes_52503 + bytes_52503 +
                  bytes_52503 + bytes_52503 + bytes_52503,
                  sext_i32_i64(local_memory_capacity_52956)) &&
            intra_suff_and_fits_45477) {
            struct memblock_device mem_52615;
            
            mem_52615.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52615, bytes_52461,
                                      "mem_52615")) {
                err = 1;
                goto cleanup;
            }
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    3, bytes_52503, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    4, bytes_52503, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    5, bytes_52503, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    6, bytes_52503, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    7, bytes_52503, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    8, bytes_52503, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    9, bytes_52503, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    10, sizeof(xdim_41931),
                                                    &xdim_41931));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    11, sizeof(ydim_41932),
                                                    &ydim_41932));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    12, sizeof(zzdim_41933),
                                                    &zzdim_41933));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    13, sizeof(ydim_41976),
                                                    &ydim_41976));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    14, sizeof(ydim_41978),
                                                    &ydim_41978));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    15, sizeof(zzdim_41979),
                                                    &zzdim_41979));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    16, sizeof(ydim_41981),
                                                    &ydim_41981));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    17, sizeof(zzdim_41982),
                                                    &zzdim_41982));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    18, sizeof(ydim_41984),
                                                    &ydim_41984));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    19, sizeof(zzdim_41985),
                                                    &zzdim_41985));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    20, sizeof(ydim_41987),
                                                    &ydim_41987));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    21, sizeof(y_42127),
                                                    &y_42127));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    22, sizeof(y_42128),
                                                    &y_42128));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    23, sizeof(distance_42130),
                                                    &distance_42130));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    24, sizeof(m_42138),
                                                    &m_42138));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    25,
                                                    sizeof(tketau_mem_52436.mem),
                                                    &tketau_mem_52436.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    26,
                                                    sizeof(dzzt_mem_52452.mem),
                                                    &dzzt_mem_52452.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    27,
                                                    sizeof(dzzw_mem_52453.mem),
                                                    &dzzw_mem_52453.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    28,
                                                    sizeof(kbot_mem_52456.mem),
                                                    &kbot_mem_52456.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    29,
                                                    sizeof(kappaM_mem_52457.mem),
                                                    &kappaM_mem_52457.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    30,
                                                    sizeof(mxl_mem_52458.mem),
                                                    &mxl_mem_52458.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    31,
                                                    sizeof(forc_mem_52459.mem),
                                                    &forc_mem_52459.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    32,
                                                    sizeof(forc_tke_surface_mem_52460.mem),
                                                    &forc_tke_surface_mem_52460.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_44093,
                                                    33, sizeof(mem_52615.mem),
                                                    &mem_52615.mem));
            if (1 * ((size_t) intra_num_groups_45472 * (size_t) zzdim_41933) !=
                0) {
                const size_t global_work_sizze_53050[1] =
                             {(size_t) intra_num_groups_45472 *
                             (size_t) zzdim_41933};
                const size_t local_work_sizze_53054[1] = {zzdim_41933};
                int64_t time_start_53051 = 0, time_end_53052 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_intragroup_44093");
                    fprintf(stderr, "%zu", global_work_sizze_53050[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_53054[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) (0 + bytes_52503 + bytes_52503 + bytes_52503 +
                                   bytes_52503 + bytes_52503 + bytes_52503 +
                                   bytes_52503));
                    time_start_53051 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_intragroup_44093,
                                                                1, NULL,
                                                                global_work_sizze_53050,
                                                                local_work_sizze_53054,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_intragroup_44093_runs,
                                                                                                          &ctx->integrate_tkezisegmap_intragroup_44093_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_53052 = get_wall_time();
                    
                    long time_diff_53053 = time_end_53052 - time_start_53051;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_intragroup_44093",
                            time_diff_53053);
                }
            }
            ctx->failure_is_an_option = 1;
            if (memblock_set_device(ctx, &lifted_11_map_res_mem_52748,
                                    &mem_52615, "mem_52615") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52615, "mem_52615") != 0)
                return 1;
        } else {
            int64_t segmap_usable_groups_45697 = sdiv_up64(nest_sizze_45695,
                                                           segmap_group_sizze_45696);
            struct memblock_device mem_52620;
            
            mem_52620.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52620, bytes_52461,
                                      "mem_52620")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52624;
            
            mem_52624.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52624, bytes_52461,
                                      "mem_52624")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52628;
            
            mem_52628.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52628, bytes_52461,
                                      "mem_52628")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52632;
            
            mem_52632.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52632, bytes_52461,
                                      "mem_52632")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    3, sizeof(xdim_41931),
                                                    &xdim_41931));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    4, sizeof(ydim_41932),
                                                    &ydim_41932));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    5, sizeof(zzdim_41933),
                                                    &zzdim_41933));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    6, sizeof(ydim_41976),
                                                    &ydim_41976));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    7, sizeof(ydim_41978),
                                                    &ydim_41978));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    8, sizeof(zzdim_41979),
                                                    &zzdim_41979));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    9, sizeof(ydim_41981),
                                                    &ydim_41981));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    10, sizeof(zzdim_41982),
                                                    &zzdim_41982));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    11, sizeof(ydim_41984),
                                                    &ydim_41984));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    12, sizeof(zzdim_41985),
                                                    &zzdim_41985));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    13, sizeof(ydim_41987),
                                                    &ydim_41987));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    14, sizeof(y_42127),
                                                    &y_42127));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    15, sizeof(y_42128),
                                                    &y_42128));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    16,
                                                    sizeof(tketau_mem_52436.mem),
                                                    &tketau_mem_52436.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    17,
                                                    sizeof(dzzt_mem_52452.mem),
                                                    &dzzt_mem_52452.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    18,
                                                    sizeof(dzzw_mem_52453.mem),
                                                    &dzzw_mem_52453.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    19,
                                                    sizeof(kbot_mem_52456.mem),
                                                    &kbot_mem_52456.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    20,
                                                    sizeof(kappaM_mem_52457.mem),
                                                    &kappaM_mem_52457.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    21,
                                                    sizeof(mxl_mem_52458.mem),
                                                    &mxl_mem_52458.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    22,
                                                    sizeof(forc_mem_52459.mem),
                                                    &forc_mem_52459.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    23,
                                                    sizeof(forc_tke_surface_mem_52460.mem),
                                                    &forc_tke_surface_mem_52460.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    24, sizeof(mem_52620.mem),
                                                    &mem_52620.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    25, sizeof(mem_52624.mem),
                                                    &mem_52624.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    26, sizeof(mem_52628.mem),
                                                    &mem_52628.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44880,
                                                    27, sizeof(mem_52632.mem),
                                                    &mem_52632.mem));
            if (1 * ((size_t) segmap_usable_groups_45697 *
                     (size_t) segmap_group_sizze_45696) != 0) {
                const size_t global_work_sizze_53055[1] =
                             {(size_t) segmap_usable_groups_45697 *
                             (size_t) segmap_group_sizze_45696};
                const size_t local_work_sizze_53059[1] =
                             {segmap_group_sizze_45696};
                int64_t time_start_53056 = 0, time_end_53057 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_44880");
                    fprintf(stderr, "%zu", global_work_sizze_53055[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_53059[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_53056 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_44880,
                                                                1, NULL,
                                                                global_work_sizze_53055,
                                                                local_work_sizze_53059,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_44880_runs,
                                                                                                          &ctx->integrate_tkezisegmap_44880_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_53057 = get_wall_time();
                    
                    long time_diff_53058 = time_end_53057 - time_start_53056;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_44880", time_diff_53058);
                }
            }
            ctx->failure_is_an_option = 1;
            
            int64_t segmap_usable_groups_45838 = sdiv_up64(nest_sizze_45695,
                                                           segmap_group_sizze_45837);
            struct memblock_device mem_52637;
            
            mem_52637.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52637, bytes_52461,
                                      "mem_52637")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52641;
            
            mem_52641.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52641, bytes_52461,
                                      "mem_52641")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    3, sizeof(xdim_41931),
                                                    &xdim_41931));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    4, sizeof(ydim_41932),
                                                    &ydim_41932));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    5, sizeof(zzdim_41933),
                                                    &zzdim_41933));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    6, sizeof(mem_52624.mem),
                                                    &mem_52624.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    7, sizeof(mem_52628.mem),
                                                    &mem_52628.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    8, sizeof(mem_52632.mem),
                                                    &mem_52632.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    9, sizeof(mem_52637.mem),
                                                    &mem_52637.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44690,
                                                    10, sizeof(mem_52641.mem),
                                                    &mem_52641.mem));
            if (1 * ((size_t) segmap_usable_groups_45838 *
                     (size_t) segmap_group_sizze_45837) != 0) {
                const size_t global_work_sizze_53060[1] =
                             {(size_t) segmap_usable_groups_45838 *
                             (size_t) segmap_group_sizze_45837};
                const size_t local_work_sizze_53064[1] =
                             {segmap_group_sizze_45837};
                int64_t time_start_53061 = 0, time_end_53062 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_44690");
                    fprintf(stderr, "%zu", global_work_sizze_53060[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_53064[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_53061 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_44690,
                                                                1, NULL,
                                                                global_work_sizze_53060,
                                                                local_work_sizze_53064,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_44690_runs,
                                                                                                          &ctx->integrate_tkezisegmap_44690_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_53062 = get_wall_time();
                    
                    long time_diff_53063 = time_end_53062 - time_start_53061;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_44690", time_diff_53063);
                }
            }
            ctx->failure_is_an_option = 1;
            
            struct memblock_device mem_52645;
            
            mem_52645.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52645, bytes_52642,
                                      "mem_52645")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52645, 0,
                                                      mem_52637, 0, 1,
                                                      zzdim_41933, xdim_41931 *
                                                      ydim_41932) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_52637, "mem_52637") != 0)
                return 1;
            
            struct memblock_device mem_52649;
            
            mem_52649.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52649, bytes_52642,
                                      "mem_52649")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52649, 0,
                                                      mem_52641, 0, 1,
                                                      zzdim_41933, xdim_41931 *
                                                      ydim_41932) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_52641, "mem_52641") != 0)
                return 1;
            
            struct memblock_device mem_52653;
            
            mem_52653.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52653, bytes_52642,
                                      "mem_52653")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52653, 0,
                                                      mem_52624, 0, 1,
                                                      zzdim_41933, xdim_41931 *
                                                      ydim_41932) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_52624, "mem_52624") != 0)
                return 1;
            
            struct memblock_device mem_52657;
            
            mem_52657.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52657, bytes_52642,
                                      "mem_52657")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52657, 0,
                                                      mem_52620, 0, 1,
                                                      zzdim_41933, xdim_41931 *
                                                      ydim_41932) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_52620, "mem_52620") != 0)
                return 1;
            
            struct memblock_device mem_52661;
            
            mem_52661.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52661, bytes_52642,
                                      "mem_52661")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52661, 0,
                                                      mem_52628, 0, 1,
                                                      zzdim_41933, xdim_41931 *
                                                      ydim_41932) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_52628, "mem_52628") != 0)
                return 1;
            
            struct memblock_device mem_52665;
            
            mem_52665.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52665, bytes_52642,
                                      "mem_52665")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52665, 0,
                                                      mem_52632, 0, 1,
                                                      zzdim_41933, xdim_41931 *
                                                      ydim_41932) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_52632, "mem_52632") != 0)
                return 1;
            
            struct memblock_device mem_52706;
            
            mem_52706.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52706, bytes_52642,
                                      "mem_52706")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52710;
            
            mem_52710.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52710, bytes_52642,
                                      "mem_52710")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52684;
            
            mem_52684.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52684, total_sizze_52835,
                                      "mem_52684")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52689;
            
            mem_52689.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52689, total_sizze_52836,
                                      "mem_52689")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    3, sizeof(xdim_41931),
                                                    &xdim_41931));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    4, sizeof(ydim_41932),
                                                    &ydim_41932));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    5, sizeof(zzdim_41933),
                                                    &zzdim_41933));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    6, sizeof(distance_42130),
                                                    &distance_42130));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    7, sizeof(num_groups_45863),
                                                    &num_groups_45863));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    8, sizeof(mem_52645.mem),
                                                    &mem_52645.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    9, sizeof(mem_52649.mem),
                                                    &mem_52649.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    10, sizeof(mem_52653.mem),
                                                    &mem_52653.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    11, sizeof(mem_52657.mem),
                                                    &mem_52657.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    12, sizeof(mem_52661.mem),
                                                    &mem_52661.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    13, sizeof(mem_52665.mem),
                                                    &mem_52665.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    14, sizeof(mem_52684.mem),
                                                    &mem_52684.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    15, sizeof(mem_52689.mem),
                                                    &mem_52689.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    16, sizeof(mem_52706.mem),
                                                    &mem_52706.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44613,
                                                    17, sizeof(mem_52710.mem),
                                                    &mem_52710.mem));
            if (1 * ((size_t) num_groups_45863 *
                     (size_t) segmap_group_sizze_45862) != 0) {
                const size_t global_work_sizze_53065[1] =
                             {(size_t) num_groups_45863 *
                             (size_t) segmap_group_sizze_45862};
                const size_t local_work_sizze_53069[1] =
                             {segmap_group_sizze_45862};
                int64_t time_start_53066 = 0, time_end_53067 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_44613");
                    fprintf(stderr, "%zu", global_work_sizze_53065[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_53069[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_53066 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_44613,
                                                                1, NULL,
                                                                global_work_sizze_53065,
                                                                local_work_sizze_53069,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_44613_runs,
                                                                                                          &ctx->integrate_tkezisegmap_44613_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_53067 = get_wall_time();
                    
                    long time_diff_53068 = time_end_53067 - time_start_53066;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_44613", time_diff_53068);
                }
            }
            ctx->failure_is_an_option = 1;
            if (memblock_unref_device(ctx, &mem_52645, "mem_52645") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52649, "mem_52649") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52653, "mem_52653") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52657, "mem_52657") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52661, "mem_52661") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52665, "mem_52665") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52684, "mem_52684") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52689, "mem_52689") != 0)
                return 1;
            
            int64_t segmap_usable_groups_45931 =
                    sdiv_up64(intra_num_groups_45472, segmap_group_sizze_45930);
            struct memblock_device mem_52714;
            
            mem_52714.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52714, bytes_52461,
                                      "mem_52714")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52714, 0,
                                                      mem_52710, 0, 1,
                                                      xdim_41931 * ydim_41932,
                                                      zzdim_41933) != 0) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44535,
                                                    1, sizeof(xdim_41931),
                                                    &xdim_41931));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44535,
                                                    2, sizeof(ydim_41932),
                                                    &ydim_41932));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44535,
                                                    3, sizeof(zzdim_41933),
                                                    &zzdim_41933));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44535,
                                                    4, sizeof(distance_42130),
                                                    &distance_42130));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44535,
                                                    5, sizeof(mem_52464.mem),
                                                    &mem_52464.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44535,
                                                    6, sizeof(mem_52714.mem),
                                                    &mem_52714.mem));
            if (1 * ((size_t) segmap_usable_groups_45931 *
                     (size_t) segmap_group_sizze_45930) != 0) {
                const size_t global_work_sizze_53070[1] =
                             {(size_t) segmap_usable_groups_45931 *
                             (size_t) segmap_group_sizze_45930};
                const size_t local_work_sizze_53074[1] =
                             {segmap_group_sizze_45930};
                int64_t time_start_53071 = 0, time_end_53072 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_44535");
                    fprintf(stderr, "%zu", global_work_sizze_53070[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_53074[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_53071 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_44535,
                                                                1, NULL,
                                                                global_work_sizze_53070,
                                                                local_work_sizze_53074,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_44535_runs,
                                                                                                          &ctx->integrate_tkezisegmap_44535_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_53072 = get_wall_time();
                    
                    long time_diff_53073 = time_end_53072 - time_start_53071;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_44535", time_diff_53073);
                }
            }
            if (memblock_unref_device(ctx, &mem_52714, "mem_52714") != 0)
                return 1;
            
            struct memblock_device mem_52719;
            
            mem_52719.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52719, bytes_52642,
                                      "mem_52719")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52719, 0,
                                                      mem_52464, 0, 1,
                                                      zzdim_41933, xdim_41931 *
                                                      ydim_41932) != 0) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52742;
            
            mem_52742.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52742, bytes_52642,
                                      "mem_52742")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_52730;
            
            mem_52730.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52730, total_sizze_52839,
                                      "mem_52730")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    3, sizeof(xdim_41931),
                                                    &xdim_41931));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    4, sizeof(ydim_41932),
                                                    &ydim_41932));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    5, sizeof(zzdim_41933),
                                                    &zzdim_41933));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    6, sizeof(distance_42130),
                                                    &distance_42130));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    7, sizeof(m_42138),
                                                    &m_42138));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    8, sizeof(num_groups_45942),
                                                    &num_groups_45942));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    9, sizeof(mem_52706.mem),
                                                    &mem_52706.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    10, sizeof(mem_52710.mem),
                                                    &mem_52710.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    11, sizeof(mem_52719.mem),
                                                    &mem_52719.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    12, sizeof(mem_52730.mem),
                                                    &mem_52730.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_44495,
                                                    13, sizeof(mem_52742.mem),
                                                    &mem_52742.mem));
            if (1 * ((size_t) num_groups_45942 *
                     (size_t) segmap_group_sizze_45941) != 0) {
                const size_t global_work_sizze_53075[1] =
                             {(size_t) num_groups_45942 *
                             (size_t) segmap_group_sizze_45941};
                const size_t local_work_sizze_53079[1] =
                             {segmap_group_sizze_45941};
                int64_t time_start_53076 = 0, time_end_53077 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_44495");
                    fprintf(stderr, "%zu", global_work_sizze_53075[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_53079[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_53076 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_44495,
                                                                1, NULL,
                                                                global_work_sizze_53075,
                                                                local_work_sizze_53079,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_44495_runs,
                                                                                                          &ctx->integrate_tkezisegmap_44495_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_53077 = get_wall_time();
                    
                    long time_diff_53078 = time_end_53077 - time_start_53076;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_44495", time_diff_53078);
                }
            }
            ctx->failure_is_an_option = 1;
            if (memblock_unref_device(ctx, &mem_52706, "mem_52706") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52710, "mem_52710") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52719, "mem_52719") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52730, "mem_52730") != 0)
                return 1;
            
            struct memblock_device mem_52746;
            
            mem_52746.references = NULL;
            if (memblock_alloc_device(ctx, &mem_52746, bytes_52461,
                                      "mem_52746")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52746, 0,
                                                      mem_52742, 0, 1,
                                                      xdim_41931 * ydim_41932,
                                                      zzdim_41933) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_52742, "mem_52742") != 0)
                return 1;
            if (memblock_set_device(ctx, &lifted_11_map_res_mem_52748,
                                    &mem_52746, "mem_52746") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52746, "mem_52746") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52730, "mem_52730") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52742, "mem_52742") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52719, "mem_52719") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52714, "mem_52714") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52689, "mem_52689") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52684, "mem_52684") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52710, "mem_52710") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52706, "mem_52706") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52665, "mem_52665") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52661, "mem_52661") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52657, "mem_52657") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52653, "mem_52653") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52649, "mem_52649") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52645, "mem_52645") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52641, "mem_52641") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52637, "mem_52637") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52632, "mem_52632") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52628, "mem_52628") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52624, "mem_52624") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_52620, "mem_52620") != 0)
                return 1;
        }
        if (memblock_set_device(ctx, &lifted_11_map_res_mem_52749,
                                &lifted_11_map_res_mem_52748,
                                "lifted_11_map_res_mem_52748") != 0)
            return 1;
        if (memblock_unref_device(ctx, &lifted_11_map_res_mem_52748,
                                  "lifted_11_map_res_mem_52748") != 0)
            return 1;
    }
    if (memblock_unref_device(ctx, &mem_52464, "mem_52464") != 0)
        return 1;
    
    int64_t segmap_group_sizze_46368;
    
    segmap_group_sizze_46368 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_46247;
    
    int64_t segmap_usable_groups_46369 = sdiv_up64(intra_num_groups_45472,
                                                   segmap_group_sizze_46368);
    int64_t bytes_52751 = 4 * intra_num_groups_45472;
    struct memblock_device mem_52753;
    
    mem_52753.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52753, bytes_52751, "mem_52753")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52755;
    
    mem_52755.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52755, intra_num_groups_45472,
                              "mem_52755")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52757;
    
    mem_52757.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52757, intra_num_groups_45472,
                              "mem_52757")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 1,
                                            sizeof(xdim_41931), &xdim_41931));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 2,
                                            sizeof(ydim_41932), &ydim_41932));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 3,
                                            sizeof(ydim_41976), &ydim_41976));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 4,
                                            sizeof(y_42127), &y_42127));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 5,
                                            sizeof(y_42128), &y_42128));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 6,
                                            sizeof(kbot_mem_52456.mem),
                                            &kbot_mem_52456.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 7,
                                            sizeof(mem_52753.mem),
                                            &mem_52753.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 8,
                                            sizeof(mem_52755.mem),
                                            &mem_52755.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46244, 9,
                                            sizeof(mem_52757.mem),
                                            &mem_52757.mem));
    if (1 * ((size_t) segmap_usable_groups_46369 *
             (size_t) segmap_group_sizze_46368) != 0) {
        const size_t global_work_sizze_53080[1] =
                     {(size_t) segmap_usable_groups_46369 *
                     (size_t) segmap_group_sizze_46368};
        const size_t local_work_sizze_53084[1] = {segmap_group_sizze_46368};
        int64_t time_start_53081 = 0, time_end_53082 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_46244");
            fprintf(stderr, "%zu", global_work_sizze_53080[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53084[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53081 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_46244,
                                                        1, NULL,
                                                        global_work_sizze_53080,
                                                        local_work_sizze_53084,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_46244_runs,
                                                                                                  &ctx->integrate_tkezisegmap_46244_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53082 = get_wall_time();
            
            long time_diff_53083 = time_end_53082 - time_start_53081;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_46244", time_diff_53083);
        }
    }
    if (memblock_unref_device(ctx, &mem_52757, "mem_52757") != 0)
        return 1;
    
    int64_t segmap_group_sizze_46402;
    
    segmap_group_sizze_46402 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_46179;
    
    int64_t segmap_usable_groups_46403 = sdiv_up64(nest_sizze_45695,
                                                   segmap_group_sizze_46402);
    struct memblock_device mem_52762;
    
    mem_52762.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52762, bytes_52461, "mem_52762")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 1,
                                            sizeof(xdim_41931), &xdim_41931));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 2,
                                            sizeof(ydim_41932), &ydim_41932));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 3,
                                            sizeof(zzdim_41933), &zzdim_41933));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 4,
                                            sizeof(ydim_41935), &ydim_41935));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 5,
                                            sizeof(zzdim_41936), &zzdim_41936));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 6,
                                            sizeof(y_42127), &y_42127));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 7,
                                            sizeof(y_42128), &y_42128));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 8,
                                            sizeof(tketaup1_mem_52437.mem),
                                            &tketaup1_mem_52437.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175, 9,
                                            sizeof(lifted_11_map_res_mem_52749.mem),
                                            &lifted_11_map_res_mem_52749.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175,
                                            10, sizeof(mem_52753.mem),
                                            &mem_52753.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175,
                                            11, sizeof(mem_52755.mem),
                                            &mem_52755.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46175,
                                            12, sizeof(mem_52762.mem),
                                            &mem_52762.mem));
    if (1 * ((size_t) segmap_usable_groups_46403 *
             (size_t) segmap_group_sizze_46402) != 0) {
        const size_t global_work_sizze_53085[1] =
                     {(size_t) segmap_usable_groups_46403 *
                     (size_t) segmap_group_sizze_46402};
        const size_t local_work_sizze_53089[1] = {segmap_group_sizze_46402};
        int64_t time_start_53086 = 0, time_end_53087 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_46175");
            fprintf(stderr, "%zu", global_work_sizze_53085[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53089[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53086 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_46175,
                                                        1, NULL,
                                                        global_work_sizze_53085,
                                                        local_work_sizze_53089,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_46175_runs,
                                                                                                  &ctx->integrate_tkezisegmap_46175_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53087 = get_wall_time();
            
            long time_diff_53088 = time_end_53087 - time_start_53086;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_46175", time_diff_53088);
        }
    }
    if (memblock_unref_device(ctx, &lifted_11_map_res_mem_52749,
                              "lifted_11_map_res_mem_52749") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52753, "mem_52753") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52755, "mem_52755") != 0)
        return 1;
    
    int64_t y_42385 = sub64(xdim_41931, 1);
    int64_t y_42386 = sub64(ydim_41932, 1);
    int64_t segmap_group_sizze_47428;
    
    segmap_group_sizze_47428 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_47208;
    
    int64_t segmap_usable_groups_47429 = sdiv_up64(intra_num_groups_45472,
                                                   segmap_group_sizze_47428);
    struct memblock_device mem_52766;
    
    mem_52766.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52766, bytes_52642, "mem_52766")) {
        err = 1;
        goto cleanup;
    }
    if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_52766, 0, mem_52762, 0,
                                              1, zzdim_41933, xdim_41931 *
                                              ydim_41932) != 0) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52769;
    
    mem_52769.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52769, intra_num_groups_45472,
                              "mem_52769")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52771;
    
    mem_52771.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52771, intra_num_groups_45472,
                              "mem_52771")) {
        err = 1;
        goto cleanup;
    }
    
    int64_t bytes_52772 = 8 * intra_num_groups_45472;
    struct memblock_device mem_52774;
    
    mem_52774.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52774, bytes_52772, "mem_52774")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 3,
                                            sizeof(xdim_41931), &xdim_41931));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 4,
                                            sizeof(ydim_41932), &ydim_41932));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 5,
                                            sizeof(zzdim_41933), &zzdim_41933));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 6,
                                            sizeof(y_42127), &y_42127));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 7,
                                            sizeof(y_42128), &y_42128));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 8,
                                            sizeof(y_42386), &y_42386));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205, 9,
                                            sizeof(dzzw_mem_52453.mem),
                                            &dzzw_mem_52453.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205,
                                            10, sizeof(mem_52766.mem),
                                            &mem_52766.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205,
                                            11, sizeof(mem_52769.mem),
                                            &mem_52769.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205,
                                            12, sizeof(mem_52771.mem),
                                            &mem_52771.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_47205,
                                            13, sizeof(mem_52774.mem),
                                            &mem_52774.mem));
    if (1 * ((size_t) segmap_usable_groups_47429 *
             (size_t) segmap_group_sizze_47428) != 0) {
        const size_t global_work_sizze_53090[1] =
                     {(size_t) segmap_usable_groups_47429 *
                     (size_t) segmap_group_sizze_47428};
        const size_t local_work_sizze_53094[1] = {segmap_group_sizze_47428};
        int64_t time_start_53091 = 0, time_end_53092 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_47205");
            fprintf(stderr, "%zu", global_work_sizze_53090[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53094[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53091 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_47205,
                                                        1, NULL,
                                                        global_work_sizze_53090,
                                                        local_work_sizze_53094,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_47205_runs,
                                                                                                  &ctx->integrate_tkezisegmap_47205_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53092 = get_wall_time();
            
            long time_diff_53093 = time_end_53092 - time_start_53091;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_47205", time_diff_53093);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_52766, "mem_52766") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52769, "mem_52769") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52771, "mem_52771") != 0)
        return 1;
    
    int64_t segmap_group_sizze_47475;
    
    segmap_group_sizze_47475 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_46994;
    
    int64_t segmap_usable_groups_47476 = sdiv_up64(nest_sizze_45695,
                                                   segmap_group_sizze_47475);
    struct memblock_device mem_52779;
    
    mem_52779.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52779, bytes_52461, "mem_52779")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52783;
    
    mem_52783.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52783, bytes_52461, "mem_52783")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52787;
    
    mem_52787.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52787, bytes_52461, "mem_52787")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 3,
                                            sizeof(xdim_41931), &xdim_41931));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 4,
                                            sizeof(ydim_41932), &ydim_41932));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 5,
                                            sizeof(zzdim_41933), &zzdim_41933));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 6,
                                            sizeof(ydim_41959), &ydim_41959));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 7,
                                            sizeof(zzdim_41960), &zzdim_41960));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 8,
                                            sizeof(ydim_41962), &ydim_41962));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990, 9,
                                            sizeof(zzdim_41963), &zzdim_41963));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            10, sizeof(y_42127), &y_42127));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            11, sizeof(y_42128), &y_42128));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            12, sizeof(distance_42130),
                                            &distance_42130));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            13, sizeof(y_42385), &y_42385));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            14, sizeof(y_42386), &y_42386));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            15, sizeof(tketau_mem_52436.mem),
                                            &tketau_mem_52436.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            16, sizeof(maskU_mem_52445.mem),
                                            &maskU_mem_52445.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            17, sizeof(maskV_mem_52446.mem),
                                            &maskV_mem_52446.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            18, sizeof(dxu_mem_52449.mem),
                                            &dxu_mem_52449.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            19, sizeof(dyu_mem_52451.mem),
                                            &dyu_mem_52451.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            20, sizeof(cost_mem_52454.mem),
                                            &cost_mem_52454.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            21, sizeof(cosu_mem_52455.mem),
                                            &cosu_mem_52455.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            22, sizeof(mem_52762.mem),
                                            &mem_52762.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            23, sizeof(mem_52779.mem),
                                            &mem_52779.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            24, sizeof(mem_52783.mem),
                                            &mem_52783.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_46990,
                                            25, sizeof(mem_52787.mem),
                                            &mem_52787.mem));
    if (1 * ((size_t) segmap_usable_groups_47476 *
             (size_t) segmap_group_sizze_47475) != 0) {
        const size_t global_work_sizze_53095[1] =
                     {(size_t) segmap_usable_groups_47476 *
                     (size_t) segmap_group_sizze_47475};
        const size_t local_work_sizze_53099[1] = {segmap_group_sizze_47475};
        int64_t time_start_53096 = 0, time_end_53097 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_46990");
            fprintf(stderr, "%zu", global_work_sizze_53095[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53099[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53096 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_46990,
                                                        1, NULL,
                                                        global_work_sizze_53095,
                                                        local_work_sizze_53099,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_46990_runs,
                                                                                                  &ctx->integrate_tkezisegmap_46990_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53097 = get_wall_time();
            
            long time_diff_53098 = time_end_53097 - time_start_53096;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_46990", time_diff_53098);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_52762, "mem_52762") != 0)
        return 1;
    
    int64_t segmap_group_sizze_50208;
    
    segmap_group_sizze_50208 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_49044;
    
    int64_t segmap_usable_groups_50209 = sdiv_up64(nest_sizze_45695,
                                                   segmap_group_sizze_50208);
    struct memblock_device mem_52792;
    
    mem_52792.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52792, bytes_52461, "mem_52792")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52796;
    
    mem_52796.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52796, bytes_52461, "mem_52796")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52800;
    
    mem_52800.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52800, bytes_52461, "mem_52800")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_52804;
    
    mem_52804.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52804, bytes_52461, "mem_52804")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 3,
                                            sizeof(xdim_41931), &xdim_41931));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 4,
                                            sizeof(ydim_41932), &ydim_41932));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 5,
                                            sizeof(zzdim_41933), &zzdim_41933));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 6,
                                            sizeof(ydim_41950), &ydim_41950));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 7,
                                            sizeof(zzdim_41951), &zzdim_41951));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 8,
                                            sizeof(ydim_41953), &ydim_41953));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040, 9,
                                            sizeof(zzdim_41954), &zzdim_41954));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            10, sizeof(ydim_41956),
                                            &ydim_41956));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            11, sizeof(zzdim_41957),
                                            &zzdim_41957));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            12, sizeof(ydim_41965),
                                            &ydim_41965));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            13, sizeof(zzdim_41966),
                                            &zzdim_41966));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            14, sizeof(y_42127), &y_42127));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            15, sizeof(y_42128), &y_42128));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            16, sizeof(distance_42130),
                                            &distance_42130));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            17, sizeof(tketau_mem_52436.mem),
                                            &tketau_mem_52436.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            18, sizeof(utau_mem_52442.mem),
                                            &utau_mem_52442.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            19, sizeof(vtau_mem_52443.mem),
                                            &vtau_mem_52443.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            20, sizeof(wtau_mem_52444.mem),
                                            &wtau_mem_52444.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            21, sizeof(maskW_mem_52447.mem),
                                            &maskW_mem_52447.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            22, sizeof(dxt_mem_52448.mem),
                                            &dxt_mem_52448.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            23, sizeof(dyt_mem_52450.mem),
                                            &dyt_mem_52450.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            24, sizeof(dzzw_mem_52453.mem),
                                            &dzzw_mem_52453.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            25, sizeof(cost_mem_52454.mem),
                                            &cost_mem_52454.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            26, sizeof(cosu_mem_52455.mem),
                                            &cosu_mem_52455.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            27, sizeof(mem_52779.mem),
                                            &mem_52779.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            28, sizeof(mem_52783.mem),
                                            &mem_52783.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            29, sizeof(mem_52787.mem),
                                            &mem_52787.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            30, sizeof(mem_52792.mem),
                                            &mem_52792.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            31, sizeof(mem_52796.mem),
                                            &mem_52796.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            32, sizeof(mem_52800.mem),
                                            &mem_52800.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49040,
                                            33, sizeof(mem_52804.mem),
                                            &mem_52804.mem));
    if (1 * ((size_t) segmap_usable_groups_50209 *
             (size_t) segmap_group_sizze_50208) != 0) {
        const size_t global_work_sizze_53100[1] =
                     {(size_t) segmap_usable_groups_50209 *
                     (size_t) segmap_group_sizze_50208};
        const size_t local_work_sizze_53104[1] = {segmap_group_sizze_50208};
        int64_t time_start_53101 = 0, time_end_53102 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_49040");
            fprintf(stderr, "%zu", global_work_sizze_53100[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53104[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53101 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_49040,
                                                        1, NULL,
                                                        global_work_sizze_53100,
                                                        local_work_sizze_53104,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_49040_runs,
                                                                                                  &ctx->integrate_tkezisegmap_49040_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53102 = get_wall_time();
            
            long time_diff_53103 = time_end_53102 - time_start_53101;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_49040", time_diff_53103);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_52779, "mem_52779") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52783, "mem_52783") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52787, "mem_52787") != 0)
        return 1;
    
    int64_t segmap_group_sizze_51694;
    
    segmap_group_sizze_51694 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_51181;
    
    int64_t segmap_usable_groups_51695 = sdiv_up64(nest_sizze_45695,
                                                   segmap_group_sizze_51694);
    struct memblock_device mem_52809;
    
    mem_52809.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52809, bytes_52461, "mem_52809")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 3,
                                            sizeof(xdim_41931), &xdim_41931));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 4,
                                            sizeof(ydim_41932), &ydim_41932));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 5,
                                            sizeof(zzdim_41933), &zzdim_41933));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 6,
                                            sizeof(ydim_41941), &ydim_41941));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 7,
                                            sizeof(zzdim_41942), &zzdim_41942));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 8,
                                            sizeof(ydim_41965), &ydim_41965));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177, 9,
                                            sizeof(zzdim_41966), &zzdim_41966));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            10, sizeof(y_42127), &y_42127));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            11, sizeof(y_42128), &y_42128));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            12, sizeof(distance_42130),
                                            &distance_42130));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            13, sizeof(dtketau_mem_52439.mem),
                                            &dtketau_mem_52439.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            14, sizeof(maskW_mem_52447.mem),
                                            &maskW_mem_52447.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            15, sizeof(dxt_mem_52448.mem),
                                            &dxt_mem_52448.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            16, sizeof(dyt_mem_52450.mem),
                                            &dyt_mem_52450.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            17, sizeof(dzzw_mem_52453.mem),
                                            &dzzw_mem_52453.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            18, sizeof(cost_mem_52454.mem),
                                            &cost_mem_52454.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            19, sizeof(mem_52792.mem),
                                            &mem_52792.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            20, sizeof(mem_52796.mem),
                                            &mem_52796.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            21, sizeof(mem_52800.mem),
                                            &mem_52800.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51177,
                                            22, sizeof(mem_52809.mem),
                                            &mem_52809.mem));
    if (1 * ((size_t) segmap_usable_groups_51695 *
             (size_t) segmap_group_sizze_51694) != 0) {
        const size_t global_work_sizze_53105[1] =
                     {(size_t) segmap_usable_groups_51695 *
                     (size_t) segmap_group_sizze_51694};
        const size_t local_work_sizze_53109[1] = {segmap_group_sizze_51694};
        int64_t time_start_53106 = 0, time_end_53107 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_51177");
            fprintf(stderr, "%zu", global_work_sizze_53105[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53109[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53106 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_51177,
                                                        1, NULL,
                                                        global_work_sizze_53105,
                                                        local_work_sizze_53109,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_51177_runs,
                                                                                                  &ctx->integrate_tkezisegmap_51177_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53107 = get_wall_time();
            
            long time_diff_53108 = time_end_53107 - time_start_53106;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_51177", time_diff_53108);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_52792, "mem_52792") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52796, "mem_52796") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52800, "mem_52800") != 0)
        return 1;
    
    int64_t segmap_group_sizze_52125;
    
    segmap_group_sizze_52125 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_51978;
    
    int64_t segmap_usable_groups_52126 = sdiv_up64(nest_sizze_45695,
                                                   segmap_group_sizze_52125);
    struct memblock_device mem_52814;
    
    mem_52814.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52814, bytes_52461, "mem_52814")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 1,
                                            sizeof(xdim_41931), &xdim_41931));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 2,
                                            sizeof(ydim_41932), &ydim_41932));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 3,
                                            sizeof(zzdim_41933), &zzdim_41933));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 4,
                                            sizeof(ydim_41947), &ydim_41947));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 5,
                                            sizeof(zzdim_41948), &zzdim_41948));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 6,
                                            sizeof(dtketaum1_mem_52441.mem),
                                            &dtketaum1_mem_52441.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 7,
                                            sizeof(mem_52804.mem),
                                            &mem_52804.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 8,
                                            sizeof(mem_52809.mem),
                                            &mem_52809.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_51974, 9,
                                            sizeof(mem_52814.mem),
                                            &mem_52814.mem));
    if (1 * ((size_t) segmap_usable_groups_52126 *
             (size_t) segmap_group_sizze_52125) != 0) {
        const size_t global_work_sizze_53110[1] =
                     {(size_t) segmap_usable_groups_52126 *
                     (size_t) segmap_group_sizze_52125};
        const size_t local_work_sizze_53114[1] = {segmap_group_sizze_52125};
        int64_t time_start_53111 = 0, time_end_53112 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_51974");
            fprintf(stderr, "%zu", global_work_sizze_53110[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_53114[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_53111 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_51974,
                                                        1, NULL,
                                                        global_work_sizze_53110,
                                                        local_work_sizze_53114,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_51974_runs,
                                                                                                  &ctx->integrate_tkezisegmap_51974_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_53112 = get_wall_time();
            
            long time_diff_53113 = time_end_53112 - time_start_53111;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_51974", time_diff_53113);
        }
    }
    if (memblock_unref_device(ctx, &mem_52804, "mem_52804") != 0)
        return 1;
    
    struct memblock_device mem_52818;
    
    mem_52818.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52818, bytes_52461, "mem_52818")) {
        err = 1;
        goto cleanup;
    }
    if (xdim_41931 * ydim_41932 * zzdim_41933 * (int64_t) sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     tketaum1_mem_52438.mem,
                                                     mem_52818.mem, 0, 0,
                                                     xdim_41931 * ydim_41932 *
                                                     zzdim_41933 *
                                                     (int64_t) sizeof(double),
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_52823;
    
    mem_52823.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52823, bytes_52461, "mem_52823")) {
        err = 1;
        goto cleanup;
    }
    if (xdim_41931 * ydim_41932 * zzdim_41933 * (int64_t) sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     dtketaup1_mem_52440.mem,
                                                     mem_52823.mem, 0, 0,
                                                     xdim_41931 * ydim_41932 *
                                                     zzdim_41933 *
                                                     (int64_t) sizeof(double),
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_52828;
    
    mem_52828.references = NULL;
    if (memblock_alloc_device(ctx, &mem_52828, bytes_52461, "mem_52828")) {
        err = 1;
        goto cleanup;
    }
    if (xdim_41931 * ydim_41932 * zzdim_41933 * (int64_t) sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     dtketaum1_mem_52441.mem,
                                                     mem_52828.mem, 0, 0,
                                                     xdim_41931 * ydim_41932 *
                                                     zzdim_41933 *
                                                     (int64_t) sizeof(double),
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    out_arrsizze_52848 = xdim_41931;
    out_arrsizze_52849 = ydim_41932;
    out_arrsizze_52850 = zzdim_41933;
    out_arrsizze_52852 = xdim_41931;
    out_arrsizze_52853 = ydim_41932;
    out_arrsizze_52854 = zzdim_41933;
    out_arrsizze_52856 = xdim_41931;
    out_arrsizze_52857 = ydim_41932;
    out_arrsizze_52858 = zzdim_41933;
    out_arrsizze_52860 = xdim_41931;
    out_arrsizze_52861 = ydim_41932;
    out_arrsizze_52862 = zzdim_41933;
    out_arrsizze_52864 = xdim_41931;
    out_arrsizze_52865 = ydim_41932;
    out_arrsizze_52866 = zzdim_41933;
    out_arrsizze_52868 = xdim_41931;
    out_arrsizze_52869 = ydim_41932;
    out_arrsizze_52870 = zzdim_41933;
    out_arrsizze_52872 = xdim_41931;
    out_arrsizze_52873 = ydim_41932;
    if (memblock_set_device(ctx, &out_mem_52847, &tketau_mem_52436,
                            "tketau_mem_52436") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_52851, &mem_52814, "mem_52814") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_52855, &mem_52818, "mem_52818") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_52859, &mem_52809, "mem_52809") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_52863, &mem_52823, "mem_52823") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_52867, &mem_52828, "mem_52828") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_52871, &mem_52774, "mem_52774") != 0)
        return 1;
    (*out_mem_p_53018).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_53018, &out_mem_52847,
                            "out_mem_52847") != 0)
        return 1;
    *out_out_arrsizze_53019 = out_arrsizze_52848;
    *out_out_arrsizze_53020 = out_arrsizze_52849;
    *out_out_arrsizze_53021 = out_arrsizze_52850;
    (*out_mem_p_53022).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_53022, &out_mem_52851,
                            "out_mem_52851") != 0)
        return 1;
    *out_out_arrsizze_53023 = out_arrsizze_52852;
    *out_out_arrsizze_53024 = out_arrsizze_52853;
    *out_out_arrsizze_53025 = out_arrsizze_52854;
    (*out_mem_p_53026).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_53026, &out_mem_52855,
                            "out_mem_52855") != 0)
        return 1;
    *out_out_arrsizze_53027 = out_arrsizze_52856;
    *out_out_arrsizze_53028 = out_arrsizze_52857;
    *out_out_arrsizze_53029 = out_arrsizze_52858;
    (*out_mem_p_53030).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_53030, &out_mem_52859,
                            "out_mem_52859") != 0)
        return 1;
    *out_out_arrsizze_53031 = out_arrsizze_52860;
    *out_out_arrsizze_53032 = out_arrsizze_52861;
    *out_out_arrsizze_53033 = out_arrsizze_52862;
    (*out_mem_p_53034).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_53034, &out_mem_52863,
                            "out_mem_52863") != 0)
        return 1;
    *out_out_arrsizze_53035 = out_arrsizze_52864;
    *out_out_arrsizze_53036 = out_arrsizze_52865;
    *out_out_arrsizze_53037 = out_arrsizze_52866;
    (*out_mem_p_53038).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_53038, &out_mem_52867,
                            "out_mem_52867") != 0)
        return 1;
    *out_out_arrsizze_53039 = out_arrsizze_52868;
    *out_out_arrsizze_53040 = out_arrsizze_52869;
    *out_out_arrsizze_53041 = out_arrsizze_52870;
    (*out_mem_p_53042).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_53042, &out_mem_52871,
                            "out_mem_52871") != 0)
        return 1;
    *out_out_arrsizze_53043 = out_arrsizze_52872;
    *out_out_arrsizze_53044 = out_arrsizze_52873;
    if (memblock_unref_device(ctx, &mem_52828, "mem_52828") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52823, "mem_52823") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52818, "mem_52818") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52814, "mem_52814") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52809, "mem_52809") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52804, "mem_52804") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52800, "mem_52800") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52796, "mem_52796") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52792, "mem_52792") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52787, "mem_52787") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52783, "mem_52783") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52779, "mem_52779") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52774, "mem_52774") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52771, "mem_52771") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52769, "mem_52769") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52766, "mem_52766") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52762, "mem_52762") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52757, "mem_52757") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52755, "mem_52755") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52753, "mem_52753") != 0)
        return 1;
    if (memblock_unref_device(ctx, &lifted_11_map_res_mem_52749,
                              "lifted_11_map_res_mem_52749") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_52464, "mem_52464") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_52871, "out_mem_52871") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_52867, "out_mem_52867") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_52863, "out_mem_52863") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_52859, "out_mem_52859") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_52855, "out_mem_52855") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_52851, "out_mem_52851") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_52847, "out_mem_52847") != 0)
        return 1;
    
  cleanup:
    { }
    return err;
}
struct futhark_f64_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_f64_2d *futhark_new_f64_2d(struct futhark_context *ctx, const
                                          double *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_f64_2d *bad = NULL;
    struct futhark_f64_2d *arr =
                          (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1) *
                              sizeof(double), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if ((size_t) (dim0 * dim1) * sizeof(double) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      (size_t) (dim0 * dim1) *
                                                      sizeof(double), data + 0,
                                                      0, NULL,
                                                      ctx->profiling_paused ||
                                                      !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                &ctx->copy_dev_to_host_runs,
                                                                                                &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f64_2d *futhark_new_raw_f64_2d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_f64_2d *bad = NULL;
    struct futhark_f64_2d *arr =
                          (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1) *
                              sizeof(double), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if ((size_t) (dim0 * dim1) * sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     (size_t) (dim0 * dim1) *
                                                     sizeof(double), 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_2d(struct futhark_context *ctx, struct futhark_f64_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_2d(struct futhark_context *ctx,
                          struct futhark_f64_2d *arr, double *data)
{
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1]) * sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     0,
                                                     (size_t) (arr->shape[0] *
                                                               arr->shape[1]) *
                                                     sizeof(double), data + 0,
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_host_to_dev_runs,
                                                                                               &ctx->copy_host_to_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
    }
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f64_2d(struct futhark_context *ctx,
                                 struct futhark_f64_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_2d(struct futhark_context *ctx,
                                    struct futhark_f64_2d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_i32_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_i32_2d *futhark_new_i32_2d(struct futhark_context *ctx, const
                                          int32_t *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_i32_2d *bad = NULL;
    struct futhark_i32_2d *arr =
                          (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1) *
                              sizeof(int32_t), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if ((size_t) (dim0 * dim1) * sizeof(int32_t) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      (size_t) (dim0 * dim1) *
                                                      sizeof(int32_t), data + 0,
                                                      0, NULL,
                                                      ctx->profiling_paused ||
                                                      !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                &ctx->copy_dev_to_host_runs,
                                                                                                &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_2d *futhark_new_raw_i32_2d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_i32_2d *bad = NULL;
    struct futhark_i32_2d *arr =
                          (struct futhark_i32_2d *) malloc(sizeof(struct futhark_i32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1) *
                              sizeof(int32_t), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    if ((size_t) (dim0 * dim1) * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     (size_t) (dim0 * dim1) *
                                                     sizeof(int32_t), 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_2d(struct futhark_context *ctx, struct futhark_i32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_2d(struct futhark_context *ctx,
                          struct futhark_i32_2d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1]) * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     0,
                                                     (size_t) (arr->shape[0] *
                                                               arr->shape[1]) *
                                                     sizeof(int32_t), data + 0,
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_host_to_dev_runs,
                                                                                               &ctx->copy_host_to_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
    }
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_i32_2d(struct futhark_context *ctx,
                                 struct futhark_i32_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_i32_2d(struct futhark_context *ctx,
                                    struct futhark_i32_2d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f64_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_f64_1d *futhark_new_f64_1d(struct futhark_context *ctx, const
                                          double *data, int64_t dim0)
{
    struct futhark_f64_1d *bad = NULL;
    struct futhark_f64_1d *arr =
                          (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(double),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if ((size_t) dim0 * sizeof(double) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      (size_t) dim0 *
                                                      sizeof(double), data + 0,
                                                      0, NULL,
                                                      ctx->profiling_paused ||
                                                      !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                &ctx->copy_dev_to_host_runs,
                                                                                                &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f64_1d *futhark_new_raw_f64_1d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0)
{
    struct futhark_f64_1d *bad = NULL;
    struct futhark_f64_1d *arr =
                          (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(double),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if ((size_t) dim0 * sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     (size_t) dim0 *
                                                     sizeof(double), 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_1d(struct futhark_context *ctx,
                          struct futhark_f64_1d *arr, double *data)
{
    lock_lock(&ctx->lock);
    if ((size_t) arr->shape[0] * sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     0, (size_t) arr->shape[0] *
                                                     sizeof(double), data + 0,
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_host_to_dev_runs,
                                                                                               &ctx->copy_host_to_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
    }
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f64_1d(struct futhark_context *ctx,
                                 struct futhark_f64_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_1d(struct futhark_context *ctx,
                                    struct futhark_f64_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f64_3d {
    struct memblock_device mem;
    int64_t shape[3];
} ;
struct futhark_f64_3d *futhark_new_f64_3d(struct futhark_context *ctx, const
                                          double *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2)
{
    struct futhark_f64_3d *bad = NULL;
    struct futhark_f64_3d *arr =
                          (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1 * dim2) *
                              sizeof(double), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    if ((size_t) (dim0 * dim1 * dim2) * sizeof(double) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      (size_t) (dim0 * dim1 *
                                                                dim2) *
                                                      sizeof(double), data + 0,
                                                      0, NULL,
                                                      ctx->profiling_paused ||
                                                      !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                &ctx->copy_dev_to_host_runs,
                                                                                                &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f64_3d *futhark_new_raw_f64_3d(struct futhark_context *ctx, const
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2)
{
    struct futhark_f64_3d *bad = NULL;
    struct futhark_f64_3d *arr =
                          (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1 * dim2) *
                              sizeof(double), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    if ((size_t) (dim0 * dim1 * dim2) * sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     (size_t) (dim0 * dim1 *
                                                               dim2) *
                                                     sizeof(double), 0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_3d(struct futhark_context *ctx, struct futhark_f64_3d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_3d(struct futhark_context *ctx,
                          struct futhark_f64_3d *arr, double *data)
{
    lock_lock(&ctx->lock);
    if ((size_t) (arr->shape[0] * arr->shape[1] * arr->shape[2]) *
        sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem,
                                                     ctx->failure_is_an_option ? CL_FALSE : CL_TRUE,
                                                     0,
                                                     (size_t) (arr->shape[0] *
                                                               arr->shape[1] *
                                                               arr->shape[2]) *
                                                     sizeof(double), data + 0,
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_host_to_dev_runs,
                                                                                               &ctx->copy_host_to_dev_total_runtime)));
        if (ctx->failure_is_an_option && futhark_context_sync(ctx) != 0)
            return 1;
    }
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f64_3d(struct futhark_context *ctx,
                                 struct futhark_f64_3d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_3d(struct futhark_context *ctx,
                                    struct futhark_f64_3d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_integrate_tke(struct futhark_context *ctx,
                                struct futhark_f64_3d **out0,
                                struct futhark_f64_3d **out1,
                                struct futhark_f64_3d **out2,
                                struct futhark_f64_3d **out3,
                                struct futhark_f64_3d **out4,
                                struct futhark_f64_3d **out5,
                                struct futhark_f64_2d **out6, const
                                struct futhark_f64_3d *in0, const
                                struct futhark_f64_3d *in1, const
                                struct futhark_f64_3d *in2, const
                                struct futhark_f64_3d *in3, const
                                struct futhark_f64_3d *in4, const
                                struct futhark_f64_3d *in5, const
                                struct futhark_f64_3d *in6, const
                                struct futhark_f64_3d *in7, const
                                struct futhark_f64_3d *in8, const
                                struct futhark_f64_3d *in9, const
                                struct futhark_f64_3d *in10, const
                                struct futhark_f64_3d *in11, const
                                struct futhark_f64_1d *in12, const
                                struct futhark_f64_1d *in13, const
                                struct futhark_f64_1d *in14, const
                                struct futhark_f64_1d *in15, const
                                struct futhark_f64_1d *in16, const
                                struct futhark_f64_1d *in17, const
                                struct futhark_f64_1d *in18, const
                                struct futhark_f64_1d *in19, const
                                struct futhark_i32_2d *in20, const
                                struct futhark_f64_3d *in21, const
                                struct futhark_f64_3d *in22, const
                                struct futhark_f64_3d *in23, const
                                struct futhark_f64_2d *in24)
{
    struct memblock_device tketau_mem_52436;
    
    tketau_mem_52436.references = NULL;
    
    struct memblock_device tketaup1_mem_52437;
    
    tketaup1_mem_52437.references = NULL;
    
    struct memblock_device tketaum1_mem_52438;
    
    tketaum1_mem_52438.references = NULL;
    
    struct memblock_device dtketau_mem_52439;
    
    dtketau_mem_52439.references = NULL;
    
    struct memblock_device dtketaup1_mem_52440;
    
    dtketaup1_mem_52440.references = NULL;
    
    struct memblock_device dtketaum1_mem_52441;
    
    dtketaum1_mem_52441.references = NULL;
    
    struct memblock_device utau_mem_52442;
    
    utau_mem_52442.references = NULL;
    
    struct memblock_device vtau_mem_52443;
    
    vtau_mem_52443.references = NULL;
    
    struct memblock_device wtau_mem_52444;
    
    wtau_mem_52444.references = NULL;
    
    struct memblock_device maskU_mem_52445;
    
    maskU_mem_52445.references = NULL;
    
    struct memblock_device maskV_mem_52446;
    
    maskV_mem_52446.references = NULL;
    
    struct memblock_device maskW_mem_52447;
    
    maskW_mem_52447.references = NULL;
    
    struct memblock_device dxt_mem_52448;
    
    dxt_mem_52448.references = NULL;
    
    struct memblock_device dxu_mem_52449;
    
    dxu_mem_52449.references = NULL;
    
    struct memblock_device dyt_mem_52450;
    
    dyt_mem_52450.references = NULL;
    
    struct memblock_device dyu_mem_52451;
    
    dyu_mem_52451.references = NULL;
    
    struct memblock_device dzzt_mem_52452;
    
    dzzt_mem_52452.references = NULL;
    
    struct memblock_device dzzw_mem_52453;
    
    dzzw_mem_52453.references = NULL;
    
    struct memblock_device cost_mem_52454;
    
    cost_mem_52454.references = NULL;
    
    struct memblock_device cosu_mem_52455;
    
    cosu_mem_52455.references = NULL;
    
    struct memblock_device kbot_mem_52456;
    
    kbot_mem_52456.references = NULL;
    
    struct memblock_device kappaM_mem_52457;
    
    kappaM_mem_52457.references = NULL;
    
    struct memblock_device mxl_mem_52458;
    
    mxl_mem_52458.references = NULL;
    
    struct memblock_device forc_mem_52459;
    
    forc_mem_52459.references = NULL;
    
    struct memblock_device forc_tke_surface_mem_52460;
    
    forc_tke_surface_mem_52460.references = NULL;
    
    int64_t xdim_41931;
    int64_t ydim_41932;
    int64_t zzdim_41933;
    int64_t xdim_41934;
    int64_t ydim_41935;
    int64_t zzdim_41936;
    int64_t xdim_41937;
    int64_t ydim_41938;
    int64_t zzdim_41939;
    int64_t xdim_41940;
    int64_t ydim_41941;
    int64_t zzdim_41942;
    int64_t xdim_41943;
    int64_t ydim_41944;
    int64_t zzdim_41945;
    int64_t xdim_41946;
    int64_t ydim_41947;
    int64_t zzdim_41948;
    int64_t xdim_41949;
    int64_t ydim_41950;
    int64_t zzdim_41951;
    int64_t xdim_41952;
    int64_t ydim_41953;
    int64_t zzdim_41954;
    int64_t xdim_41955;
    int64_t ydim_41956;
    int64_t zzdim_41957;
    int64_t xdim_41958;
    int64_t ydim_41959;
    int64_t zzdim_41960;
    int64_t xdim_41961;
    int64_t ydim_41962;
    int64_t zzdim_41963;
    int64_t xdim_41964;
    int64_t ydim_41965;
    int64_t zzdim_41966;
    int64_t xdim_41967;
    int64_t xdim_41968;
    int64_t ydim_41969;
    int64_t ydim_41970;
    int64_t zzdim_41971;
    int64_t zzdim_41972;
    int64_t ydim_41973;
    int64_t ydim_41974;
    int64_t xdim_41975;
    int64_t ydim_41976;
    int64_t xdim_41977;
    int64_t ydim_41978;
    int64_t zzdim_41979;
    int64_t xdim_41980;
    int64_t ydim_41981;
    int64_t zzdim_41982;
    int64_t xdim_41983;
    int64_t ydim_41984;
    int64_t zzdim_41985;
    int64_t xdim_41986;
    int64_t ydim_41987;
    struct memblock_device out_mem_52847;
    
    out_mem_52847.references = NULL;
    
    int64_t out_arrsizze_52848;
    int64_t out_arrsizze_52849;
    int64_t out_arrsizze_52850;
    struct memblock_device out_mem_52851;
    
    out_mem_52851.references = NULL;
    
    int64_t out_arrsizze_52852;
    int64_t out_arrsizze_52853;
    int64_t out_arrsizze_52854;
    struct memblock_device out_mem_52855;
    
    out_mem_52855.references = NULL;
    
    int64_t out_arrsizze_52856;
    int64_t out_arrsizze_52857;
    int64_t out_arrsizze_52858;
    struct memblock_device out_mem_52859;
    
    out_mem_52859.references = NULL;
    
    int64_t out_arrsizze_52860;
    int64_t out_arrsizze_52861;
    int64_t out_arrsizze_52862;
    struct memblock_device out_mem_52863;
    
    out_mem_52863.references = NULL;
    
    int64_t out_arrsizze_52864;
    int64_t out_arrsizze_52865;
    int64_t out_arrsizze_52866;
    struct memblock_device out_mem_52867;
    
    out_mem_52867.references = NULL;
    
    int64_t out_arrsizze_52868;
    int64_t out_arrsizze_52869;
    int64_t out_arrsizze_52870;
    struct memblock_device out_mem_52871;
    
    out_mem_52871.references = NULL;
    
    int64_t out_arrsizze_52872;
    int64_t out_arrsizze_52873;
    
    lock_lock(&ctx->lock);
    tketau_mem_52436 = in0->mem;
    xdim_41931 = in0->shape[0];
    ydim_41932 = in0->shape[1];
    zzdim_41933 = in0->shape[2];
    tketaup1_mem_52437 = in1->mem;
    xdim_41934 = in1->shape[0];
    ydim_41935 = in1->shape[1];
    zzdim_41936 = in1->shape[2];
    tketaum1_mem_52438 = in2->mem;
    xdim_41937 = in2->shape[0];
    ydim_41938 = in2->shape[1];
    zzdim_41939 = in2->shape[2];
    dtketau_mem_52439 = in3->mem;
    xdim_41940 = in3->shape[0];
    ydim_41941 = in3->shape[1];
    zzdim_41942 = in3->shape[2];
    dtketaup1_mem_52440 = in4->mem;
    xdim_41943 = in4->shape[0];
    ydim_41944 = in4->shape[1];
    zzdim_41945 = in4->shape[2];
    dtketaum1_mem_52441 = in5->mem;
    xdim_41946 = in5->shape[0];
    ydim_41947 = in5->shape[1];
    zzdim_41948 = in5->shape[2];
    utau_mem_52442 = in6->mem;
    xdim_41949 = in6->shape[0];
    ydim_41950 = in6->shape[1];
    zzdim_41951 = in6->shape[2];
    vtau_mem_52443 = in7->mem;
    xdim_41952 = in7->shape[0];
    ydim_41953 = in7->shape[1];
    zzdim_41954 = in7->shape[2];
    wtau_mem_52444 = in8->mem;
    xdim_41955 = in8->shape[0];
    ydim_41956 = in8->shape[1];
    zzdim_41957 = in8->shape[2];
    maskU_mem_52445 = in9->mem;
    xdim_41958 = in9->shape[0];
    ydim_41959 = in9->shape[1];
    zzdim_41960 = in9->shape[2];
    maskV_mem_52446 = in10->mem;
    xdim_41961 = in10->shape[0];
    ydim_41962 = in10->shape[1];
    zzdim_41963 = in10->shape[2];
    maskW_mem_52447 = in11->mem;
    xdim_41964 = in11->shape[0];
    ydim_41965 = in11->shape[1];
    zzdim_41966 = in11->shape[2];
    dxt_mem_52448 = in12->mem;
    xdim_41967 = in12->shape[0];
    dxu_mem_52449 = in13->mem;
    xdim_41968 = in13->shape[0];
    dyt_mem_52450 = in14->mem;
    ydim_41969 = in14->shape[0];
    dyu_mem_52451 = in15->mem;
    ydim_41970 = in15->shape[0];
    dzzt_mem_52452 = in16->mem;
    zzdim_41971 = in16->shape[0];
    dzzw_mem_52453 = in17->mem;
    zzdim_41972 = in17->shape[0];
    cost_mem_52454 = in18->mem;
    ydim_41973 = in18->shape[0];
    cosu_mem_52455 = in19->mem;
    ydim_41974 = in19->shape[0];
    kbot_mem_52456 = in20->mem;
    xdim_41975 = in20->shape[0];
    ydim_41976 = in20->shape[1];
    kappaM_mem_52457 = in21->mem;
    xdim_41977 = in21->shape[0];
    ydim_41978 = in21->shape[1];
    zzdim_41979 = in21->shape[2];
    mxl_mem_52458 = in22->mem;
    xdim_41980 = in22->shape[0];
    ydim_41981 = in22->shape[1];
    zzdim_41982 = in22->shape[2];
    forc_mem_52459 = in23->mem;
    xdim_41983 = in23->shape[0];
    ydim_41984 = in23->shape[1];
    zzdim_41985 = in23->shape[2];
    forc_tke_surface_mem_52460 = in24->mem;
    xdim_41986 = in24->shape[0];
    ydim_41987 = in24->shape[1];
    
    int ret = futrts_integrate_tke(ctx, &out_mem_52847, &out_arrsizze_52848,
                                   &out_arrsizze_52849, &out_arrsizze_52850,
                                   &out_mem_52851, &out_arrsizze_52852,
                                   &out_arrsizze_52853, &out_arrsizze_52854,
                                   &out_mem_52855, &out_arrsizze_52856,
                                   &out_arrsizze_52857, &out_arrsizze_52858,
                                   &out_mem_52859, &out_arrsizze_52860,
                                   &out_arrsizze_52861, &out_arrsizze_52862,
                                   &out_mem_52863, &out_arrsizze_52864,
                                   &out_arrsizze_52865, &out_arrsizze_52866,
                                   &out_mem_52867, &out_arrsizze_52868,
                                   &out_arrsizze_52869, &out_arrsizze_52870,
                                   &out_mem_52871, &out_arrsizze_52872,
                                   &out_arrsizze_52873, tketau_mem_52436,
                                   tketaup1_mem_52437, tketaum1_mem_52438,
                                   dtketau_mem_52439, dtketaup1_mem_52440,
                                   dtketaum1_mem_52441, utau_mem_52442,
                                   vtau_mem_52443, wtau_mem_52444,
                                   maskU_mem_52445, maskV_mem_52446,
                                   maskW_mem_52447, dxt_mem_52448,
                                   dxu_mem_52449, dyt_mem_52450, dyu_mem_52451,
                                   dzzt_mem_52452, dzzw_mem_52453,
                                   cost_mem_52454, cosu_mem_52455,
                                   kbot_mem_52456, kappaM_mem_52457,
                                   mxl_mem_52458, forc_mem_52459,
                                   forc_tke_surface_mem_52460, xdim_41931,
                                   ydim_41932, zzdim_41933, xdim_41934,
                                   ydim_41935, zzdim_41936, xdim_41937,
                                   ydim_41938, zzdim_41939, xdim_41940,
                                   ydim_41941, zzdim_41942, xdim_41943,
                                   ydim_41944, zzdim_41945, xdim_41946,
                                   ydim_41947, zzdim_41948, xdim_41949,
                                   ydim_41950, zzdim_41951, xdim_41952,
                                   ydim_41953, zzdim_41954, xdim_41955,
                                   ydim_41956, zzdim_41957, xdim_41958,
                                   ydim_41959, zzdim_41960, xdim_41961,
                                   ydim_41962, zzdim_41963, xdim_41964,
                                   ydim_41965, zzdim_41966, xdim_41967,
                                   xdim_41968, ydim_41969, ydim_41970,
                                   zzdim_41971, zzdim_41972, ydim_41973,
                                   ydim_41974, xdim_41975, ydim_41976,
                                   xdim_41977, ydim_41978, zzdim_41979,
                                   xdim_41980, ydim_41981, zzdim_41982,
                                   xdim_41983, ydim_41984, zzdim_41985,
                                   xdim_41986, ydim_41987);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out0)->mem = out_mem_52847;
        (*out0)->shape[0] = out_arrsizze_52848;
        (*out0)->shape[1] = out_arrsizze_52849;
        (*out0)->shape[2] = out_arrsizze_52850;
        assert((*out1 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out1)->mem = out_mem_52851;
        (*out1)->shape[0] = out_arrsizze_52852;
        (*out1)->shape[1] = out_arrsizze_52853;
        (*out1)->shape[2] = out_arrsizze_52854;
        assert((*out2 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out2)->mem = out_mem_52855;
        (*out2)->shape[0] = out_arrsizze_52856;
        (*out2)->shape[1] = out_arrsizze_52857;
        (*out2)->shape[2] = out_arrsizze_52858;
        assert((*out3 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out3)->mem = out_mem_52859;
        (*out3)->shape[0] = out_arrsizze_52860;
        (*out3)->shape[1] = out_arrsizze_52861;
        (*out3)->shape[2] = out_arrsizze_52862;
        assert((*out4 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out4)->mem = out_mem_52863;
        (*out4)->shape[0] = out_arrsizze_52864;
        (*out4)->shape[1] = out_arrsizze_52865;
        (*out4)->shape[2] = out_arrsizze_52866;
        assert((*out5 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out5)->mem = out_mem_52867;
        (*out5)->shape[0] = out_arrsizze_52868;
        (*out5)->shape[1] = out_arrsizze_52869;
        (*out5)->shape[2] = out_arrsizze_52870;
        assert((*out6 =
                (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) !=
            NULL);
        (*out6)->mem = out_mem_52871;
        (*out6)->shape[0] = out_arrsizze_52872;
        (*out6)->shape[1] = out_arrsizze_52873;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
