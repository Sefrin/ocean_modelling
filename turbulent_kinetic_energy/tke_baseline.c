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
    
    struct futhark_f64_3d *read_value_61401;
    int64_t read_shape_61402[3];
    double *read_arr_61403 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61403, read_shape_61402, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61404;
    int64_t read_shape_61405[3];
    double *read_arr_61406 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61406, read_shape_61405, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61407;
    int64_t read_shape_61408[3];
    double *read_arr_61409 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61409, read_shape_61408, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61410;
    int64_t read_shape_61411[3];
    double *read_arr_61412 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61412, read_shape_61411, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61413;
    int64_t read_shape_61414[3];
    double *read_arr_61415 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61415, read_shape_61414, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61416;
    int64_t read_shape_61417[3];
    double *read_arr_61418 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61418, read_shape_61417, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61419;
    int64_t read_shape_61420[3];
    double *read_arr_61421 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61421, read_shape_61420, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61422;
    int64_t read_shape_61423[3];
    double *read_arr_61424 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61424, read_shape_61423, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61425;
    int64_t read_shape_61426[3];
    double *read_arr_61427 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61427, read_shape_61426, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 8,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61428;
    int64_t read_shape_61429[3];
    double *read_arr_61430 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61430, read_shape_61429, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 9,
                      "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61431;
    int64_t read_shape_61432[3];
    double *read_arr_61433 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61433, read_shape_61432, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      10, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61434;
    int64_t read_shape_61435[3];
    double *read_arr_61436 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61436, read_shape_61435, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      11, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61437;
    int64_t read_shape_61438[1];
    double *read_arr_61439 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61439, read_shape_61438, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      12, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61440;
    int64_t read_shape_61441[1];
    double *read_arr_61442 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61442, read_shape_61441, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      13, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61443;
    int64_t read_shape_61444[1];
    double *read_arr_61445 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61445, read_shape_61444, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      14, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61446;
    int64_t read_shape_61447[1];
    double *read_arr_61448 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61448, read_shape_61447, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      15, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61449;
    int64_t read_shape_61450[1];
    double *read_arr_61451 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61451, read_shape_61450, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      16, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61452;
    int64_t read_shape_61453[1];
    double *read_arr_61454 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61454, read_shape_61453, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      17, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61455;
    int64_t read_shape_61456[1];
    double *read_arr_61457 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61457, read_shape_61456, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      18, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_1d *read_value_61458;
    int64_t read_shape_61459[1];
    double *read_arr_61460 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61460, read_shape_61459, 1) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      19, "[]", f64_info.type_name, strerror(errno));
    
    struct futhark_i32_2d *read_value_61461;
    int64_t read_shape_61462[2];
    int32_t *read_arr_61463 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_61463, read_shape_61462, 2) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      20, "[][]", i32_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61464;
    int64_t read_shape_61465[3];
    double *read_arr_61466 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61466, read_shape_61465, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      21, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61467;
    int64_t read_shape_61468[3];
    double *read_arr_61469 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61469, read_shape_61468, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      22, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_3d *read_value_61470;
    int64_t read_shape_61471[3];
    double *read_arr_61472 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61472, read_shape_61471, 3) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      23, "[][][]", f64_info.type_name, strerror(errno));
    
    struct futhark_f64_2d *read_value_61473;
    int64_t read_shape_61474[2];
    double *read_arr_61475 = NULL;
    
    errno = 0;
    if (read_array(&f64_info, (void **) &read_arr_61475, read_shape_61474, 2) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n",
                      24, "[][]", f64_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"integrate_tke\"");
    
    struct futhark_f64_3d *result_61476;
    struct futhark_f64_3d *result_61477;
    struct futhark_f64_3d *result_61478;
    struct futhark_f64_3d *result_61479;
    struct futhark_f64_3d *result_61480;
    struct futhark_f64_3d *result_61481;
    struct futhark_f64_2d *result_61482;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_61401 = futhark_new_f64_3d(ctx, read_arr_61403,
                                                      read_shape_61402[0],
                                                      read_shape_61402[1],
                                                      read_shape_61402[2])) !=
            0);
        assert((read_value_61404 = futhark_new_f64_3d(ctx, read_arr_61406,
                                                      read_shape_61405[0],
                                                      read_shape_61405[1],
                                                      read_shape_61405[2])) !=
            0);
        assert((read_value_61407 = futhark_new_f64_3d(ctx, read_arr_61409,
                                                      read_shape_61408[0],
                                                      read_shape_61408[1],
                                                      read_shape_61408[2])) !=
            0);
        assert((read_value_61410 = futhark_new_f64_3d(ctx, read_arr_61412,
                                                      read_shape_61411[0],
                                                      read_shape_61411[1],
                                                      read_shape_61411[2])) !=
            0);
        assert((read_value_61413 = futhark_new_f64_3d(ctx, read_arr_61415,
                                                      read_shape_61414[0],
                                                      read_shape_61414[1],
                                                      read_shape_61414[2])) !=
            0);
        assert((read_value_61416 = futhark_new_f64_3d(ctx, read_arr_61418,
                                                      read_shape_61417[0],
                                                      read_shape_61417[1],
                                                      read_shape_61417[2])) !=
            0);
        assert((read_value_61419 = futhark_new_f64_3d(ctx, read_arr_61421,
                                                      read_shape_61420[0],
                                                      read_shape_61420[1],
                                                      read_shape_61420[2])) !=
            0);
        assert((read_value_61422 = futhark_new_f64_3d(ctx, read_arr_61424,
                                                      read_shape_61423[0],
                                                      read_shape_61423[1],
                                                      read_shape_61423[2])) !=
            0);
        assert((read_value_61425 = futhark_new_f64_3d(ctx, read_arr_61427,
                                                      read_shape_61426[0],
                                                      read_shape_61426[1],
                                                      read_shape_61426[2])) !=
            0);
        assert((read_value_61428 = futhark_new_f64_3d(ctx, read_arr_61430,
                                                      read_shape_61429[0],
                                                      read_shape_61429[1],
                                                      read_shape_61429[2])) !=
            0);
        assert((read_value_61431 = futhark_new_f64_3d(ctx, read_arr_61433,
                                                      read_shape_61432[0],
                                                      read_shape_61432[1],
                                                      read_shape_61432[2])) !=
            0);
        assert((read_value_61434 = futhark_new_f64_3d(ctx, read_arr_61436,
                                                      read_shape_61435[0],
                                                      read_shape_61435[1],
                                                      read_shape_61435[2])) !=
            0);
        assert((read_value_61437 = futhark_new_f64_1d(ctx, read_arr_61439,
                                                      read_shape_61438[0])) !=
            0);
        assert((read_value_61440 = futhark_new_f64_1d(ctx, read_arr_61442,
                                                      read_shape_61441[0])) !=
            0);
        assert((read_value_61443 = futhark_new_f64_1d(ctx, read_arr_61445,
                                                      read_shape_61444[0])) !=
            0);
        assert((read_value_61446 = futhark_new_f64_1d(ctx, read_arr_61448,
                                                      read_shape_61447[0])) !=
            0);
        assert((read_value_61449 = futhark_new_f64_1d(ctx, read_arr_61451,
                                                      read_shape_61450[0])) !=
            0);
        assert((read_value_61452 = futhark_new_f64_1d(ctx, read_arr_61454,
                                                      read_shape_61453[0])) !=
            0);
        assert((read_value_61455 = futhark_new_f64_1d(ctx, read_arr_61457,
                                                      read_shape_61456[0])) !=
            0);
        assert((read_value_61458 = futhark_new_f64_1d(ctx, read_arr_61460,
                                                      read_shape_61459[0])) !=
            0);
        assert((read_value_61461 = futhark_new_i32_2d(ctx, read_arr_61463,
                                                      read_shape_61462[0],
                                                      read_shape_61462[1])) !=
            0);
        assert((read_value_61464 = futhark_new_f64_3d(ctx, read_arr_61466,
                                                      read_shape_61465[0],
                                                      read_shape_61465[1],
                                                      read_shape_61465[2])) !=
            0);
        assert((read_value_61467 = futhark_new_f64_3d(ctx, read_arr_61469,
                                                      read_shape_61468[0],
                                                      read_shape_61468[1],
                                                      read_shape_61468[2])) !=
            0);
        assert((read_value_61470 = futhark_new_f64_3d(ctx, read_arr_61472,
                                                      read_shape_61471[0],
                                                      read_shape_61471[1],
                                                      read_shape_61471[2])) !=
            0);
        assert((read_value_61473 = futhark_new_f64_2d(ctx, read_arr_61475,
                                                      read_shape_61474[0],
                                                      read_shape_61474[1])) !=
            0);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_integrate_tke(ctx, &result_61476, &result_61477,
                                        &result_61478, &result_61479,
                                        &result_61480, &result_61481,
                                        &result_61482, read_value_61401,
                                        read_value_61404, read_value_61407,
                                        read_value_61410, read_value_61413,
                                        read_value_61416, read_value_61419,
                                        read_value_61422, read_value_61425,
                                        read_value_61428, read_value_61431,
                                        read_value_61434, read_value_61437,
                                        read_value_61440, read_value_61443,
                                        read_value_61446, read_value_61449,
                                        read_value_61452, read_value_61455,
                                        read_value_61458, read_value_61461,
                                        read_value_61464, read_value_61467,
                                        read_value_61470, read_value_61473);
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
        assert(futhark_free_f64_3d(ctx, read_value_61401) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61404) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61407) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61410) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61413) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61416) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61419) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61422) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61425) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61428) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61431) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61434) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61437) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61440) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61443) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61446) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61449) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61452) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61455) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61458) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_61461) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61464) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61467) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61470) == 0);
        assert(futhark_free_f64_2d(ctx, read_value_61473) == 0);
        assert(futhark_free_f64_3d(ctx, result_61476) == 0);
        assert(futhark_free_f64_3d(ctx, result_61477) == 0);
        assert(futhark_free_f64_3d(ctx, result_61478) == 0);
        assert(futhark_free_f64_3d(ctx, result_61479) == 0);
        assert(futhark_free_f64_3d(ctx, result_61480) == 0);
        assert(futhark_free_f64_3d(ctx, result_61481) == 0);
        assert(futhark_free_f64_2d(ctx, result_61482) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_61401 = futhark_new_f64_3d(ctx, read_arr_61403,
                                                      read_shape_61402[0],
                                                      read_shape_61402[1],
                                                      read_shape_61402[2])) !=
            0);
        assert((read_value_61404 = futhark_new_f64_3d(ctx, read_arr_61406,
                                                      read_shape_61405[0],
                                                      read_shape_61405[1],
                                                      read_shape_61405[2])) !=
            0);
        assert((read_value_61407 = futhark_new_f64_3d(ctx, read_arr_61409,
                                                      read_shape_61408[0],
                                                      read_shape_61408[1],
                                                      read_shape_61408[2])) !=
            0);
        assert((read_value_61410 = futhark_new_f64_3d(ctx, read_arr_61412,
                                                      read_shape_61411[0],
                                                      read_shape_61411[1],
                                                      read_shape_61411[2])) !=
            0);
        assert((read_value_61413 = futhark_new_f64_3d(ctx, read_arr_61415,
                                                      read_shape_61414[0],
                                                      read_shape_61414[1],
                                                      read_shape_61414[2])) !=
            0);
        assert((read_value_61416 = futhark_new_f64_3d(ctx, read_arr_61418,
                                                      read_shape_61417[0],
                                                      read_shape_61417[1],
                                                      read_shape_61417[2])) !=
            0);
        assert((read_value_61419 = futhark_new_f64_3d(ctx, read_arr_61421,
                                                      read_shape_61420[0],
                                                      read_shape_61420[1],
                                                      read_shape_61420[2])) !=
            0);
        assert((read_value_61422 = futhark_new_f64_3d(ctx, read_arr_61424,
                                                      read_shape_61423[0],
                                                      read_shape_61423[1],
                                                      read_shape_61423[2])) !=
            0);
        assert((read_value_61425 = futhark_new_f64_3d(ctx, read_arr_61427,
                                                      read_shape_61426[0],
                                                      read_shape_61426[1],
                                                      read_shape_61426[2])) !=
            0);
        assert((read_value_61428 = futhark_new_f64_3d(ctx, read_arr_61430,
                                                      read_shape_61429[0],
                                                      read_shape_61429[1],
                                                      read_shape_61429[2])) !=
            0);
        assert((read_value_61431 = futhark_new_f64_3d(ctx, read_arr_61433,
                                                      read_shape_61432[0],
                                                      read_shape_61432[1],
                                                      read_shape_61432[2])) !=
            0);
        assert((read_value_61434 = futhark_new_f64_3d(ctx, read_arr_61436,
                                                      read_shape_61435[0],
                                                      read_shape_61435[1],
                                                      read_shape_61435[2])) !=
            0);
        assert((read_value_61437 = futhark_new_f64_1d(ctx, read_arr_61439,
                                                      read_shape_61438[0])) !=
            0);
        assert((read_value_61440 = futhark_new_f64_1d(ctx, read_arr_61442,
                                                      read_shape_61441[0])) !=
            0);
        assert((read_value_61443 = futhark_new_f64_1d(ctx, read_arr_61445,
                                                      read_shape_61444[0])) !=
            0);
        assert((read_value_61446 = futhark_new_f64_1d(ctx, read_arr_61448,
                                                      read_shape_61447[0])) !=
            0);
        assert((read_value_61449 = futhark_new_f64_1d(ctx, read_arr_61451,
                                                      read_shape_61450[0])) !=
            0);
        assert((read_value_61452 = futhark_new_f64_1d(ctx, read_arr_61454,
                                                      read_shape_61453[0])) !=
            0);
        assert((read_value_61455 = futhark_new_f64_1d(ctx, read_arr_61457,
                                                      read_shape_61456[0])) !=
            0);
        assert((read_value_61458 = futhark_new_f64_1d(ctx, read_arr_61460,
                                                      read_shape_61459[0])) !=
            0);
        assert((read_value_61461 = futhark_new_i32_2d(ctx, read_arr_61463,
                                                      read_shape_61462[0],
                                                      read_shape_61462[1])) !=
            0);
        assert((read_value_61464 = futhark_new_f64_3d(ctx, read_arr_61466,
                                                      read_shape_61465[0],
                                                      read_shape_61465[1],
                                                      read_shape_61465[2])) !=
            0);
        assert((read_value_61467 = futhark_new_f64_3d(ctx, read_arr_61469,
                                                      read_shape_61468[0],
                                                      read_shape_61468[1],
                                                      read_shape_61468[2])) !=
            0);
        assert((read_value_61470 = futhark_new_f64_3d(ctx, read_arr_61472,
                                                      read_shape_61471[0],
                                                      read_shape_61471[1],
                                                      read_shape_61471[2])) !=
            0);
        assert((read_value_61473 = futhark_new_f64_2d(ctx, read_arr_61475,
                                                      read_shape_61474[0],
                                                      read_shape_61474[1])) !=
            0);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_integrate_tke(ctx, &result_61476, &result_61477,
                                        &result_61478, &result_61479,
                                        &result_61480, &result_61481,
                                        &result_61482, read_value_61401,
                                        read_value_61404, read_value_61407,
                                        read_value_61410, read_value_61413,
                                        read_value_61416, read_value_61419,
                                        read_value_61422, read_value_61425,
                                        read_value_61428, read_value_61431,
                                        read_value_61434, read_value_61437,
                                        read_value_61440, read_value_61443,
                                        read_value_61446, read_value_61449,
                                        read_value_61452, read_value_61455,
                                        read_value_61458, read_value_61461,
                                        read_value_61464, read_value_61467,
                                        read_value_61470, read_value_61473);
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
        assert(futhark_free_f64_3d(ctx, read_value_61401) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61404) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61407) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61410) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61413) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61416) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61419) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61422) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61425) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61428) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61431) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61434) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61437) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61440) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61443) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61446) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61449) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61452) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61455) == 0);
        assert(futhark_free_f64_1d(ctx, read_value_61458) == 0);
        assert(futhark_free_i32_2d(ctx, read_value_61461) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61464) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61467) == 0);
        assert(futhark_free_f64_3d(ctx, read_value_61470) == 0);
        assert(futhark_free_f64_2d(ctx, read_value_61473) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f64_3d(ctx, result_61476) == 0);
            assert(futhark_free_f64_3d(ctx, result_61477) == 0);
            assert(futhark_free_f64_3d(ctx, result_61478) == 0);
            assert(futhark_free_f64_3d(ctx, result_61479) == 0);
            assert(futhark_free_f64_3d(ctx, result_61480) == 0);
            assert(futhark_free_f64_3d(ctx, result_61481) == 0);
            assert(futhark_free_f64_2d(ctx, result_61482) == 0);
        }
    }
    free(read_arr_61403);
    free(read_arr_61406);
    free(read_arr_61409);
    free(read_arr_61412);
    free(read_arr_61415);
    free(read_arr_61418);
    free(read_arr_61421);
    free(read_arr_61424);
    free(read_arr_61427);
    free(read_arr_61430);
    free(read_arr_61433);
    free(read_arr_61436);
    free(read_arr_61439);
    free(read_arr_61442);
    free(read_arr_61445);
    free(read_arr_61448);
    free(read_arr_61451);
    free(read_arr_61454);
    free(read_arr_61457);
    free(read_arr_61460);
    free(read_arr_61463);
    free(read_arr_61466);
    free(read_arr_61469);
    free(read_arr_61472);
    free(read_arr_61475);
    if (binary_output)
        set_binary_mode(stdout);
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_61476)[0] *
                             futhark_shape_f64_3d(ctx, result_61476)[1] *
                             futhark_shape_f64_3d(ctx, result_61476)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_61476, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_61476), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_61477)[0] *
                             futhark_shape_f64_3d(ctx, result_61477)[1] *
                             futhark_shape_f64_3d(ctx, result_61477)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_61477, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_61477), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_61478)[0] *
                             futhark_shape_f64_3d(ctx, result_61478)[1] *
                             futhark_shape_f64_3d(ctx, result_61478)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_61478, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_61478), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_61479)[0] *
                             futhark_shape_f64_3d(ctx, result_61479)[1] *
                             futhark_shape_f64_3d(ctx, result_61479)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_61479, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_61479), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_61480)[0] *
                             futhark_shape_f64_3d(ctx, result_61480)[1] *
                             futhark_shape_f64_3d(ctx, result_61480)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_61480, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_61480), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_3d(ctx,
                                                                  result_61481)[0] *
                             futhark_shape_f64_3d(ctx, result_61481)[1] *
                             futhark_shape_f64_3d(ctx, result_61481)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_3d(ctx, result_61481, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_3d(ctx, result_61481), 3);
        free(arr);
    }
    printf("\n");
    {
        double *arr = calloc(sizeof(double), futhark_shape_f64_2d(ctx,
                                                                  result_61482)[0] *
                             futhark_shape_f64_2d(ctx, result_61482)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f64_2d(ctx, result_61482, arr) == 0);
        write_array(stdout, binary_output, &f64_info, arr,
                    futhark_shape_f64_2d(ctx, result_61482), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f64_3d(ctx, result_61476) == 0);
    assert(futhark_free_f64_3d(ctx, result_61477) == 0);
    assert(futhark_free_f64_3d(ctx, result_61478) == 0);
    assert(futhark_free_f64_3d(ctx, result_61479) == 0);
    assert(futhark_free_f64_3d(ctx, result_61480) == 0);
    assert(futhark_free_f64_3d(ctx, result_61481) == 0);
    assert(futhark_free_f64_2d(ctx, result_61482) == 0);
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
                   "urn atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\n// End of atomics.h\n\n\n\n\n__kernel void builtinzhreplicate_f64zireplicate_61160(__global\n                                                      unsigned char *mem_61156,\n                                                      int32_t num_elems_61157,\n                                                      double val_61158)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_61160;\n    int32_t replicate_ltid_61161;\n    int32_t replicate_gid_61162;\n    \n    replicate_gtid_61160 = get_global_id(0);\n    replicate_ltid_61161 = get_local_id(0);\n    replicate_gid_61162 = get_group_id(0);\n    if (slt64(replicate_gtid_61160, num_elems_61157)",
                   ") {\n        ((__global double *) mem_61156)[sext_i32_i64(replicate_gtid_61160)] =\n            val_61158;\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64(__local volatile\n                                    int64_t *block_9_backing_aligned_0,\n                                    int32_t destoffset_1, int32_t srcoffset_3,\n                                    int32_t num_arrays_4, int32_t x_elems_5,\n                                    int32_t y_elems_6, int32_t mulx_7,\n                                    int32_t muly_8, __global\n                                    unsigned char *destmem_0, __global\n                                    unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t x_index_31 = get_global_id_0_37;\n    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;\n    \n    if (slt32(x_index_31, x_elems_5)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_in_35 ",
                   "= (y_index_32 + j_43 * 8) * x_elems_5 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {\n                ((__local double *) block_9)[sext_i32_i64((get_local_id_1_39 +\n                                                           j_43 * 8) * 33 +\n                                             get_local_id_0_38)] = ((__global\n                                                                     double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                                         index_in_35)];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;\n    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;\n    if (slt32(x_index_31, y_elems_6)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {\n                ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                                index_out_36)] = ((__local\n                                                                   double *) block_9)[sext_i32_i64(get_local_id_0_38 *\n                                                                                      33 +\n                                                                                      get_local_id_1_39 +\n                                                                                      j_43 *\n                                                                                      8)];\n            }\n        }\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64_low_height(__local volatile\n                                               int64_t *block_9_backing_aligned_0,\n                                               int32_t destoffset_1,\n               ",
                   "                                int32_t srcoffset_3,\n                                               int32_t num_arrays_4,\n                                               int32_t x_elems_5,\n                                               int32_t y_elems_6,\n                                               int32_t mulx_7, int32_t muly_8,\n                                               __global\n                                               unsigned char *destmem_0,\n                                               __global unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +\n            srem32(get_local_id_1_39, mulx_7) * 16;\n    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,\n                                                          mulx_7);\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && sl",
                   "t32(y_index_32, y_elems_6)) {\n        ((__local double *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +\n                                     get_local_id_0_38)] = ((__global\n                                                             double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                                 index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);\n    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +\n        srem32(get_local_id_0_38, mulx_7) * 16;\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {\n        ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                        index_out_36)] = ((__local\n                                                           double *) block_9)[sext_i32_i64(get_local_id_0_38 *\n                                                                              17 +\n                                                                              get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64_low_width(__local volatile\n                                              int64_t *block_9_backing_aligned_0,\n                                              int32_t destoffset_1,\n                                              int32_t srcoffset_3,\n                                              int32_t num_arrays_4,\n                                              int32_t x_elems_5,\n                                              int32_t y_elems_6, int32_t mulx_7,\n                                              int32_t muly_8, __global\n                                              unsigned char *destmem_0, __global\n                                              unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int bloc",
                   "k_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,\n                                                          muly_8);\n    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +\n            srem32(get_local_id_0_38, muly_8) * 16;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {\n        ((__local double *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +\n                                     get_local_id_0_38)] = ((__global\n                                                             double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                                 index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +\n        srem32(get_local_id_1_39, muly_8) * 16;\n    y_index_32 = get_group_id_0_40 * 16 + squot32(g",
                   "et_local_id_1_39, muly_8);\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {\n        ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                        index_out_36)] = ((__local\n                                                           double *) block_9)[sext_i32_i64(get_local_id_0_38 *\n                                                                              17 +\n                                                                              get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void gpu_map_transpose_f64_small(__local volatile\n                                          int64_t *block_9_backing_aligned_0,\n                                          int32_t destoffset_1,\n                                          int32_t srcoffset_3,\n                                          int32_t num_arrays_4,\n                                          int32_t x_elems_5, int32_t y_elems_6,\n                                          int32_t mulx_7, int32_t muly_8,\n                                          __global unsigned char *destmem_0,\n                                          __global unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_9_backing_0 = (__local volatile\n                                                         char *) block_9_backing_aligned_0;\n    __local char *block_9;\n    \n    block_9 = (__local char *) block_9_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group",
                   "_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *\n                                          x_elems_5) * (y_elems_6 * x_elems_5);\n    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *\n                                        x_elems_5), y_elems_6);\n    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);\n    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;\n    \n    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {\n        ((__global double *) destmem_0)[sext_i32_i64(odata_offset_33 +\n                                        index_out_36)] = ((__global\n                                                           double *) srcmem_2)[sext_i32_i64(idata_offset_34 +\n                                                                               index_in_35)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void integrate_tkezisegmap_49654(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114,\n                                          int64_t ydim_48159,\n                                          int64_t zzdim_48160, int64_t y_48308,\n                                          int64_t y_48309, int64_t y_48310,\n                                          __global\n                                          unsigned char *tketau_mem_60702,\n                                      ",
                   "    __global\n                                          unsigned char *dzzt_mem_60718,\n                                          __global\n                                          unsigned char *kappaM_mem_60723,\n                                          __global unsigned char *mem_60731,\n                                          __global unsigned char *mem_60735)\n{\n    #define segmap_group_sizze_49904 (integrate_tkezisegmap_group_sizze_49658)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61150;\n    int32_t local_tid_61151;\n    int64_t group_sizze_61154;\n    int32_t wave_sizze_61153;\n    int32_t group_tid_61152;\n    \n    global_tid_61150 = get_global_id(0);\n    local_tid_61151 = get_local_id(0);\n    group_sizze_61154 = get_local_size(0);\n    wave_sizze_61153 = LOCKSTEP_WIDTH;\n    group_tid_61152 = get_group_id(0);\n    \n    int32_t phys_tid_49654;\n    \n    phys_tid_49654 = global_tid_61150;\n    \n    int64_t gtid_49651;\n    \n    gtid_49651 = squot64(sext_i32_i64(group_tid_61152) *\n                         segmap_group_sizze_49904 +\n                         sext_i32_i64(local_tid_61151), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_49652;\n    \n    gtid_49652 = squot64(sext_i32_i64(group_tid_61152) *\n                         segmap_group_sizze_49904 +\n                         sext_i32_i64(local_tid_61151) -\n                         squot64(sext_i32_i64(group_tid_61152) *\n                                 segmap_group_sizze_49904 +\n                                 sext_i32_i64(local_tid_61151), ydim_48113 *\n                                 zzdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_49653;\n    \n    gtid_49653 = sext_i32_i64(group_tid_61152) * segmap_group_sizze_49904 +\n        sext_i32_i64(local_tid_61151) - squot64(sext_i32_i64(group_tid_61152) *\n            ",
                   "                                    segmap_group_sizze_49904 +\n                                                sext_i32_i64(local_tid_61151),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61152) *\n                                             segmap_group_sizze_49904 +\n                                             sext_i32_i64(local_tid_61151) -\n                                             squot64(sext_i32_i64(group_tid_61152) *\n                                                     segmap_group_sizze_49904 +\n                                                     sext_i32_i64(local_tid_61151),\n                                                     ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 * zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_49651, xdim_48112) && slt64(gtid_49652, ydim_48113)) &&\n        slt64(gtid_49653, zzdim_48114)) {\n        bool binop_x_60622 = sle64(2, gtid_49652);\n        bool binop_x_60623 = sle64(2, gtid_49651);\n        bool binop_y_60624 = slt64(gtid_49651, y_48308);\n        bool binop_y_60625 = binop_x_60623 && binop_y_60624;\n        bool binop_x_60626 = binop_x_60622 && binop_y_60625;\n        bool binop_y_60627 = slt64(gtid_49652, y_48309);\n        bool index_primexp_60628 = binop_x_60626 && binop_y_60627;\n        double max_arg_49920 = ((__global\n                                 double *) tketau_mem_60702)[gtid_49651 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_49652 *\n                                                             zzdim_48114 +\n                                                             gtid_49653];\n        double max_res_49921 = fmax64(0.0, max_arg_49920);\n       ",
                   " double sqrt_res_49922;\n        \n        sqrt_res_49922 = futrts_sqrt64(max_res_49921);\n        \n        bool cond_t_res_49923 = slt64(gtid_49653, y_48310);\n        bool x_49924 = cond_t_res_49923 && index_primexp_60628;\n        double lifted_0_f_res_49925;\n        \n        if (x_49924) {\n            int64_t i_49926 = add64(1, gtid_49653);\n            bool x_49927 = sle64(0, i_49926);\n            bool y_49928 = slt64(i_49926, zzdim_48114);\n            bool bounds_check_49929 = x_49927 && y_49928;\n            bool index_certs_49930;\n            \n            if (!bounds_check_49929) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 0) ==\n                        -1) {\n                        global_failure_args[0] = i_49926;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_49931 = ((__global double *) dzzt_mem_60718)[i_49926];\n            double x_49932 = 1.0 / y_49931;\n            double x_49933 = 0.5 * x_49932;\n            double x_49943 = ((__global double *) kappaM_mem_60723)[gtid_49651 *\n                                                                    (zzdim_48160 *\n                                                                     ydim_48159) +\n                                                                    gtid_49652 *\n                                                                    zzdim_48160 +\n                                                                    gtid_49653];\n            bool index_certs_49946;\n            \n            if (!bounds_check_49929) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 1) ==\n                        -1) {\n                        global_failure_args[0] = gtid_49651;\n                        global_failure_args[1] = gtid_49652;\n                        global_failure_args[2] = i_49926",
                   ";\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_49947 = ((__global double *) kappaM_mem_60723)[gtid_49651 *\n                                                                    (zzdim_48160 *\n                                                                     ydim_48159) +\n                                                                    gtid_49652 *\n                                                                    zzdim_48160 +\n                                                                    i_49926];\n            double y_49948 = x_49943 + y_49947;\n            double lifted_0_f_res_t_res_49949 = x_49933 * y_49948;\n            \n            lifted_0_f_res_49925 = lifted_0_f_res_t_res_49949;\n        } else {\n            lifted_0_f_res_49925 = 0.0;\n        }\n        ((__global double *) mem_60731)[gtid_49651 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_49652 *\n                                        zzdim_48114 + gtid_49653] =\n            lifted_0_f_res_49925;\n        ((__global double *) mem_60735)[gtid_49651 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_49652 *\n                                        zzdim_48114 + gtid_49653] =\n            sqrt_res_49922;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_49904\n}\n__kernel void integrate_tkezisegmap_52110(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                  ",
                   "        int64_t zzdim_48114, int64_t y_48310,\n                                          int64_t m_48377,\n                                          int64_t num_groups_54169, __global\n                                          unsigned char *mem_60981, __global\n                                          unsigned char *mem_60985, __global\n                                          unsigned char *mem_60994, __global\n                                          unsigned char *mem_61005, __global\n                                          unsigned char *mem_61017)\n{\n    #define segmap_group_sizze_54168 (integrate_tkezisegmap_group_sizze_52113)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_61226;\n    int32_t local_tid_61227;\n    int64_t group_sizze_61230;\n    int32_t wave_sizze_61229;\n    int32_t group_tid_61228;\n    \n    global_tid_61226 = get_global_id(0);\n    local_tid_61227 = get_local_id(0);\n    group_sizze_61230 = get_local_size(0);\n    wave_sizze_61229 = LOCKSTEP_WIDTH;\n    group_tid_61228 = get_group_id(0);\n    \n    int32_t phys_tid_52110;\n    \n    phys_tid_52110 = global_tid_61226;\n    \n    int32_t phys_group_id_61231;\n    \n    phys_group_id_61231 = get_group_id(0);\n    for (int32_t i_61232 = 0; i_61232 <\n         sdiv_up32(sext_i64_i32(sdiv_up64(xdim_48112 * ydim_48113,\n                                          segmap_group_sizze_54168)) -\n                   phys_group_id_61231, sext_i64_i32(num_groups_54169));\n         i_61232++) {\n        int32_t virt_group_id_61233 = phys_group_id_61231 + i_61232 *\n                sext_i64_i32(num_groups_54169);\n        int64_t gtid_52108 = squot64(sext_i32_i64(virt_group_id_61233) *\n                                     segma",
                   "p_group_sizze_54168 +\n                                     sext_i32_i64(local_tid_61227), ydim_48113);\n        int64_t gtid_52109 = sext_i32_i64(virt_group_id_61233) *\n                segmap_group_sizze_54168 + sext_i32_i64(local_tid_61227) -\n                squot64(sext_i32_i64(virt_group_id_61233) *\n                        segmap_group_sizze_54168 +\n                        sext_i32_i64(local_tid_61227), ydim_48113) * ydim_48113;\n        \n        if (slt64(gtid_52108, xdim_48112) && slt64(gtid_52109, ydim_48113)) {\n            for (int64_t i_61234 = 0; i_61234 < zzdim_48114; i_61234++) {\n                ((__global double *) mem_61005)[phys_tid_52110 + i_61234 *\n                                                (num_groups_54169 *\n                                                 segmap_group_sizze_54168)] =\n                    ((__global double *) mem_60994)[gtid_52108 * ydim_48113 +\n                                                    gtid_52109 + i_61234 *\n                                                    (ydim_48113 * xdim_48112)];\n            }\n            for (int64_t i_54176 = 0; i_54176 < y_48310; i_54176++) {\n                int64_t binop_y_54178 = -1 * i_54176;\n                int64_t binop_x_54179 = m_48377 + binop_y_54178;\n                bool x_54180 = sle64(0, binop_x_54179);\n                bool y_54181 = slt64(binop_x_54179, zzdim_48114);\n                bool bounds_check_54182 = x_54180 && y_54181;\n                bool index_certs_54183;\n                \n                if (!bounds_check_54182) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 30) ==\n                            -1) {\n                            global_failure_args[0] = binop_x_54179;\n                            global_failure_args[1] = zzdim_48114;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n            ",
                   "    \n                double x_54184 = ((__global double *) mem_60985)[binop_x_54179 *\n                                                                 (ydim_48113 *\n                                                                  xdim_48112) +\n                                                                 gtid_52108 *\n                                                                 ydim_48113 +\n                                                                 gtid_52109];\n                double x_54185 = ((__global double *) mem_60981)[binop_x_54179 *\n                                                                 (ydim_48113 *\n                                                                  xdim_48112) +\n                                                                 gtid_52108 *\n                                                                 ydim_48113 +\n                                                                 gtid_52109];\n                int64_t i_54186 = add64(1, binop_x_54179);\n                bool x_54187 = sle64(0, i_54186);\n                bool y_54188 = slt64(i_54186, zzdim_48114);\n                bool bounds_check_54189 = x_54187 && y_54188;\n                bool index_certs_54190;\n                \n                if (!bounds_check_54189) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 31) ==\n                            -1) {\n                            global_failure_args[0] = i_54186;\n                            global_failure_args[1] = zzdim_48114;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double y_54191 = ((__global\n                                   double *) mem_61005)[phys_tid_52110 +\n                                                        i_54186 *\n                                                        (num_groups_541",
                   "69 *\n                                                         segmap_group_sizze_54168)];\n                double y_54192 = x_54185 * y_54191;\n                double lw_val_54193 = x_54184 - y_54192;\n                \n                ((__global double *) mem_61005)[phys_tid_52110 + binop_x_54179 *\n                                                (num_groups_54169 *\n                                                 segmap_group_sizze_54168)] =\n                    lw_val_54193;\n            }\n            for (int64_t i_61236 = 0; i_61236 < zzdim_48114; i_61236++) {\n                ((__global double *) mem_61017)[i_61236 * (ydim_48113 *\n                                                           xdim_48112) +\n                                                gtid_52108 * ydim_48113 +\n                                                gtid_52109] = ((__global\n                                                                double *) mem_61005)[phys_tid_52110 +\n                                                                                     i_61236 *\n                                                                                     (num_groups_54169 *\n                                                                                      segmap_group_sizze_54168)];\n            }\n        }\n        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_54168\n}\n__kernel void integrate_tkezisegmap_52150(__global int *global_failure,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114, int64_t y_48310,\n                                          __global unsigned char *mem_60739,\n                                          __global unsigned char *mem_60989)\n{\n    #define segmap_group_sizze_54157 (integrate_tkezisegmap_group_sizze_52154)\n    \n    const int block_dim0 = 0;\n    const int block_dim1",
                   " = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61221;\n    int32_t local_tid_61222;\n    int64_t group_sizze_61225;\n    int32_t wave_sizze_61224;\n    int32_t group_tid_61223;\n    \n    global_tid_61221 = get_global_id(0);\n    local_tid_61222 = get_local_id(0);\n    group_sizze_61225 = get_local_size(0);\n    wave_sizze_61224 = LOCKSTEP_WIDTH;\n    group_tid_61223 = get_group_id(0);\n    \n    int32_t phys_tid_52150;\n    \n    phys_tid_52150 = global_tid_61221;\n    \n    int64_t gtid_52145;\n    \n    gtid_52145 = squot64(sext_i32_i64(group_tid_61223) *\n                         segmap_group_sizze_54157 +\n                         sext_i32_i64(local_tid_61222), ydim_48113);\n    \n    int64_t gtid_52146;\n    \n    gtid_52146 = sext_i32_i64(group_tid_61223) * segmap_group_sizze_54157 +\n        sext_i32_i64(local_tid_61222) - squot64(sext_i32_i64(group_tid_61223) *\n                                                segmap_group_sizze_54157 +\n                                                sext_i32_i64(local_tid_61222),\n                                                ydim_48113) * ydim_48113;\n    \n    int64_t gtid_slice_52147;\n    \n    gtid_slice_52147 = sext_i32_i64(group_tid_61223) *\n        segmap_group_sizze_54157 + sext_i32_i64(local_tid_61222) -\n        squot64(sext_i32_i64(group_tid_61223) * segmap_group_sizze_54157 +\n                sext_i32_i64(local_tid_61222), ydim_48113) * ydim_48113 -\n        (sext_i32_i64(group_tid_61223) * segmap_group_sizze_54157 +\n         sext_i32_i64(local_tid_61222) - squot64(sext_i32_i64(group_tid_61223) *\n                                                 segmap_group_sizze_54157 +\n                                                 sext_i32_i64(local_tid_61222),\n                                                 ydim_48113) * ydim_48113);\n    if ((slt64(gtid_52145, xdim_48112) && slt64(gtid_52146, ydim_48113)) &&\n        slt64(gtid_slice_52147, 1)) {\n        int64_t index_primexp_60459 = y_",
                   "48310 + gtid_slice_52147;\n        double v_54164 = ((__global double *) mem_60989)[gtid_52145 *\n                                                         (zzdim_48114 *\n                                                          ydim_48113) +\n                                                         gtid_52146 *\n                                                         zzdim_48114 +\n                                                         index_primexp_60459];\n        \n        if (((sle64(0, gtid_52145) && slt64(gtid_52145, xdim_48112)) &&\n             (sle64(0, gtid_52146) && slt64(gtid_52146, ydim_48113))) &&\n            (sle64(0, index_primexp_60459) && slt64(index_primexp_60459,\n                                                    zzdim_48114))) {\n            ((__global double *) mem_60739)[gtid_52145 * (zzdim_48114 *\n                                                          ydim_48113) +\n                                            gtid_52146 * zzdim_48114 +\n                                            index_primexp_60459] = v_54164;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_54157\n}\n__kernel void integrate_tkezisegmap_52228(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114, int64_t y_48310,\n                                          int64_t num_groups_54090, __global\n                                          unsigned char *mem_60920, __global\n                                          unsigned char *mem_60924, __global\n                                          unsigned char *mem_60928, __global\n                                          unsigned char *mem_60932, __global\n                                          unsigned char *mem_",
                   "60936, __global\n                                          unsigned char *mem_60940, __global\n                                          unsigned char *mem_60959, __global\n                                          unsigned char *mem_60964, __global\n                                          unsigned char *mem_60981, __global\n                                          unsigned char *mem_60985)\n{\n    #define segmap_group_sizze_54089 (integrate_tkezisegmap_group_sizze_52231)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_61207;\n    int32_t local_tid_61208;\n    int64_t group_sizze_61211;\n    int32_t wave_sizze_61210;\n    int32_t group_tid_61209;\n    \n    global_tid_61207 = get_global_id(0);\n    local_tid_61208 = get_local_id(0);\n    group_sizze_61211 = get_local_size(0);\n    wave_sizze_61210 = LOCKSTEP_WIDTH;\n    group_tid_61209 = get_group_id(0);\n    \n    int32_t phys_tid_52228;\n    \n    phys_tid_52228 = global_tid_61207;\n    \n    int32_t phys_group_id_61212;\n    \n    phys_group_id_61212 = get_group_id(0);\n    for (int32_t i_61213 = 0; i_61213 <\n         sdiv_up32(sext_i64_i32(sdiv_up64(xdim_48112 * ydim_48113,\n                                          segmap_group_sizze_54089)) -\n                   phys_group_id_61212, sext_i64_i32(num_groups_54090));\n         i_61213++) {\n        int32_t virt_group_id_61214 = phys_group_id_61212 + i_61213 *\n                sext_i64_i32(num_groups_54090);\n        int64_t gtid_52226 = squot64(sext_i32_i64(virt_group_id_61214) *\n                                     segmap_group_sizze_54089 +\n                                     sext_i32_i64(local_tid_61208), ydim_48113);\n        int64_t gtid_52227 = sext_i32_i64(virt_group_id_61214) ",
                   "*\n                segmap_group_sizze_54089 + sext_i32_i64(local_tid_61208) -\n                squot64(sext_i32_i64(virt_group_id_61214) *\n                        segmap_group_sizze_54089 +\n                        sext_i32_i64(local_tid_61208), ydim_48113) * ydim_48113;\n        \n        if (slt64(gtid_52226, xdim_48112) && slt64(gtid_52227, ydim_48113)) {\n            for (int64_t i_61215 = 0; i_61215 < zzdim_48114; i_61215++) {\n                ((__global double *) mem_60959)[phys_tid_52228 + i_61215 *\n                                                (num_groups_54090 *\n                                                 segmap_group_sizze_54089)] =\n                    ((__global double *) mem_60924)[gtid_52226 * ydim_48113 +\n                                                    gtid_52227 + i_61215 *\n                                                    (ydim_48113 * xdim_48112)];\n            }\n            for (int64_t i_61216 = 0; i_61216 < zzdim_48114; i_61216++) {\n                ((__global double *) mem_60964)[phys_tid_52228 + i_61216 *\n                                                (num_groups_54090 *\n                                                 segmap_group_sizze_54089)] =\n                    ((__global double *) mem_60920)[gtid_52226 * ydim_48113 +\n                                                    gtid_52227 + i_61216 *\n                                                    (ydim_48113 * xdim_48112)];\n            }\n            for (int64_t i_54102 = 0; i_54102 < y_48310; i_54102++) {\n                int64_t index_primexp_54105 = add64(1, i_54102);\n                bool x_54106 = sle64(0, index_primexp_54105);\n                bool y_54107 = slt64(index_primexp_54105, zzdim_48114);\n                bool bounds_check_54108 = x_54106 && y_54107;\n                bool index_certs_54109;\n                \n                if (!bounds_check_54108) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 28) ==\n                          ",
                   "  -1) {\n                            global_failure_args[0] = index_primexp_54105;\n                            global_failure_args[1] = zzdim_48114;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double x_54110 = ((__global\n                                   double *) mem_60928)[index_primexp_54105 *\n                                                        (ydim_48113 *\n                                                         xdim_48112) +\n                                                        gtid_52226 *\n                                                        ydim_48113 +\n                                                        gtid_52227];\n                double x_54111 = ((__global\n                                   double *) mem_60932)[index_primexp_54105 *\n                                                        (ydim_48113 *\n                                                         xdim_48112) +\n                                                        gtid_52226 *\n                                                        ydim_48113 +\n                                                        gtid_52227];\n                bool y_54112 = slt64(i_54102, zzdim_48114);\n                bool index_certs_54113;\n                \n                if (!y_54112) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 29) ==\n                            -1) {\n                            global_failure_args[0] = i_54102;\n                            global_failure_args[1] = zzdim_48114;\n                            ;\n                        }\n                        local_failure = true;\n                        goto error_0;\n                    }\n                }\n                \n                double y_54114 = ((__global\n                                   double *) mem_60959)[phys",
                   "_tid_52228 +\n                                                        i_54102 *\n                                                        (num_groups_54090 *\n                                                         segmap_group_sizze_54089)];\n                double y_54115 = x_54111 * y_54114;\n                double y_54116 = x_54110 - y_54115;\n                double norm_factor_54117 = 1.0 / y_54116;\n                double x_54118 = ((__global\n                                   double *) mem_60936)[index_primexp_54105 *\n                                                        (ydim_48113 *\n                                                         xdim_48112) +\n                                                        gtid_52226 *\n                                                        ydim_48113 +\n                                                        gtid_52227];\n                double lw_val_54119 = norm_factor_54117 * x_54118;\n                \n                ((__global double *) mem_60959)[phys_tid_52228 +\n                                                index_primexp_54105 *\n                                                (num_groups_54090 *\n                                                 segmap_group_sizze_54089)] =\n                    lw_val_54119;\n                \n                double x_54121 = ((__global\n                                   double *) mem_60940)[index_primexp_54105 *\n                                                        (ydim_48113 *\n                                                         xdim_48112) +\n                                                        gtid_52226 *\n                                                        ydim_48113 +\n                                                        gtid_52227];\n                double y_54122 = ((__global\n                                   double *) mem_60964)[phys_tid_52228 +\n                                                        i_54102 *\n                                                        (nu",
                   "m_groups_54090 *\n                                                         segmap_group_sizze_54089)];\n                double y_54123 = x_54111 * y_54122;\n                double x_54124 = x_54121 - y_54123;\n                double lw_val_54125 = norm_factor_54117 * x_54124;\n                \n                ((__global double *) mem_60964)[phys_tid_52228 +\n                                                index_primexp_54105 *\n                                                (num_groups_54090 *\n                                                 segmap_group_sizze_54089)] =\n                    lw_val_54125;\n            }\n            for (int64_t i_61219 = 0; i_61219 < zzdim_48114; i_61219++) {\n                ((__global double *) mem_60981)[i_61219 * (ydim_48113 *\n                                                           xdim_48112) +\n                                                gtid_52226 * ydim_48113 +\n                                                gtid_52227] = ((__global\n                                                                double *) mem_60959)[phys_tid_52228 +\n                                                                                     i_61219 *\n                                                                                     (num_groups_54090 *\n                                                                                      segmap_group_sizze_54089)];\n            }\n            for (int64_t i_61220 = 0; i_61220 < zzdim_48114; i_61220++) {\n                ((__global double *) mem_60985)[i_61220 * (ydim_48113 *\n                                                           xdim_48112) +\n                                                gtid_52226 * ydim_48113 +\n                                                gtid_52227] = ((__global\n                                                                double *) mem_60964)[phys_tid_52228 +\n                                                                                     i_61220 *\n                      ",
                   "                                                               (num_groups_54090 *\n                                                                                      segmap_group_sizze_54089)];\n            }\n        }\n        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_54089\n}\n__kernel void integrate_tkezisegmap_52305(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114, __global\n                                          unsigned char *mem_60895, __global\n                                          unsigned char *mem_60899, __global\n                                          unsigned char *mem_60903, __global\n                                          unsigned char *mem_60912, __global\n                                          unsigned char *mem_60916)\n{\n    #define segmap_group_sizze_54064 (integrate_tkezisegmap_group_sizze_52309)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61202;\n    int32_t local_tid_61203;\n    int64_t group_sizze_61206;\n    int32_t wave_sizze_61205;\n    int32_t group_tid_61204;\n    \n    global_tid_61202 = get_global_id(0);\n    local_tid_61203 = get_local_id(0);\n    group_sizze_61206 = get_local_size(0);\n    wave_sizze_61205 = LOCKSTEP_WIDTH;\n    group_tid_61204 = get_group_id(0);\n    \n    int32_t phys_tid_52305;\n    \n    phys_tid_52305 = global_tid_61202;\n    \n    int64_t gtid_52302;\n    \n    gtid_52302 = squot64(sext_i32_i64(group_tid_61204) *\n                         segmap_group_sizze_54064 +\n                         sext_i32_i64(local_ti",
                   "d_61203), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_52303;\n    \n    gtid_52303 = squot64(sext_i32_i64(group_tid_61204) *\n                         segmap_group_sizze_54064 +\n                         sext_i32_i64(local_tid_61203) -\n                         squot64(sext_i32_i64(group_tid_61204) *\n                                 segmap_group_sizze_54064 +\n                                 sext_i32_i64(local_tid_61203), ydim_48113 *\n                                 zzdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_52304;\n    \n    gtid_52304 = sext_i32_i64(group_tid_61204) * segmap_group_sizze_54064 +\n        sext_i32_i64(local_tid_61203) - squot64(sext_i32_i64(group_tid_61204) *\n                                                segmap_group_sizze_54064 +\n                                                sext_i32_i64(local_tid_61203),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61204) *\n                                             segmap_group_sizze_54064 +\n                                             sext_i32_i64(local_tid_61203) -\n                                             squot64(sext_i32_i64(group_tid_61204) *\n                                                     segmap_group_sizze_54064 +\n                                                     sext_i32_i64(local_tid_61203),\n                                                     ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 * zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_52302, xdim_48112) && slt64(gtid_52303, ydim_48113)) &&\n        slt64(gtid_52304, zzdim_48114)) {\n        bool cond_54072 = gtid_52304 == 0;\n        double lifted_0_f_res_54073;\n        \n        if (cond_54072) {\n            bool y_54074 = slt64(0, zzdim_48114);\n            bool",
                   " index_certs_54075;\n            \n            if (!y_54074) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 26) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_54076 = ((__global double *) mem_60899)[gtid_52302 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_52303 *\n                                                             zzdim_48114];\n            double y_54077 = ((__global double *) mem_60903)[gtid_52302 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_52303 *\n                                                             zzdim_48114];\n            double lifted_0_f_res_t_res_54078 = x_54076 / y_54077;\n            \n            lifted_0_f_res_54073 = lifted_0_f_res_t_res_54078;\n        } else {\n            lifted_0_f_res_54073 = 0.0;\n        }\n        \n        double lifted_0_f_res_54079;\n        \n        if (cond_54072) {\n            bool y_54080 = slt64(0, zzdim_48114);\n            bool index_certs_54081;\n            \n            if (!y_54080) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 27) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_54082 = ((__global double *) m",
                   "em_60895)[gtid_52302 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_52303 *\n                                                             zzdim_48114];\n            double y_54083 = ((__global double *) mem_60903)[gtid_52302 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_52303 *\n                                                             zzdim_48114];\n            double lifted_0_f_res_t_res_54084 = x_54082 / y_54083;\n            \n            lifted_0_f_res_54079 = lifted_0_f_res_t_res_54084;\n        } else {\n            lifted_0_f_res_54079 = 0.0;\n        }\n        ((__global double *) mem_60912)[gtid_52302 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_52303 *\n                                        zzdim_48114 + gtid_52304] =\n            lifted_0_f_res_54079;\n        ((__global double *) mem_60916)[gtid_52302 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_52303 *\n                                        zzdim_48114 + gtid_52304] =\n            lifted_0_f_res_54073;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_54064\n}\n__kernel void integrate_tkezisegmap_52597(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114,\n                                          int64_t ydim_48157,\n                                          int64",
                   "_t ydim_48162,\n                                          int64_t zzdim_48163,\n                                          int64_t ydim_48165,\n                                          int64_t zzdim_48166,\n                                          int64_t ydim_48168, int64_t y_48308,\n                                          int64_t y_48309, int64_t y_48310,\n                                          __global\n                                          unsigned char *tketau_mem_60702,\n                                          __global\n                                          unsigned char *dzzw_mem_60719,\n                                          __global\n                                          unsigned char *kbot_mem_60722,\n                                          __global unsigned char *mxl_mem_60724,\n                                          __global\n                                          unsigned char *forc_mem_60725,\n                                          __global\n                                          unsigned char *forc_tke_surface_mem_60726,\n                                          __global unsigned char *mem_60731,\n                                          __global unsigned char *mem_60735,\n                                          __global unsigned char *mem_60895,\n                                          __global unsigned char *mem_60899,\n                                          __global unsigned char *mem_60903,\n                                          __global unsigned char *mem_60907)\n{\n    #define segmap_group_sizze_53821 (integrate_tkezisegmap_group_sizze_52601)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61197;\n    int32_t local_tid_61198;\n    int64_t group_sizze_61201;\n    int32_t wave_sizze_61200;\n    int32_t group_tid_61199;\n    \n    global_tid_61197 = get_global_id(0);\n    local_tid_61198 = get_local_id(0);",
                   "\n    group_sizze_61201 = get_local_size(0);\n    wave_sizze_61200 = LOCKSTEP_WIDTH;\n    group_tid_61199 = get_group_id(0);\n    \n    int32_t phys_tid_52597;\n    \n    phys_tid_52597 = global_tid_61197;\n    \n    int64_t gtid_52594;\n    \n    gtid_52594 = squot64(sext_i32_i64(group_tid_61199) *\n                         segmap_group_sizze_53821 +\n                         sext_i32_i64(local_tid_61198), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_52595;\n    \n    gtid_52595 = squot64(sext_i32_i64(group_tid_61199) *\n                         segmap_group_sizze_53821 +\n                         sext_i32_i64(local_tid_61198) -\n                         squot64(sext_i32_i64(group_tid_61199) *\n                                 segmap_group_sizze_53821 +\n                                 sext_i32_i64(local_tid_61198), ydim_48113 *\n                                 zzdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_52596;\n    \n    gtid_52596 = sext_i32_i64(group_tid_61199) * segmap_group_sizze_53821 +\n        sext_i32_i64(local_tid_61198) - squot64(sext_i32_i64(group_tid_61199) *\n                                                segmap_group_sizze_53821 +\n                                                sext_i32_i64(local_tid_61198),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61199) *\n                                             segmap_group_sizze_53821 +\n                                             sext_i32_i64(local_tid_61198) -\n                                             squot64(sext_i32_i64(group_tid_61199) *\n                                                     segmap_group_sizze_53821 +\n                                                     sext_i32_i64(local_tid_61198),\n                                                     ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 ",
                   "* zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_52594, xdim_48112) && slt64(gtid_52595, ydim_48113)) &&\n        slt64(gtid_52596, zzdim_48114)) {\n        bool binop_x_60629 = sle64(2, gtid_52595);\n        bool binop_x_60630 = sle64(2, gtid_52594);\n        bool binop_y_60631 = slt64(gtid_52594, y_48308);\n        bool binop_y_60632 = binop_x_60630 && binop_y_60631;\n        bool binop_x_60633 = binop_x_60629 && binop_y_60632;\n        bool binop_y_60634 = slt64(gtid_52595, y_48309);\n        bool index_primexp_60635 = binop_x_60633 && binop_y_60634;\n        double lifted_0_f_res_53831;\n        \n        if (index_primexp_60635) {\n            int32_t x_53840 = ((__global int32_t *) kbot_mem_60722)[gtid_52594 *\n                                                                    ydim_48157 +\n                                                                    gtid_52595];\n            int32_t ks_val_53841 = sub32(x_53840, 1);\n            bool land_mask_53842 = sle32(0, ks_val_53841);\n            int32_t i64_res_53843 = sext_i64_i32(gtid_52596);\n            bool edge_mask_t_res_53844 = i64_res_53843 == ks_val_53841;\n            bool x_53845 = land_mask_53842 && edge_mask_t_res_53844;\n            bool water_mask_t_res_53846 = sle32(ks_val_53841, i64_res_53843);\n            bool x_53847 = land_mask_53842 && water_mask_t_res_53846;\n            bool cond_f_res_53848 = !x_53847;\n            bool x_53849 = !x_53845;\n            bool y_53850 = cond_f_res_53848 && x_53849;\n            bool cond_53851 = x_53845 || y_53850;\n            double lifted_0_f_res_t_res_53852;\n            \n            if (cond_53851) {\n                lifted_0_f_res_t_res_53852 = 0.0;\n            } else {\n                bool cond_53853 = slt64(0, gtid_52596);\n                int64_t y_53854 = sub64(zzdim_48114, 1);\n                bool cond_t_res_53855 = slt64(gtid_52596, y_53854);\n                bool x_53856 = cond_53853 && cond_t_res_53855;\n      ",
                   "          double lifted_0_f_res_t_res_f_res_53857;\n                \n                if (x_53856) {\n                    int64_t i_53858 = sub64(gtid_52596, 1);\n                    bool x_53859 = sle64(0, i_53858);\n                    bool y_53860 = slt64(i_53858, zzdim_48114);\n                    bool bounds_check_53861 = x_53859 && y_53860;\n                    bool index_certs_53864;\n                    \n                    if (!bounds_check_53861) {\n                        {\n                            if (atomic_cmpxchg_i32_global(global_failure, -1,\n                                                          22) == -1) {\n                                global_failure_args[0] = gtid_52594;\n                                global_failure_args[1] = gtid_52595;\n                                global_failure_args[2] = i_53858;\n                                global_failure_args[3] = xdim_48112;\n                                global_failure_args[4] = ydim_48113;\n                                global_failure_args[5] = zzdim_48114;\n                                ;\n                            }\n                            return;\n                        }\n                    }\n                    \n                    double x_53865 = ((__global\n                                       double *) mem_60731)[gtid_52594 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_52595 *\n                                                            zzdim_48114 +\n                                                            i_53858];\n                    double y_53870 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_52596];\n                    double negate_arg_53871 = x_53865 / y_53870;\n                    double lifted_0_f_res_t_res_f_res_t_res_53872 = 0.0 -\n                           negate_ar",
                   "g_53871;\n                    \n                    lifted_0_f_res_t_res_f_res_53857 =\n                        lifted_0_f_res_t_res_f_res_t_res_53872;\n                } else {\n                    bool cond_53873 = gtid_52596 == y_53854;\n                    double lifted_0_f_res_t_res_f_res_f_res_53874;\n                    \n                    if (cond_53873) {\n                        int64_t i_53875 = sub64(gtid_52596, 1);\n                        bool x_53876 = sle64(0, i_53875);\n                        bool y_53877 = slt64(i_53875, zzdim_48114);\n                        bool bounds_check_53878 = x_53876 && y_53877;\n                        bool index_certs_53881;\n                        \n                        if (!bounds_check_53878) {\n                            {\n                                if (atomic_cmpxchg_i32_global(global_failure,\n                                                              -1, 23) == -1) {\n                                    global_failure_args[0] = gtid_52594;\n                                    global_failure_args[1] = gtid_52595;\n                                    global_failure_args[2] = i_53875;\n                                    global_failure_args[3] = xdim_48112;\n                                    global_failure_args[4] = ydim_48113;\n                                    global_failure_args[5] = zzdim_48114;\n                                    ;\n                                }\n                                return;\n                            }\n                        }\n                        \n                        double x_53882 = ((__global\n                                           double *) mem_60731)[gtid_52594 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_52595 *\n                                                                zzdim_48114",
                   " +\n                                                                i_53875];\n                        double y_53887 = ((__global\n                                           double *) dzzw_mem_60719)[gtid_52596];\n                        double y_53888 = 0.5 * y_53887;\n                        double negate_arg_53889 = x_53882 / y_53888;\n                        double lifted_0_f_res_t_res_f_res_f_res_t_res_53890 =\n                               0.0 - negate_arg_53889;\n                        \n                        lifted_0_f_res_t_res_f_res_f_res_53874 =\n                            lifted_0_f_res_t_res_f_res_f_res_t_res_53890;\n                    } else {\n                        lifted_0_f_res_t_res_f_res_f_res_53874 = 0.0;\n                    }\n                    lifted_0_f_res_t_res_f_res_53857 =\n                        lifted_0_f_res_t_res_f_res_f_res_53874;\n                }\n                lifted_0_f_res_t_res_53852 = lifted_0_f_res_t_res_f_res_53857;\n            }\n            lifted_0_f_res_53831 = lifted_0_f_res_t_res_53852;\n        } else {\n            lifted_0_f_res_53831 = 0.0;\n        }\n        \n        double lifted_0_f_res_53891;\n        \n        if (index_primexp_60635) {\n            int32_t x_53900 = ((__global int32_t *) kbot_mem_60722)[gtid_52594 *\n                                                                    ydim_48157 +\n                                                                    gtid_52595];\n            int32_t ks_val_53901 = sub32(x_53900, 1);\n            bool land_mask_53902 = sle32(0, ks_val_53901);\n            int32_t i64_res_53903 = sext_i64_i32(gtid_52596);\n            bool edge_mask_t_res_53904 = i64_res_53903 == ks_val_53901;\n            bool x_53905 = land_mask_53902 && edge_mask_t_res_53904;\n            bool water_mask_t_res_53906 = sle32(ks_val_53901, i64_res_53903);\n            bool x_53907 = land_mask_53902 && water_mask_t_res_53906;\n            bool cond_53908 = !x_53907;\n            double lifted_0_f_res_t_res_53909;\n  ",
                   "          \n            if (cond_53908) {\n                lifted_0_f_res_t_res_53909 = 1.0;\n            } else {\n                double lifted_0_f_res_t_res_f_res_53910;\n                \n                if (x_53905) {\n                    double x_53917 = ((__global\n                                       double *) mem_60731)[gtid_52594 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_52595 *\n                                                            zzdim_48114 +\n                                                            gtid_52596];\n                    double y_53919 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_52596];\n                    double y_53920 = x_53917 / y_53919;\n                    double x_53921 = 1.0 + y_53920;\n                    double y_53922 = ((__global\n                                       double *) mxl_mem_60724)[gtid_52594 *\n                                                                (zzdim_48163 *\n                                                                 ydim_48162) +\n                                                                gtid_52595 *\n                                                                zzdim_48163 +\n                                                                gtid_52596];\n                    double x_53923 = 0.7 / y_53922;\n                    double y_53924 = ((__global\n                                       double *) mem_60735)[gtid_52594 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_52595 *\n                                                            zzdim_48114 +\n                                                            gtid_52596];\n",
                   "                    double y_53925 = x_53923 * y_53924;\n                    double lifted_0_f_res_t_res_f_res_t_res_53926 = x_53921 +\n                           y_53925;\n                    \n                    lifted_0_f_res_t_res_f_res_53910 =\n                        lifted_0_f_res_t_res_f_res_t_res_53926;\n                } else {\n                    bool cond_53927 = slt64(0, gtid_52596);\n                    int64_t y_53928 = sub64(zzdim_48114, 1);\n                    bool cond_t_res_53929 = slt64(gtid_52596, y_53928);\n                    bool x_53930 = cond_53927 && cond_t_res_53929;\n                    double lifted_0_f_res_t_res_f_res_f_res_53931;\n                    \n                    if (x_53930) {\n                        double x_53938 = ((__global\n                                           double *) mem_60731)[gtid_52594 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_52595 *\n                                                                zzdim_48114 +\n                                                                gtid_52596];\n                        int64_t i_53939 = sub64(gtid_52596, 1);\n                        bool x_53940 = sle64(0, i_53939);\n                        bool y_53941 = slt64(i_53939, zzdim_48114);\n                        bool bounds_check_53942 = x_53940 && y_53941;\n                        bool index_certs_53945;\n                        \n                        if (!bounds_check_53942) {\n                            {\n                                if (atomic_cmpxchg_i32_global(global_failure,\n                                                              -1, 24) == -1) {\n                                    global_failure_args[0] = gtid_52594;\n                                    global_failure_args[1] = gtid_52595;\n                                    global_fai",
                   "lure_args[2] = i_53939;\n                                    global_failure_args[3] = xdim_48112;\n                                    global_failure_args[4] = ydim_48113;\n                                    global_failure_args[5] = zzdim_48114;\n                                    ;\n                                }\n                                return;\n                            }\n                        }\n                        \n                        double y_53946 = ((__global\n                                           double *) mem_60731)[gtid_52594 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_52595 *\n                                                                zzdim_48114 +\n                                                                i_53939];\n                        double x_53947 = x_53938 + y_53946;\n                        double y_53949 = ((__global\n                                           double *) dzzw_mem_60719)[gtid_52596];\n                        double y_53950 = x_53947 / y_53949;\n                        double x_53951 = 1.0 + y_53950;\n                        double y_53952 = ((__global\n                                           double *) mem_60735)[gtid_52594 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_52595 *\n                                                                zzdim_48114 +\n                                                                gtid_52596];\n                        double x_53953 = 0.7 * y_53952;\n                        double y_53954 = ((__global\n                                           double *) mxl_mem_60724)[gtid_52594 *\n                             ",
                   "                                       (zzdim_48163 *\n                                                                     ydim_48162) +\n                                                                    gtid_52595 *\n                                                                    zzdim_48163 +\n                                                                    gtid_52596];\n                        double y_53955 = x_53953 / y_53954;\n                        double lifted_0_f_res_t_res_f_res_f_res_t_res_53956 =\n                               x_53951 + y_53955;\n                        \n                        lifted_0_f_res_t_res_f_res_f_res_53931 =\n                            lifted_0_f_res_t_res_f_res_f_res_t_res_53956;\n                    } else {\n                        bool cond_53957 = gtid_52596 == y_53928;\n                        double lifted_0_f_res_t_res_f_res_f_res_f_res_53958;\n                        \n                        if (cond_53957) {\n                            int64_t i_53959 = sub64(gtid_52596, 1);\n                            bool x_53960 = sle64(0, i_53959);\n                            bool y_53961 = slt64(i_53959, zzdim_48114);\n                            bool bounds_check_53962 = x_53960 && y_53961;\n                            bool index_certs_53965;\n                            \n                            if (!bounds_check_53962) {\n                                {\n                                    if (atomic_cmpxchg_i32_global(global_failure,\n                                                                  -1, 25) ==\n                                        -1) {\n                                        global_failure_args[0] = gtid_52594;\n                                        global_failure_args[1] = gtid_52595;\n                                        global_failure_args[2] = i_53959;\n                                        global_failure_args[3] = xdim_48112;\n                                        global_failure_args[4] = ydim_481",
                   "13;\n                                        global_failure_args[5] = zzdim_48114;\n                                        ;\n                                    }\n                                    return;\n                                }\n                            }\n                            \n                            double x_53966 = ((__global\n                                               double *) mem_60731)[gtid_52594 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    gtid_52595 *\n                                                                    zzdim_48114 +\n                                                                    i_53959];\n                            double y_53971 = ((__global\n                                               double *) dzzw_mem_60719)[gtid_52596];\n                            double y_53972 = 0.5 * y_53971;\n                            double y_53973 = x_53966 / y_53972;\n                            double x_53974 = 1.0 + y_53973;\n                            double y_53978 = ((__global\n                                               double *) mxl_mem_60724)[gtid_52594 *\n                                                                        (zzdim_48163 *\n                                                                         ydim_48162) +\n                                                                        gtid_52595 *\n                                                                        zzdim_48163 +\n                                                                        gtid_52596];\n                            double x_53979 = 0.7 / y_53978;\n                            double y_53980 = ((__global\n                                               double *) mem_60735)[gtid_52594 *\n                                                               ",
                   "     (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    gtid_52595 *\n                                                                    zzdim_48114 +\n                                                                    gtid_52596];\n                            double y_53981 = x_53979 * y_53980;\n                            double\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_t_res_53982 =\n                            x_53974 + y_53981;\n                            \n                            lifted_0_f_res_t_res_f_res_f_res_f_res_53958 =\n                                lifted_0_f_res_t_res_f_res_f_res_f_res_t_res_53982;\n                        } else {\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_53958 = 0.0;\n                        }\n                        lifted_0_f_res_t_res_f_res_f_res_53931 =\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_53958;\n                    }\n                    lifted_0_f_res_t_res_f_res_53910 =\n                        lifted_0_f_res_t_res_f_res_f_res_53931;\n                }\n                lifted_0_f_res_t_res_53909 = lifted_0_f_res_t_res_f_res_53910;\n            }\n            lifted_0_f_res_53891 = lifted_0_f_res_t_res_53909;\n        } else {\n            lifted_0_f_res_53891 = 0.0;\n        }\n        \n        bool cond_t_res_53983 = slt64(gtid_52596, y_48310);\n        bool x_53984 = cond_t_res_53983 && index_primexp_60635;\n        double lifted_0_f_res_53985;\n        \n        if (x_53984) {\n            int32_t x_53994 = ((__global int32_t *) kbot_mem_60722)[gtid_52594 *\n                                                                    ydim_48157 +\n                                                                    gtid_52595];\n            int32_t ks_val_53995 = sub32(x_53994, 1);\n            bool land_mask_53996 = sle32(0, ks_val_53995);\n            int32_t ",
                   "i64_res_53997 = sext_i64_i32(gtid_52596);\n            bool water_mask_t_res_53998 = sle32(ks_val_53995, i64_res_53997);\n            bool x_53999 = land_mask_53996 && water_mask_t_res_53998;\n            bool cond_54000 = !x_53999;\n            double lifted_0_f_res_t_res_54001;\n            \n            if (cond_54000) {\n                lifted_0_f_res_t_res_54001 = 0.0;\n            } else {\n                double x_54008 = ((__global double *) mem_60731)[gtid_52594 *\n                                                                 (zzdim_48114 *\n                                                                  ydim_48113) +\n                                                                 gtid_52595 *\n                                                                 zzdim_48114 +\n                                                                 gtid_52596];\n                double y_54010 = ((__global\n                                   double *) dzzw_mem_60719)[gtid_52596];\n                double negate_arg_54011 = x_54008 / y_54010;\n                double lifted_0_f_res_t_res_f_res_54012 = 0.0 -\n                       negate_arg_54011;\n                \n                lifted_0_f_res_t_res_54001 = lifted_0_f_res_t_res_f_res_54012;\n            }\n            lifted_0_f_res_53985 = lifted_0_f_res_t_res_54001;\n        } else {\n            lifted_0_f_res_53985 = 0.0;\n        }\n        \n        double lifted_0_f_res_54013;\n        \n        if (index_primexp_60635) {\n            int32_t x_54022 = ((__global int32_t *) kbot_mem_60722)[gtid_52594 *\n                                                                    ydim_48157 +\n                                                                    gtid_52595];\n            int32_t ks_val_54023 = sub32(x_54022, 1);\n            bool land_mask_54024 = sle32(0, ks_val_54023);\n            int32_t i64_res_54025 = sext_i64_i32(gtid_52596);\n            bool water_mask_t_res_54026 = sle32(ks_val_54023, i64_res_54025);\n            bool x_54027 =",
                   " land_mask_54024 && water_mask_t_res_54026;\n            bool cond_54028 = !x_54027;\n            double lifted_0_f_res_t_res_54029;\n            \n            if (cond_54028) {\n                lifted_0_f_res_t_res_54029 = 0.0;\n            } else {\n                double x_54036 = ((__global\n                                   double *) tketau_mem_60702)[gtid_52594 *\n                                                               (zzdim_48114 *\n                                                                ydim_48113) +\n                                                               gtid_52595 *\n                                                               zzdim_48114 +\n                                                               gtid_52596];\n                double y_54037 = ((__global\n                                   double *) forc_mem_60725)[gtid_52594 *\n                                                             (zzdim_48166 *\n                                                              ydim_48165) +\n                                                             gtid_52595 *\n                                                             zzdim_48166 +\n                                                             gtid_52596];\n                double tmp_54038 = x_54036 + y_54037;\n                int64_t y_54039 = sub64(zzdim_48114, 1);\n                bool cond_54040 = gtid_52596 == y_54039;\n                double lifted_0_f_res_t_res_f_res_54041;\n                \n                if (cond_54040) {\n                    double y_54042 = ((__global\n                                       double *) forc_tke_surface_mem_60726)[gtid_52594 *\n                                                                             ydim_48168 +\n                                                                             gtid_52595];\n                    double y_54044 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_52596];\n                    double y_54045 = ",
                   "0.5 * y_54044;\n                    double y_54046 = y_54042 / y_54045;\n                    double lifted_0_f_res_t_res_f_res_t_res_54047 = tmp_54038 +\n                           y_54046;\n                    \n                    lifted_0_f_res_t_res_f_res_54041 =\n                        lifted_0_f_res_t_res_f_res_t_res_54047;\n                } else {\n                    lifted_0_f_res_t_res_f_res_54041 = tmp_54038;\n                }\n                lifted_0_f_res_t_res_54029 = lifted_0_f_res_t_res_f_res_54041;\n            }\n            lifted_0_f_res_54013 = lifted_0_f_res_t_res_54029;\n        } else {\n            lifted_0_f_res_54013 = 0.0;\n        }\n        ((__global double *) mem_60895)[gtid_52594 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_52595 *\n                                        zzdim_48114 + gtid_52596] =\n            lifted_0_f_res_54013;\n        ((__global double *) mem_60899)[gtid_52594 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_52595 *\n                                        zzdim_48114 + gtid_52596] =\n            lifted_0_f_res_53985;\n        ((__global double *) mem_60903)[gtid_52594 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_52595 *\n                                        zzdim_48114 + gtid_52596] =\n            lifted_0_f_res_53891;\n        ((__global double *) mem_60907)[gtid_52594 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_52595 *\n                                        zzdim_48114 + gtid_52596] =\n            lifted_0_f_res_53831;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_53821\n}\n__kernel void integrate_tkezisegmap_54402(__global int *global_failure,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114,\n      ",
                   "                                    int64_t ydim_48116,\n                                          int64_t zzdim_48117, int64_t y_48308,\n                                          int64_t y_48309, __global\n                                          unsigned char *tketaup1_mem_60703,\n                                          __global\n                                          unsigned char *lifted_11_map_res_mem_61024,\n                                          __global unsigned char *mem_61028,\n                                          __global unsigned char *mem_61030,\n                                          __global unsigned char *mem_61037)\n{\n    #define segmap_group_sizze_54629 (integrate_tkezisegmap_group_sizze_54406)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61244;\n    int32_t local_tid_61245;\n    int64_t group_sizze_61248;\n    int32_t wave_sizze_61247;\n    int32_t group_tid_61246;\n    \n    global_tid_61244 = get_global_id(0);\n    local_tid_61245 = get_local_id(0);\n    group_sizze_61248 = get_local_size(0);\n    wave_sizze_61247 = LOCKSTEP_WIDTH;\n    group_tid_61246 = get_group_id(0);\n    \n    int32_t phys_tid_54402;\n    \n    phys_tid_54402 = global_tid_61244;\n    \n    int64_t gtid_54399;\n    \n    gtid_54399 = squot64(sext_i32_i64(group_tid_61246) *\n                         segmap_group_sizze_54629 +\n                         sext_i32_i64(local_tid_61245), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_54400;\n    \n    gtid_54400 = squot64(sext_i32_i64(group_tid_61246) *\n                         segmap_group_sizze_54629 +\n                         sext_i32_i64(local_tid_61245) -\n                         squot64(sext_i32_i64(group_tid_61246) *\n                                 segmap_group_sizze_54629 +\n                                 sext_i32_i64(local_tid_61245), ydim_48113 *\n                                 z",
                   "zdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_54401;\n    \n    gtid_54401 = sext_i32_i64(group_tid_61246) * segmap_group_sizze_54629 +\n        sext_i32_i64(local_tid_61245) - squot64(sext_i32_i64(group_tid_61246) *\n                                                segmap_group_sizze_54629 +\n                                                sext_i32_i64(local_tid_61245),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61246) *\n                                             segmap_group_sizze_54629 +\n                                             sext_i32_i64(local_tid_61245) -\n                                             squot64(sext_i32_i64(group_tid_61246) *\n                                                     segmap_group_sizze_54629 +\n                                                     sext_i32_i64(local_tid_61245),\n                                                     ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 * zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_54399, xdim_48112) && slt64(gtid_54400, ydim_48113)) &&\n        slt64(gtid_54401, zzdim_48114)) {\n        int32_t ks_val_54636 = ((__global int32_t *) mem_61028)[gtid_54399 *\n                                                                ydim_48113 +\n                                                                gtid_54400];\n        bool cond_54637 = ((__global bool *) mem_61030)[gtid_54399 *\n                                                        ydim_48113 +\n                                                        gtid_54400];\n        bool binop_x_60636 = sle64(2, gtid_54400);\n        bool binop_x_60637 = sle64(2, gtid_54399);\n        bool binop_y_60638 = slt64(gtid_54399, y_48308);\n        bool binop_y_60639 = binop_x_60637 && binop_y_60638;\n        bool binop_x_",
                   "60640 = binop_x_60636 && binop_y_60639;\n        bool binop_y_60641 = slt64(gtid_54400, y_48309);\n        bool index_primexp_60642 = binop_x_60640 && binop_y_60641;\n        int32_t i64_res_54640 = sext_i64_i32(gtid_54401);\n        bool water_mask_t_res_54641 = sle32(ks_val_54636, i64_res_54640);\n        bool x_54642 = cond_54637 && water_mask_t_res_54641;\n        bool x_54643 = x_54642 && index_primexp_60642;\n        double lifted_0_f_res_54644;\n        \n        if (x_54643) {\n            double lifted_0_f_res_t_res_54651 = ((__global\n                                                  double *) lifted_11_map_res_mem_61024)[gtid_54399 *\n                                                                                         (zzdim_48114 *\n                                                                                          ydim_48113) +\n                                                                                         gtid_54400 *\n                                                                                         zzdim_48114 +\n                                                                                         gtid_54401];\n            \n            lifted_0_f_res_54644 = lifted_0_f_res_t_res_54651;\n        } else {\n            double lifted_0_f_res_f_res_54658 = ((__global\n                                                  double *) tketaup1_mem_60703)[gtid_54399 *\n                                                                                (zzdim_48117 *\n                                                                                 ydim_48116) +\n                                                                                gtid_54400 *\n                                                                                zzdim_48117 +\n                                                                                gtid_54401];\n            \n            lifted_0_f_res_54644 = lifted_0_f_res_f_res_54658;\n        }\n        ((__global double *) mem_61037)[gt",
                   "id_54399 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_54400 *\n                                        zzdim_48114 + gtid_54401] =\n            lifted_0_f_res_54644;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_54629\n}\n__kernel void integrate_tkezisegmap_54471(__global int *global_failure,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t ydim_48157, int64_t y_48308,\n                                          int64_t y_48309, __global\n                                          unsigned char *kbot_mem_60722,\n                                          __global unsigned char *mem_61028,\n                                          __global unsigned char *mem_61030,\n                                          __global unsigned char *mem_61032)\n{\n    #define segmap_group_sizze_54595 (integrate_tkezisegmap_group_sizze_54474)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61239;\n    int32_t local_tid_61240;\n    int64_t group_sizze_61243;\n    int32_t wave_sizze_61242;\n    int32_t group_tid_61241;\n    \n    global_tid_61239 = get_global_id(0);\n    local_tid_61240 = get_local_id(0);\n    group_sizze_61243 = get_local_size(0);\n    wave_sizze_61242 = LOCKSTEP_WIDTH;\n    group_tid_61241 = get_group_id(0);\n    \n    int32_t phys_tid_54471;\n    \n    phys_tid_54471 = global_tid_61239;\n    \n    int64_t gtid_54469;\n    \n    gtid_54469 = squot64(sext_i32_i64(group_tid_61241) *\n                         segmap_group_sizze_54595 +\n                         sext_i32_i64(local_tid_61240), ydim_48113);\n    \n    int64_t gtid_54470;\n    \n    gtid_54470 = sext_i32_i64(group_tid_61241) * segmap_group_sizze_54595 +\n        sext_i32_i64(local_tid_61240) - squot64(sext_i32_i64(group_tid_61241) *\n      ",
                   "                                          segmap_group_sizze_54595 +\n                                                sext_i32_i64(local_tid_61240),\n                                                ydim_48113) * ydim_48113;\n    if (slt64(gtid_54469, xdim_48112) && slt64(gtid_54470, ydim_48113)) {\n        bool binop_x_60466 = sle64(2, gtid_54469);\n        bool binop_y_60469 = slt64(gtid_54469, y_48308);\n        bool index_primexp_60470 = binop_x_60466 && binop_y_60469;\n        int32_t x_54610 = ((__global int32_t *) kbot_mem_60722)[gtid_54469 *\n                                                                ydim_48157 +\n                                                                gtid_54470];\n        int32_t ks_val_54611 = sub32(x_54610, 1);\n        bool cond_54612 = sle32(0, ks_val_54611);\n        bool cond_t_res_54613 = sle64(2, gtid_54470);\n        bool x_54614 = cond_t_res_54613 && index_primexp_60470;\n        bool cond_t_res_54615 = slt64(gtid_54470, y_48309);\n        bool x_54616 = x_54614 && cond_t_res_54615;\n        \n        ((__global int32_t *) mem_61028)[gtid_54469 * ydim_48113 + gtid_54470] =\n            ks_val_54611;\n        ((__global bool *) mem_61030)[gtid_54469 * ydim_48113 + gtid_54470] =\n            cond_54612;\n        ((__global bool *) mem_61032)[gtid_54469 * ydim_48113 + gtid_54470] =\n            x_54616;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_54595\n}\n__kernel void integrate_tkezisegmap_55217(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114,\n                                          int64_t ydim_48140,\n                                          int64_t zzdim_48141,\n                                          int6",
                   "4_t ydim_48143,\n                                          int64_t zzdim_48144, int64_t y_48308,\n                                          int64_t y_48309, int64_t y_48310,\n                                          int64_t y_48726, int64_t y_48727,\n                                          __global\n                                          unsigned char *tketau_mem_60702,\n                                          __global\n                                          unsigned char *maskU_mem_60711,\n                                          __global\n                                          unsigned char *maskV_mem_60712,\n                                          __global unsigned char *dxu_mem_60715,\n                                          __global unsigned char *dyu_mem_60717,\n                                          __global\n                                          unsigned char *cost_mem_60720,\n                                          __global\n                                          unsigned char *cosu_mem_60721,\n                                          __global unsigned char *mem_61037,\n                                          __global unsigned char *mem_61054,\n                                          __global unsigned char *mem_61058,\n                                          __global unsigned char *mem_61062)\n{\n    #define segmap_group_sizze_55702 (integrate_tkezisegmap_group_sizze_55221)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61254;\n    int32_t local_tid_61255;\n    int64_t group_sizze_61258;\n    int32_t wave_sizze_61257;\n    int32_t group_tid_61256;\n    \n    global_tid_61254 = get_global_id(0);\n    local_tid_61255 = get_local_id(0);\n    group_sizze_61258 = get_local_size(0);\n    wave_sizze_61257 = LOCKSTEP_WIDTH;\n    group_tid_61256 = get_group_id(0);\n    \n    int32_t phys_tid_55217;\n    \n    phys_tid_55217 = global_tid_6",
                   "1254;\n    \n    int64_t gtid_55214;\n    \n    gtid_55214 = squot64(sext_i32_i64(group_tid_61256) *\n                         segmap_group_sizze_55702 +\n                         sext_i32_i64(local_tid_61255), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_55215;\n    \n    gtid_55215 = squot64(sext_i32_i64(group_tid_61256) *\n                         segmap_group_sizze_55702 +\n                         sext_i32_i64(local_tid_61255) -\n                         squot64(sext_i32_i64(group_tid_61256) *\n                                 segmap_group_sizze_55702 +\n                                 sext_i32_i64(local_tid_61255), ydim_48113 *\n                                 zzdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_55216;\n    \n    gtid_55216 = sext_i32_i64(group_tid_61256) * segmap_group_sizze_55702 +\n        sext_i32_i64(local_tid_61255) - squot64(sext_i32_i64(group_tid_61256) *\n                                                segmap_group_sizze_55702 +\n                                                sext_i32_i64(local_tid_61255),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61256) *\n                                             segmap_group_sizze_55702 +\n                                             sext_i32_i64(local_tid_61255) -\n                                             squot64(sext_i32_i64(group_tid_61256) *\n                                                     segmap_group_sizze_55702 +\n                                                     sext_i32_i64(local_tid_61255),\n                                                     ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 * zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_55214, xdim_48112) && slt64(gtid_55215, ydim_48113)) &&\n        slt64(gtid_55216, zz",
                   "dim_48114)) {\n        bool index_primexp_60522 = slt64(gtid_55214, y_48726);\n        bool binop_x_60643 = sle64(2, gtid_55215);\n        bool binop_x_60644 = sle64(2, gtid_55214);\n        bool binop_y_60645 = slt64(gtid_55214, y_48308);\n        bool binop_y_60646 = binop_x_60644 && binop_y_60645;\n        bool binop_x_60647 = binop_x_60643 && binop_y_60646;\n        bool binop_y_60648 = slt64(gtid_55215, y_48309);\n        bool index_primexp_60649 = binop_x_60647 && binop_y_60648;\n        bool index_primexp_60517 = slt64(gtid_55215, y_48727);\n        bool cond_t_res_55713 = gtid_55216 == y_48310;\n        bool x_55714 = cond_t_res_55713 && index_primexp_60649;\n        double lifted_0_f_res_55715;\n        \n        if (x_55714) {\n            double tke_val_55728 = ((__global double *) mem_61037)[gtid_55214 *\n                                                                   (zzdim_48114 *\n                                                                    ydim_48113) +\n                                                                   gtid_55215 *\n                                                                   zzdim_48114 +\n                                                                   gtid_55216];\n            bool cond_55729 = tke_val_55728 < 0.0;\n            double lifted_0_f_res_t_res_55730;\n            \n            if (cond_55729) {\n                lifted_0_f_res_t_res_55730 = 0.0;\n            } else {\n                lifted_0_f_res_t_res_55730 = tke_val_55728;\n            }\n            lifted_0_f_res_55715 = lifted_0_f_res_t_res_55730;\n        } else {\n            double lifted_0_f_res_f_res_55743 = ((__global\n                                                  double *) mem_61037)[gtid_55214 *\n                                                                       (zzdim_48114 *\n                                                                        ydim_48113) +\n                                                                       gtid_55215 *\n                 ",
                   "                                                      zzdim_48114 +\n                                                                       gtid_55216];\n            \n            lifted_0_f_res_55715 = lifted_0_f_res_f_res_55743;\n        }\n        \n        double lifted_0_f_res_55744;\n        \n        if (index_primexp_60522) {\n            int64_t i_55745 = add64(1, gtid_55214);\n            bool x_55746 = sle64(0, i_55745);\n            bool y_55747 = slt64(i_55745, xdim_48112);\n            bool bounds_check_55748 = x_55746 && y_55747;\n            bool index_certs_55757;\n            \n            if (!bounds_check_55748) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 34) ==\n                        -1) {\n                        global_failure_args[0] = i_55745;\n                        global_failure_args[1] = gtid_55215;\n                        global_failure_args[2] = gtid_55216;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_55758 = ((__global double *) tketau_mem_60702)[i_55745 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    gtid_55215 *\n                                                                    zzdim_48114 +\n                                                                    gtid_55216];\n            double y_55765 = ((__global double *) tketau_mem_60702)[gtid_55214 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                            ",
                   "                                        gtid_55215 *\n                                                                    zzdim_48114 +\n                                                                    gtid_55216];\n            double y_55766 = x_55758 - y_55765;\n            double x_55767 = 2000.0 * y_55766;\n            double x_55769 = ((__global double *) cost_mem_60720)[gtid_55215];\n            double y_55771 = ((__global double *) dxu_mem_60715)[gtid_55214];\n            double y_55772 = x_55769 * y_55771;\n            double x_55773 = x_55767 / y_55772;\n            double y_55774 = ((__global double *) maskU_mem_60711)[gtid_55214 *\n                                                                   (zzdim_48141 *\n                                                                    ydim_48140) +\n                                                                   gtid_55215 *\n                                                                   zzdim_48141 +\n                                                                   gtid_55216];\n            double lifted_0_f_res_t_res_55775 = x_55773 * y_55774;\n            \n            lifted_0_f_res_55744 = lifted_0_f_res_t_res_55775;\n        } else {\n            lifted_0_f_res_55744 = 0.0;\n        }\n        \n        double lifted_0_f_res_55776;\n        \n        if (index_primexp_60517) {\n            int64_t i_55780 = add64(1, gtid_55215);\n            bool x_55781 = sle64(0, i_55780);\n            bool y_55782 = slt64(i_55780, ydim_48113);\n            bool bounds_check_55783 = x_55781 && y_55782;\n            bool index_certs_55789;\n            \n            if (!bounds_check_55783) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 35) ==\n                        -1) {\n                        global_failure_args[0] = gtid_55214;\n                        global_failure_args[1] = i_55780;\n                        global_failure_args[2] = gtid_55216;\n                        global_failure_args[3] ",
                   "= xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_55790 = ((__global double *) tketau_mem_60702)[gtid_55214 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    i_55780 *\n                                                                    zzdim_48114 +\n                                                                    gtid_55216];\n            double y_55796 = ((__global double *) tketau_mem_60702)[gtid_55214 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    gtid_55215 *\n                                                                    zzdim_48114 +\n                                                                    gtid_55216];\n            double y_55797 = x_55790 - y_55796;\n            double x_55798 = 2000.0 * y_55797;\n            double y_55800 = ((__global double *) dyu_mem_60717)[gtid_55215];\n            double x_55801 = x_55798 / y_55800;\n            double y_55802 = ((__global double *) maskV_mem_60712)[gtid_55214 *\n                                                                   (zzdim_48144 *\n                                                                    ydim_48143) +\n                                                                   gtid_55215 *\n                                                                   zzdim_48144 +\n                                                                   gtid_55216];\n            double x_55803 = x_55801 * y_55802;\n         ",
                   "   double y_55804 = ((__global double *) cosu_mem_60721)[gtid_55215];\n            double lifted_0_f_res_t_res_55805 = x_55803 * y_55804;\n            \n            lifted_0_f_res_55776 = lifted_0_f_res_t_res_55805;\n        } else {\n            lifted_0_f_res_55776 = 0.0;\n        }\n        ((__global double *) mem_61054)[gtid_55214 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_55215 *\n                                        zzdim_48114 + gtid_55216] =\n            lifted_0_f_res_55776;\n        ((__global double *) mem_61058)[gtid_55214 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_55215 *\n                                        zzdim_48114 + gtid_55216] =\n            lifted_0_f_res_55744;\n        ((__global double *) mem_61062)[gtid_55214 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_55215 *\n                                        zzdim_48114 + gtid_55216] =\n            lifted_0_f_res_55715;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_55702\n}\n__kernel void integrate_tkezisegmap_55432(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114, int64_t y_48308,\n                                          int64_t y_48309, int64_t y_48727,\n                                          __global\n                                          unsigned char *dzzw_mem_60719,\n                                          __global unsigned char *mem_61041,\n                                          __global unsigned char *mem_61044,\n                                          __global unsigned char *mem_61046,\n                                       ",
                   "   __global unsigned char *mem_61049)\n{\n    #define segmap_group_sizze_55655 (integrate_tkezisegmap_group_sizze_55435)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61249;\n    int32_t local_tid_61250;\n    int64_t group_sizze_61253;\n    int32_t wave_sizze_61252;\n    int32_t group_tid_61251;\n    \n    global_tid_61249 = get_global_id(0);\n    local_tid_61250 = get_local_id(0);\n    group_sizze_61253 = get_local_size(0);\n    wave_sizze_61252 = LOCKSTEP_WIDTH;\n    group_tid_61251 = get_group_id(0);\n    \n    int32_t phys_tid_55432;\n    \n    phys_tid_55432 = global_tid_61249;\n    \n    int64_t gtid_55430;\n    \n    gtid_55430 = squot64(sext_i32_i64(group_tid_61251) *\n                         segmap_group_sizze_55655 +\n                         sext_i32_i64(local_tid_61250), ydim_48113);\n    \n    int64_t gtid_55431;\n    \n    gtid_55431 = sext_i32_i64(group_tid_61251) * segmap_group_sizze_55655 +\n        sext_i32_i64(local_tid_61250) - squot64(sext_i32_i64(group_tid_61251) *\n                                                segmap_group_sizze_55655 +\n                                                sext_i32_i64(local_tid_61250),\n                                                ydim_48113) * ydim_48113;\n    if (slt64(gtid_55430, xdim_48112) && slt64(gtid_55431, ydim_48113)) {\n        bool binop_x_60506 = sle64(2, gtid_55430);\n        bool binop_y_60509 = slt64(gtid_55430, y_48308);\n        bool index_primexp_60510 = binop_x_60506 && binop_y_60509;\n        bool cond_t_res_55663 = sle64(2, gtid_55431);\n        bool x_55664 = cond_t_res_55663 && index_primexp_60510;\n        bool cond_t_res_55665 = slt64(gtid_55431, y_48309);\n        bool x_55666 = x_55664 && cond_t_res_55665;\n        double lifted_0_f_res_55667;\n        \n        if (x_55666) {\n            int64_t i_55674 = sub64(zzdim_48114, 1);\n            bool x_55675 = sle64(0, i_55674);\n            bool y_55",
                   "676 = slt64(i_55674, zzdim_48114);\n            bool bounds_check_55677 = x_55675 && y_55676;\n            bool index_certs_55680;\n            \n            if (!bounds_check_55677) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 32) ==\n                        -1) {\n                        global_failure_args[0] = gtid_55430;\n                        global_failure_args[1] = gtid_55431;\n                        global_failure_args[2] = i_55674;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double tke_val_55681 = ((__global double *) mem_61041)[i_55674 *\n                                                                   (ydim_48113 *\n                                                                    xdim_48112) +\n                                                                   gtid_55430 *\n                                                                   ydim_48113 +\n                                                                   gtid_55431];\n            bool cond_55682 = tke_val_55681 < 0.0;\n            double lifted_0_f_res_t_res_55683;\n            \n            if (cond_55682) {\n                double x_55684 = 0.5 * tke_val_55681;\n                bool index_certs_55685;\n                \n                if (!bounds_check_55677) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 33) ==\n                            -1) {\n                            global_failure_args[0] = i_55674;\n                            global_failure_args[1] = zzdim_48114;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double y",
                   "_55686 = ((__global double *) dzzw_mem_60719)[i_55674];\n                double x_55687 = x_55684 * y_55686;\n                double lifted_0_f_res_t_res_t_res_55688 = 0.0 - x_55687;\n                \n                lifted_0_f_res_t_res_55683 = lifted_0_f_res_t_res_t_res_55688;\n            } else {\n                lifted_0_f_res_t_res_55683 = 0.0;\n            }\n            lifted_0_f_res_55667 = lifted_0_f_res_t_res_55683;\n        } else {\n            lifted_0_f_res_55667 = 0.0;\n        }\n        \n        bool cond_55689 = slt64(gtid_55431, y_48727);\n        \n        ((__global bool *) mem_61044)[gtid_55430 * ydim_48113 + gtid_55431] =\n            x_55666;\n        ((__global bool *) mem_61046)[gtid_55430 * ydim_48113 + gtid_55431] =\n            cond_55689;\n        ((__global double *) mem_61049)[gtid_55430 * ydim_48113 + gtid_55431] =\n            lifted_0_f_res_55667;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_55655\n}\n__kernel void integrate_tkezisegmap_57267(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114,\n                                          int64_t ydim_48131,\n                                          int64_t zzdim_48132,\n                                          int64_t ydim_48134,\n                                          int64_t zzdim_48135,\n                                          int64_t ydim_48137,\n                                          int64_t zzdim_48138,\n                                          int64_t ydim_48146,\n                                          int64_t zzdim_48147, int64_t y_48308,\n                                          int64_t y_48309, int64_t y_48310,\n                                          __glob",
                   "al\n                                          unsigned char *tketau_mem_60702,\n                                          __global\n                                          unsigned char *utau_mem_60708,\n                                          __global\n                                          unsigned char *vtau_mem_60709,\n                                          __global\n                                          unsigned char *wtau_mem_60710,\n                                          __global\n                                          unsigned char *maskW_mem_60713,\n                                          __global unsigned char *dxt_mem_60714,\n                                          __global unsigned char *dyt_mem_60716,\n                                          __global\n                                          unsigned char *dzzw_mem_60719,\n                                          __global\n                                          unsigned char *cost_mem_60720,\n                                          __global\n                                          unsigned char *cosu_mem_60721,\n                                          __global unsigned char *mem_61054,\n                                          __global unsigned char *mem_61058,\n                                          __global unsigned char *mem_61062,\n                                          __global unsigned char *mem_61067,\n                                          __global unsigned char *mem_61071,\n                                          __global unsigned char *mem_61075,\n                                          __global unsigned char *mem_61079)\n{\n    #define segmap_group_sizze_58435 (integrate_tkezisegmap_group_sizze_57271)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61259;\n    int32_t local_tid_61260;\n    int64_t group_sizze_61263;\n    int32_t wave_sizze_61262;\n",
                   "    int32_t group_tid_61261;\n    \n    global_tid_61259 = get_global_id(0);\n    local_tid_61260 = get_local_id(0);\n    group_sizze_61263 = get_local_size(0);\n    wave_sizze_61262 = LOCKSTEP_WIDTH;\n    group_tid_61261 = get_group_id(0);\n    \n    int32_t phys_tid_57267;\n    \n    phys_tid_57267 = global_tid_61259;\n    \n    int64_t gtid_57264;\n    \n    gtid_57264 = squot64(sext_i32_i64(group_tid_61261) *\n                         segmap_group_sizze_58435 +\n                         sext_i32_i64(local_tid_61260), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_57265;\n    \n    gtid_57265 = squot64(sext_i32_i64(group_tid_61261) *\n                         segmap_group_sizze_58435 +\n                         sext_i32_i64(local_tid_61260) -\n                         squot64(sext_i32_i64(group_tid_61261) *\n                                 segmap_group_sizze_58435 +\n                                 sext_i32_i64(local_tid_61260), ydim_48113 *\n                                 zzdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_57266;\n    \n    gtid_57266 = sext_i32_i64(group_tid_61261) * segmap_group_sizze_58435 +\n        sext_i32_i64(local_tid_61260) - squot64(sext_i32_i64(group_tid_61261) *\n                                                segmap_group_sizze_58435 +\n                                                sext_i32_i64(local_tid_61260),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61261) *\n                                             segmap_group_sizze_58435 +\n                                             sext_i32_i64(local_tid_61260) -\n                                             squot64(sext_i32_i64(group_tid_61261) *\n                                                     segmap_group_sizze_58435 +\n                                                     sext_i32_i64(local_tid_61260),\n                         ",
                   "                            ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 * zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_57264, xdim_48112) && slt64(gtid_57265, ydim_48113)) &&\n        slt64(gtid_57266, zzdim_48114)) {\n        bool index_primexp_60571 = sle64(2, gtid_57264);\n        bool index_primexp_60568 = slt64(gtid_57264, y_48308);\n        bool index_primexp_60556 = sle64(2, gtid_57265);\n        bool index_primexp_60553 = slt64(gtid_57265, y_48309);\n        bool binop_y_60667 = index_primexp_60568 && index_primexp_60571;\n        bool binop_x_60668 = index_primexp_60556 && binop_y_60667;\n        bool index_primexp_60670 = index_primexp_60553 && binop_x_60668;\n        bool binop_y_60660 = sle64(1, gtid_57264);\n        bool binop_y_60661 = index_primexp_60568 && binop_y_60660;\n        bool binop_y_60662 = index_primexp_60556 && binop_y_60661;\n        bool index_primexp_60663 = index_primexp_60553 && binop_y_60662;\n        bool binop_y_60651 = sle64(1, gtid_57265);\n        bool binop_x_60652 = index_primexp_60553 && binop_y_60651;\n        bool binop_x_60654 = index_primexp_60571 && binop_x_60652;\n        bool index_primexp_60656 = index_primexp_60568 && binop_x_60654;\n        double previous_58459 = ((__global double *) mem_61062)[gtid_57264 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_57265 *\n                                                                zzdim_48114 +\n                                                                gtid_57266];\n        bool cond_t_res_58460 = gtid_57266 == y_48310;\n        bool x_58461 = cond_t_res_58460 && index_primexp_60670;\n        double lifted_0_f_res_58462;\n        \n        if (x_58461) {\n            double y_58472 = ((__global double *) maskW_",
                   "mem_60713)[gtid_57264 *\n                                                                   (zzdim_48147 *\n                                                                    ydim_48146) +\n                                                                   gtid_57265 *\n                                                                   zzdim_48147 +\n                                                                   gtid_57266];\n            double x_58473 = ((__global double *) mem_61058)[gtid_57264 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_57265 *\n                                                             zzdim_48114 +\n                                                             gtid_57266];\n            int64_t i_58474 = sub64(gtid_57264, 1);\n            bool x_58475 = sle64(0, i_58474);\n            bool y_58476 = slt64(i_58474, xdim_48112);\n            bool bounds_check_58477 = x_58475 && y_58476;\n            bool index_certs_58480;\n            \n            if (!bounds_check_58477) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 36) ==\n                        -1) {\n                        global_failure_args[0] = i_58474;\n                        global_failure_args[1] = gtid_57265;\n                        global_failure_args[2] = gtid_57266;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_58481 = ((__global double *) mem_61058)[i_58474 *\n                                                             (zzdim_48114 *\n                                                              ydim",
                   "_48113) +\n                                                             gtid_57265 *\n                                                             zzdim_48114 +\n                                                             gtid_57266];\n            double x_58482 = x_58473 - y_58481;\n            double x_58484 = ((__global double *) cost_mem_60720)[gtid_57265];\n            double y_58486 = ((__global double *) dxt_mem_60714)[gtid_57264];\n            double y_58487 = x_58484 * y_58486;\n            double x_58488 = x_58482 / y_58487;\n            double x_58489 = ((__global double *) mem_61054)[gtid_57264 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_57265 *\n                                                             zzdim_48114 +\n                                                             gtid_57266];\n            int64_t i_58490 = sub64(gtid_57265, 1);\n            bool x_58491 = sle64(0, i_58490);\n            bool y_58492 = slt64(i_58490, ydim_48113);\n            bool bounds_check_58493 = x_58491 && y_58492;\n            bool index_certs_58495;\n            \n            if (!bounds_check_58493) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 37) ==\n                        -1) {\n                        global_failure_args[0] = gtid_57264;\n                        global_failure_args[1] = i_58490;\n                        global_failure_args[2] = gtid_57266;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_58496 = ((__global double *) mem_61054)[gtid_57264 *\n                                  ",
                   "                           (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             i_58490 *\n                                                             zzdim_48114 +\n                                                             gtid_57266];\n            double x_58497 = x_58489 - y_58496;\n            double y_58498 = ((__global double *) dyt_mem_60716)[gtid_57265];\n            double y_58499 = x_58484 * y_58498;\n            double y_58500 = x_58497 / y_58499;\n            double y_58501 = x_58488 + y_58500;\n            double y_58502 = y_58472 * y_58501;\n            double lifted_0_f_res_t_res_58503 = previous_58459 + y_58502;\n            \n            lifted_0_f_res_58462 = lifted_0_f_res_t_res_58503;\n        } else {\n            lifted_0_f_res_58462 = previous_58459;\n        }\n        \n        double lifted_0_f_res_58504;\n        \n        if (index_primexp_60663) {\n            double x_58509 = ((__global double *) cost_mem_60720)[gtid_57265];\n            double y_58514 = ((__global double *) dxt_mem_60714)[gtid_57264];\n            double dx_58515 = x_58509 * y_58514;\n            double velS_58519 = ((__global\n                                  double *) utau_mem_60708)[gtid_57264 *\n                                                            (zzdim_48132 *\n                                                             ydim_48131) +\n                                                            gtid_57265 *\n                                                            zzdim_48132 +\n                                                            gtid_57266];\n            int64_t i_58520 = sub64(gtid_57264, 1);\n            bool x_58521 = sle64(0, i_58520);\n            bool y_58522 = slt64(i_58520, xdim_48112);\n            bool bounds_check_58523 = x_58521 && y_58522;\n            bool index_certs_58526;\n            \n            if (!bounds_check_58523) {\n                {\n                 ",
                   "   if (atomic_cmpxchg_i32_global(global_failure, -1, 38) ==\n                        -1) {\n                        global_failure_args[0] = i_58520;\n                        global_failure_args[1] = gtid_57265;\n                        global_failure_args[2] = gtid_57266;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double maskWs_58528 = ((__global\n                                    double *) maskW_mem_60713)[gtid_57264 *\n                                                               (zzdim_48147 *\n                                                                ydim_48146) +\n                                                               gtid_57265 *\n                                                               zzdim_48147 +\n                                                               gtid_57266];\n            int64_t i_58529 = add64(1, gtid_57264);\n            bool x_58530 = sle64(0, i_58529);\n            bool y_58531 = slt64(i_58529, xdim_48112);\n            bool bounds_check_58532 = x_58530 && y_58531;\n            bool index_certs_58535;\n            \n            if (!bounds_check_58532) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 39) ==\n                        -1) {\n                        global_failure_args[0] = i_58529;\n                        global_failure_args[1] = gtid_57265;\n                        global_failure_args[2] = gtid_57266;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n          ",
                   "  double maskWp1_58536 = ((__global\n                                     double *) maskW_mem_60713)[i_58529 *\n                                                                (zzdim_48147 *\n                                                                 ydim_48146) +\n                                                                gtid_57265 *\n                                                                zzdim_48147 +\n                                                                gtid_57266];\n            int64_t i_58537 = add64(2, gtid_57264);\n            bool x_58538 = sle64(0, i_58537);\n            bool y_58539 = slt64(i_58537, xdim_48112);\n            bool bounds_check_58540 = x_58538 && y_58539;\n            bool index_certs_58543;\n            \n            if (!bounds_check_58540) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 40) ==\n                        -1) {\n                        global_failure_args[0] = i_58537;\n                        global_failure_args[1] = gtid_57265;\n                        global_failure_args[2] = gtid_57266;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double varSM1_58545 = ((__global\n                                    double *) tketau_mem_60702)[i_58520 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_57265 *\n                                                                zzdim_48114 +\n                                                                gtid_57266];\n            double varS_58546 = ((__global\n                                  doubl",
                   "e *) tketau_mem_60702)[gtid_57264 *\n                                                              (zzdim_48114 *\n                                                               ydim_48113) +\n                                                              gtid_57265 *\n                                                              zzdim_48114 +\n                                                              gtid_57266];\n            double varSP1_58547 = ((__global\n                                    double *) tketau_mem_60702)[i_58529 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_57265 *\n                                                                zzdim_48114 +\n                                                                gtid_57266];\n            double varSP2_58548 = ((__global\n                                    double *) tketau_mem_60702)[i_58537 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_57265 *\n                                                                zzdim_48114 +\n                                                                gtid_57266];\n            int64_t y_58549 = sub64(xdim_48112, 1);\n            bool cond_58550 = slt64(gtid_57264, y_58549);\n            double maskUtr_58551;\n            \n            if (cond_58550) {\n                double maskUtr_t_res_58552 = maskWs_58528 * maskWp1_58536;\n                \n                maskUtr_58551 = maskUtr_t_res_58552;\n            } else {\n                maskUtr_58551 = 0.0;\n            }\n            \n            double maskUtrP1_58553;\n            \n            if (cond_58550) {\n                double maskwp2_58544 = ((__global\n                    ",
                   "                     double *) maskW_mem_60713)[i_58537 *\n                                                                    (zzdim_48147 *\n                                                                     ydim_48146) +\n                                                                    gtid_57265 *\n                                                                    zzdim_48147 +\n                                                                    gtid_57266];\n                double maskUtrP1_t_res_58554 = maskWp1_58536 * maskwp2_58544;\n                \n                maskUtrP1_58553 = maskUtrP1_t_res_58554;\n            } else {\n                maskUtrP1_58553 = 0.0;\n            }\n            \n            double maskUtrM1_58555;\n            \n            if (cond_58550) {\n                double maskWm1_58527 = ((__global\n                                         double *) maskW_mem_60713)[i_58520 *\n                                                                    (zzdim_48147 *\n                                                                     ydim_48146) +\n                                                                    gtid_57265 *\n                                                                    zzdim_48147 +\n                                                                    gtid_57266];\n                double maskUtrM1_t_res_58556 = maskWm1_58527 * maskWs_58528;\n                \n                maskUtrM1_58555 = maskUtrM1_t_res_58556;\n            } else {\n                maskUtrM1_58555 = 0.0;\n            }\n            \n            double abs_arg_58557 = velS_58519 / dx_58515;\n            double abs_res_58558 = fabs(abs_arg_58557);\n            double x_58559 = varSP2_58548 - varSP1_58547;\n            double rjp_58560 = maskUtrP1_58553 * x_58559;\n            double x_58561 = varSP1_58547 - varS_58546;\n            double rj_58562 = maskUtr_58551 * x_58561;\n            double x_58563 = varS_58546 - varSM1_58545;\n            double rjm_58564 = maskUtrM1_",
                   "58555 * x_58563;\n            double abs_res_58565 = fabs(rj_58562);\n            bool cond_58566 = abs_res_58565 < 1.0e-20;\n            double divisor_58567;\n            \n            if (cond_58566) {\n                divisor_58567 = 1.0e-20;\n            } else {\n                divisor_58567 = rj_58562;\n            }\n            \n            bool cond_58568 = 0.0 < velS_58519;\n            double cr_58569;\n            \n            if (cond_58568) {\n                double cr_t_res_58570 = rjm_58564 / divisor_58567;\n                \n                cr_58569 = cr_t_res_58570;\n            } else {\n                double cr_f_res_58571 = rjp_58560 / divisor_58567;\n                \n                cr_58569 = cr_f_res_58571;\n            }\n            \n            double min_res_58572 = fmin64(2.0, cr_58569);\n            double min_arg_58573 = 2.0 * cr_58569;\n            double min_res_58574 = fmin64(1.0, min_arg_58573);\n            double max_res_58575 = fmax64(min_res_58572, min_res_58574);\n            double max_res_58576 = fmax64(0.0, max_res_58575);\n            double y_58577 = varS_58546 + varSP1_58547;\n            double x_58578 = velS_58519 * y_58577;\n            double x_58579 = 0.5 * x_58578;\n            double abs_res_58580 = fabs(velS_58519);\n            double x_58581 = 1.0 - max_res_58576;\n            double y_58582 = abs_res_58558 * max_res_58576;\n            double y_58583 = x_58581 + y_58582;\n            double x_58584 = abs_res_58580 * y_58583;\n            double x_58585 = rj_58562 * x_58584;\n            double y_58586 = 0.5 * x_58585;\n            double calcflux_res_58587 = x_58579 - y_58586;\n            \n            lifted_0_f_res_58504 = calcflux_res_58587;\n        } else {\n            lifted_0_f_res_58504 = 0.0;\n        }\n        \n        double lifted_0_f_res_58588;\n        \n        if (index_primexp_60656) {\n            double x_58593 = ((__global double *) cost_mem_60720)[gtid_57265];\n            double y_58594 = ((__global double *) dyt_mem_60716)[gt",
                   "id_57265];\n            double dx_58595 = x_58593 * y_58594;\n            double velS_58602 = ((__global\n                                  double *) vtau_mem_60709)[gtid_57264 *\n                                                            (zzdim_48135 *\n                                                             ydim_48134) +\n                                                            gtid_57265 *\n                                                            zzdim_48135 +\n                                                            gtid_57266];\n            int64_t i_58603 = sub64(gtid_57265, 1);\n            bool x_58604 = sle64(0, i_58603);\n            bool y_58605 = slt64(i_58603, ydim_48113);\n            bool bounds_check_58606 = x_58604 && y_58605;\n            bool index_certs_58608;\n            \n            if (!bounds_check_58606) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 41) ==\n                        -1) {\n                        global_failure_args[0] = gtid_57264;\n                        global_failure_args[1] = i_58603;\n                        global_failure_args[2] = gtid_57266;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double maskWs_58610 = ((__global\n                                    double *) maskW_mem_60713)[gtid_57264 *\n                                                               (zzdim_48147 *\n                                                                ydim_48146) +\n                                                               gtid_57265 *\n                                                               zzdim_48147 +\n                                                               gtid_57266];\n            int64_t i_58611 = add64(1, ",
                   "gtid_57265);\n            bool x_58612 = sle64(0, i_58611);\n            bool y_58613 = slt64(i_58611, ydim_48113);\n            bool bounds_check_58614 = x_58612 && y_58613;\n            bool index_certs_58616;\n            \n            if (!bounds_check_58614) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 42) ==\n                        -1) {\n                        global_failure_args[0] = gtid_57264;\n                        global_failure_args[1] = i_58611;\n                        global_failure_args[2] = gtid_57266;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double maskWp1_58617 = ((__global\n                                     double *) maskW_mem_60713)[gtid_57264 *\n                                                                (zzdim_48147 *\n                                                                 ydim_48146) +\n                                                                i_58611 *\n                                                                zzdim_48147 +\n                                                                gtid_57266];\n            int64_t i_58618 = add64(2, gtid_57265);\n            bool x_58619 = sle64(0, i_58618);\n            bool y_58620 = slt64(i_58618, ydim_48113);\n            bool bounds_check_58621 = x_58619 && y_58620;\n            bool index_certs_58623;\n            \n            if (!bounds_check_58621) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 43) ==\n                        -1) {\n                        global_failure_args[0] = gtid_57264;\n                        global_failure_args[1] = i_58618;\n                        global_failure_args[2] = gtid_57266;\n                ",
                   "        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double varSM1_58625 = ((__global\n                                    double *) tketau_mem_60702)[gtid_57264 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                i_58603 *\n                                                                zzdim_48114 +\n                                                                gtid_57266];\n            double varS_58626 = ((__global\n                                  double *) tketau_mem_60702)[gtid_57264 *\n                                                              (zzdim_48114 *\n                                                               ydim_48113) +\n                                                              gtid_57265 *\n                                                              zzdim_48114 +\n                                                              gtid_57266];\n            double varSP1_58627 = ((__global\n                                    double *) tketau_mem_60702)[gtid_57264 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                i_58611 *\n                                                                zzdim_48114 +\n                                                                gtid_57266];\n            double varSP2_58628 = ((__global\n                                    double *) tketau_mem_60702)[gtid_57264 *\n                                                                (zzdim_48",
                   "114 *\n                                                                 ydim_48113) +\n                                                                i_58618 *\n                                                                zzdim_48114 +\n                                                                gtid_57266];\n            int64_t y_58629 = sub64(ydim_48113, 1);\n            bool cond_58630 = slt64(gtid_57265, y_58629);\n            double maskVtr_58631;\n            \n            if (cond_58630) {\n                double maskVtr_t_res_58632 = maskWs_58610 * maskWp1_58617;\n                \n                maskVtr_58631 = maskVtr_t_res_58632;\n            } else {\n                maskVtr_58631 = 0.0;\n            }\n            \n            double maskVtrP1_58633;\n            \n            if (cond_58630) {\n                double maskwp2_58624 = ((__global\n                                         double *) maskW_mem_60713)[gtid_57264 *\n                                                                    (zzdim_48147 *\n                                                                     ydim_48146) +\n                                                                    i_58618 *\n                                                                    zzdim_48147 +\n                                                                    gtid_57266];\n                double maskVtrP1_t_res_58634 = maskWp1_58617 * maskwp2_58624;\n                \n                maskVtrP1_58633 = maskVtrP1_t_res_58634;\n            } else {\n                maskVtrP1_58633 = 0.0;\n            }\n            \n            double maskVtrM1_58635;\n            \n            if (cond_58630) {\n                double maskWm1_58609 = ((__global\n                                         double *) maskW_mem_60713)[gtid_57264 *\n                                                                    (zzdim_48147 *\n                                                                     ydim_48146) +\n                                     ",
                   "                               i_58603 *\n                                                                    zzdim_48147 +\n                                                                    gtid_57266];\n                double maskVtrM1_t_res_58636 = maskWm1_58609 * maskWs_58610;\n                \n                maskVtrM1_58635 = maskVtrM1_t_res_58636;\n            } else {\n                maskVtrM1_58635 = 0.0;\n            }\n            \n            double calcflux_arg_58637 = ((__global\n                                          double *) cosu_mem_60721)[gtid_57265];\n            double scaledVel_58638 = velS_58602 * calcflux_arg_58637;\n            double abs_arg_58639 = scaledVel_58638 / dx_58595;\n            double abs_res_58640 = fabs(abs_arg_58639);\n            double x_58641 = varSP2_58628 - varSP1_58627;\n            double rjp_58642 = maskVtrP1_58633 * x_58641;\n            double x_58643 = varSP1_58627 - varS_58626;\n            double rj_58644 = maskVtr_58631 * x_58643;\n            double x_58645 = varS_58626 - varSM1_58625;\n            double rjm_58646 = maskVtrM1_58635 * x_58645;\n            double abs_res_58647 = fabs(rj_58644);\n            bool cond_58648 = abs_res_58647 < 1.0e-20;\n            double divisor_58649;\n            \n            if (cond_58648) {\n                divisor_58649 = 1.0e-20;\n            } else {\n                divisor_58649 = rj_58644;\n            }\n            \n            bool cond_58650 = 0.0 < velS_58602;\n            double cr_58651;\n            \n            if (cond_58650) {\n                double cr_t_res_58652 = rjm_58646 / divisor_58649;\n                \n                cr_58651 = cr_t_res_58652;\n            } else {\n                double cr_f_res_58653 = rjp_58642 / divisor_58649;\n                \n                cr_58651 = cr_f_res_58653;\n            }\n            \n            double min_res_58654 = fmin64(2.0, cr_58651);\n            double min_arg_58655 = 2.0 * cr_58651;\n            double min_res_58656 = fmin64(1.0, min_",
                   "arg_58655);\n            double max_res_58657 = fmax64(min_res_58654, min_res_58656);\n            double max_res_58658 = fmax64(0.0, max_res_58657);\n            double y_58659 = varS_58626 + varSP1_58627;\n            double x_58660 = scaledVel_58638 * y_58659;\n            double x_58661 = 0.5 * x_58660;\n            double abs_res_58662 = fabs(scaledVel_58638);\n            double x_58663 = 1.0 - max_res_58658;\n            double y_58664 = abs_res_58640 * max_res_58658;\n            double y_58665 = x_58663 + y_58664;\n            double x_58666 = abs_res_58662 * y_58665;\n            double x_58667 = rj_58644 * x_58666;\n            double y_58668 = 0.5 * x_58667;\n            double calcflux_res_58669 = x_58661 - y_58668;\n            \n            lifted_0_f_res_58588 = calcflux_res_58669;\n        } else {\n            lifted_0_f_res_58588 = 0.0;\n        }\n        \n        bool cond_58670 = slt64(gtid_57266, y_48310);\n        bool x_58671 = cond_58670 && index_primexp_60571;\n        bool x_58672 = x_58671 && index_primexp_60568;\n        bool x_58673 = x_58672 && index_primexp_60556;\n        bool x_58674 = x_58673 && index_primexp_60553;\n        double lifted_0_f_res_58675;\n        \n        if (x_58674) {\n            double velS_58685 = ((__global\n                                  double *) wtau_mem_60710)[gtid_57264 *\n                                                            (zzdim_48138 *\n                                                             ydim_48137) +\n                                                            gtid_57265 *\n                                                            zzdim_48138 +\n                                                            gtid_57266];\n            bool cond_58686 = gtid_57266 == 0;\n            bool cond_58687 = !cond_58686;\n            double varSM1_58688;\n            \n            if (cond_58687) {\n                int64_t i_58689 = sub64(gtid_57266, 1);\n                bool x_58690 = sle64(0, i_58689);\n                bool y_586",
                   "91 = slt64(i_58689, zzdim_48114);\n                bool bounds_check_58692 = x_58690 && y_58691;\n                bool index_certs_58695;\n                \n                if (!bounds_check_58692) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 44) ==\n                            -1) {\n                            global_failure_args[0] = gtid_57264;\n                            global_failure_args[1] = gtid_57265;\n                            global_failure_args[2] = i_58689;\n                            global_failure_args[3] = xdim_48112;\n                            global_failure_args[4] = ydim_48113;\n                            global_failure_args[5] = zzdim_48114;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double varSM1_t_res_58696 = ((__global\n                                              double *) tketau_mem_60702)[gtid_57264 *\n                                                                          (zzdim_48114 *\n                                                                           ydim_48113) +\n                                                                          gtid_57265 *\n                                                                          zzdim_48114 +\n                                                                          i_58689];\n                \n                varSM1_58688 = varSM1_t_res_58696;\n            } else {\n                varSM1_58688 = 0.0;\n            }\n            \n            double varS_58697 = ((__global\n                                  double *) tketau_mem_60702)[gtid_57264 *\n                                                              (zzdim_48114 *\n                                                               ydim_48113) +\n                                                              gtid_57265 *\n                                                        ",
                   "      zzdim_48114 +\n                                                              gtid_57266];\n            int64_t y_58698 = sub64(zzdim_48114, 2);\n            bool cond_58699 = slt64(gtid_57266, y_58698);\n            double varSP2_58700;\n            \n            if (cond_58699) {\n                int64_t i_58701 = add64(2, gtid_57266);\n                bool x_58702 = sle64(0, i_58701);\n                bool y_58703 = slt64(i_58701, zzdim_48114);\n                bool bounds_check_58704 = x_58702 && y_58703;\n                bool index_certs_58707;\n                \n                if (!bounds_check_58704) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 45) ==\n                            -1) {\n                            global_failure_args[0] = gtid_57264;\n                            global_failure_args[1] = gtid_57265;\n                            global_failure_args[2] = i_58701;\n                            global_failure_args[3] = xdim_48112;\n                            global_failure_args[4] = ydim_48113;\n                            global_failure_args[5] = zzdim_48114;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double varSP2_t_res_58708 = ((__global\n                                              double *) tketau_mem_60702)[gtid_57264 *\n                                                                          (zzdim_48114 *\n                                                                           ydim_48113) +\n                                                                          gtid_57265 *\n                                                                          zzdim_48114 +\n                                                                          i_58701];\n                \n                varSP2_58700 = varSP2_t_res_58708;\n            } else {\n                varSP2_58700 = 0.0;\n            }\n  ",
                   "          \n            int64_t i_58709 = add64(1, gtid_57266);\n            bool x_58710 = sle64(0, i_58709);\n            bool y_58711 = slt64(i_58709, zzdim_48114);\n            bool bounds_check_58712 = x_58710 && y_58711;\n            bool index_certs_58715;\n            \n            if (!bounds_check_58712) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 46) ==\n                        -1) {\n                        global_failure_args[0] = gtid_57264;\n                        global_failure_args[1] = gtid_57265;\n                        global_failure_args[2] = i_58709;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double varSP1_58716 = ((__global\n                                    double *) tketau_mem_60702)[gtid_57264 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_57265 *\n                                                                zzdim_48114 +\n                                                                i_58709];\n            double maskWm1_58717;\n            \n            if (cond_58687) {\n                int64_t i_58718 = sub64(gtid_57266, 1);\n                bool x_58719 = sle64(0, i_58718);\n                bool y_58720 = slt64(i_58718, zzdim_48114);\n                bool bounds_check_58721 = x_58719 && y_58720;\n                bool index_certs_58724;\n                \n                if (!bounds_check_58721) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 47) ==\n                            -1) {\n                            ",
                   "global_failure_args[0] = gtid_57264;\n                            global_failure_args[1] = gtid_57265;\n                            global_failure_args[2] = i_58718;\n                            global_failure_args[3] = xdim_48112;\n                            global_failure_args[4] = ydim_48113;\n                            global_failure_args[5] = zzdim_48114;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double maskWm1_t_res_58725 = ((__global\n                                               double *) maskW_mem_60713)[gtid_57264 *\n                                                                          (zzdim_48147 *\n                                                                           ydim_48146) +\n                                                                          gtid_57265 *\n                                                                          zzdim_48147 +\n                                                                          i_58718];\n                \n                maskWm1_58717 = maskWm1_t_res_58725;\n            } else {\n                maskWm1_58717 = 0.0;\n            }\n            \n            double maskWs_58726 = ((__global\n                                    double *) maskW_mem_60713)[gtid_57264 *\n                                                               (zzdim_48147 *\n                                                                ydim_48146) +\n                                                               gtid_57265 *\n                                                               zzdim_48147 +\n                                                               gtid_57266];\n            double maskWp1_58727 = ((__global\n                                     double *) maskW_mem_60713)[gtid_57264 *\n                                                                (zzdim_48147 *\n                                                    ",
                   "             ydim_48146) +\n                                                                gtid_57265 *\n                                                                zzdim_48147 +\n                                                                i_58709];\n            double maskwp2_58728;\n            \n            if (cond_58699) {\n                int64_t i_58729 = add64(2, gtid_57266);\n                bool x_58730 = sle64(0, i_58729);\n                bool y_58731 = slt64(i_58729, zzdim_48114);\n                bool bounds_check_58732 = x_58730 && y_58731;\n                bool index_certs_58735;\n                \n                if (!bounds_check_58732) {\n                    {\n                        if (atomic_cmpxchg_i32_global(global_failure, -1, 48) ==\n                            -1) {\n                            global_failure_args[0] = gtid_57264;\n                            global_failure_args[1] = gtid_57265;\n                            global_failure_args[2] = i_58729;\n                            global_failure_args[3] = xdim_48112;\n                            global_failure_args[4] = ydim_48113;\n                            global_failure_args[5] = zzdim_48114;\n                            ;\n                        }\n                        return;\n                    }\n                }\n                \n                double maskwp2_t_res_58736 = ((__global\n                                               double *) maskW_mem_60713)[gtid_57264 *\n                                                                          (zzdim_48147 *\n                                                                           ydim_48146) +\n                                                                          gtid_57265 *\n                                                                          zzdim_48147 +\n                                                                          i_58729];\n                \n                maskwp2_58728 = maskwp2_t_res_58736;\n            } else ",
                   "{\n                maskwp2_58728 = 0.0;\n            }\n            \n            double maskWtr_58737 = maskWs_58726 * maskWp1_58727;\n            double maskWtrP1_58738 = maskWp1_58727 * maskwp2_58728;\n            double maskWtrM1_58739 = maskWm1_58717 * maskWs_58726;\n            double dx_58741 = ((__global double *) dzzw_mem_60719)[gtid_57266];\n            double abs_arg_58742 = velS_58685 / dx_58741;\n            double abs_res_58743 = fabs(abs_arg_58742);\n            double x_58744 = varSP2_58700 - varSP1_58716;\n            double rjp_58745 = maskWtrP1_58738 * x_58744;\n            double x_58746 = varSP1_58716 - varS_58697;\n            double rj_58747 = maskWtr_58737 * x_58746;\n            double x_58748 = varS_58697 - varSM1_58688;\n            double rjm_58749 = maskWtrM1_58739 * x_58748;\n            double abs_res_58750 = fabs(rj_58747);\n            bool cond_58751 = abs_res_58750 < 1.0e-20;\n            double divisor_58752;\n            \n            if (cond_58751) {\n                divisor_58752 = 1.0e-20;\n            } else {\n                divisor_58752 = rj_58747;\n            }\n            \n            bool cond_58753 = 0.0 < velS_58685;\n            double cr_58754;\n            \n            if (cond_58753) {\n                double cr_t_res_58755 = rjm_58749 / divisor_58752;\n                \n                cr_58754 = cr_t_res_58755;\n            } else {\n                double cr_f_res_58756 = rjp_58745 / divisor_58752;\n                \n                cr_58754 = cr_f_res_58756;\n            }\n            \n            double min_res_58757 = fmin64(2.0, cr_58754);\n            double min_arg_58758 = 2.0 * cr_58754;\n            double min_res_58759 = fmin64(1.0, min_arg_58758);\n            double max_res_58760 = fmax64(min_res_58757, min_res_58759);\n            double max_res_58761 = fmax64(0.0, max_res_58760);\n            double y_58762 = varS_58697 + varSP1_58716;\n            double x_58763 = velS_58685 * y_58762;\n            double x_58764 = 0.5 * x_58763;\n    ",
                   "        double abs_res_58765 = fabs(velS_58685);\n            double x_58766 = 1.0 - max_res_58761;\n            double y_58767 = abs_res_58743 * max_res_58761;\n            double y_58768 = x_58766 + y_58767;\n            double x_58769 = abs_res_58765 * y_58768;\n            double x_58770 = rj_58747 * x_58769;\n            double y_58771 = 0.5 * x_58770;\n            double calcflux_res_58772 = x_58764 - y_58771;\n            \n            lifted_0_f_res_58675 = calcflux_res_58772;\n        } else {\n            lifted_0_f_res_58675 = 0.0;\n        }\n        ((__global double *) mem_61067)[gtid_57264 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_57265 *\n                                        zzdim_48114 + gtid_57266] =\n            lifted_0_f_res_58675;\n        ((__global double *) mem_61071)[gtid_57264 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_57265 *\n                                        zzdim_48114 + gtid_57266] =\n            lifted_0_f_res_58588;\n        ((__global double *) mem_61075)[gtid_57264 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_57265 *\n                                        zzdim_48114 + gtid_57266] =\n            lifted_0_f_res_58504;\n        ((__global double *) mem_61079)[gtid_57264 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_57265 *\n                                        zzdim_48114 + gtid_57266] =\n            lifted_0_f_res_58462;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_58435\n}\n__kernel void integrate_tkezisegmap_59404(__global int *global_failure,\n                                          int failure_is_an_option, __global\n                                          int64_t *global_failure_args,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                               ",
                   "           int64_t zzdim_48114,\n                                          int64_t ydim_48122,\n                                          int64_t zzdim_48123,\n                                          int64_t ydim_48146,\n                                          int64_t zzdim_48147, int64_t y_48308,\n                                          int64_t y_48309, int64_t y_48310,\n                                          __global\n                                          unsigned char *dtketau_mem_60705,\n                                          __global\n                                          unsigned char *maskW_mem_60713,\n                                          __global unsigned char *dxt_mem_60714,\n                                          __global unsigned char *dyt_mem_60716,\n                                          __global\n                                          unsigned char *dzzw_mem_60719,\n                                          __global\n                                          unsigned char *cost_mem_60720,\n                                          __global unsigned char *mem_61067,\n                                          __global unsigned char *mem_61071,\n                                          __global unsigned char *mem_61075,\n                                          __global unsigned char *mem_61084)\n{\n    #define segmap_group_sizze_59921 (integrate_tkezisegmap_group_sizze_59408)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61264;\n    int32_t local_tid_61265;\n    int64_t group_sizze_61268;\n    int32_t wave_sizze_61267;\n    int32_t group_tid_61266;\n    \n    global_tid_61264 = get_global_id(0);\n    local_tid_61265 = get_local_id(0);\n    group_sizze_61268 = get_local_size(0);\n    wave_sizze_61267 = LOCKSTEP_WIDTH;\n    group_tid_61266 = get_group_id(0);\n    \n    int32_t phys_tid_59404;\n    \n    phys_tid_59404 = global_ti",
                   "d_61264;\n    \n    int64_t gtid_59401;\n    \n    gtid_59401 = squot64(sext_i32_i64(group_tid_61266) *\n                         segmap_group_sizze_59921 +\n                         sext_i32_i64(local_tid_61265), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_59402;\n    \n    gtid_59402 = squot64(sext_i32_i64(group_tid_61266) *\n                         segmap_group_sizze_59921 +\n                         sext_i32_i64(local_tid_61265) -\n                         squot64(sext_i32_i64(group_tid_61266) *\n                                 segmap_group_sizze_59921 +\n                                 sext_i32_i64(local_tid_61265), ydim_48113 *\n                                 zzdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_59403;\n    \n    gtid_59403 = sext_i32_i64(group_tid_61266) * segmap_group_sizze_59921 +\n        sext_i32_i64(local_tid_61265) - squot64(sext_i32_i64(group_tid_61266) *\n                                                segmap_group_sizze_59921 +\n                                                sext_i32_i64(local_tid_61265),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61266) *\n                                             segmap_group_sizze_59921 +\n                                             sext_i32_i64(local_tid_61265) -\n                                             squot64(sext_i32_i64(group_tid_61266) *\n                                                     segmap_group_sizze_59921 +\n                                                     sext_i32_i64(local_tid_61265),\n                                                     ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 * zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_59401, xdim_48112) && slt64(gtid_59402, ydim_48113)) &&\n        slt64(gtid_59403,",
                   " zzdim_48114)) {\n        bool binop_x_60671 = sle64(2, gtid_59402);\n        bool binop_x_60672 = sle64(2, gtid_59401);\n        bool binop_y_60673 = slt64(gtid_59401, y_48308);\n        bool binop_y_60674 = binop_x_60672 && binop_y_60673;\n        bool binop_x_60675 = binop_x_60671 && binop_y_60674;\n        bool binop_y_60676 = slt64(gtid_59402, y_48309);\n        bool index_primexp_60677 = binop_x_60675 && binop_y_60676;\n        double tmp_59928;\n        \n        if (index_primexp_60677) {\n            double x_59941 = ((__global double *) maskW_mem_60713)[gtid_59401 *\n                                                                   (zzdim_48147 *\n                                                                    ydim_48146) +\n                                                                   gtid_59402 *\n                                                                   zzdim_48147 +\n                                                                   gtid_59403];\n            double x_59942 = ((__global double *) mem_61075)[gtid_59401 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114 +\n                                                             gtid_59403];\n            int64_t i_59943 = sub64(gtid_59401, 1);\n            bool x_59944 = sle64(0, i_59943);\n            bool y_59945 = slt64(i_59943, xdim_48112);\n            bool bounds_check_59946 = x_59944 && y_59945;\n            bool index_certs_59949;\n            \n            if (!bounds_check_59946) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 49) ==\n                        -1) {\n                        global_failure_args[0] = i_59943;\n                        global_failure_args[1] = gtid_59402;\n                        global_",
                   "failure_args[2] = gtid_59403;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_59950 = ((__global double *) mem_61075)[i_59943 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114 +\n                                                             gtid_59403];\n            double x_59951 = x_59942 - y_59950;\n            double x_59953 = ((__global double *) cost_mem_60720)[gtid_59402];\n            double y_59955 = ((__global double *) dxt_mem_60714)[gtid_59401];\n            double y_59956 = x_59953 * y_59955;\n            double negate_arg_59957 = x_59951 / y_59956;\n            double x_59958 = 0.0 - negate_arg_59957;\n            double x_59959 = ((__global double *) mem_61071)[gtid_59401 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114 +\n                                                             gtid_59403];\n            int64_t i_59960 = sub64(gtid_59402, 1);\n            bool x_59961 = sle64(0, i_59960);\n            bool y_59962 = slt64(i_59960, ydim_48113);\n            bool bounds_check_59963 = x_59961 && y_59962;\n            bool index_certs_59965;\n            \n            if (!bounds_check_59963) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 50) ==\n             ",
                   "           -1) {\n                        global_failure_args[0] = gtid_59401;\n                        global_failure_args[1] = i_59960;\n                        global_failure_args[2] = gtid_59403;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_59966 = ((__global double *) mem_61071)[gtid_59401 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             i_59960 *\n                                                             zzdim_48114 +\n                                                             gtid_59403];\n            double x_59967 = x_59959 - y_59966;\n            double y_59968 = ((__global double *) dyt_mem_60716)[gtid_59402];\n            double y_59969 = x_59953 * y_59968;\n            double y_59970 = x_59967 / y_59969;\n            double y_59971 = x_59958 - y_59970;\n            double tmp_t_res_59972 = x_59941 * y_59971;\n            \n            tmp_59928 = tmp_t_res_59972;\n        } else {\n            double tmp_f_res_59985 = ((__global\n                                       double *) dtketau_mem_60705)[gtid_59401 *\n                                                                    (zzdim_48123 *\n                                                                     ydim_48122) +\n                                                                    gtid_59402 *\n                                                                    zzdim_48123 +\n                                                                    gtid_59403];\n            \n            tmp_59928 = tmp_f_res_59985;\n        }\n        \n        bool cond_59986 = gtid_59403 == 0;\n",
                   "        double zz0_update_59987;\n        \n        if (cond_59986) {\n            bool y_59994 = slt64(0, zzdim_48114);\n            bool index_certs_59997;\n            \n            if (!y_59994) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 51) ==\n                        -1) {\n                        global_failure_args[0] = gtid_59401;\n                        global_failure_args[1] = gtid_59402;\n                        global_failure_args[2] = 0;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double x_59998 = ((__global double *) mem_61067)[gtid_59401 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114];\n            bool index_certs_59999;\n            \n            if (!y_59994) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 52) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_60000 = ((__global double *) dzzw_mem_60719)[0];\n            double y_60001 = x_59998 / y_60000;\n            double zz0_update_t_res_60002 = tmp_59928 - y_60001;\n            \n            zz0_update_59987 = zz0_update_t_res_60002;\n        } else {\n            zz0_update_59987 = tmp_59928;\n        }\n        \n        bool cond_60003 = sle64(1, gtid_59403);\n        bool",
                   " cond_t_res_60004 = slt64(gtid_59403, y_48310);\n        bool x_60005 = cond_60003 && cond_t_res_60004;\n        double zz_middle_update_60006;\n        \n        if (x_60005) {\n            double x_60019 = ((__global double *) mem_61067)[gtid_59401 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114 +\n                                                             gtid_59403];\n            int64_t i_60020 = sub64(gtid_59403, 1);\n            bool x_60021 = sle64(0, i_60020);\n            bool y_60022 = slt64(i_60020, zzdim_48114);\n            bool bounds_check_60023 = x_60021 && y_60022;\n            bool index_certs_60026;\n            \n            if (!bounds_check_60023) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 53) ==\n                        -1) {\n                        global_failure_args[0] = gtid_59401;\n                        global_failure_args[1] = gtid_59402;\n                        global_failure_args[2] = i_60020;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_60027 = ((__global double *) mem_61067)[gtid_59401 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114 +\n                                                             i_60020];\n            double x_",
                   "60028 = x_60019 - y_60027;\n            double y_60030 = ((__global double *) dzzw_mem_60719)[gtid_59403];\n            double y_60031 = x_60028 / y_60030;\n            double zz_middle_update_t_res_60032 = zz0_update_59987 - y_60031;\n            \n            zz_middle_update_60006 = zz_middle_update_t_res_60032;\n        } else {\n            zz_middle_update_60006 = zz0_update_59987;\n        }\n        \n        bool cond_60033 = gtid_59403 == y_48310;\n        double lifted_0_f_res_60034;\n        \n        if (cond_60033) {\n            double x_60047 = ((__global double *) mem_61067)[gtid_59401 *\n                                                             (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114 +\n                                                             gtid_59403];\n            int64_t i_60048 = sub64(gtid_59403, 1);\n            bool x_60049 = sle64(0, i_60048);\n            bool y_60050 = slt64(i_60048, zzdim_48114);\n            bool bounds_check_60051 = x_60049 && y_60050;\n            bool index_certs_60054;\n            \n            if (!bounds_check_60051) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 54) ==\n                        -1) {\n                        global_failure_args[0] = gtid_59401;\n                        global_failure_args[1] = gtid_59402;\n                        global_failure_args[2] = i_60048;\n                        global_failure_args[3] = xdim_48112;\n                        global_failure_args[4] = ydim_48113;\n                        global_failure_args[5] = zzdim_48114;\n                        ;\n                    }\n                    return;\n                }\n            }\n            \n            double y_60055 = ((__global double *) mem_61067)[gtid_59401 *\n                                           ",
                   "                  (zzdim_48114 *\n                                                              ydim_48113) +\n                                                             gtid_59402 *\n                                                             zzdim_48114 +\n                                                             i_60048];\n            double x_60056 = x_60047 - y_60055;\n            double y_60058 = ((__global double *) dzzw_mem_60719)[gtid_59403];\n            double y_60059 = 0.5 * y_60058;\n            double y_60060 = x_60056 / y_60059;\n            double lifted_0_f_res_t_res_60061 = zz_middle_update_60006 -\n                   y_60060;\n            \n            lifted_0_f_res_60034 = lifted_0_f_res_t_res_60061;\n        } else {\n            lifted_0_f_res_60034 = zz_middle_update_60006;\n        }\n        ((__global double *) mem_61084)[gtid_59401 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_59402 *\n                                        zzdim_48114 + gtid_59403] =\n            lifted_0_f_res_60034;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_59921\n}\n__kernel void integrate_tkezisegmap_60201(__global int *global_failure,\n                                          int64_t xdim_48112,\n                                          int64_t ydim_48113,\n                                          int64_t zzdim_48114,\n                                          int64_t ydim_48128,\n                                          int64_t zzdim_48129, __global\n                                          unsigned char *dtketaum1_mem_60707,\n                                          __global unsigned char *mem_61079,\n                                          __global unsigned char *mem_61084,\n                                          __global unsigned char *mem_61089)\n{\n    #define segmap_group_sizze_60352 (integrate_tkezisegmap_group_sizze_60205)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int bloc",
                   "k_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_61269;\n    int32_t local_tid_61270;\n    int64_t group_sizze_61273;\n    int32_t wave_sizze_61272;\n    int32_t group_tid_61271;\n    \n    global_tid_61269 = get_global_id(0);\n    local_tid_61270 = get_local_id(0);\n    group_sizze_61273 = get_local_size(0);\n    wave_sizze_61272 = LOCKSTEP_WIDTH;\n    group_tid_61271 = get_group_id(0);\n    \n    int32_t phys_tid_60201;\n    \n    phys_tid_60201 = global_tid_61269;\n    \n    int64_t gtid_60198;\n    \n    gtid_60198 = squot64(sext_i32_i64(group_tid_61271) *\n                         segmap_group_sizze_60352 +\n                         sext_i32_i64(local_tid_61270), ydim_48113 *\n                         zzdim_48114);\n    \n    int64_t gtid_60199;\n    \n    gtid_60199 = squot64(sext_i32_i64(group_tid_61271) *\n                         segmap_group_sizze_60352 +\n                         sext_i32_i64(local_tid_61270) -\n                         squot64(sext_i32_i64(group_tid_61271) *\n                                 segmap_group_sizze_60352 +\n                                 sext_i32_i64(local_tid_61270), ydim_48113 *\n                                 zzdim_48114) * (ydim_48113 * zzdim_48114),\n                         zzdim_48114);\n    \n    int64_t gtid_60200;\n    \n    gtid_60200 = sext_i32_i64(group_tid_61271) * segmap_group_sizze_60352 +\n        sext_i32_i64(local_tid_61270) - squot64(sext_i32_i64(group_tid_61271) *\n                                                segmap_group_sizze_60352 +\n                                                sext_i32_i64(local_tid_61270),\n                                                ydim_48113 * zzdim_48114) *\n        (ydim_48113 * zzdim_48114) - squot64(sext_i32_i64(group_tid_61271) *\n                                             segmap_group_sizze_60352 +\n                                             sext_i32_i64(local_tid_61270) -\n                                             squot64(sext_i32_i64(group_tid_61271) *\n ",
                   "                                                    segmap_group_sizze_60352 +\n                                                     sext_i32_i64(local_tid_61270),\n                                                     ydim_48113 * zzdim_48114) *\n                                             (ydim_48113 * zzdim_48114),\n                                             zzdim_48114) * zzdim_48114;\n    if ((slt64(gtid_60198, xdim_48112) && slt64(gtid_60199, ydim_48113)) &&\n        slt64(gtid_60200, zzdim_48114)) {\n        double x_60366 = ((__global double *) mem_61079)[gtid_60198 *\n                                                         (zzdim_48114 *\n                                                          ydim_48113) +\n                                                         gtid_60199 *\n                                                         zzdim_48114 +\n                                                         gtid_60200];\n        double y_60367 = ((__global double *) mem_61084)[gtid_60198 *\n                                                         (zzdim_48114 *\n                                                          ydim_48113) +\n                                                         gtid_60199 *\n                                                         zzdim_48114 +\n                                                         gtid_60200];\n        double x_60368 = 1.6 * y_60367;\n        double y_60369 = ((__global double *) dtketaum1_mem_60707)[gtid_60198 *\n                                                                   (zzdim_48129 *\n                                                                    ydim_48128) +\n                                                                   gtid_60199 *\n                                                                   zzdim_48129 +\n                                                                   gtid_60200];\n        double y_60370 = 0.6 * y_60369;\n        double y_60371 = x_60368 - y_60370;\n        double lifted_0_f_res_603",
                   "72 = x_60366 + y_60371;\n        \n        ((__global double *) mem_61089)[gtid_60198 * (zzdim_48114 *\n                                                      ydim_48113) + gtid_60199 *\n                                        zzdim_48114 + gtid_60200] =\n            lifted_0_f_res_60372;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_60352\n}\n__kernel void integrate_tkezisegmap_intragroup_50310(__global\n                                                     int *global_failure,\n                                                     int failure_is_an_option,\n                                                     __global\n                                                     int64_t *global_failure_args,\n                                                     __local volatile\n                                                     int64_t *mem_60818_backing_aligned_0,\n                                                     __local volatile\n                                                     int64_t *mem_60829_backing_aligned_1,\n                                                     __local volatile\n                                                     int64_t *mem_60806_backing_aligned_2,\n                                                     __local volatile\n                                                     int64_t *mem_60784_backing_aligned_3,\n                                                     __local volatile\n                                                     int64_t *mem_60779_backing_aligned_4,\n                                                     __local volatile\n                                                     int64_t *mem_60803_backing_aligned_5,\n                                                     __local volatile\n                                                     int64_t *mem_60800_backing_aligned_6,\n                                                     __local volatile\n                                                     int64_t *mem_60760_backing_aligned",
                   "_7,\n                                                     __local volatile\n                                                     int64_t *mem_60757_backing_aligned_8,\n                                                     __local volatile\n                                                     int64_t *mem_60753_backing_aligned_9,\n                                                     __local volatile\n                                                     int64_t *mem_60750_backing_aligned_10,\n                                                     __local volatile\n                                                     int64_t *mem_60747_backing_aligned_11,\n                                                     __local volatile\n                                                     int64_t *mem_60744_backing_aligned_12,\n                                                     int64_t xdim_48112,\n                                                     int64_t ydim_48113,\n                                                     int64_t zzdim_48114,\n                                                     int64_t ydim_48157,\n                                                     int64_t ydim_48162,\n                                                     int64_t zzdim_48163,\n                                                     int64_t ydim_48165,\n                                                     int64_t zzdim_48166,\n                                                     int64_t ydim_48168,\n                                                     int64_t y_48308,\n                                                     int64_t y_48309,\n                                                     int64_t y_48310,\n                                                     int64_t m_48377,\n                                                     int64_t computed_group_sizze_49958,\n                                                     __global\n                                                     unsigned char *tketau_mem_60702,\n           ",
                   "                                          __global\n                                                     unsigned char *dzzw_mem_60719,\n                                                     __global\n                                                     unsigned char *kbot_mem_60722,\n                                                     __global\n                                                     unsigned char *mxl_mem_60724,\n                                                     __global\n                                                     unsigned char *forc_mem_60725,\n                                                     __global\n                                                     unsigned char *forc_tke_surface_mem_60726,\n                                                     __global\n                                                     unsigned char *mem_60731,\n                                                     __global\n                                                     unsigned char *mem_60735,\n                                                     __global\n                                                     unsigned char *mem_60833)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_60818_backing_12 = (__local volatile\n                                                            char *) mem_60818_backing_aligned_0;\n    __local volatile char *restrict mem_60829_backing_11 = (__local volatile\n                                                            char *) mem_60829_backing_aligned_1;\n    __local volatile char *restrict mem_60806_backing_10 = (__local volatile\n                                                            char *) mem_60806_backing_aligned_2;\n    __local volatile char *restrict mem_60784_backing_9 = (__local volatile\n                                                           char *) mem_60784_backing_aligned_3;\n    __local volatile char *restrict mem_60779_backing_8 = (__",
                   "local volatile\n                                                           char *) mem_60779_backing_aligned_4;\n    __local volatile char *restrict mem_60803_backing_7 = (__local volatile\n                                                           char *) mem_60803_backing_aligned_5;\n    __local volatile char *restrict mem_60800_backing_6 = (__local volatile\n                                                           char *) mem_60800_backing_aligned_6;\n    __local volatile char *restrict mem_60760_backing_5 = (__local volatile\n                                                           char *) mem_60760_backing_aligned_7;\n    __local volatile char *restrict mem_60757_backing_4 = (__local volatile\n                                                           char *) mem_60757_backing_aligned_8;\n    __local volatile char *restrict mem_60753_backing_3 = (__local volatile\n                                                           char *) mem_60753_backing_aligned_9;\n    __local volatile char *restrict mem_60750_backing_2 = (__local volatile\n                                                           char *) mem_60750_backing_aligned_10;\n    __local volatile char *restrict mem_60747_backing_1 = (__local volatile\n                                                           char *) mem_60747_backing_aligned_11;\n    __local volatile char *restrict mem_60744_backing_0 = (__local volatile\n                                                           char *) mem_60744_backing_aligned_12;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_61166;\n    int32_t local_tid_61167;\n    int64_t group_sizze_61170;\n    int32_t wave_sizze_61169;\n    int32_t group_tid_61168;\n    \n    global_tid_61166 = get_global_id(0);\n    local_tid_61167 = get_local_id(0);\n    group_sizze_61170 = get_local_",
                   "size(0);\n    wave_sizze_61169 = LOCKSTEP_WIDTH;\n    group_tid_61168 = get_group_id(0);\n    \n    int32_t phys_tid_50310;\n    \n    phys_tid_50310 = group_tid_61168;\n    \n    int32_t ltid_pre_61171;\n    \n    ltid_pre_61171 = local_tid_61167;\n    \n    int32_t ltid_pre_61172;\n    \n    ltid_pre_61172 = local_tid_61167;\n    \n    int32_t ltid_pre_61173;\n    \n    ltid_pre_61173 = local_tid_61167 - local_tid_61167;\n    \n    int32_t ltid_pre_61174;\n    \n    ltid_pre_61174 = squot32(local_tid_61167, sext_i64_i32(zzdim_48114));\n    \n    int32_t ltid_pre_61175;\n    \n    ltid_pre_61175 = local_tid_61167 - squot32(local_tid_61167,\n                                               sext_i64_i32(zzdim_48114)) *\n        sext_i64_i32(zzdim_48114);\n    \n    int64_t gtid_49956;\n    \n    gtid_49956 = sext_i32_i64(group_tid_61168);\n    \n    bool cond_50927;\n    \n    cond_50927 = sle64(2, gtid_49956);\n    \n    bool cond_t_res_50928 = slt64(gtid_49956, y_48308);\n    bool x_50929 = cond_50927 && cond_t_res_50928;\n    __local char *mem_60744;\n    \n    mem_60744 = (__local char *) mem_60744_backing_0;\n    \n    __local char *mem_60747;\n    \n    mem_60747 = (__local char *) mem_60747_backing_1;\n    \n    __local char *mem_60750;\n    \n    mem_60750 = (__local char *) mem_60750_backing_2;\n    \n    __local char *mem_60753;\n    \n    mem_60753 = (__local char *) mem_60753_backing_3;\n    \n    int64_t gtid_50073 = sext_i32_i64(ltid_pre_61174);\n    int64_t gtid_50074 = sext_i32_i64(ltid_pre_61175);\n    int32_t phys_tid_50075 = local_tid_61167;\n    \n    if (slt64(gtid_50073, ydim_48113) && slt64(gtid_50074, zzdim_48114)) {\n        bool binop_y_60412 = sle64(2, gtid_50073);\n        bool binop_x_60413 = x_50929 && binop_y_60412;\n        bool binop_y_60416 = slt64(gtid_50073, y_48309);\n        bool index_primexp_60417 = binop_x_60413 && binop_y_60416;\n        double lifted_0_f_res_50943;\n        \n        if (index_primexp_60417) {\n            int32_t x_50952 = ((__global int32_t *) kbot_mem_60722)[gtid_49956 *\n  ",
                   "                                                                  ydim_48157 +\n                                                                    gtid_50073];\n            int32_t ks_val_50953 = sub32(x_50952, 1);\n            bool land_mask_50954 = sle32(0, ks_val_50953);\n            int32_t i64_res_50955 = sext_i64_i32(gtid_50074);\n            bool edge_mask_t_res_50956 = i64_res_50955 == ks_val_50953;\n            bool x_50957 = land_mask_50954 && edge_mask_t_res_50956;\n            bool water_mask_t_res_50958 = sle32(ks_val_50953, i64_res_50955);\n            bool x_50959 = land_mask_50954 && water_mask_t_res_50958;\n            bool cond_f_res_50960 = !x_50959;\n            bool x_50961 = !x_50957;\n            bool y_50962 = cond_f_res_50960 && x_50961;\n            bool cond_50963 = x_50957 || y_50962;\n            double lifted_0_f_res_t_res_50964;\n            \n            if (cond_50963) {\n                lifted_0_f_res_t_res_50964 = 0.0;\n            } else {\n                bool cond_50965 = slt64(0, gtid_50074);\n                int64_t y_50966 = sub64(zzdim_48114, 1);\n                bool cond_t_res_50967 = slt64(gtid_50074, y_50966);\n                bool x_50968 = cond_50965 && cond_t_res_50967;\n                double lifted_0_f_res_t_res_f_res_50969;\n                \n                if (x_50968) {\n                    int64_t i_50970 = sub64(gtid_50074, 1);\n                    bool x_50971 = sle64(0, i_50970);\n                    bool y_50972 = slt64(i_50970, zzdim_48114);\n                    bool bounds_check_50973 = x_50971 && y_50972;\n                    bool index_certs_50976;\n                    \n                    if (!bounds_check_50973) {\n                        {\n                            if (atomic_cmpxchg_i32_global(global_failure, -1,\n                                                          2) == -1) {\n                                global_failure_args[0] = gtid_49956;\n                                global_failure_args[1] = gtid_50073;\n         ",
                   "                       global_failure_args[2] = i_50970;\n                                global_failure_args[3] = xdim_48112;\n                                global_failure_args[4] = ydim_48113;\n                                global_failure_args[5] = zzdim_48114;\n                                ;\n                            }\n                            local_failure = true;\n                            goto error_0;\n                        }\n                    }\n                    \n                    double x_50977 = ((__global\n                                       double *) mem_60731)[gtid_49956 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_50073 *\n                                                            zzdim_48114 +\n                                                            i_50970];\n                    double y_50982 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_50074];\n                    double negate_arg_50983 = x_50977 / y_50982;\n                    double lifted_0_f_res_t_res_f_res_t_res_50984 = 0.0 -\n                           negate_arg_50983;\n                    \n                    lifted_0_f_res_t_res_f_res_50969 =\n                        lifted_0_f_res_t_res_f_res_t_res_50984;\n                } else {\n                    bool cond_50985 = gtid_50074 == y_50966;\n                    double lifted_0_f_res_t_res_f_res_f_res_50986;\n                    \n                    if (cond_50985) {\n                        int64_t i_50987 = sub64(gtid_50074, 1);\n                        bool x_50988 = sle64(0, i_50987);\n                        bool y_50989 = slt64(i_50987, zzdim_48114);\n                        bool bounds_check_50990 = x_50988 && y_50989;\n                        bool index_certs_50993;\n                        \n                        if ",
                   "(!bounds_check_50990) {\n                            {\n                                if (atomic_cmpxchg_i32_global(global_failure,\n                                                              -1, 3) == -1) {\n                                    global_failure_args[0] = gtid_49956;\n                                    global_failure_args[1] = gtid_50073;\n                                    global_failure_args[2] = i_50987;\n                                    global_failure_args[3] = xdim_48112;\n                                    global_failure_args[4] = ydim_48113;\n                                    global_failure_args[5] = zzdim_48114;\n                                    ;\n                                }\n                                local_failure = true;\n                                goto error_0;\n                            }\n                        }\n                        \n                        double x_50994 = ((__global\n                                           double *) mem_60731)[gtid_49956 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_50073 *\n                                                                zzdim_48114 +\n                                                                i_50987];\n                        double y_50999 = ((__global\n                                           double *) dzzw_mem_60719)[gtid_50074];\n                        double y_51000 = 0.5 * y_50999;\n                        double negate_arg_51001 = x_50994 / y_51000;\n                        double lifted_0_f_res_t_res_f_res_f_res_t_res_51002 =\n                               0.0 - negate_arg_51001;\n                        \n                        lifted_0_f_res_t_res_f_res_f_res_50986 =\n                            lifted_0_f_res_t_res_f_res_f_res_t_res_51002;\n                    } else {\n",
                   "                        lifted_0_f_res_t_res_f_res_f_res_50986 = 0.0;\n                    }\n                    lifted_0_f_res_t_res_f_res_50969 =\n                        lifted_0_f_res_t_res_f_res_f_res_50986;\n                }\n                lifted_0_f_res_t_res_50964 = lifted_0_f_res_t_res_f_res_50969;\n            }\n            lifted_0_f_res_50943 = lifted_0_f_res_t_res_50964;\n        } else {\n            lifted_0_f_res_50943 = 0.0;\n        }\n        \n        double lifted_0_f_res_51003;\n        \n        if (index_primexp_60417) {\n            int32_t x_51012 = ((__global int32_t *) kbot_mem_60722)[gtid_49956 *\n                                                                    ydim_48157 +\n                                                                    gtid_50073];\n            int32_t ks_val_51013 = sub32(x_51012, 1);\n            bool land_mask_51014 = sle32(0, ks_val_51013);\n            int32_t i64_res_51015 = sext_i64_i32(gtid_50074);\n            bool edge_mask_t_res_51016 = i64_res_51015 == ks_val_51013;\n            bool x_51017 = land_mask_51014 && edge_mask_t_res_51016;\n            bool water_mask_t_res_51018 = sle32(ks_val_51013, i64_res_51015);\n            bool x_51019 = land_mask_51014 && water_mask_t_res_51018;\n            bool cond_51020 = !x_51019;\n            double lifted_0_f_res_t_res_51021;\n            \n            if (cond_51020) {\n                lifted_0_f_res_t_res_51021 = 1.0;\n            } else {\n                double lifted_0_f_res_t_res_f_res_51022;\n                \n                if (x_51017) {\n                    double x_51029 = ((__global\n                                       double *) mem_60731)[gtid_49956 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_50073 *\n                                                            zzdim_48114 +\n                           ",
                   "                                 gtid_50074];\n                    double y_51031 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_50074];\n                    double y_51032 = x_51029 / y_51031;\n                    double x_51033 = 1.0 + y_51032;\n                    double y_51034 = ((__global\n                                       double *) mxl_mem_60724)[gtid_49956 *\n                                                                (zzdim_48163 *\n                                                                 ydim_48162) +\n                                                                gtid_50073 *\n                                                                zzdim_48163 +\n                                                                gtid_50074];\n                    double x_51035 = 0.7 / y_51034;\n                    double y_51036 = ((__global\n                                       double *) mem_60735)[gtid_49956 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_50073 *\n                                                            zzdim_48114 +\n                                                            gtid_50074];\n                    double y_51037 = x_51035 * y_51036;\n                    double lifted_0_f_res_t_res_f_res_t_res_51038 = x_51033 +\n                           y_51037;\n                    \n                    lifted_0_f_res_t_res_f_res_51022 =\n                        lifted_0_f_res_t_res_f_res_t_res_51038;\n                } else {\n                    bool cond_51039 = slt64(0, gtid_50074);\n                    int64_t y_51040 = sub64(zzdim_48114, 1);\n                    bool cond_t_res_51041 = slt64(gtid_50074, y_51040);\n                    bool x_51042 = cond_51039 && cond_t_res_51041;\n                    double lifted_0_f_res_t_res_f_res_f_res_51043;\n",
                   "                    \n                    if (x_51042) {\n                        double x_51050 = ((__global\n                                           double *) mem_60731)[gtid_49956 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_50073 *\n                                                                zzdim_48114 +\n                                                                gtid_50074];\n                        int64_t i_51051 = sub64(gtid_50074, 1);\n                        bool x_51052 = sle64(0, i_51051);\n                        bool y_51053 = slt64(i_51051, zzdim_48114);\n                        bool bounds_check_51054 = x_51052 && y_51053;\n                        bool index_certs_51057;\n                        \n                        if (!bounds_check_51054) {\n                            {\n                                if (atomic_cmpxchg_i32_global(global_failure,\n                                                              -1, 4) == -1) {\n                                    global_failure_args[0] = gtid_49956;\n                                    global_failure_args[1] = gtid_50073;\n                                    global_failure_args[2] = i_51051;\n                                    global_failure_args[3] = xdim_48112;\n                                    global_failure_args[4] = ydim_48113;\n                                    global_failure_args[5] = zzdim_48114;\n                                    ;\n                                }\n                                local_failure = true;\n                                goto error_0;\n                            }\n                        }\n                        \n                        double y_51058 = ((__global\n                                           double *) mem_60731)[gtid_49956 *\n                                     ",
                   "                           (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_50073 *\n                                                                zzdim_48114 +\n                                                                i_51051];\n                        double x_51059 = x_51050 + y_51058;\n                        double y_51061 = ((__global\n                                           double *) dzzw_mem_60719)[gtid_50074];\n                        double y_51062 = x_51059 / y_51061;\n                        double x_51063 = 1.0 + y_51062;\n                        double y_51064 = ((__global\n                                           double *) mem_60735)[gtid_49956 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_50073 *\n                                                                zzdim_48114 +\n                                                                gtid_50074];\n                        double x_51065 = 0.7 * y_51064;\n                        double y_51066 = ((__global\n                                           double *) mxl_mem_60724)[gtid_49956 *\n                                                                    (zzdim_48163 *\n                                                                     ydim_48162) +\n                                                                    gtid_50073 *\n                                                                    zzdim_48163 +\n                                                                    gtid_50074];\n                        double y_51067 = x_51065 / y_51066;\n                        double lifted_0_f_res_t_res_f_res_f_res_t_res_51068 =\n                               x_51063 + y_51067;\n                        \n         ",
                   "               lifted_0_f_res_t_res_f_res_f_res_51043 =\n                            lifted_0_f_res_t_res_f_res_f_res_t_res_51068;\n                    } else {\n                        bool cond_51069 = gtid_50074 == y_51040;\n                        double lifted_0_f_res_t_res_f_res_f_res_f_res_51070;\n                        \n                        if (cond_51069) {\n                            int64_t i_51071 = sub64(gtid_50074, 1);\n                            bool x_51072 = sle64(0, i_51071);\n                            bool y_51073 = slt64(i_51071, zzdim_48114);\n                            bool bounds_check_51074 = x_51072 && y_51073;\n                            bool index_certs_51077;\n                            \n                            if (!bounds_check_51074) {\n                                {\n                                    if (atomic_cmpxchg_i32_global(global_failure,\n                                                                  -1, 5) ==\n                                        -1) {\n                                        global_failure_args[0] = gtid_49956;\n                                        global_failure_args[1] = gtid_50073;\n                                        global_failure_args[2] = i_51071;\n                                        global_failure_args[3] = xdim_48112;\n                                        global_failure_args[4] = ydim_48113;\n                                        global_failure_args[5] = zzdim_48114;\n                                        ;\n                                    }\n                                    local_failure = true;\n                                    goto error_0;\n                                }\n                            }\n                            \n                            double x_51078 = ((__global\n                                               double *) mem_60731)[gtid_49956 *\n                                                                    (zzdim_48114 *\n                     ",
                   "                                                ydim_48113) +\n                                                                    gtid_50073 *\n                                                                    zzdim_48114 +\n                                                                    i_51071];\n                            double y_51083 = ((__global\n                                               double *) dzzw_mem_60719)[gtid_50074];\n                            double y_51084 = 0.5 * y_51083;\n                            double y_51085 = x_51078 / y_51084;\n                            double x_51086 = 1.0 + y_51085;\n                            double y_51090 = ((__global\n                                               double *) mxl_mem_60724)[gtid_49956 *\n                                                                        (zzdim_48163 *\n                                                                         ydim_48162) +\n                                                                        gtid_50073 *\n                                                                        zzdim_48163 +\n                                                                        gtid_50074];\n                            double x_51091 = 0.7 / y_51090;\n                            double y_51092 = ((__global\n                                               double *) mem_60735)[gtid_49956 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    gtid_50073 *\n                                                                    zzdim_48114 +\n                                                                    gtid_50074];\n                            double y_51093 = x_51091 * y_51092;\n                            double\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_t_res_51094 =\n             ",
                   "               x_51086 + y_51093;\n                            \n                            lifted_0_f_res_t_res_f_res_f_res_f_res_51070 =\n                                lifted_0_f_res_t_res_f_res_f_res_f_res_t_res_51094;\n                        } else {\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_51070 = 0.0;\n                        }\n                        lifted_0_f_res_t_res_f_res_f_res_51043 =\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_51070;\n                    }\n                    lifted_0_f_res_t_res_f_res_51022 =\n                        lifted_0_f_res_t_res_f_res_f_res_51043;\n                }\n                lifted_0_f_res_t_res_51021 = lifted_0_f_res_t_res_f_res_51022;\n            }\n            lifted_0_f_res_51003 = lifted_0_f_res_t_res_51021;\n        } else {\n            lifted_0_f_res_51003 = 0.0;\n        }\n        \n        bool cond_t_res_51095 = slt64(gtid_50074, y_48310);\n        bool x_51096 = cond_t_res_51095 && index_primexp_60417;\n        double lifted_0_f_res_51097;\n        \n        if (x_51096) {\n            int32_t x_51106 = ((__global int32_t *) kbot_mem_60722)[gtid_49956 *\n                                                                    ydim_48157 +\n                                                                    gtid_50073];\n            int32_t ks_val_51107 = sub32(x_51106, 1);\n            bool land_mask_51108 = sle32(0, ks_val_51107);\n            int32_t i64_res_51109 = sext_i64_i32(gtid_50074);\n            bool water_mask_t_res_51110 = sle32(ks_val_51107, i64_res_51109);\n            bool x_51111 = land_mask_51108 && water_mask_t_res_51110;\n            bool cond_51112 = !x_51111;\n            double lifted_0_f_res_t_res_51113;\n            \n            if (cond_51112) {\n                lifted_0_f_res_t_res_51113 = 0.0;\n            } else {\n                double x_51120 = ((__global double *) mem_60731)[gtid_49956 *\n                                                                 (zzdim",
                   "_48114 *\n                                                                  ydim_48113) +\n                                                                 gtid_50073 *\n                                                                 zzdim_48114 +\n                                                                 gtid_50074];\n                double y_51122 = ((__global\n                                   double *) dzzw_mem_60719)[gtid_50074];\n                double negate_arg_51123 = x_51120 / y_51122;\n                double lifted_0_f_res_t_res_f_res_51124 = 0.0 -\n                       negate_arg_51123;\n                \n                lifted_0_f_res_t_res_51113 = lifted_0_f_res_t_res_f_res_51124;\n            }\n            lifted_0_f_res_51097 = lifted_0_f_res_t_res_51113;\n        } else {\n            lifted_0_f_res_51097 = 0.0;\n        }\n        \n        double lifted_0_f_res_51125;\n        \n        if (index_primexp_60417) {\n            int32_t x_51134 = ((__global int32_t *) kbot_mem_60722)[gtid_49956 *\n                                                                    ydim_48157 +\n                                                                    gtid_50073];\n            int32_t ks_val_51135 = sub32(x_51134, 1);\n            bool land_mask_51136 = sle32(0, ks_val_51135);\n            int32_t i64_res_51137 = sext_i64_i32(gtid_50074);\n            bool water_mask_t_res_51138 = sle32(ks_val_51135, i64_res_51137);\n            bool x_51139 = land_mask_51136 && water_mask_t_res_51138;\n            bool cond_51140 = !x_51139;\n            double lifted_0_f_res_t_res_51141;\n            \n            if (cond_51140) {\n                lifted_0_f_res_t_res_51141 = 0.0;\n            } else {\n                double x_51148 = ((__global\n                                   double *) tketau_mem_60702)[gtid_49956 *\n                                                               (zzdim_48114 *\n                                                                ydim_48113) +\n                   ",
                   "                                            gtid_50073 *\n                                                               zzdim_48114 +\n                                                               gtid_50074];\n                double y_51149 = ((__global\n                                   double *) forc_mem_60725)[gtid_49956 *\n                                                             (zzdim_48166 *\n                                                              ydim_48165) +\n                                                             gtid_50073 *\n                                                             zzdim_48166 +\n                                                             gtid_50074];\n                double tmp_51150 = x_51148 + y_51149;\n                int64_t y_51151 = sub64(zzdim_48114, 1);\n                bool cond_51152 = gtid_50074 == y_51151;\n                double lifted_0_f_res_t_res_f_res_51153;\n                \n                if (cond_51152) {\n                    double y_51154 = ((__global\n                                       double *) forc_tke_surface_mem_60726)[gtid_49956 *\n                                                                             ydim_48168 +\n                                                                             gtid_50073];\n                    double y_51156 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_50074];\n                    double y_51157 = 0.5 * y_51156;\n                    double y_51158 = y_51154 / y_51157;\n                    double lifted_0_f_res_t_res_f_res_t_res_51159 = tmp_51150 +\n                           y_51158;\n                    \n                    lifted_0_f_res_t_res_f_res_51153 =\n                        lifted_0_f_res_t_res_f_res_t_res_51159;\n                } else {\n                    lifted_0_f_res_t_res_f_res_51153 = tmp_51150;\n                }\n                lifted_0_f_res_t_res_51141 = lifted_0_f_res_t_res_f_res_51153;\n            }\n           ",
                   " lifted_0_f_res_51125 = lifted_0_f_res_t_res_51141;\n        } else {\n            lifted_0_f_res_51125 = 0.0;\n        }\n        ((__local double *) mem_60744)[gtid_50073 * zzdim_48114 + gtid_50074] =\n            lifted_0_f_res_51125;\n        ((__local double *) mem_60747)[gtid_50073 * zzdim_48114 + gtid_50074] =\n            lifted_0_f_res_51097;\n        ((__local double *) mem_60750)[gtid_50073 * zzdim_48114 + gtid_50074] =\n            lifted_0_f_res_51003;\n        ((__local double *) mem_60753)[gtid_50073 * zzdim_48114 + gtid_50074] =\n            lifted_0_f_res_50943;\n    }\n    \n  error_0:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_60757;\n    \n    mem_60757 = (__local char *) mem_60757_backing_4;\n    \n    __local char *mem_60760;\n    \n    mem_60760 = (__local char *) mem_60760_backing_5;\n    \n    int64_t gtid_50052 = sext_i32_i64(ltid_pre_61174);\n    int64_t gtid_50053 = sext_i32_i64(ltid_pre_61175);\n    int32_t phys_tid_50054 = local_tid_61167;\n    \n    if (slt64(gtid_50052, ydim_48113) && slt64(gtid_50053, zzdim_48114)) {\n        bool cond_51166 = gtid_50053 == 0;\n        double lifted_0_f_res_51167;\n        \n        if (cond_51166) {\n            bool y_51168 = slt64(0, zzdim_48114);\n            bool index_certs_51169;\n            \n            if (!y_51168) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 6) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_51170 = ((__local double *) mem_60747)[gtid_50052 *\n                                                            zzdim_48114];\n            double y_51171 = ((__local double *) mem_60750)[gtid_50052 *\n",
                   "                                                            zzdim_48114];\n            double lifted_0_f_res_t_res_51172 = x_51170 / y_51171;\n            \n            lifted_0_f_res_51167 = lifted_0_f_res_t_res_51172;\n        } else {\n            lifted_0_f_res_51167 = 0.0;\n        }\n        \n        double lifted_0_f_res_51173;\n        \n        if (cond_51166) {\n            bool y_51174 = slt64(0, zzdim_48114);\n            bool index_certs_51175;\n            \n            if (!y_51174) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 7) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_51176 = ((__local double *) mem_60744)[gtid_50052 *\n                                                            zzdim_48114];\n            double y_51177 = ((__local double *) mem_60750)[gtid_50052 *\n                                                            zzdim_48114];\n            double lifted_0_f_res_t_res_51178 = x_51176 / y_51177;\n            \n            lifted_0_f_res_51173 = lifted_0_f_res_t_res_51178;\n        } else {\n            lifted_0_f_res_51173 = 0.0;\n        }\n        ((__local double *) mem_60757)[gtid_50052 * zzdim_48114 + gtid_50053] =\n            lifted_0_f_res_51173;\n        ((__local double *) mem_60760)[gtid_50052 * zzdim_48114 + gtid_50053] =\n            lifted_0_f_res_51167;\n    }\n    \n  error_1:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_60800;\n    \n    mem_60800 = (__local char *) mem_60800_backing_6;\n    \n    __local char *mem_60803;\n    \n    mem_60803 = (__local char *) mem_60803_backing_7;\n    \n    int64_t gtid_50017 = sext_i32_i64(ltid_pre_61",
                   "171);\n    int32_t phys_tid_50018 = local_tid_61167;\n    \n    if (slt64(gtid_50017, ydim_48113)) {\n        __local char *mem_60779;\n        \n        mem_60779 = (__local char *) mem_60779_backing_8;\n        for (int64_t i_61176 = 0; i_61176 < zzdim_48114; i_61176++) {\n            ((__local double *) mem_60779)[i_61176] = ((__local\n                                                        double *) mem_60760)[gtid_50017 *\n                                                                             zzdim_48114 +\n                                                                             i_61176];\n        }\n        \n        __local char *mem_60784;\n        \n        mem_60784 = (__local char *) mem_60784_backing_9;\n        for (int64_t i_61177 = 0; i_61177 < zzdim_48114; i_61177++) {\n            ((__local double *) mem_60784)[i_61177] = ((__local\n                                                        double *) mem_60757)[gtid_50017 *\n                                                                             zzdim_48114 +\n                                                                             i_61177];\n        }\n        for (int64_t i_51189 = 0; i_51189 < y_48310; i_51189++) {\n            int64_t index_primexp_51192 = add64(1, i_51189);\n            bool x_51193 = sle64(0, index_primexp_51192);\n            bool y_51194 = slt64(index_primexp_51192, zzdim_48114);\n            bool bounds_check_51195 = x_51193 && y_51194;\n            bool index_certs_51196;\n            \n            if (!bounds_check_51195) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 8) ==\n                        -1) {\n                        global_failure_args[0] = index_primexp_51192;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_2;\n                }\n            }\n            \n            double x_51197 = ((__local double",
                   " *) mem_60750)[gtid_50017 *\n                                                            zzdim_48114 +\n                                                            index_primexp_51192];\n            double x_51198 = ((__local double *) mem_60753)[gtid_50017 *\n                                                            zzdim_48114 +\n                                                            index_primexp_51192];\n            bool y_51199 = slt64(i_51189, zzdim_48114);\n            bool index_certs_51200;\n            \n            if (!y_51199) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 9) ==\n                        -1) {\n                        global_failure_args[0] = i_51189;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_2;\n                }\n            }\n            \n            double y_51201 = ((__local double *) mem_60779)[i_51189];\n            double y_51202 = x_51198 * y_51201;\n            double y_51203 = x_51197 - y_51202;\n            double norm_factor_51204 = 1.0 / y_51203;\n            double x_51205 = ((__local double *) mem_60747)[gtid_50017 *\n                                                            zzdim_48114 +\n                                                            index_primexp_51192];\n            double lw_val_51206 = norm_factor_51204 * x_51205;\n            \n            ((__local double *) mem_60779)[index_primexp_51192] = lw_val_51206;\n            \n            double x_51208 = ((__local double *) mem_60744)[gtid_50017 *\n                                                            zzdim_48114 +\n                                                            index_primexp_51192];\n            double y_51209 = ((__local double *) mem_60784)[i_51189];\n            double y_51210 = x_51198 * y_51209;\n            double x_51211 = x_51208 - y_51210;\n            double lw_val_51",
                   "212 = norm_factor_51204 * x_51211;\n            \n            ((__local double *) mem_60784)[index_primexp_51192] = lw_val_51212;\n        }\n        for (int64_t i_61180 = 0; i_61180 < zzdim_48114; i_61180++) {\n            ((__local double *) mem_60800)[gtid_50017 * zzdim_48114 + i_61180] =\n                ((__local double *) mem_60779)[i_61180];\n        }\n        for (int64_t i_61181 = 0; i_61181 < zzdim_48114; i_61181++) {\n            ((__local double *) mem_60803)[gtid_50017 * zzdim_48114 + i_61181] =\n                ((__local double *) mem_60784)[i_61181];\n        }\n    }\n    \n  error_2:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_60806;\n    \n    mem_60806 = (__local char *) mem_60806_backing_10;\n    for (int64_t i_61182 = 0; i_61182 < sdiv_up64(ydim_48113 * zzdim_48114 -\n                                                  sext_i32_i64(local_tid_61167),\n                                                  computed_group_sizze_49958);\n         i_61182++) {\n        ((__local double *) mem_60806)[squot64(i_61182 *\n                                               computed_group_sizze_49958 +\n                                               sext_i32_i64(local_tid_61167),\n                                               zzdim_48114) * zzdim_48114 +\n                                       (i_61182 * computed_group_sizze_49958 +\n                                        sext_i32_i64(local_tid_61167) -\n                                        squot64(i_61182 *\n                                                computed_group_sizze_49958 +\n                                                sext_i32_i64(local_tid_61167),\n                                                zzdim_48114) * zzdim_48114)] =\n            0.0;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int64_t gtid_49989 = sext_i32_i64(ltid_pre_61172);\n    int64_t gtid_slice_49990 = sext_i32_i64(ltid_pre_61173);\n    int32_t phys_tid_49993 = local_",
                   "tid_61167;\n    \n    if (slt64(gtid_49989, ydim_48113) && slt64(gtid_slice_49990, 1)) {\n        int64_t index_primexp_60422 = y_48310 + gtid_slice_49990;\n        double v_51223 = ((__local double *) mem_60803)[gtid_49989 *\n                                                        zzdim_48114 +\n                                                        index_primexp_60422];\n        \n        if ((sle64(0, gtid_49989) && slt64(gtid_49989, ydim_48113)) && (sle64(0,\n                                                                              index_primexp_60422) &&\n                                                                        slt64(index_primexp_60422,\n                                                                              zzdim_48114))) {\n            ((__local double *) mem_60806)[gtid_49989 * zzdim_48114 +\n                                           index_primexp_60422] = v_51223;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_60829;\n    \n    mem_60829 = (__local char *) mem_60829_backing_11;\n    \n    int64_t gtid_49962 = sext_i32_i64(ltid_pre_61171);\n    int32_t phys_tid_49963 = local_tid_61167;\n    \n    if (slt64(gtid_49962, ydim_48113)) {\n        __local char *mem_60818;\n        \n        mem_60818 = (__local char *) mem_60818_backing_12;\n        for (int64_t i_61183 = 0; i_61183 < zzdim_48114; i_61183++) {\n            ((__local double *) mem_60818)[i_61183] = ((__local\n                                                        double *) mem_60806)[gtid_49962 *\n                                                                             zzdim_48114 +\n                                                                             i_61183];\n        }\n        for (int64_t i_51230 = 0; i_51230 < y_48310; i_51230++) {\n            int64_t binop_y_51232 = -1 * i_51230;\n            int64_t binop_x_51233 = m_48377 + binop_y_51232;\n            bool x_51234 = sle64(0, binop_x_51233);\n            bool y_51235 = slt64(binop_x_51233, zzdim_48114);",
                   "\n            bool bounds_check_51236 = x_51234 && y_51235;\n            bool index_certs_51237;\n            \n            if (!bounds_check_51236) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 10) ==\n                        -1) {\n                        global_failure_args[0] = binop_x_51233;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_4;\n                }\n            }\n            \n            double x_51238 = ((__local double *) mem_60803)[gtid_49962 *\n                                                            zzdim_48114 +\n                                                            binop_x_51233];\n            double x_51239 = ((__local double *) mem_60800)[gtid_49962 *\n                                                            zzdim_48114 +\n                                                            binop_x_51233];\n            int64_t i_51240 = add64(1, binop_x_51233);\n            bool x_51241 = sle64(0, i_51240);\n            bool y_51242 = slt64(i_51240, zzdim_48114);\n            bool bounds_check_51243 = x_51241 && y_51242;\n            bool index_certs_51244;\n            \n            if (!bounds_check_51243) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 11) ==\n                        -1) {\n                        global_failure_args[0] = i_51240;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_4;\n                }\n            }\n            \n            double y_51245 = ((__local double *) mem_60818)[i_51240];\n            double y_51246 = x_51239 * y_51245;\n            double lw_val_51247 = x_51238 - y_51246;\n            \n            ((__local double *) mem_60818)[binop_x_51233] = lw_val_51247;\n        }\n  ",
                   "      for (int64_t i_61185 = 0; i_61185 < zzdim_48114; i_61185++) {\n            ((__local double *) mem_60829)[gtid_49962 * zzdim_48114 + i_61185] =\n                ((__local double *) mem_60818)[i_61185];\n        }\n    }\n    \n  error_4:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_61186 = 0; i_61186 < sdiv_up64(ydim_48113 * zzdim_48114 -\n                                                  sext_i32_i64(local_tid_61167),\n                                                  computed_group_sizze_49958);\n         i_61186++) {\n        ((__global double *) mem_60833)[gtid_49956 * (zzdim_48114 *\n                                                      ydim_48113) +\n                                        squot64(i_61186 *\n                                                computed_group_sizze_49958 +\n                                                sext_i32_i64(local_tid_61167),\n                                                zzdim_48114) * zzdim_48114 +\n                                        (i_61186 * computed_group_sizze_49958 +\n                                         sext_i32_i64(local_tid_61167) -\n                                         squot64(i_61186 *\n                                                 computed_group_sizze_49958 +\n                                                 sext_i32_i64(local_tid_61167),\n                                                 zzdim_48114) * zzdim_48114)] =\n            ((__local double *) mem_60829)[squot64(i_61186 *\n                                                   computed_group_sizze_49958 +\n                                                   sext_i32_i64(local_tid_61167),\n                                                   zzdim_48114) * zzdim_48114 +\n                                           (i_61186 *\n                                            computed_group_sizze_49958 +\n                                            sext_i32_i64(local_tid_61167) -\n             ",
                   "                               squot64(i_61186 *\n                                                    computed_group_sizze_49958 +\n                                                    sext_i32_i64(local_tid_61167),\n                                                    zzdim_48114) *\n                                            zzdim_48114)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n  error_5:\n    return;\n}\n__kernel void integrate_tkezisegmap_intragroup_51504(__global\n                                                     int *global_failure,\n                                                     int failure_is_an_option,\n                                                     __global\n                                                     int64_t *global_failure_args,\n                                                     __local volatile\n                                                     int64_t *mem_60874_backing_aligned_0,\n                                                     __local volatile\n                                                     int64_t *mem_60848_backing_aligned_1,\n                                                     __local volatile\n                                                     int64_t *mem_60846_backing_aligned_2,\n                                                     __local volatile\n                                                     int64_t *mem_60843_backing_aligned_3,\n                                                     __local volatile\n                                                     int64_t *mem_60841_backing_aligned_4,\n                                                     __local volatile\n                                                     int64_t *mem_60839_backing_aligned_5,\n                                                     __local volatile\n                                                     int64_t *mem_60837_backing_aligned_6,\n                                                     int64_t xdim_48112,\n                               ",
                   "                      int64_t ydim_48113,\n                                                     int64_t zzdim_48114,\n                                                     int64_t ydim_48157,\n                                                     int64_t ydim_48162,\n                                                     int64_t zzdim_48163,\n                                                     int64_t ydim_48165,\n                                                     int64_t zzdim_48166,\n                                                     int64_t ydim_48168,\n                                                     int64_t y_48308,\n                                                     int64_t y_48309,\n                                                     int64_t y_48310,\n                                                     int64_t m_48377, __global\n                                                     unsigned char *tketau_mem_60702,\n                                                     __global\n                                                     unsigned char *dzzw_mem_60719,\n                                                     __global\n                                                     unsigned char *kbot_mem_60722,\n                                                     __global\n                                                     unsigned char *mxl_mem_60724,\n                                                     __global\n                                                     unsigned char *forc_mem_60725,\n                                                     __global\n                                                     unsigned char *forc_tke_surface_mem_60726,\n                                                     __global\n                                                     unsigned char *mem_60731,\n                                                     __global\n                                                     unsigned char *mem_60735,\n                                              ",
                   "       __global\n                                                     unsigned char *mem_60890)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_60874_backing_6 = (__local volatile\n                                                           char *) mem_60874_backing_aligned_0;\n    __local volatile char *restrict mem_60848_backing_5 = (__local volatile\n                                                           char *) mem_60848_backing_aligned_1;\n    __local volatile char *restrict mem_60846_backing_4 = (__local volatile\n                                                           char *) mem_60846_backing_aligned_2;\n    __local volatile char *restrict mem_60843_backing_3 = (__local volatile\n                                                           char *) mem_60843_backing_aligned_3;\n    __local volatile char *restrict mem_60841_backing_2 = (__local volatile\n                                                           char *) mem_60841_backing_aligned_4;\n    __local volatile char *restrict mem_60839_backing_1 = (__local volatile\n                                                           char *) mem_60839_backing_aligned_5;\n    __local volatile char *restrict mem_60837_backing_0 = (__local volatile\n                                                           char *) mem_60837_backing_aligned_6;\n    volatile __local bool local_failure;\n    \n    if (failure_is_an_option) {\n        int failed = *global_failure >= 0;\n        \n        if (failed)\n            return;\n    }\n    local_failure = false;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t global_tid_61187;\n    int32_t local_tid_61188;\n    int64_t group_sizze_61191;\n    int32_t wave_sizze_61190;\n    int32_t group_tid_61189;\n    \n    global_tid_61187 = get_global_id(0);\n    local_tid_61188 = get_local_id(0);\n    group_sizze_61191 = get_local_size(0);\n    wave_sizze_61190 = LOCKSTEP_WIDTH;\n    group_tid_61189 = get_group_id(0);\n    \n    int",
                   "32_t phys_tid_51504;\n    \n    phys_tid_51504 = group_tid_61189;\n    \n    int32_t ltid_pre_61192;\n    \n    ltid_pre_61192 = local_tid_61188;\n    \n    int64_t gtid_51263;\n    \n    gtid_51263 = squot64(sext_i32_i64(group_tid_61189), ydim_48113);\n    \n    int64_t gtid_51264;\n    \n    gtid_51264 = sext_i32_i64(group_tid_61189) -\n        squot64(sext_i32_i64(group_tid_61189), ydim_48113) * ydim_48113;\n    \n    bool binop_x_60435;\n    \n    binop_x_60435 = sle64(2, gtid_51263);\n    \n    bool binop_y_60438 = slt64(gtid_51263, y_48308);\n    bool index_primexp_60439 = binop_x_60435 && binop_y_60438;\n    bool cond_t_res_53506 = sle64(2, gtid_51264);\n    bool x_53507 = cond_t_res_53506 && index_primexp_60439;\n    bool cond_t_res_53508 = slt64(gtid_51264, y_48309);\n    bool x_53509 = x_53507 && cond_t_res_53508;\n    __local char *mem_60837;\n    \n    mem_60837 = (__local char *) mem_60837_backing_0;\n    \n    __local char *mem_60839;\n    \n    mem_60839 = (__local char *) mem_60839_backing_1;\n    \n    __local char *mem_60841;\n    \n    mem_60841 = (__local char *) mem_60841_backing_2;\n    \n    __local char *mem_60843;\n    \n    mem_60843 = (__local char *) mem_60843_backing_3;\n    \n    int64_t gtid_51268 = sext_i32_i64(ltid_pre_61192);\n    int32_t phys_tid_51269 = local_tid_61188;\n    \n    if (slt64(gtid_51268, zzdim_48114)) {\n        double lifted_0_f_res_53515;\n        \n        if (x_53509) {\n            int32_t x_53524 = ((__global int32_t *) kbot_mem_60722)[gtid_51263 *\n                                                                    ydim_48157 +\n                                                                    gtid_51264];\n            int32_t ks_val_53525 = sub32(x_53524, 1);\n            bool land_mask_53526 = sle32(0, ks_val_53525);\n            int32_t i64_res_53527 = sext_i64_i32(gtid_51268);\n            bool edge_mask_t_res_53528 = i64_res_53527 == ks_val_53525;\n            bool x_53529 = land_mask_53526 && edge_mask_t_res_53528;\n            bool water_mask_t_res_53530 = ",
                   "sle32(ks_val_53525, i64_res_53527);\n            bool x_53531 = land_mask_53526 && water_mask_t_res_53530;\n            bool cond_f_res_53532 = !x_53531;\n            bool x_53533 = !x_53529;\n            bool y_53534 = cond_f_res_53532 && x_53533;\n            bool cond_53535 = x_53529 || y_53534;\n            double lifted_0_f_res_t_res_53536;\n            \n            if (cond_53535) {\n                lifted_0_f_res_t_res_53536 = 0.0;\n            } else {\n                bool cond_53537 = slt64(0, gtid_51268);\n                int64_t y_53538 = sub64(zzdim_48114, 1);\n                bool cond_t_res_53539 = slt64(gtid_51268, y_53538);\n                bool x_53540 = cond_53537 && cond_t_res_53539;\n                double lifted_0_f_res_t_res_f_res_53541;\n                \n                if (x_53540) {\n                    int64_t i_53542 = sub64(gtid_51268, 1);\n                    bool x_53543 = sle64(0, i_53542);\n                    bool y_53544 = slt64(i_53542, zzdim_48114);\n                    bool bounds_check_53545 = x_53543 && y_53544;\n                    bool index_certs_53548;\n                    \n                    if (!bounds_check_53545) {\n                        {\n                            if (atomic_cmpxchg_i32_global(global_failure, -1,\n                                                          12) == -1) {\n                                global_failure_args[0] = gtid_51263;\n                                global_failure_args[1] = gtid_51264;\n                                global_failure_args[2] = i_53542;\n                                global_failure_args[3] = xdim_48112;\n                                global_failure_args[4] = ydim_48113;\n                                global_failure_args[5] = zzdim_48114;\n                                ;\n                            }\n                            local_failure = true;\n                            goto error_0;\n                        }\n                    }\n                    \n                    double ",
                   "x_53549 = ((__global\n                                       double *) mem_60731)[gtid_51263 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_51264 *\n                                                            zzdim_48114 +\n                                                            i_53542];\n                    double y_53554 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_51268];\n                    double negate_arg_53555 = x_53549 / y_53554;\n                    double lifted_0_f_res_t_res_f_res_t_res_53556 = 0.0 -\n                           negate_arg_53555;\n                    \n                    lifted_0_f_res_t_res_f_res_53541 =\n                        lifted_0_f_res_t_res_f_res_t_res_53556;\n                } else {\n                    bool cond_53557 = gtid_51268 == y_53538;\n                    double lifted_0_f_res_t_res_f_res_f_res_53558;\n                    \n                    if (cond_53557) {\n                        int64_t i_53559 = sub64(gtid_51268, 1);\n                        bool x_53560 = sle64(0, i_53559);\n                        bool y_53561 = slt64(i_53559, zzdim_48114);\n                        bool bounds_check_53562 = x_53560 && y_53561;\n                        bool index_certs_53565;\n                        \n                        if (!bounds_check_53562) {\n                            {\n                                if (atomic_cmpxchg_i32_global(global_failure,\n                                                              -1, 13) == -1) {\n                                    global_failure_args[0] = gtid_51263;\n                                    global_failure_args[1] = gtid_51264;\n                                    global_failure_args[2] = i_53559;\n                                    global_failure_args[3] = xdim_48112;\n                 ",
                   "                   global_failure_args[4] = ydim_48113;\n                                    global_failure_args[5] = zzdim_48114;\n                                    ;\n                                }\n                                local_failure = true;\n                                goto error_0;\n                            }\n                        }\n                        \n                        double x_53566 = ((__global\n                                           double *) mem_60731)[gtid_51263 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_51264 *\n                                                                zzdim_48114 +\n                                                                i_53559];\n                        double y_53571 = ((__global\n                                           double *) dzzw_mem_60719)[gtid_51268];\n                        double y_53572 = 0.5 * y_53571;\n                        double negate_arg_53573 = x_53566 / y_53572;\n                        double lifted_0_f_res_t_res_f_res_f_res_t_res_53574 =\n                               0.0 - negate_arg_53573;\n                        \n                        lifted_0_f_res_t_res_f_res_f_res_53558 =\n                            lifted_0_f_res_t_res_f_res_f_res_t_res_53574;\n                    } else {\n                        lifted_0_f_res_t_res_f_res_f_res_53558 = 0.0;\n                    }\n                    lifted_0_f_res_t_res_f_res_53541 =\n                        lifted_0_f_res_t_res_f_res_f_res_53558;\n                }\n                lifted_0_f_res_t_res_53536 = lifted_0_f_res_t_res_f_res_53541;\n            }\n            lifted_0_f_res_53515 = lifted_0_f_res_t_res_53536;\n        } else {\n            lifted_0_f_res_53515 = 0.0;\n        }\n        \n        double lifted_0_f_res_53575;\n        \n        i",
                   "f (x_53509) {\n            int32_t x_53584 = ((__global int32_t *) kbot_mem_60722)[gtid_51263 *\n                                                                    ydim_48157 +\n                                                                    gtid_51264];\n            int32_t ks_val_53585 = sub32(x_53584, 1);\n            bool land_mask_53586 = sle32(0, ks_val_53585);\n            int32_t i64_res_53587 = sext_i64_i32(gtid_51268);\n            bool edge_mask_t_res_53588 = i64_res_53587 == ks_val_53585;\n            bool x_53589 = land_mask_53586 && edge_mask_t_res_53588;\n            bool water_mask_t_res_53590 = sle32(ks_val_53585, i64_res_53587);\n            bool x_53591 = land_mask_53586 && water_mask_t_res_53590;\n            bool cond_53592 = !x_53591;\n            double lifted_0_f_res_t_res_53593;\n            \n            if (cond_53592) {\n                lifted_0_f_res_t_res_53593 = 1.0;\n            } else {\n                double lifted_0_f_res_t_res_f_res_53594;\n                \n                if (x_53589) {\n                    double x_53601 = ((__global\n                                       double *) mem_60731)[gtid_51263 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_51264 *\n                                                            zzdim_48114 +\n                                                            gtid_51268];\n                    double y_53603 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_51268];\n                    double y_53604 = x_53601 / y_53603;\n                    double x_53605 = 1.0 + y_53604;\n                    double y_53606 = ((__global\n                                       double *) mxl_mem_60724)[gtid_51263 *\n                                                                (zzdim_48163 *\n                                            ",
                   "                     ydim_48162) +\n                                                                gtid_51264 *\n                                                                zzdim_48163 +\n                                                                gtid_51268];\n                    double x_53607 = 0.7 / y_53606;\n                    double y_53608 = ((__global\n                                       double *) mem_60735)[gtid_51263 *\n                                                            (zzdim_48114 *\n                                                             ydim_48113) +\n                                                            gtid_51264 *\n                                                            zzdim_48114 +\n                                                            gtid_51268];\n                    double y_53609 = x_53607 * y_53608;\n                    double lifted_0_f_res_t_res_f_res_t_res_53610 = x_53605 +\n                           y_53609;\n                    \n                    lifted_0_f_res_t_res_f_res_53594 =\n                        lifted_0_f_res_t_res_f_res_t_res_53610;\n                } else {\n                    bool cond_53611 = slt64(0, gtid_51268);\n                    int64_t y_53612 = sub64(zzdim_48114, 1);\n                    bool cond_t_res_53613 = slt64(gtid_51268, y_53612);\n                    bool x_53614 = cond_53611 && cond_t_res_53613;\n                    double lifted_0_f_res_t_res_f_res_f_res_53615;\n                    \n                    if (x_53614) {\n                        double x_53622 = ((__global\n                                           double *) mem_60731)[gtid_51263 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_51264 *\n                                                                zzdim_48114 +\n                              ",
                   "                                  gtid_51268];\n                        int64_t i_53623 = sub64(gtid_51268, 1);\n                        bool x_53624 = sle64(0, i_53623);\n                        bool y_53625 = slt64(i_53623, zzdim_48114);\n                        bool bounds_check_53626 = x_53624 && y_53625;\n                        bool index_certs_53629;\n                        \n                        if (!bounds_check_53626) {\n                            {\n                                if (atomic_cmpxchg_i32_global(global_failure,\n                                                              -1, 14) == -1) {\n                                    global_failure_args[0] = gtid_51263;\n                                    global_failure_args[1] = gtid_51264;\n                                    global_failure_args[2] = i_53623;\n                                    global_failure_args[3] = xdim_48112;\n                                    global_failure_args[4] = ydim_48113;\n                                    global_failure_args[5] = zzdim_48114;\n                                    ;\n                                }\n                                local_failure = true;\n                                goto error_0;\n                            }\n                        }\n                        \n                        double y_53630 = ((__global\n                                           double *) mem_60731)[gtid_51263 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_51264 *\n                                                                zzdim_48114 +\n                                                                i_53623];\n                        double x_53631 = x_53622 + y_53630;\n                        double y_53633 = ((__global\n                                           double *) dzzw_mem_607",
                   "19)[gtid_51268];\n                        double y_53634 = x_53631 / y_53633;\n                        double x_53635 = 1.0 + y_53634;\n                        double y_53636 = ((__global\n                                           double *) mem_60735)[gtid_51263 *\n                                                                (zzdim_48114 *\n                                                                 ydim_48113) +\n                                                                gtid_51264 *\n                                                                zzdim_48114 +\n                                                                gtid_51268];\n                        double x_53637 = 0.7 * y_53636;\n                        double y_53638 = ((__global\n                                           double *) mxl_mem_60724)[gtid_51263 *\n                                                                    (zzdim_48163 *\n                                                                     ydim_48162) +\n                                                                    gtid_51264 *\n                                                                    zzdim_48163 +\n                                                                    gtid_51268];\n                        double y_53639 = x_53637 / y_53638;\n                        double lifted_0_f_res_t_res_f_res_f_res_t_res_53640 =\n                               x_53635 + y_53639;\n                        \n                        lifted_0_f_res_t_res_f_res_f_res_53615 =\n                            lifted_0_f_res_t_res_f_res_f_res_t_res_53640;\n                    } else {\n                        bool cond_53641 = gtid_51268 == y_53612;\n                        double lifted_0_f_res_t_res_f_res_f_res_f_res_53642;\n                        \n                        if (cond_53641) {\n                            int64_t i_53643 = sub64(gtid_51268, 1);\n                            bool x_53644 = sle64(0, i_53643);\n                            b",
                   "ool y_53645 = slt64(i_53643, zzdim_48114);\n                            bool bounds_check_53646 = x_53644 && y_53645;\n                            bool index_certs_53649;\n                            \n                            if (!bounds_check_53646) {\n                                {\n                                    if (atomic_cmpxchg_i32_global(global_failure,\n                                                                  -1, 15) ==\n                                        -1) {\n                                        global_failure_args[0] = gtid_51263;\n                                        global_failure_args[1] = gtid_51264;\n                                        global_failure_args[2] = i_53643;\n                                        global_failure_args[3] = xdim_48112;\n                                        global_failure_args[4] = ydim_48113;\n                                        global_failure_args[5] = zzdim_48114;\n                                        ;\n                                    }\n                                    local_failure = true;\n                                    goto error_0;\n                                }\n                            }\n                            \n                            double x_53650 = ((__global\n                                               double *) mem_60731)[gtid_51263 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    gtid_51264 *\n                                                                    zzdim_48114 +\n                                                                    i_53643];\n                            double y_53655 = ((__global\n                                               double *) dzzw_mem_60719)[gtid_51268];\n                            double y_53656 = 0.5 * y_53655;\n                     ",
                   "       double y_53657 = x_53650 / y_53656;\n                            double x_53658 = 1.0 + y_53657;\n                            double y_53662 = ((__global\n                                               double *) mxl_mem_60724)[gtid_51263 *\n                                                                        (zzdim_48163 *\n                                                                         ydim_48162) +\n                                                                        gtid_51264 *\n                                                                        zzdim_48163 +\n                                                                        gtid_51268];\n                            double x_53663 = 0.7 / y_53662;\n                            double y_53664 = ((__global\n                                               double *) mem_60735)[gtid_51263 *\n                                                                    (zzdim_48114 *\n                                                                     ydim_48113) +\n                                                                    gtid_51264 *\n                                                                    zzdim_48114 +\n                                                                    gtid_51268];\n                            double y_53665 = x_53663 * y_53664;\n                            double\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_t_res_53666 =\n                            x_53658 + y_53665;\n                            \n                            lifted_0_f_res_t_res_f_res_f_res_f_res_53642 =\n                                lifted_0_f_res_t_res_f_res_f_res_f_res_t_res_53666;\n                        } else {\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_53642 = 0.0;\n                        }\n                        lifted_0_f_res_t_res_f_res_f_res_53615 =\n                            lifted_0_f_res_t_res_f_res_f_res_f_res_53642;\n                    }\n    ",
                   "                lifted_0_f_res_t_res_f_res_53594 =\n                        lifted_0_f_res_t_res_f_res_f_res_53615;\n                }\n                lifted_0_f_res_t_res_53593 = lifted_0_f_res_t_res_f_res_53594;\n            }\n            lifted_0_f_res_53575 = lifted_0_f_res_t_res_53593;\n        } else {\n            lifted_0_f_res_53575 = 0.0;\n        }\n        \n        bool cond_t_res_53667 = slt64(gtid_51268, y_48310);\n        bool x_53668 = x_53509 && cond_t_res_53667;\n        double lifted_0_f_res_53669;\n        \n        if (x_53668) {\n            int32_t x_53678 = ((__global int32_t *) kbot_mem_60722)[gtid_51263 *\n                                                                    ydim_48157 +\n                                                                    gtid_51264];\n            int32_t ks_val_53679 = sub32(x_53678, 1);\n            bool land_mask_53680 = sle32(0, ks_val_53679);\n            int32_t i64_res_53681 = sext_i64_i32(gtid_51268);\n            bool water_mask_t_res_53682 = sle32(ks_val_53679, i64_res_53681);\n            bool x_53683 = land_mask_53680 && water_mask_t_res_53682;\n            bool cond_53684 = !x_53683;\n            double lifted_0_f_res_t_res_53685;\n            \n            if (cond_53684) {\n                lifted_0_f_res_t_res_53685 = 0.0;\n            } else {\n                double x_53692 = ((__global double *) mem_60731)[gtid_51263 *\n                                                                 (zzdim_48114 *\n                                                                  ydim_48113) +\n                                                                 gtid_51264 *\n                                                                 zzdim_48114 +\n                                                                 gtid_51268];\n                double y_53694 = ((__global\n                                   double *) dzzw_mem_60719)[gtid_51268];\n                double negate_arg_53695 = x_53692 / y_53694;\n                double lifted_0_f_r",
                   "es_t_res_f_res_53696 = 0.0 -\n                       negate_arg_53695;\n                \n                lifted_0_f_res_t_res_53685 = lifted_0_f_res_t_res_f_res_53696;\n            }\n            lifted_0_f_res_53669 = lifted_0_f_res_t_res_53685;\n        } else {\n            lifted_0_f_res_53669 = 0.0;\n        }\n        \n        double lifted_0_f_res_53697;\n        \n        if (x_53509) {\n            int32_t x_53706 = ((__global int32_t *) kbot_mem_60722)[gtid_51263 *\n                                                                    ydim_48157 +\n                                                                    gtid_51264];\n            int32_t ks_val_53707 = sub32(x_53706, 1);\n            bool land_mask_53708 = sle32(0, ks_val_53707);\n            int32_t i64_res_53709 = sext_i64_i32(gtid_51268);\n            bool water_mask_t_res_53710 = sle32(ks_val_53707, i64_res_53709);\n            bool x_53711 = land_mask_53708 && water_mask_t_res_53710;\n            bool cond_53712 = !x_53711;\n            double lifted_0_f_res_t_res_53713;\n            \n            if (cond_53712) {\n                lifted_0_f_res_t_res_53713 = 0.0;\n            } else {\n                double x_53720 = ((__global\n                                   double *) tketau_mem_60702)[gtid_51263 *\n                                                               (zzdim_48114 *\n                                                                ydim_48113) +\n                                                               gtid_51264 *\n                                                               zzdim_48114 +\n                                                               gtid_51268];\n                double y_53721 = ((__global\n                                   double *) forc_mem_60725)[gtid_51263 *\n                                                             (zzdim_48166 *\n                                                              ydim_48165) +\n                                                             gtid_5126",
                   "4 *\n                                                             zzdim_48166 +\n                                                             gtid_51268];\n                double tmp_53722 = x_53720 + y_53721;\n                int64_t y_53723 = sub64(zzdim_48114, 1);\n                bool cond_53724 = gtid_51268 == y_53723;\n                double lifted_0_f_res_t_res_f_res_53725;\n                \n                if (cond_53724) {\n                    double y_53726 = ((__global\n                                       double *) forc_tke_surface_mem_60726)[gtid_51263 *\n                                                                             ydim_48168 +\n                                                                             gtid_51264];\n                    double y_53728 = ((__global\n                                       double *) dzzw_mem_60719)[gtid_51268];\n                    double y_53729 = 0.5 * y_53728;\n                    double y_53730 = y_53726 / y_53729;\n                    double lifted_0_f_res_t_res_f_res_t_res_53731 = tmp_53722 +\n                           y_53730;\n                    \n                    lifted_0_f_res_t_res_f_res_53725 =\n                        lifted_0_f_res_t_res_f_res_t_res_53731;\n                } else {\n                    lifted_0_f_res_t_res_f_res_53725 = tmp_53722;\n                }\n                lifted_0_f_res_t_res_53713 = lifted_0_f_res_t_res_f_res_53725;\n            }\n            lifted_0_f_res_53697 = lifted_0_f_res_t_res_53713;\n        } else {\n            lifted_0_f_res_53697 = 0.0;\n        }\n        ((__local double *) mem_60837)[gtid_51268] = lifted_0_f_res_53697;\n        ((__local double *) mem_60839)[gtid_51268] = lifted_0_f_res_53669;\n        ((__local double *) mem_60841)[gtid_51268] = lifted_0_f_res_53575;\n        ((__local double *) mem_60843)[gtid_51268] = lifted_0_f_res_53515;\n    }\n    \n  error_0:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n  ",
                   "  \n    __local char *mem_60846;\n    \n    mem_60846 = (__local char *) mem_60846_backing_4;\n    \n    __local char *mem_60848;\n    \n    mem_60848 = (__local char *) mem_60848_backing_5;\n    \n    int64_t gtid_51488 = sext_i32_i64(ltid_pre_61192);\n    int32_t phys_tid_51489 = local_tid_61188;\n    \n    if (slt64(gtid_51488, zzdim_48114)) {\n        bool cond_53735 = gtid_51488 == 0;\n        double lifted_0_f_res_53736;\n        \n        if (cond_53735) {\n            bool y_53737 = slt64(0, zzdim_48114);\n            bool index_certs_53738;\n            \n            if (!y_53737) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 16) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_53739 = ((__local double *) mem_60839)[0];\n            double y_53740 = ((__local double *) mem_60841)[0];\n            double lifted_0_f_res_t_res_53741 = x_53739 / y_53740;\n            \n            lifted_0_f_res_53736 = lifted_0_f_res_t_res_53741;\n        } else {\n            lifted_0_f_res_53736 = 0.0;\n        }\n        \n        double lifted_0_f_res_53742;\n        \n        if (cond_53735) {\n            bool y_53743 = slt64(0, zzdim_48114);\n            bool index_certs_53744;\n            \n            if (!y_53743) {\n                {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 17) ==\n                        -1) {\n                        global_failure_args[0] = 0;\n                        global_failure_args[1] = zzdim_48114;\n                        ;\n                    }\n                    local_failure = true;\n                    goto error_1;\n                }\n            }\n            \n            double x_53745 = ((__local double *) mem_60837)",
                   "[0];\n            double y_53746 = ((__local double *) mem_60841)[0];\n            double lifted_0_f_res_t_res_53747 = x_53745 / y_53746;\n            \n            lifted_0_f_res_53742 = lifted_0_f_res_t_res_53747;\n        } else {\n            lifted_0_f_res_53742 = 0.0;\n        }\n        ((__local double *) mem_60846)[gtid_51488] = lifted_0_f_res_53742;\n        ((__local double *) mem_60848)[gtid_51488] = lifted_0_f_res_53736;\n    }\n    \n  error_1:\n    barrier(CLK_LOCAL_MEM_FENCE);\n    if (local_failure)\n        return;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_53750 = 0; i_53750 < y_48310; i_53750++) {\n        int64_t index_primexp_53753 = add64(1, i_53750);\n        bool x_53754 = sle64(0, index_primexp_53753);\n        bool y_53755 = slt64(index_primexp_53753, zzdim_48114);\n        bool bounds_check_53756 = x_53754 && y_53755;\n        bool index_certs_53757;\n        \n        if (!bounds_check_53756) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 18) == -1) {\n                    global_failure_args[0] = index_primexp_53753;\n                    global_failure_args[1] = zzdim_48114;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double x_53758 = ((__local double *) mem_60841)[index_primexp_53753];\n        double x_53759 = ((__local double *) mem_60843)[index_primexp_53753];\n        bool y_53760 = slt64(i_53750, zzdim_48114);\n        bool index_certs_53761;\n        \n        if (!y_53760) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 19) == -1) {\n                    global_failure_args[0] = i_53750;\n                    global_failure_args[1] = zzdim_48114;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double y_53762 = ((__local double *) mem_60848)[i_53750];\n        double y_53763 = x_53759",
                   " * y_53762;\n        double y_53764 = x_53758 - y_53763;\n        double norm_factor_53765 = 1.0 / y_53764;\n        double x_53766 = ((__local double *) mem_60839)[index_primexp_53753];\n        double lw_val_53767 = norm_factor_53765 * x_53766;\n        \n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_61188 == 0) {\n            ((__local double *) mem_60848)[index_primexp_53753] = lw_val_53767;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        double x_53769 = ((__local double *) mem_60837)[index_primexp_53753];\n        double y_53770 = ((__local double *) mem_60846)[i_53750];\n        double y_53771 = x_53759 * y_53770;\n        double x_53772 = x_53769 - y_53771;\n        double lw_val_53773 = norm_factor_53765 * x_53772;\n        \n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_61188 == 0) {\n            ((__local double *) mem_60846)[index_primexp_53753] = lw_val_53773;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    \n    __local char *mem_60874;\n    \n    mem_60874 = (__local char *) mem_60874_backing_6;\n    ((__local double *) mem_60874)[sext_i32_i64(local_tid_61188)] = 0.0;\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_61195 = 0; i_61195 < sdiv_up64(1 -\n                                                  sext_i32_i64(local_tid_61188),\n                                                  zzdim_48114); i_61195++) {\n        ((__local double *) mem_60874)[y_48310 + (i_61195 * zzdim_48114 +\n                                                  sext_i32_i64(local_tid_61188))] =\n            ((__local double *) mem_60846)[y_48310 + (i_61195 * zzdim_48114 +\n                                                      sext_i32_i64(local_tid_61188))];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    for (int64_t i_53779 = 0; i_53779 < y_48310; i_53779++) {\n        int64_t binop_y_53781 = -1 * i_53779;\n        int64_t binop_x_53782 = m_48377 + binop_y_53781;\n        bool x_53783 = sle64(0, binop_x_53782);\n        bool y_53784 = slt64(binop_x_5378",
                   "2, zzdim_48114);\n        bool bounds_check_53785 = x_53783 && y_53784;\n        bool index_certs_53786;\n        \n        if (!bounds_check_53785) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 20) == -1) {\n                    global_failure_args[0] = binop_x_53782;\n                    global_failure_args[1] = zzdim_48114;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double x_53787 = ((__local double *) mem_60846)[binop_x_53782];\n        double x_53788 = ((__local double *) mem_60848)[binop_x_53782];\n        int64_t i_53789 = add64(1, binop_x_53782);\n        bool x_53790 = sle64(0, i_53789);\n        bool y_53791 = slt64(i_53789, zzdim_48114);\n        bool bounds_check_53792 = x_53790 && y_53791;\n        bool index_certs_53793;\n        \n        if (!bounds_check_53792) {\n            {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 21) == -1) {\n                    global_failure_args[0] = i_53789;\n                    global_failure_args[1] = zzdim_48114;\n                    ;\n                }\n                local_failure = true;\n                goto error_2;\n            }\n        }\n        \n        double y_53794 = ((__local double *) mem_60874)[i_53789];\n        double y_53795 = x_53788 * y_53794;\n        double lw_val_53796 = x_53787 - y_53795;\n        \n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (local_tid_61188 == 0) {\n            ((__local double *) mem_60874)[binop_x_53782] = lw_val_53796;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    ((__global double *) mem_60890)[gtid_51263 * (zzdim_48114 * ydim_48113) +\n                                    gtid_51264 * zzdim_48114 +\n                                    sext_i32_i64(local_tid_61188)] = ((__local\n                                                                       double *) mem_60874)[sext_i32_i64(local_tid_61188)];\n    barrier(CLK_LOCAL_",
                   "MEM_FENCE);\n    \n  error_2:\n    return;\n}\n", NULL};
static const char *size_names[] = {"builtin#replicate_f64.group_size_61163",
                                   "integrate_tke.segmap_group_size_49658",
                                   "integrate_tke.segmap_group_size_52113",
                                   "integrate_tke.segmap_group_size_52154",
                                   "integrate_tke.segmap_group_size_52231",
                                   "integrate_tke.segmap_group_size_52309",
                                   "integrate_tke.segmap_group_size_52601",
                                   "integrate_tke.segmap_group_size_54406",
                                   "integrate_tke.segmap_group_size_54474",
                                   "integrate_tke.segmap_group_size_55221",
                                   "integrate_tke.segmap_group_size_55435",
                                   "integrate_tke.segmap_group_size_57271",
                                   "integrate_tke.segmap_group_size_59408",
                                   "integrate_tke.segmap_group_size_60205",
                                   "integrate_tke.segmap_num_groups_52115",
                                   "integrate_tke.segmap_num_groups_52233",
                                   "integrate_tke.suff_intra_par_4",
                                   "integrate_tke.suff_intra_par_6"};
static const char *size_vars[] = {"builtinzhreplicate_f64zigroup_sizze_61163",
                                  "integrate_tkezisegmap_group_sizze_49658",
                                  "integrate_tkezisegmap_group_sizze_52113",
                                  "integrate_tkezisegmap_group_sizze_52154",
                                  "integrate_tkezisegmap_group_sizze_52231",
                                  "integrate_tkezisegmap_group_sizze_52309",
                                  "integrate_tkezisegmap_group_sizze_52601",
                                  "integrate_tkezisegmap_group_sizze_54406",
                                  "integrate_tkezisegmap_group_sizze_54474",
                                  "integrate_tkezisegmap_group_sizze_55221",
                                  "integrate_tkezisegmap_group_sizze_55435",
                                  "integrate_tkezisegmap_group_sizze_57271",
                                  "integrate_tkezisegmap_group_sizze_59408",
                                  "integrate_tkezisegmap_group_sizze_60205",
                                  "integrate_tkezisegmap_num_groups_52115",
                                  "integrate_tkezisegmap_num_groups_52233",
                                  "integrate_tkezisuff_intra_par_4",
                                  "integrate_tkezisuff_intra_par_6"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "num_groups",
                                     "num_groups", "threshold ()",
                                     "threshold (!integrate_tke.suff_intra_par_4)"};
int futhark_get_num_sizes(void)
{
    return 18;
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
    size_t builtinzhreplicate_f64zigroup_sizze_61163;
    size_t integrate_tkezisegmap_group_sizze_49658;
    size_t integrate_tkezisegmap_group_sizze_52113;
    size_t integrate_tkezisegmap_group_sizze_52154;
    size_t integrate_tkezisegmap_group_sizze_52231;
    size_t integrate_tkezisegmap_group_sizze_52309;
    size_t integrate_tkezisegmap_group_sizze_52601;
    size_t integrate_tkezisegmap_group_sizze_54406;
    size_t integrate_tkezisegmap_group_sizze_54474;
    size_t integrate_tkezisegmap_group_sizze_55221;
    size_t integrate_tkezisegmap_group_sizze_55435;
    size_t integrate_tkezisegmap_group_sizze_57271;
    size_t integrate_tkezisegmap_group_sizze_59408;
    size_t integrate_tkezisegmap_group_sizze_60205;
    size_t integrate_tkezisegmap_num_groups_52115;
    size_t integrate_tkezisegmap_num_groups_52233;
    size_t integrate_tkezisuff_intra_par_4;
    size_t integrate_tkezisuff_intra_par_6;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[18];
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
    cfg->sizes[15] = 0;
    cfg->sizes[16] = 32;
    cfg->sizes[17] = 32;
    opencl_config_init(&cfg->opencl, 18, size_names, size_vars, cfg->sizes,
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
    for (int i = 0; i < 18; i++) {
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
    cl_kernel builtinzhreplicate_f64zireplicate_61160;
    cl_kernel gpu_map_transpose_f64;
    cl_kernel gpu_map_transpose_f64_low_height;
    cl_kernel gpu_map_transpose_f64_low_width;
    cl_kernel gpu_map_transpose_f64_small;
    cl_kernel integrate_tkezisegmap_49654;
    cl_kernel integrate_tkezisegmap_52110;
    cl_kernel integrate_tkezisegmap_52150;
    cl_kernel integrate_tkezisegmap_52228;
    cl_kernel integrate_tkezisegmap_52305;
    cl_kernel integrate_tkezisegmap_52597;
    cl_kernel integrate_tkezisegmap_54402;
    cl_kernel integrate_tkezisegmap_54471;
    cl_kernel integrate_tkezisegmap_55217;
    cl_kernel integrate_tkezisegmap_55432;
    cl_kernel integrate_tkezisegmap_57267;
    cl_kernel integrate_tkezisegmap_59404;
    cl_kernel integrate_tkezisegmap_60201;
    cl_kernel integrate_tkezisegmap_intragroup_50310;
    cl_kernel integrate_tkezisegmap_intragroup_51504;
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
    int64_t builtinzhreplicate_f64zireplicate_61160_total_runtime;
    int builtinzhreplicate_f64zireplicate_61160_runs;
    int64_t gpu_map_transpose_f64_total_runtime;
    int gpu_map_transpose_f64_runs;
    int64_t gpu_map_transpose_f64_low_height_total_runtime;
    int gpu_map_transpose_f64_low_height_runs;
    int64_t gpu_map_transpose_f64_low_width_total_runtime;
    int gpu_map_transpose_f64_low_width_runs;
    int64_t gpu_map_transpose_f64_small_total_runtime;
    int gpu_map_transpose_f64_small_runs;
    int64_t integrate_tkezisegmap_49654_total_runtime;
    int integrate_tkezisegmap_49654_runs;
    int64_t integrate_tkezisegmap_52110_total_runtime;
    int integrate_tkezisegmap_52110_runs;
    int64_t integrate_tkezisegmap_52150_total_runtime;
    int integrate_tkezisegmap_52150_runs;
    int64_t integrate_tkezisegmap_52228_total_runtime;
    int integrate_tkezisegmap_52228_runs;
    int64_t integrate_tkezisegmap_52305_total_runtime;
    int integrate_tkezisegmap_52305_runs;
    int64_t integrate_tkezisegmap_52597_total_runtime;
    int integrate_tkezisegmap_52597_runs;
    int64_t integrate_tkezisegmap_54402_total_runtime;
    int integrate_tkezisegmap_54402_runs;
    int64_t integrate_tkezisegmap_54471_total_runtime;
    int integrate_tkezisegmap_54471_runs;
    int64_t integrate_tkezisegmap_55217_total_runtime;
    int integrate_tkezisegmap_55217_runs;
    int64_t integrate_tkezisegmap_55432_total_runtime;
    int integrate_tkezisegmap_55432_runs;
    int64_t integrate_tkezisegmap_57267_total_runtime;
    int integrate_tkezisegmap_57267_runs;
    int64_t integrate_tkezisegmap_59404_total_runtime;
    int integrate_tkezisegmap_59404_runs;
    int64_t integrate_tkezisegmap_60201_total_runtime;
    int integrate_tkezisegmap_60201_runs;
    int64_t integrate_tkezisegmap_intragroup_50310_total_runtime;
    int integrate_tkezisegmap_intragroup_50310_runs;
    int64_t integrate_tkezisegmap_intragroup_51504_total_runtime;
    int integrate_tkezisegmap_intragroup_51504_runs;
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
    ctx->builtinzhreplicate_f64zireplicate_61160_total_runtime = 0;
    ctx->builtinzhreplicate_f64zireplicate_61160_runs = 0;
    ctx->gpu_map_transpose_f64_total_runtime = 0;
    ctx->gpu_map_transpose_f64_runs = 0;
    ctx->gpu_map_transpose_f64_low_height_total_runtime = 0;
    ctx->gpu_map_transpose_f64_low_height_runs = 0;
    ctx->gpu_map_transpose_f64_low_width_total_runtime = 0;
    ctx->gpu_map_transpose_f64_low_width_runs = 0;
    ctx->gpu_map_transpose_f64_small_total_runtime = 0;
    ctx->gpu_map_transpose_f64_small_runs = 0;
    ctx->integrate_tkezisegmap_49654_total_runtime = 0;
    ctx->integrate_tkezisegmap_49654_runs = 0;
    ctx->integrate_tkezisegmap_52110_total_runtime = 0;
    ctx->integrate_tkezisegmap_52110_runs = 0;
    ctx->integrate_tkezisegmap_52150_total_runtime = 0;
    ctx->integrate_tkezisegmap_52150_runs = 0;
    ctx->integrate_tkezisegmap_52228_total_runtime = 0;
    ctx->integrate_tkezisegmap_52228_runs = 0;
    ctx->integrate_tkezisegmap_52305_total_runtime = 0;
    ctx->integrate_tkezisegmap_52305_runs = 0;
    ctx->integrate_tkezisegmap_52597_total_runtime = 0;
    ctx->integrate_tkezisegmap_52597_runs = 0;
    ctx->integrate_tkezisegmap_54402_total_runtime = 0;
    ctx->integrate_tkezisegmap_54402_runs = 0;
    ctx->integrate_tkezisegmap_54471_total_runtime = 0;
    ctx->integrate_tkezisegmap_54471_runs = 0;
    ctx->integrate_tkezisegmap_55217_total_runtime = 0;
    ctx->integrate_tkezisegmap_55217_runs = 0;
    ctx->integrate_tkezisegmap_55432_total_runtime = 0;
    ctx->integrate_tkezisegmap_55432_runs = 0;
    ctx->integrate_tkezisegmap_57267_total_runtime = 0;
    ctx->integrate_tkezisegmap_57267_runs = 0;
    ctx->integrate_tkezisegmap_59404_total_runtime = 0;
    ctx->integrate_tkezisegmap_59404_runs = 0;
    ctx->integrate_tkezisegmap_60201_total_runtime = 0;
    ctx->integrate_tkezisegmap_60201_runs = 0;
    ctx->integrate_tkezisegmap_intragroup_50310_total_runtime = 0;
    ctx->integrate_tkezisegmap_intragroup_50310_runs = 0;
    ctx->integrate_tkezisegmap_intragroup_51504_total_runtime = 0;
    ctx->integrate_tkezisegmap_intragroup_51504_runs = 0;
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
        ctx->builtinzhreplicate_f64zireplicate_61160 = clCreateKernel(prog,
                                                                      "builtinzhreplicate_f64zireplicate_61160",
                                                                      &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "builtin#replicate_f64.replicate_61160");
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
        ctx->integrate_tkezisegmap_49654 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_49654",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_49654");
    }
    {
        ctx->integrate_tkezisegmap_52110 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_52110",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52110, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52110, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_52110");
    }
    {
        ctx->integrate_tkezisegmap_52150 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_52150",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52150, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_52150");
    }
    {
        ctx->integrate_tkezisegmap_52228 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_52228",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52228, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52228, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_52228");
    }
    {
        ctx->integrate_tkezisegmap_52305 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_52305",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52305, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52305, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_52305");
    }
    {
        ctx->integrate_tkezisegmap_52597 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_52597",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52597, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_52597, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_52597");
    }
    {
        ctx->integrate_tkezisegmap_54402 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_54402",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_54402");
    }
    {
        ctx->integrate_tkezisegmap_54471 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_54471",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_54471");
    }
    {
        ctx->integrate_tkezisegmap_55217 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_55217",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_55217");
    }
    {
        ctx->integrate_tkezisegmap_55432 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_55432",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_55432");
    }
    {
        ctx->integrate_tkezisegmap_57267 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_57267",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_57267");
    }
    {
        ctx->integrate_tkezisegmap_59404 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_59404",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 2,
                                            sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_59404");
    }
    {
        ctx->integrate_tkezisegmap_60201 = clCreateKernel(prog,
                                                          "integrate_tkezisegmap_60201",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 0,
                                            sizeof(cl_mem),
                                            &ctx->global_failure));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_60201");
    }
    {
        ctx->integrate_tkezisegmap_intragroup_50310 = clCreateKernel(prog,
                                                                     "integrate_tkezisegmap_intragroup_50310",
                                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                            0, sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                            2, sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_intragroup_50310");
    }
    {
        ctx->integrate_tkezisegmap_intragroup_51504 = clCreateKernel(prog,
                                                                     "integrate_tkezisegmap_intragroup_51504",
                                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                            0, sizeof(cl_mem),
                                            &ctx->global_failure));
        OPENCL_SUCCEED_FATAL(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                            2, sizeof(cl_mem),
                                            &ctx->global_failure_args));
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "integrate_tke.segmap_intragroup_51504");
    }
    ctx->sizes.builtinzhreplicate_f64zigroup_sizze_61163 = cfg->sizes[0];
    ctx->sizes.integrate_tkezisegmap_group_sizze_49658 = cfg->sizes[1];
    ctx->sizes.integrate_tkezisegmap_group_sizze_52113 = cfg->sizes[2];
    ctx->sizes.integrate_tkezisegmap_group_sizze_52154 = cfg->sizes[3];
    ctx->sizes.integrate_tkezisegmap_group_sizze_52231 = cfg->sizes[4];
    ctx->sizes.integrate_tkezisegmap_group_sizze_52309 = cfg->sizes[5];
    ctx->sizes.integrate_tkezisegmap_group_sizze_52601 = cfg->sizes[6];
    ctx->sizes.integrate_tkezisegmap_group_sizze_54406 = cfg->sizes[7];
    ctx->sizes.integrate_tkezisegmap_group_sizze_54474 = cfg->sizes[8];
    ctx->sizes.integrate_tkezisegmap_group_sizze_55221 = cfg->sizes[9];
    ctx->sizes.integrate_tkezisegmap_group_sizze_55435 = cfg->sizes[10];
    ctx->sizes.integrate_tkezisegmap_group_sizze_57271 = cfg->sizes[11];
    ctx->sizes.integrate_tkezisegmap_group_sizze_59408 = cfg->sizes[12];
    ctx->sizes.integrate_tkezisegmap_group_sizze_60205 = cfg->sizes[13];
    ctx->sizes.integrate_tkezisegmap_num_groups_52115 = cfg->sizes[14];
    ctx->sizes.integrate_tkezisegmap_num_groups_52233 = cfg->sizes[15];
    ctx->sizes.integrate_tkezisuff_intra_par_4 = cfg->sizes[16];
    ctx->sizes.integrate_tkezisuff_intra_par_6 = cfg->sizes[17];
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
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->builtinzhreplicate_f64zireplicate_61160));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64_low_height));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64_low_width));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->gpu_map_transpose_f64_small));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_49654));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_52110));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_52150));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_52228));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_52305));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_52597));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_54402));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_54471));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_55217));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_55432));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_57267));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_59404));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_60201));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_intragroup_50310));
    OPENCL_SUCCEED_FATAL(clReleaseKernel(ctx->integrate_tkezisegmap_intragroup_51504));
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
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:116:35-42\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:114:17-119:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 1:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:117:46-62\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:114:17-119:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 2:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:131:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:120:17-136:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 3:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:133:43-58\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:120:17-136:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 4:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:153:60-75\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:138:17-161:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 5:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:157:42-57\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:138:17-161:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 6:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:4:37-40\n   #1  tke_baseline.fut:4:13-65\n   #2  tke_baseline.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke_baseline.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke_baseline.fut:20:4-24:12\n   #9  tke_baseline.fut:198:15-45\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 7:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:5:37-40\n   #1  tke_baseline.fut:5:13-65\n   #2  tke_baseline.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke_baseline.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke_baseline.fut:20:4-24:12\n   #9  tke_baseline.fut:198:15-45\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 8:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:8:35-38\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 9:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:8:49-55\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 10:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:16:25-34\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 11:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:16:51-63\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 12:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:131:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:120:17-136:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 13:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:133:43-58\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:120:17-136:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 14:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:153:60-75\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:138:17-161:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 15:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:157:42-57\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:138:17-161:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 16:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:4:37-40\n   #1  tke_baseline.fut:4:13-65\n   #2  tke_baseline.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke_baseline.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke_baseline.fut:20:4-24:12\n   #9  tke_baseline.fut:198:15-45\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 17:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:5:37-40\n   #1  tke_baseline.fut:5:13-65\n   #2  tke_baseline.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke_baseline.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke_baseline.fut:20:4-24:12\n   #9  tke_baseline.fut:198:15-45\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 18:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:8:35-38\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 19:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:8:49-55\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 20:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:16:25-34\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 21:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:16:51-63\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 22:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:131:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:120:17-136:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 23:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:133:43-58\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:120:17-136:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 24:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:153:60-75\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:138:17-161:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 25:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:157:42-57\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:138:17-161:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 26:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:4:37-40\n   #1  tke_baseline.fut:4:13-65\n   #2  tke_baseline.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke_baseline.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke_baseline.fut:20:4-24:12\n   #9  tke_baseline.fut:198:15-45\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 27:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:5:37-40\n   #1  tke_baseline.fut:5:13-65\n   #2  tke_baseline.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke_baseline.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke_baseline.fut:20:4-24:12\n   #9  tke_baseline.fut:198:15-45\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 28:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:8:35-38\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 29:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:8:49-55\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 30:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:16:25-34\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 31:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:16:51-63\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 32:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:212:39-60\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  tke_baseline.fut:209:25-217:21\n   #7  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 33:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:214:51-61\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  tke_baseline.fut:209:25-217:21\n   #7  tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 34:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:232:40-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:229:21-236:5\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 35:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:241:40-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:238:22-244:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 36:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:250:54-73\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:246:20-255:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 37:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:252:56-76\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:246:20-255:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 38:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:263:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:257:21-276:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 39:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:265:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:257:21-276:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 40:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:266:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:257:21-276:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 41:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:282:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:277:22-295:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 42:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:284:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:277:22-295:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 43:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:285:43-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:277:22-295:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 44:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:300:57-71\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:296:20-314:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 45:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:302:61-75\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:296:20-314:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 46:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:303:42-56\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:296:20-314:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 47:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:304:58-71\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:296:20-314:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 48:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:307:62-75\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:296:20-314:21\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 49:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:319:66-85\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:315:19-332:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 50:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:321:56-76\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:315:19-332:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 51:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:325:62-78\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:315:19-332:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 52:
            {
                ctx->error =
                    msgprintf("Index [%lld] out of bounds for array of shape [%lld].\n-> #0  tke_baseline.fut:325:82-87\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:315:19-332:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1]);
                break;
            }
            
          case 53:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:327:87-105\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:315:19-332:17\n   #10 tke_baseline.fut:62:1-338:81\n",
                              args[0], args[1], args[2], args[3], args[4],
                              args[5]);
                break;
            }
            
          case 54:
            {
                ctx->error =
                    msgprintf("Index [%lld, %lld, %lld] out of bounds for array of shape [%lld][%lld][%lld].\n-> #0  tke_baseline.fut:329:87-105\n   #1  /prelude/soacs.fut:48:3-10\n   #2  /prelude/array.fut:126:3-17\n   #3  /prelude/functional.fut:39:59-65\n   #4  /prelude/soacs.fut:48:3-10\n   #5  /prelude/array.fut:130:3-34\n   #6  /prelude/functional.fut:39:59-65\n   #7  /prelude/soacs.fut:48:3-10\n   #8  /prelude/array.fut:134:3-39\n   #9  tke_baseline.fut:315:19-332:17\n   #10 tke_baseline.fut:62:1-338:81\n",
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
                    "builtin#replicate_f64.replicate_61160 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->builtinzhreplicate_f64zireplicate_61160_runs,
                    (long) ctx->builtinzhreplicate_f64zireplicate_61160_total_runtime /
                    (ctx->builtinzhreplicate_f64zireplicate_61160_runs !=
                     0 ? ctx->builtinzhreplicate_f64zireplicate_61160_runs : 1),
                    (long) ctx->builtinzhreplicate_f64zireplicate_61160_total_runtime);
        ctx->total_runtime +=
            ctx->builtinzhreplicate_f64zireplicate_61160_total_runtime;
        ctx->total_runs += ctx->builtinzhreplicate_f64zireplicate_61160_runs;
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
                    "integrate_tke.segmap_49654            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_49654_runs,
                    (long) ctx->integrate_tkezisegmap_49654_total_runtime /
                    (ctx->integrate_tkezisegmap_49654_runs !=
                     0 ? ctx->integrate_tkezisegmap_49654_runs : 1),
                    (long) ctx->integrate_tkezisegmap_49654_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_49654_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_49654_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_52110            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_52110_runs,
                    (long) ctx->integrate_tkezisegmap_52110_total_runtime /
                    (ctx->integrate_tkezisegmap_52110_runs !=
                     0 ? ctx->integrate_tkezisegmap_52110_runs : 1),
                    (long) ctx->integrate_tkezisegmap_52110_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_52110_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_52110_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_52150            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_52150_runs,
                    (long) ctx->integrate_tkezisegmap_52150_total_runtime /
                    (ctx->integrate_tkezisegmap_52150_runs !=
                     0 ? ctx->integrate_tkezisegmap_52150_runs : 1),
                    (long) ctx->integrate_tkezisegmap_52150_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_52150_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_52150_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_52228            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_52228_runs,
                    (long) ctx->integrate_tkezisegmap_52228_total_runtime /
                    (ctx->integrate_tkezisegmap_52228_runs !=
                     0 ? ctx->integrate_tkezisegmap_52228_runs : 1),
                    (long) ctx->integrate_tkezisegmap_52228_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_52228_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_52228_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_52305            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_52305_runs,
                    (long) ctx->integrate_tkezisegmap_52305_total_runtime /
                    (ctx->integrate_tkezisegmap_52305_runs !=
                     0 ? ctx->integrate_tkezisegmap_52305_runs : 1),
                    (long) ctx->integrate_tkezisegmap_52305_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_52305_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_52305_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_52597            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_52597_runs,
                    (long) ctx->integrate_tkezisegmap_52597_total_runtime /
                    (ctx->integrate_tkezisegmap_52597_runs !=
                     0 ? ctx->integrate_tkezisegmap_52597_runs : 1),
                    (long) ctx->integrate_tkezisegmap_52597_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_52597_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_52597_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_54402            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_54402_runs,
                    (long) ctx->integrate_tkezisegmap_54402_total_runtime /
                    (ctx->integrate_tkezisegmap_54402_runs !=
                     0 ? ctx->integrate_tkezisegmap_54402_runs : 1),
                    (long) ctx->integrate_tkezisegmap_54402_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_54402_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_54402_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_54471            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_54471_runs,
                    (long) ctx->integrate_tkezisegmap_54471_total_runtime /
                    (ctx->integrate_tkezisegmap_54471_runs !=
                     0 ? ctx->integrate_tkezisegmap_54471_runs : 1),
                    (long) ctx->integrate_tkezisegmap_54471_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_54471_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_54471_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_55217            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_55217_runs,
                    (long) ctx->integrate_tkezisegmap_55217_total_runtime /
                    (ctx->integrate_tkezisegmap_55217_runs !=
                     0 ? ctx->integrate_tkezisegmap_55217_runs : 1),
                    (long) ctx->integrate_tkezisegmap_55217_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_55217_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_55217_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_55432            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_55432_runs,
                    (long) ctx->integrate_tkezisegmap_55432_total_runtime /
                    (ctx->integrate_tkezisegmap_55432_runs !=
                     0 ? ctx->integrate_tkezisegmap_55432_runs : 1),
                    (long) ctx->integrate_tkezisegmap_55432_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_55432_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_55432_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_57267            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_57267_runs,
                    (long) ctx->integrate_tkezisegmap_57267_total_runtime /
                    (ctx->integrate_tkezisegmap_57267_runs !=
                     0 ? ctx->integrate_tkezisegmap_57267_runs : 1),
                    (long) ctx->integrate_tkezisegmap_57267_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_57267_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_57267_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_59404            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_59404_runs,
                    (long) ctx->integrate_tkezisegmap_59404_total_runtime /
                    (ctx->integrate_tkezisegmap_59404_runs !=
                     0 ? ctx->integrate_tkezisegmap_59404_runs : 1),
                    (long) ctx->integrate_tkezisegmap_59404_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_59404_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_59404_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_60201            ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_60201_runs,
                    (long) ctx->integrate_tkezisegmap_60201_total_runtime /
                    (ctx->integrate_tkezisegmap_60201_runs !=
                     0 ? ctx->integrate_tkezisegmap_60201_runs : 1),
                    (long) ctx->integrate_tkezisegmap_60201_total_runtime);
        ctx->total_runtime += ctx->integrate_tkezisegmap_60201_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_60201_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_intragroup_50310 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_intragroup_50310_runs,
                    (long) ctx->integrate_tkezisegmap_intragroup_50310_total_runtime /
                    (ctx->integrate_tkezisegmap_intragroup_50310_runs !=
                     0 ? ctx->integrate_tkezisegmap_intragroup_50310_runs : 1),
                    (long) ctx->integrate_tkezisegmap_intragroup_50310_total_runtime);
        ctx->total_runtime +=
            ctx->integrate_tkezisegmap_intragroup_50310_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_intragroup_50310_runs;
        str_builder(&builder,
                    "integrate_tke.segmap_intragroup_51504 ran %5d times; avg: %8ldus; total: %8ldus\n",
                    ctx->integrate_tkezisegmap_intragroup_51504_runs,
                    (long) ctx->integrate_tkezisegmap_intragroup_51504_total_runtime /
                    (ctx->integrate_tkezisegmap_intragroup_51504_runs !=
                     0 ? ctx->integrate_tkezisegmap_intragroup_51504_runs : 1),
                    (long) ctx->integrate_tkezisegmap_intragroup_51504_total_runtime);
        ctx->total_runtime +=
            ctx->integrate_tkezisegmap_intragroup_51504_total_runtime;
        ctx->total_runs += ctx->integrate_tkezisegmap_intragroup_51504_runs;
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
                                         struct memblock_device mem_61156,
                                         int32_t num_elems_61157,
                                         double val_61158);
static int futrts_integrate_tke(struct futhark_context *ctx,
                                struct memblock_device *out_mem_p_61299,
                                int64_t *out_out_arrsizze_61300,
                                int64_t *out_out_arrsizze_61301,
                                int64_t *out_out_arrsizze_61302,
                                struct memblock_device *out_mem_p_61303,
                                int64_t *out_out_arrsizze_61304,
                                int64_t *out_out_arrsizze_61305,
                                int64_t *out_out_arrsizze_61306,
                                struct memblock_device *out_mem_p_61307,
                                int64_t *out_out_arrsizze_61308,
                                int64_t *out_out_arrsizze_61309,
                                int64_t *out_out_arrsizze_61310,
                                struct memblock_device *out_mem_p_61311,
                                int64_t *out_out_arrsizze_61312,
                                int64_t *out_out_arrsizze_61313,
                                int64_t *out_out_arrsizze_61314,
                                struct memblock_device *out_mem_p_61315,
                                int64_t *out_out_arrsizze_61316,
                                int64_t *out_out_arrsizze_61317,
                                int64_t *out_out_arrsizze_61318,
                                struct memblock_device *out_mem_p_61319,
                                int64_t *out_out_arrsizze_61320,
                                int64_t *out_out_arrsizze_61321,
                                int64_t *out_out_arrsizze_61322,
                                struct memblock_device *out_mem_p_61323,
                                int64_t *out_out_arrsizze_61324,
                                int64_t *out_out_arrsizze_61325,
                                struct memblock_device tketau_mem_60702,
                                struct memblock_device tketaup1_mem_60703,
                                struct memblock_device tketaum1_mem_60704,
                                struct memblock_device dtketau_mem_60705,
                                struct memblock_device dtketaup1_mem_60706,
                                struct memblock_device dtketaum1_mem_60707,
                                struct memblock_device utau_mem_60708,
                                struct memblock_device vtau_mem_60709,
                                struct memblock_device wtau_mem_60710,
                                struct memblock_device maskU_mem_60711,
                                struct memblock_device maskV_mem_60712,
                                struct memblock_device maskW_mem_60713,
                                struct memblock_device dxt_mem_60714,
                                struct memblock_device dxu_mem_60715,
                                struct memblock_device dyt_mem_60716,
                                struct memblock_device dyu_mem_60717,
                                struct memblock_device dzzt_mem_60718,
                                struct memblock_device dzzw_mem_60719,
                                struct memblock_device cost_mem_60720,
                                struct memblock_device cosu_mem_60721,
                                struct memblock_device kbot_mem_60722,
                                struct memblock_device kappaM_mem_60723,
                                struct memblock_device mxl_mem_60724,
                                struct memblock_device forc_mem_60725,
                                struct memblock_device forc_tke_surface_mem_60726,
                                int64_t xdim_48112, int64_t ydim_48113,
                                int64_t zzdim_48114, int64_t xdim_48115,
                                int64_t ydim_48116, int64_t zzdim_48117,
                                int64_t xdim_48118, int64_t ydim_48119,
                                int64_t zzdim_48120, int64_t xdim_48121,
                                int64_t ydim_48122, int64_t zzdim_48123,
                                int64_t xdim_48124, int64_t ydim_48125,
                                int64_t zzdim_48126, int64_t xdim_48127,
                                int64_t ydim_48128, int64_t zzdim_48129,
                                int64_t xdim_48130, int64_t ydim_48131,
                                int64_t zzdim_48132, int64_t xdim_48133,
                                int64_t ydim_48134, int64_t zzdim_48135,
                                int64_t xdim_48136, int64_t ydim_48137,
                                int64_t zzdim_48138, int64_t xdim_48139,
                                int64_t ydim_48140, int64_t zzdim_48141,
                                int64_t xdim_48142, int64_t ydim_48143,
                                int64_t zzdim_48144, int64_t xdim_48145,
                                int64_t ydim_48146, int64_t zzdim_48147,
                                int64_t xdim_48148, int64_t xdim_48149,
                                int64_t ydim_48150, int64_t ydim_48151,
                                int64_t zzdim_48152, int64_t zzdim_48153,
                                int64_t ydim_48154, int64_t ydim_48155,
                                int64_t xdim_48156, int64_t ydim_48157,
                                int64_t xdim_48158, int64_t ydim_48159,
                                int64_t zzdim_48160, int64_t xdim_48161,
                                int64_t ydim_48162, int64_t zzdim_48163,
                                int64_t xdim_48164, int64_t ydim_48165,
                                int64_t zzdim_48166, int64_t xdim_48167,
                                int64_t ydim_48168);
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
                    const size_t global_work_sizze_61274[3] =
                                 {(size_t) sdiv_up32(x_elems_5, 16) *
                                  (size_t) 16,
                                  (size_t) sdiv_up32(sdiv_up32(y_elems_6,
                                                               muly_8), 16) *
                                  (size_t) 16, (size_t) num_arrays_4 *
                                  (size_t) 1};
                    const size_t local_work_sizze_61278[3] = {16, 16, 1};
                    int64_t time_start_61275 = 0, time_end_61276 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "gpu_map_transpose_f64_low_width");
                        fprintf(stderr, "%zu", global_work_sizze_61274[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_61274[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_61274[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_61278[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_61278[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_61278[2]);
                        fprintf(stderr,
                                "]; local memory parameters sum to %d bytes.\n",
                                (int) (0 + 2176));
                        time_start_61275 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->gpu_map_transpose_f64_low_width,
                                                                    3, NULL,
                                                                    global_work_sizze_61274,
                                                                    local_work_sizze_61278,
                                                                    0, NULL,
                                                                    ctx->profiling_paused ||
                                                                    !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                              &ctx->gpu_map_transpose_f64_low_width_runs,
                                                                                                              &ctx->gpu_map_transpose_f64_low_width_total_runtime)));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_61276 = get_wall_time();
                        
                        long time_diff_61277 = time_end_61276 -
                             time_start_61275;
                        
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "gpu_map_transpose_f64_low_width",
                                time_diff_61277);
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
                        const size_t global_work_sizze_61279[3] =
                                     {(size_t) sdiv_up32(sdiv_up32(x_elems_5,
                                                                   mulx_7),
                                                         16) * (size_t) 16,
                                      (size_t) sdiv_up32(y_elems_6, 16) *
                                      (size_t) 16, (size_t) num_arrays_4 *
                                      (size_t) 1};
                        const size_t local_work_sizze_61283[3] = {16, 16, 1};
                        int64_t time_start_61280 = 0, time_end_61281 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "gpu_map_transpose_f64_low_height");
                            fprintf(stderr, "%zu", global_work_sizze_61279[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_61279[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_61279[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_61283[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_61283[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_61283[2]);
                            fprintf(stderr,
                                    "]; local memory parameters sum to %d bytes.\n",
                                    (int) (0 + 2176));
                            time_start_61280 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->gpu_map_transpose_f64_low_height,
                                                                        3, NULL,
                                                                        global_work_sizze_61279,
                                                                        local_work_sizze_61283,
                                                                        0, NULL,
                                                                        ctx->profiling_paused ||
                                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                  &ctx->gpu_map_transpose_f64_low_height_runs,
                                                                                                                  &ctx->gpu_map_transpose_f64_low_height_total_runtime)));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_61281 = get_wall_time();
                            
                            long time_diff_61282 = time_end_61281 -
                                 time_start_61280;
                            
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "gpu_map_transpose_f64_low_height",
                                    time_diff_61282);
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
                            const size_t global_work_sizze_61284[1] =
                                         {(size_t) sdiv_up32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256) *
                                         (size_t) 256};
                            const size_t local_work_sizze_61288[1] = {256};
                            int64_t time_start_61285 = 0, time_end_61286 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "gpu_map_transpose_f64_small");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_61284[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_61288[0]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 1));
                                time_start_61285 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->gpu_map_transpose_f64_small,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_61284,
                                                                            local_work_sizze_61288,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ||
                                                                            !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                      &ctx->gpu_map_transpose_f64_small_runs,
                                                                                                                      &ctx->gpu_map_transpose_f64_small_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_61286 = get_wall_time();
                                
                                long time_diff_61287 = time_end_61286 -
                                     time_start_61285;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "gpu_map_transpose_f64_small",
                                        time_diff_61287);
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
                            const size_t global_work_sizze_61289[3] =
                                         {(size_t) sdiv_up32(x_elems_5, 32) *
                                          (size_t) 32,
                                          (size_t) sdiv_up32(y_elems_6, 32) *
                                          (size_t) 8, (size_t) num_arrays_4 *
                                          (size_t) 1};
                            const size_t local_work_sizze_61293[3] = {32, 8, 1};
                            int64_t time_start_61290 = 0, time_end_61291 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "gpu_map_transpose_f64");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_61289[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_61289[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_61289[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_61293[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_61293[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_61293[2]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 8448));
                                time_start_61290 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->gpu_map_transpose_f64,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_61289,
                                                                            local_work_sizze_61293,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ||
                                                                            !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                      &ctx->gpu_map_transpose_f64_runs,
                                                                                                                      &ctx->gpu_map_transpose_f64_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_61291 = get_wall_time();
                                
                                long time_diff_61292 = time_end_61291 -
                                     time_start_61290;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "gpu_map_transpose_f64",
                                        time_diff_61292);
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
                                         struct memblock_device mem_61156,
                                         int32_t num_elems_61157,
                                         double val_61158)
{
    (void) ctx;
    
    int err = 0;
    int64_t group_sizze_61163;
    
    group_sizze_61163 = ctx->sizes.builtinzhreplicate_f64zigroup_sizze_61163;
    
    int64_t num_groups_61164;
    
    num_groups_61164 = sdiv_up64(num_elems_61157, group_sizze_61163);
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->builtinzhreplicate_f64zireplicate_61160,
                                            0, sizeof(mem_61156.mem),
                                            &mem_61156.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->builtinzhreplicate_f64zireplicate_61160,
                                            1, sizeof(num_elems_61157),
                                            &num_elems_61157));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->builtinzhreplicate_f64zireplicate_61160,
                                            2, sizeof(val_61158), &val_61158));
    if (1 * ((size_t) num_groups_61164 * (size_t) group_sizze_61163) != 0) {
        const size_t global_work_sizze_61294[1] = {(size_t) num_groups_61164 *
                     (size_t) group_sizze_61163};
        const size_t local_work_sizze_61298[1] = {group_sizze_61163};
        int64_t time_start_61295 = 0, time_end_61296 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "builtin#replicate_f64.replicate_61160");
            fprintf(stderr, "%zu", global_work_sizze_61294[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61298[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61295 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->builtinzhreplicate_f64zireplicate_61160,
                                                        1, NULL,
                                                        global_work_sizze_61294,
                                                        local_work_sizze_61298,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->builtinzhreplicate_f64zireplicate_61160_runs,
                                                                                                  &ctx->builtinzhreplicate_f64zireplicate_61160_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61296 = get_wall_time();
            
            long time_diff_61297 = time_end_61296 - time_start_61295;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "builtin#replicate_f64.replicate_61160", time_diff_61297);
        }
    }
    
  cleanup:
    { }
    return err;
}
static int futrts_integrate_tke(struct futhark_context *ctx,
                                struct memblock_device *out_mem_p_61299,
                                int64_t *out_out_arrsizze_61300,
                                int64_t *out_out_arrsizze_61301,
                                int64_t *out_out_arrsizze_61302,
                                struct memblock_device *out_mem_p_61303,
                                int64_t *out_out_arrsizze_61304,
                                int64_t *out_out_arrsizze_61305,
                                int64_t *out_out_arrsizze_61306,
                                struct memblock_device *out_mem_p_61307,
                                int64_t *out_out_arrsizze_61308,
                                int64_t *out_out_arrsizze_61309,
                                int64_t *out_out_arrsizze_61310,
                                struct memblock_device *out_mem_p_61311,
                                int64_t *out_out_arrsizze_61312,
                                int64_t *out_out_arrsizze_61313,
                                int64_t *out_out_arrsizze_61314,
                                struct memblock_device *out_mem_p_61315,
                                int64_t *out_out_arrsizze_61316,
                                int64_t *out_out_arrsizze_61317,
                                int64_t *out_out_arrsizze_61318,
                                struct memblock_device *out_mem_p_61319,
                                int64_t *out_out_arrsizze_61320,
                                int64_t *out_out_arrsizze_61321,
                                int64_t *out_out_arrsizze_61322,
                                struct memblock_device *out_mem_p_61323,
                                int64_t *out_out_arrsizze_61324,
                                int64_t *out_out_arrsizze_61325,
                                struct memblock_device tketau_mem_60702,
                                struct memblock_device tketaup1_mem_60703,
                                struct memblock_device tketaum1_mem_60704,
                                struct memblock_device dtketau_mem_60705,
                                struct memblock_device dtketaup1_mem_60706,
                                struct memblock_device dtketaum1_mem_60707,
                                struct memblock_device utau_mem_60708,
                                struct memblock_device vtau_mem_60709,
                                struct memblock_device wtau_mem_60710,
                                struct memblock_device maskU_mem_60711,
                                struct memblock_device maskV_mem_60712,
                                struct memblock_device maskW_mem_60713,
                                struct memblock_device dxt_mem_60714,
                                struct memblock_device dxu_mem_60715,
                                struct memblock_device dyt_mem_60716,
                                struct memblock_device dyu_mem_60717,
                                struct memblock_device dzzt_mem_60718,
                                struct memblock_device dzzw_mem_60719,
                                struct memblock_device cost_mem_60720,
                                struct memblock_device cosu_mem_60721,
                                struct memblock_device kbot_mem_60722,
                                struct memblock_device kappaM_mem_60723,
                                struct memblock_device mxl_mem_60724,
                                struct memblock_device forc_mem_60725,
                                struct memblock_device forc_tke_surface_mem_60726,
                                int64_t xdim_48112, int64_t ydim_48113,
                                int64_t zzdim_48114, int64_t xdim_48115,
                                int64_t ydim_48116, int64_t zzdim_48117,
                                int64_t xdim_48118, int64_t ydim_48119,
                                int64_t zzdim_48120, int64_t xdim_48121,
                                int64_t ydim_48122, int64_t zzdim_48123,
                                int64_t xdim_48124, int64_t ydim_48125,
                                int64_t zzdim_48126, int64_t xdim_48127,
                                int64_t ydim_48128, int64_t zzdim_48129,
                                int64_t xdim_48130, int64_t ydim_48131,
                                int64_t zzdim_48132, int64_t xdim_48133,
                                int64_t ydim_48134, int64_t zzdim_48135,
                                int64_t xdim_48136, int64_t ydim_48137,
                                int64_t zzdim_48138, int64_t xdim_48139,
                                int64_t ydim_48140, int64_t zzdim_48141,
                                int64_t xdim_48142, int64_t ydim_48143,
                                int64_t zzdim_48144, int64_t xdim_48145,
                                int64_t ydim_48146, int64_t zzdim_48147,
                                int64_t xdim_48148, int64_t xdim_48149,
                                int64_t ydim_48150, int64_t ydim_48151,
                                int64_t zzdim_48152, int64_t zzdim_48153,
                                int64_t ydim_48154, int64_t ydim_48155,
                                int64_t xdim_48156, int64_t ydim_48157,
                                int64_t xdim_48158, int64_t ydim_48159,
                                int64_t zzdim_48160, int64_t xdim_48161,
                                int64_t ydim_48162, int64_t zzdim_48163,
                                int64_t xdim_48164, int64_t ydim_48165,
                                int64_t zzdim_48166, int64_t xdim_48167,
                                int64_t ydim_48168)
{
    (void) ctx;
    
    int err = 0;
    struct memblock_device out_mem_61123;
    
    out_mem_61123.references = NULL;
    
    int64_t out_arrsizze_61124;
    int64_t out_arrsizze_61125;
    int64_t out_arrsizze_61126;
    struct memblock_device out_mem_61127;
    
    out_mem_61127.references = NULL;
    
    int64_t out_arrsizze_61128;
    int64_t out_arrsizze_61129;
    int64_t out_arrsizze_61130;
    struct memblock_device out_mem_61131;
    
    out_mem_61131.references = NULL;
    
    int64_t out_arrsizze_61132;
    int64_t out_arrsizze_61133;
    int64_t out_arrsizze_61134;
    struct memblock_device out_mem_61135;
    
    out_mem_61135.references = NULL;
    
    int64_t out_arrsizze_61136;
    int64_t out_arrsizze_61137;
    int64_t out_arrsizze_61138;
    struct memblock_device out_mem_61139;
    
    out_mem_61139.references = NULL;
    
    int64_t out_arrsizze_61140;
    int64_t out_arrsizze_61141;
    int64_t out_arrsizze_61142;
    struct memblock_device out_mem_61143;
    
    out_mem_61143.references = NULL;
    
    int64_t out_arrsizze_61144;
    int64_t out_arrsizze_61145;
    int64_t out_arrsizze_61146;
    struct memblock_device out_mem_61147;
    
    out_mem_61147.references = NULL;
    
    int64_t out_arrsizze_61148;
    int64_t out_arrsizze_61149;
    bool dim_match_48194 = xdim_48112 == xdim_48115;
    bool dim_match_48195 = ydim_48113 == ydim_48116;
    bool dim_match_48196 = zzdim_48114 == zzdim_48117;
    bool y_48197 = dim_match_48194 && dim_match_48196;
    bool match_48198 = dim_match_48195 && y_48197;
    bool empty_or_match_cert_48199;
    
    if (!match_48198) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48200 = xdim_48112 == xdim_48118;
    bool dim_match_48201 = ydim_48113 == ydim_48119;
    bool dim_match_48202 = zzdim_48114 == zzdim_48120;
    bool y_48203 = dim_match_48200 && dim_match_48202;
    bool match_48204 = dim_match_48201 && y_48203;
    bool empty_or_match_cert_48205;
    
    if (!match_48204) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48207 = xdim_48112 == xdim_48121;
    bool dim_match_48208 = ydim_48113 == ydim_48122;
    bool dim_match_48209 = zzdim_48114 == zzdim_48123;
    bool y_48210 = dim_match_48207 && dim_match_48209;
    bool match_48211 = dim_match_48208 && y_48210;
    bool empty_or_match_cert_48212;
    
    if (!match_48211) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48213 = xdim_48112 == xdim_48124;
    bool dim_match_48214 = ydim_48113 == ydim_48125;
    bool dim_match_48215 = zzdim_48114 == zzdim_48126;
    bool y_48216 = dim_match_48213 && dim_match_48215;
    bool match_48217 = dim_match_48214 && y_48216;
    bool empty_or_match_cert_48218;
    
    if (!match_48217) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48220 = xdim_48112 == xdim_48127;
    bool dim_match_48221 = ydim_48113 == ydim_48128;
    bool dim_match_48222 = zzdim_48114 == zzdim_48129;
    bool y_48223 = dim_match_48220 && dim_match_48222;
    bool match_48224 = dim_match_48221 && y_48223;
    bool empty_or_match_cert_48225;
    
    if (!match_48224) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48227 = xdim_48112 == xdim_48130;
    bool dim_match_48228 = ydim_48113 == ydim_48131;
    bool dim_match_48229 = zzdim_48114 == zzdim_48132;
    bool y_48230 = dim_match_48227 && dim_match_48229;
    bool match_48231 = dim_match_48228 && y_48230;
    bool empty_or_match_cert_48232;
    
    if (!match_48231) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48233 = xdim_48112 == xdim_48133;
    bool dim_match_48234 = ydim_48113 == ydim_48134;
    bool dim_match_48235 = zzdim_48114 == zzdim_48135;
    bool y_48236 = dim_match_48233 && dim_match_48235;
    bool match_48237 = dim_match_48234 && y_48236;
    bool empty_or_match_cert_48238;
    
    if (!match_48237) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48239 = xdim_48112 == xdim_48136;
    bool dim_match_48240 = ydim_48113 == ydim_48137;
    bool dim_match_48241 = zzdim_48114 == zzdim_48138;
    bool y_48242 = dim_match_48239 && dim_match_48241;
    bool match_48243 = dim_match_48240 && y_48242;
    bool empty_or_match_cert_48244;
    
    if (!match_48243) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48245 = xdim_48112 == xdim_48139;
    bool dim_match_48246 = ydim_48113 == ydim_48140;
    bool dim_match_48247 = zzdim_48114 == zzdim_48141;
    bool y_48248 = dim_match_48245 && dim_match_48247;
    bool match_48249 = dim_match_48246 && y_48248;
    bool empty_or_match_cert_48250;
    
    if (!match_48249) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48251 = xdim_48112 == xdim_48142;
    bool dim_match_48252 = ydim_48113 == ydim_48143;
    bool dim_match_48253 = zzdim_48114 == zzdim_48144;
    bool y_48254 = dim_match_48251 && dim_match_48253;
    bool match_48255 = dim_match_48252 && y_48254;
    bool empty_or_match_cert_48256;
    
    if (!match_48255) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48257 = xdim_48112 == xdim_48145;
    bool dim_match_48258 = ydim_48113 == ydim_48146;
    bool dim_match_48259 = zzdim_48114 == zzdim_48147;
    bool y_48260 = dim_match_48257 && dim_match_48259;
    bool match_48261 = dim_match_48258 && y_48260;
    bool empty_or_match_cert_48262;
    
    if (!match_48261) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48263 = xdim_48112 == xdim_48148;
    bool empty_or_match_cert_48264;
    
    if (!dim_match_48263) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48265 = xdim_48112 == xdim_48149;
    bool empty_or_match_cert_48266;
    
    if (!dim_match_48265) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48267 = ydim_48113 == ydim_48150;
    bool empty_or_match_cert_48268;
    
    if (!dim_match_48267) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48269 = ydim_48113 == ydim_48151;
    bool empty_or_match_cert_48270;
    
    if (!dim_match_48269) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48271 = zzdim_48114 == zzdim_48152;
    bool empty_or_match_cert_48272;
    
    if (!dim_match_48271) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48273 = zzdim_48114 == zzdim_48153;
    bool empty_or_match_cert_48274;
    
    if (!dim_match_48273) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48275 = ydim_48113 == ydim_48154;
    bool empty_or_match_cert_48276;
    
    if (!dim_match_48275) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48277 = ydim_48113 == ydim_48155;
    bool empty_or_match_cert_48278;
    
    if (!dim_match_48277) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48279 = xdim_48112 == xdim_48156;
    bool dim_match_48280 = ydim_48113 == ydim_48157;
    bool match_48281 = dim_match_48279 && dim_match_48280;
    bool empty_or_match_cert_48282;
    
    if (!match_48281) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48283 = xdim_48112 == xdim_48158;
    bool dim_match_48284 = ydim_48113 == ydim_48159;
    bool dim_match_48285 = zzdim_48114 == zzdim_48160;
    bool y_48286 = dim_match_48283 && dim_match_48285;
    bool match_48287 = dim_match_48284 && y_48286;
    bool empty_or_match_cert_48288;
    
    if (!match_48287) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48289 = xdim_48112 == xdim_48161;
    bool dim_match_48290 = ydim_48113 == ydim_48162;
    bool dim_match_48291 = zzdim_48114 == zzdim_48163;
    bool y_48292 = dim_match_48289 && dim_match_48291;
    bool match_48293 = dim_match_48290 && y_48292;
    bool empty_or_match_cert_48294;
    
    if (!match_48293) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48295 = xdim_48112 == xdim_48164;
    bool dim_match_48296 = ydim_48113 == ydim_48165;
    bool dim_match_48297 = zzdim_48114 == zzdim_48166;
    bool y_48298 = dim_match_48295 && dim_match_48297;
    bool match_48299 = dim_match_48296 && y_48298;
    bool empty_or_match_cert_48300;
    
    if (!match_48299) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool dim_match_48301 = xdim_48112 == xdim_48167;
    bool dim_match_48302 = ydim_48113 == ydim_48168;
    bool match_48303 = dim_match_48301 && dim_match_48302;
    bool empty_or_match_cert_48304;
    
    if (!match_48303) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "function arguments of wrong shape",
                               "-> #0  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    int64_t y_48308 = sub64(xdim_48112, 2);
    int64_t y_48309 = sub64(ydim_48113, 2);
    int64_t y_48310 = sub64(zzdim_48114, 1);
    int64_t y_49902 = ydim_48113 * zzdim_48114;
    int64_t nest_sizze_49903 = xdim_48112 * y_49902;
    int64_t segmap_group_sizze_49904;
    
    segmap_group_sizze_49904 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_49658;
    
    int64_t segmap_usable_groups_49905 = sdiv_up64(nest_sizze_49903,
                                                   segmap_group_sizze_49904);
    int64_t binop_x_60729 = xdim_48112 * ydim_48113;
    int64_t binop_x_60730 = zzdim_48114 * binop_x_60729;
    int64_t bytes_60728 = 8 * binop_x_60730;
    struct memblock_device mem_60731;
    
    mem_60731.references = NULL;
    if (memblock_alloc_device(ctx, &mem_60731, bytes_60728, "mem_60731")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_60735;
    
    mem_60735.references = NULL;
    if (memblock_alloc_device(ctx, &mem_60735, bytes_60728, "mem_60735")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 3,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 4,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 5,
                                            sizeof(zzdim_48114), &zzdim_48114));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 6,
                                            sizeof(ydim_48159), &ydim_48159));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 7,
                                            sizeof(zzdim_48160), &zzdim_48160));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 8,
                                            sizeof(y_48308), &y_48308));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654, 9,
                                            sizeof(y_48309), &y_48309));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654,
                                            10, sizeof(y_48310), &y_48310));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654,
                                            11, sizeof(tketau_mem_60702.mem),
                                            &tketau_mem_60702.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654,
                                            12, sizeof(dzzt_mem_60718.mem),
                                            &dzzt_mem_60718.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654,
                                            13, sizeof(kappaM_mem_60723.mem),
                                            &kappaM_mem_60723.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654,
                                            14, sizeof(mem_60731.mem),
                                            &mem_60731.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_49654,
                                            15, sizeof(mem_60735.mem),
                                            &mem_60735.mem));
    if (1 * ((size_t) segmap_usable_groups_49905 *
             (size_t) segmap_group_sizze_49904) != 0) {
        const size_t global_work_sizze_61326[1] =
                     {(size_t) segmap_usable_groups_49905 *
                     (size_t) segmap_group_sizze_49904};
        const size_t local_work_sizze_61330[1] = {segmap_group_sizze_49904};
        int64_t time_start_61327 = 0, time_end_61328 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_49654");
            fprintf(stderr, "%zu", global_work_sizze_61326[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61330[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61327 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_49654,
                                                        1, NULL,
                                                        global_work_sizze_61326,
                                                        local_work_sizze_61330,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_49654_runs,
                                                                                                  &ctx->integrate_tkezisegmap_49654_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61328 = get_wall_time();
            
            long time_diff_61329 = time_end_61328 - time_start_61327;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_49654", time_diff_61329);
        }
    }
    ctx->failure_is_an_option = 1;
    
    bool bounds_invalid_upwards_48369 = slt64(zzdim_48114, 1);
    bool valid_48370 = !bounds_invalid_upwards_48369;
    bool range_valid_c_48371;
    
    if (!valid_48370) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", 1, "..<", zzdim_48114, " is invalid.",
                               "-> #0  tke_baseline.fut:7:30-34\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &mem_60735, "mem_60735") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_60731, "mem_60731") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool x_48372 = sle64(0, y_48310);
    bool y_48373 = slt64(y_48310, zzdim_48114);
    bool bounds_check_48374 = x_48372 && y_48373;
    bool index_certs_48375;
    
    if (!bounds_check_48374) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Index [", y_48310,
                               "] out of bounds for array of shape [",
                               zzdim_48114, "].",
                               "-> #0  tke_baseline.fut:13:24-35\n   #1  tke_baseline.fut:22:13-31\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke_baseline.fut:21:9-23:21\n   #5  /prelude/soacs.fut:72:25-33\n   #6  /prelude/soacs.fut:72:3-53\n   #7  tke_baseline.fut:20:4-24:12\n   #8  tke_baseline.fut:198:15-45\n   #9  tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &mem_60735, "mem_60735") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_60731, "mem_60731") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    bool empty_slice_48376 = y_48310 == 0;
    int64_t m_48377 = sub64(y_48310, 1);
    bool zzero_leq_i_p_m_t_s_48378 = sle64(0, m_48377);
    bool i_p_m_t_s_leq_w_48379 = slt64(m_48377, zzdim_48114);
    bool y_48380 = zzero_leq_i_p_m_t_s_48378 && i_p_m_t_s_leq_w_48379;
    bool y_48381 = x_48372 && y_48380;
    bool ok_or_empty_48382 = empty_slice_48376 || y_48381;
    bool index_certs_48383;
    
    if (!ok_or_empty_48382) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Index [", 0, ":", y_48310,
                               "] out of bounds for array of shape [",
                               zzdim_48114, "].",
                               "-> #0  /prelude/array.fut:24:29-36\n   #1  tke_baseline.fut:14:24-39\n   #2  tke_baseline.fut:22:13-31\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke_baseline.fut:21:9-23:21\n   #6  /prelude/soacs.fut:72:25-33\n   #7  /prelude/soacs.fut:72:3-53\n   #8  tke_baseline.fut:20:4-24:12\n   #9  tke_baseline.fut:198:15-45\n   #10 tke_baseline.fut:62:1-338:81\n");
        if (memblock_unref_device(ctx, &mem_60735, "mem_60735") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_60731, "mem_60731") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
            return 1;
        return 1;
    }
    
    int64_t y_50307 = smin64(ydim_48113, y_49902);
    int64_t intra_avail_par_50308 = smin64(ydim_48113, y_50307);
    int64_t computed_group_sizze_49958 = smax64(ydim_48113, y_49902);
    int64_t max_group_sizze_50922;
    
    max_group_sizze_50922 = ctx->opencl.max_group_size;
    
    bool fits_50923 = sle64(computed_group_sizze_49958, max_group_sizze_50922);
    bool suff_intra_par_50921;
    
    suff_intra_par_50921 = ctx->sizes.integrate_tkezisuff_intra_par_4 <=
        intra_avail_par_50308;
    if (ctx->logging)
        fprintf(stderr, "Compared %s <= %d.\n",
                "integrate_tke.suff_intra_par_4", intra_avail_par_50308);
    
    bool intra_suff_and_fits_50924 = suff_intra_par_50921 && fits_50923;
    bool fits_53497 = sle64(zzdim_48114, max_group_sizze_50922);
    bool suff_intra_par_53499;
    
    suff_intra_par_53499 = ctx->sizes.integrate_tkezisuff_intra_par_6 <=
        zzdim_48114;
    if (ctx->logging)
        fprintf(stderr, "Compared %s <= %d.\n",
                "integrate_tke.suff_intra_par_6", zzdim_48114);
    
    bool intra_suff_and_fits_53500 = fits_53497 && suff_intra_par_53499;
    int64_t segmap_group_sizze_53821;
    
    segmap_group_sizze_53821 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_52601;
    
    int64_t segmap_group_sizze_54064;
    
    segmap_group_sizze_54064 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_52309;
    
    int64_t segmap_group_sizze_54089;
    
    segmap_group_sizze_54089 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_52231;
    
    int64_t num_groups_54090;
    int32_t max_num_groups_61155;
    
    max_num_groups_61155 = ctx->sizes.integrate_tkezisegmap_num_groups_52233;
    num_groups_54090 = sext_i64_i32(smax64(1, smin64(sdiv_up64(binop_x_60729,
                                                               segmap_group_sizze_54089),
                                                     sext_i32_i64(max_num_groups_61155))));
    
    struct memblock_device mem_60739;
    
    mem_60739.references = NULL;
    if (memblock_alloc_device(ctx, &mem_60739, bytes_60728, "mem_60739")) {
        err = 1;
        goto cleanup;
    }
    if (futrts_builtinzhreplicate_f64(ctx, mem_60739, xdim_48112 * ydim_48113 *
                                      zzdim_48114, 0.0) != 0) {
        err = 1;
        goto cleanup;
    }
    
    int64_t segmap_group_sizze_54157;
    
    segmap_group_sizze_54157 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_52154;
    
    int64_t segmap_group_sizze_54168;
    
    segmap_group_sizze_54168 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_52113;
    
    int64_t num_groups_54169;
    int32_t max_num_groups_61165;
    
    max_num_groups_61165 = ctx->sizes.integrate_tkezisegmap_num_groups_52115;
    num_groups_54169 = sext_i64_i32(smax64(1, smin64(sdiv_up64(binop_x_60729,
                                                               segmap_group_sizze_54168),
                                                     sext_i32_i64(max_num_groups_61165))));
    
    int64_t bytes_60742 = 8 * y_49902;
    int64_t bytes_60778 = 8 * zzdim_48114;
    int64_t binop_x_60918 = xdim_48112 * zzdim_48114;
    int64_t binop_x_60919 = ydim_48113 * binop_x_60918;
    int64_t bytes_60917 = 8 * binop_x_60919;
    int64_t num_threads_61110 = segmap_group_sizze_54089 * num_groups_54090;
    int64_t total_sizze_61111 = bytes_60778 * num_threads_61110;
    int64_t total_sizze_61112 = bytes_60778 * num_threads_61110;
    int64_t num_threads_61114 = segmap_group_sizze_54168 * num_groups_54169;
    int64_t total_sizze_61115 = bytes_60778 * num_threads_61114;
    struct memblock_device lifted_11_map_res_mem_61024;
    
    lifted_11_map_res_mem_61024.references = NULL;
    
    int32_t local_memory_capacity_61238;
    
    local_memory_capacity_61238 = ctx->opencl.max_local_memory;
    if (sle64(bytes_60742 + bytes_60742 + bytes_60742 + bytes_60742 +
              bytes_60742 + bytes_60742 + bytes_60742 + bytes_60742 +
              bytes_60778 + bytes_60778 + bytes_60742 + bytes_60742 +
              bytes_60778, sext_i32_i64(local_memory_capacity_61238)) &&
        intra_suff_and_fits_50924) {
        struct memblock_device mem_60833;
        
        mem_60833.references = NULL;
        if (memblock_alloc_device(ctx, &mem_60833, bytes_60728, "mem_60833")) {
            err = 1;
            goto cleanup;
        }
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                1,
                                                sizeof(ctx->failure_is_an_option),
                                                &ctx->failure_is_an_option));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                3, bytes_60778, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                4, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                5, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                6, bytes_60778, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                7, bytes_60778, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                8, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                9, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                10, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                11, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                12, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                13, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                14, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                15, bytes_60742, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                16, sizeof(xdim_48112),
                                                &xdim_48112));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                17, sizeof(ydim_48113),
                                                &ydim_48113));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                18, sizeof(zzdim_48114),
                                                &zzdim_48114));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                19, sizeof(ydim_48157),
                                                &ydim_48157));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                20, sizeof(ydim_48162),
                                                &ydim_48162));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                21, sizeof(zzdim_48163),
                                                &zzdim_48163));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                22, sizeof(ydim_48165),
                                                &ydim_48165));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                23, sizeof(zzdim_48166),
                                                &zzdim_48166));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                24, sizeof(ydim_48168),
                                                &ydim_48168));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                25, sizeof(y_48308), &y_48308));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                26, sizeof(y_48309), &y_48309));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                27, sizeof(y_48310), &y_48310));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                28, sizeof(m_48377), &m_48377));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                29,
                                                sizeof(computed_group_sizze_49958),
                                                &computed_group_sizze_49958));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                30,
                                                sizeof(tketau_mem_60702.mem),
                                                &tketau_mem_60702.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                31, sizeof(dzzw_mem_60719.mem),
                                                &dzzw_mem_60719.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                32, sizeof(kbot_mem_60722.mem),
                                                &kbot_mem_60722.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                33, sizeof(mxl_mem_60724.mem),
                                                &mxl_mem_60724.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                34, sizeof(forc_mem_60725.mem),
                                                &forc_mem_60725.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                35,
                                                sizeof(forc_tke_surface_mem_60726.mem),
                                                &forc_tke_surface_mem_60726.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                36, sizeof(mem_60731.mem),
                                                &mem_60731.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                37, sizeof(mem_60735.mem),
                                                &mem_60735.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_50310,
                                                38, sizeof(mem_60833.mem),
                                                &mem_60833.mem));
        if (1 * ((size_t) xdim_48112 * (size_t) computed_group_sizze_49958) !=
            0) {
            const size_t global_work_sizze_61331[1] = {(size_t) xdim_48112 *
                         (size_t) computed_group_sizze_49958};
            const size_t local_work_sizze_61335[1] =
                         {computed_group_sizze_49958};
            int64_t time_start_61332 = 0, time_end_61333 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "integrate_tke.segmap_intragroup_50310");
                fprintf(stderr, "%zu", global_work_sizze_61331[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_61335[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + bytes_60778 + bytes_60742 + bytes_60742 +
                               bytes_60778 + bytes_60778 + bytes_60742 +
                               bytes_60742 + bytes_60742 + bytes_60742 +
                               bytes_60742 + bytes_60742 + bytes_60742 +
                               bytes_60742));
                time_start_61332 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->integrate_tkezisegmap_intragroup_50310,
                                                            1, NULL,
                                                            global_work_sizze_61331,
                                                            local_work_sizze_61335,
                                                            0, NULL,
                                                            ctx->profiling_paused ||
                                                            !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                      &ctx->integrate_tkezisegmap_intragroup_50310_runs,
                                                                                                      &ctx->integrate_tkezisegmap_intragroup_50310_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_61333 = get_wall_time();
                
                long time_diff_61334 = time_end_61333 - time_start_61332;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "integrate_tke.segmap_intragroup_50310",
                        time_diff_61334);
            }
        }
        ctx->failure_is_an_option = 1;
        if (memblock_set_device(ctx, &lifted_11_map_res_mem_61024, &mem_60833,
                                "mem_60833") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_60833, "mem_60833") != 0)
            return 1;
    } else {
        struct memblock_device lifted_11_map_res_mem_61023;
        
        lifted_11_map_res_mem_61023.references = NULL;
        
        int32_t local_memory_capacity_61237;
        
        local_memory_capacity_61237 = ctx->opencl.max_local_memory;
        if (sle64(bytes_60778 + bytes_60778 + bytes_60778 + bytes_60778 +
                  bytes_60778 + bytes_60778 + bytes_60778,
                  sext_i32_i64(local_memory_capacity_61237)) &&
            intra_suff_and_fits_53500) {
            struct memblock_device mem_60890;
            
            mem_60890.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60890, bytes_60728,
                                      "mem_60890")) {
                err = 1;
                goto cleanup;
            }
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    3, bytes_60778, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    4, bytes_60778, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    5, bytes_60778, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    6, bytes_60778, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    7, bytes_60778, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    8, bytes_60778, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    9, bytes_60778, NULL));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    10, sizeof(xdim_48112),
                                                    &xdim_48112));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    11, sizeof(ydim_48113),
                                                    &ydim_48113));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    12, sizeof(zzdim_48114),
                                                    &zzdim_48114));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    13, sizeof(ydim_48157),
                                                    &ydim_48157));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    14, sizeof(ydim_48162),
                                                    &ydim_48162));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    15, sizeof(zzdim_48163),
                                                    &zzdim_48163));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    16, sizeof(ydim_48165),
                                                    &ydim_48165));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    17, sizeof(zzdim_48166),
                                                    &zzdim_48166));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    18, sizeof(ydim_48168),
                                                    &ydim_48168));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    19, sizeof(y_48308),
                                                    &y_48308));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    20, sizeof(y_48309),
                                                    &y_48309));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    21, sizeof(y_48310),
                                                    &y_48310));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    22, sizeof(m_48377),
                                                    &m_48377));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    23,
                                                    sizeof(tketau_mem_60702.mem),
                                                    &tketau_mem_60702.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    24,
                                                    sizeof(dzzw_mem_60719.mem),
                                                    &dzzw_mem_60719.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    25,
                                                    sizeof(kbot_mem_60722.mem),
                                                    &kbot_mem_60722.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    26,
                                                    sizeof(mxl_mem_60724.mem),
                                                    &mxl_mem_60724.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    27,
                                                    sizeof(forc_mem_60725.mem),
                                                    &forc_mem_60725.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    28,
                                                    sizeof(forc_tke_surface_mem_60726.mem),
                                                    &forc_tke_surface_mem_60726.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    29, sizeof(mem_60731.mem),
                                                    &mem_60731.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    30, sizeof(mem_60735.mem),
                                                    &mem_60735.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_intragroup_51504,
                                                    31, sizeof(mem_60890.mem),
                                                    &mem_60890.mem));
            if (1 * ((size_t) binop_x_60729 * (size_t) zzdim_48114) != 0) {
                const size_t global_work_sizze_61336[1] =
                             {(size_t) binop_x_60729 * (size_t) zzdim_48114};
                const size_t local_work_sizze_61340[1] = {zzdim_48114};
                int64_t time_start_61337 = 0, time_end_61338 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_intragroup_51504");
                    fprintf(stderr, "%zu", global_work_sizze_61336[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_61340[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) (0 + bytes_60778 + bytes_60778 + bytes_60778 +
                                   bytes_60778 + bytes_60778 + bytes_60778 +
                                   bytes_60778));
                    time_start_61337 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_intragroup_51504,
                                                                1, NULL,
                                                                global_work_sizze_61336,
                                                                local_work_sizze_61340,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_intragroup_51504_runs,
                                                                                                          &ctx->integrate_tkezisegmap_intragroup_51504_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_61338 = get_wall_time();
                    
                    long time_diff_61339 = time_end_61338 - time_start_61337;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_intragroup_51504",
                            time_diff_61339);
                }
            }
            ctx->failure_is_an_option = 1;
            if (memblock_set_device(ctx, &lifted_11_map_res_mem_61023,
                                    &mem_60890, "mem_60890") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60890, "mem_60890") != 0)
                return 1;
        } else {
            int64_t segmap_usable_groups_53822 = sdiv_up64(nest_sizze_49903,
                                                           segmap_group_sizze_53821);
            struct memblock_device mem_60895;
            
            mem_60895.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60895, bytes_60728,
                                      "mem_60895")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_60899;
            
            mem_60899.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60899, bytes_60728,
                                      "mem_60899")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_60903;
            
            mem_60903.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60903, bytes_60728,
                                      "mem_60903")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_60907;
            
            mem_60907.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60907, bytes_60728,
                                      "mem_60907")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    3, sizeof(xdim_48112),
                                                    &xdim_48112));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    4, sizeof(ydim_48113),
                                                    &ydim_48113));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    5, sizeof(zzdim_48114),
                                                    &zzdim_48114));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    6, sizeof(ydim_48157),
                                                    &ydim_48157));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    7, sizeof(ydim_48162),
                                                    &ydim_48162));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    8, sizeof(zzdim_48163),
                                                    &zzdim_48163));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    9, sizeof(ydim_48165),
                                                    &ydim_48165));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    10, sizeof(zzdim_48166),
                                                    &zzdim_48166));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    11, sizeof(ydim_48168),
                                                    &ydim_48168));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    12, sizeof(y_48308),
                                                    &y_48308));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    13, sizeof(y_48309),
                                                    &y_48309));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    14, sizeof(y_48310),
                                                    &y_48310));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    15,
                                                    sizeof(tketau_mem_60702.mem),
                                                    &tketau_mem_60702.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    16,
                                                    sizeof(dzzw_mem_60719.mem),
                                                    &dzzw_mem_60719.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    17,
                                                    sizeof(kbot_mem_60722.mem),
                                                    &kbot_mem_60722.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    18,
                                                    sizeof(mxl_mem_60724.mem),
                                                    &mxl_mem_60724.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    19,
                                                    sizeof(forc_mem_60725.mem),
                                                    &forc_mem_60725.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    20,
                                                    sizeof(forc_tke_surface_mem_60726.mem),
                                                    &forc_tke_surface_mem_60726.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    21, sizeof(mem_60731.mem),
                                                    &mem_60731.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    22, sizeof(mem_60735.mem),
                                                    &mem_60735.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    23, sizeof(mem_60895.mem),
                                                    &mem_60895.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    24, sizeof(mem_60899.mem),
                                                    &mem_60899.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    25, sizeof(mem_60903.mem),
                                                    &mem_60903.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52597,
                                                    26, sizeof(mem_60907.mem),
                                                    &mem_60907.mem));
            if (1 * ((size_t) segmap_usable_groups_53822 *
                     (size_t) segmap_group_sizze_53821) != 0) {
                const size_t global_work_sizze_61341[1] =
                             {(size_t) segmap_usable_groups_53822 *
                             (size_t) segmap_group_sizze_53821};
                const size_t local_work_sizze_61345[1] =
                             {segmap_group_sizze_53821};
                int64_t time_start_61342 = 0, time_end_61343 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_52597");
                    fprintf(stderr, "%zu", global_work_sizze_61341[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_61345[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_61342 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_52597,
                                                                1, NULL,
                                                                global_work_sizze_61341,
                                                                local_work_sizze_61345,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_52597_runs,
                                                                                                          &ctx->integrate_tkezisegmap_52597_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_61343 = get_wall_time();
                    
                    long time_diff_61344 = time_end_61343 - time_start_61342;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_52597", time_diff_61344);
                }
            }
            ctx->failure_is_an_option = 1;
            
            int64_t segmap_usable_groups_54065 = sdiv_up64(nest_sizze_49903,
                                                           segmap_group_sizze_54064);
            struct memblock_device mem_60912;
            
            mem_60912.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60912, bytes_60728,
                                      "mem_60912")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_60916;
            
            mem_60916.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60916, bytes_60728,
                                      "mem_60916")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    3, sizeof(xdim_48112),
                                                    &xdim_48112));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    4, sizeof(ydim_48113),
                                                    &ydim_48113));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    5, sizeof(zzdim_48114),
                                                    &zzdim_48114));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    6, sizeof(mem_60895.mem),
                                                    &mem_60895.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    7, sizeof(mem_60899.mem),
                                                    &mem_60899.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    8, sizeof(mem_60903.mem),
                                                    &mem_60903.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    9, sizeof(mem_60912.mem),
                                                    &mem_60912.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52305,
                                                    10, sizeof(mem_60916.mem),
                                                    &mem_60916.mem));
            if (1 * ((size_t) segmap_usable_groups_54065 *
                     (size_t) segmap_group_sizze_54064) != 0) {
                const size_t global_work_sizze_61346[1] =
                             {(size_t) segmap_usable_groups_54065 *
                             (size_t) segmap_group_sizze_54064};
                const size_t local_work_sizze_61350[1] =
                             {segmap_group_sizze_54064};
                int64_t time_start_61347 = 0, time_end_61348 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_52305");
                    fprintf(stderr, "%zu", global_work_sizze_61346[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_61350[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_61347 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_52305,
                                                                1, NULL,
                                                                global_work_sizze_61346,
                                                                local_work_sizze_61350,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_52305_runs,
                                                                                                          &ctx->integrate_tkezisegmap_52305_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_61348 = get_wall_time();
                    
                    long time_diff_61349 = time_end_61348 - time_start_61347;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_52305", time_diff_61349);
                }
            }
            ctx->failure_is_an_option = 1;
            
            struct memblock_device mem_60920;
            
            mem_60920.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60920, bytes_60917,
                                      "mem_60920")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60920, 0,
                                                      mem_60912, 0, 1,
                                                      zzdim_48114, xdim_48112 *
                                                      ydim_48113) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_60912, "mem_60912") != 0)
                return 1;
            
            struct memblock_device mem_60924;
            
            mem_60924.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60924, bytes_60917,
                                      "mem_60924")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60924, 0,
                                                      mem_60916, 0, 1,
                                                      zzdim_48114, xdim_48112 *
                                                      ydim_48113) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_60916, "mem_60916") != 0)
                return 1;
            
            struct memblock_device mem_60928;
            
            mem_60928.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60928, bytes_60917,
                                      "mem_60928")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60928, 0,
                                                      mem_60903, 0, 1,
                                                      zzdim_48114, xdim_48112 *
                                                      ydim_48113) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_60903, "mem_60903") != 0)
                return 1;
            
            struct memblock_device mem_60932;
            
            mem_60932.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60932, bytes_60917,
                                      "mem_60932")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60932, 0,
                                                      mem_60907, 0, 1,
                                                      zzdim_48114, xdim_48112 *
                                                      ydim_48113) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_60907, "mem_60907") != 0)
                return 1;
            
            struct memblock_device mem_60936;
            
            mem_60936.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60936, bytes_60917,
                                      "mem_60936")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60936, 0,
                                                      mem_60899, 0, 1,
                                                      zzdim_48114, xdim_48112 *
                                                      ydim_48113) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_60899, "mem_60899") != 0)
                return 1;
            
            struct memblock_device mem_60940;
            
            mem_60940.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60940, bytes_60917,
                                      "mem_60940")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60940, 0,
                                                      mem_60895, 0, 1,
                                                      zzdim_48114, xdim_48112 *
                                                      ydim_48113) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_60895, "mem_60895") != 0)
                return 1;
            
            struct memblock_device mem_60981;
            
            mem_60981.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60981, bytes_60917,
                                      "mem_60981")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_60985;
            
            mem_60985.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60985, bytes_60917,
                                      "mem_60985")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_60959;
            
            mem_60959.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60959, total_sizze_61111,
                                      "mem_60959")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_60964;
            
            mem_60964.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60964, total_sizze_61112,
                                      "mem_60964")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    3, sizeof(xdim_48112),
                                                    &xdim_48112));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    4, sizeof(ydim_48113),
                                                    &ydim_48113));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    5, sizeof(zzdim_48114),
                                                    &zzdim_48114));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    6, sizeof(y_48310),
                                                    &y_48310));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    7, sizeof(num_groups_54090),
                                                    &num_groups_54090));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    8, sizeof(mem_60920.mem),
                                                    &mem_60920.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    9, sizeof(mem_60924.mem),
                                                    &mem_60924.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    10, sizeof(mem_60928.mem),
                                                    &mem_60928.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    11, sizeof(mem_60932.mem),
                                                    &mem_60932.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    12, sizeof(mem_60936.mem),
                                                    &mem_60936.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    13, sizeof(mem_60940.mem),
                                                    &mem_60940.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    14, sizeof(mem_60959.mem),
                                                    &mem_60959.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    15, sizeof(mem_60964.mem),
                                                    &mem_60964.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    16, sizeof(mem_60981.mem),
                                                    &mem_60981.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52228,
                                                    17, sizeof(mem_60985.mem),
                                                    &mem_60985.mem));
            if (1 * ((size_t) num_groups_54090 *
                     (size_t) segmap_group_sizze_54089) != 0) {
                const size_t global_work_sizze_61351[1] =
                             {(size_t) num_groups_54090 *
                             (size_t) segmap_group_sizze_54089};
                const size_t local_work_sizze_61355[1] =
                             {segmap_group_sizze_54089};
                int64_t time_start_61352 = 0, time_end_61353 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_52228");
                    fprintf(stderr, "%zu", global_work_sizze_61351[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_61355[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_61352 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_52228,
                                                                1, NULL,
                                                                global_work_sizze_61351,
                                                                local_work_sizze_61355,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_52228_runs,
                                                                                                          &ctx->integrate_tkezisegmap_52228_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_61353 = get_wall_time();
                    
                    long time_diff_61354 = time_end_61353 - time_start_61352;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_52228", time_diff_61354);
                }
            }
            ctx->failure_is_an_option = 1;
            if (memblock_unref_device(ctx, &mem_60920, "mem_60920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60924, "mem_60924") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60928, "mem_60928") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60932, "mem_60932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60936, "mem_60936") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60940, "mem_60940") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60959, "mem_60959") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60964, "mem_60964") != 0)
                return 1;
            
            int64_t segmap_usable_groups_54158 = sdiv_up64(binop_x_60729,
                                                           segmap_group_sizze_54157);
            struct memblock_device mem_60989;
            
            mem_60989.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60989, bytes_60728,
                                      "mem_60989")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60989, 0,
                                                      mem_60985, 0, 1,
                                                      xdim_48112 * ydim_48113,
                                                      zzdim_48114) != 0) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52150,
                                                    1, sizeof(xdim_48112),
                                                    &xdim_48112));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52150,
                                                    2, sizeof(ydim_48113),
                                                    &ydim_48113));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52150,
                                                    3, sizeof(zzdim_48114),
                                                    &zzdim_48114));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52150,
                                                    4, sizeof(y_48310),
                                                    &y_48310));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52150,
                                                    5, sizeof(mem_60739.mem),
                                                    &mem_60739.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52150,
                                                    6, sizeof(mem_60989.mem),
                                                    &mem_60989.mem));
            if (1 * ((size_t) segmap_usable_groups_54158 *
                     (size_t) segmap_group_sizze_54157) != 0) {
                const size_t global_work_sizze_61356[1] =
                             {(size_t) segmap_usable_groups_54158 *
                             (size_t) segmap_group_sizze_54157};
                const size_t local_work_sizze_61360[1] =
                             {segmap_group_sizze_54157};
                int64_t time_start_61357 = 0, time_end_61358 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_52150");
                    fprintf(stderr, "%zu", global_work_sizze_61356[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_61360[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_61357 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_52150,
                                                                1, NULL,
                                                                global_work_sizze_61356,
                                                                local_work_sizze_61360,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_52150_runs,
                                                                                                          &ctx->integrate_tkezisegmap_52150_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_61358 = get_wall_time();
                    
                    long time_diff_61359 = time_end_61358 - time_start_61357;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_52150", time_diff_61359);
                }
            }
            if (memblock_unref_device(ctx, &mem_60989, "mem_60989") != 0)
                return 1;
            
            struct memblock_device mem_60994;
            
            mem_60994.references = NULL;
            if (memblock_alloc_device(ctx, &mem_60994, bytes_60917,
                                      "mem_60994")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_60994, 0,
                                                      mem_60739, 0, 1,
                                                      zzdim_48114, xdim_48112 *
                                                      ydim_48113) != 0) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_61017;
            
            mem_61017.references = NULL;
            if (memblock_alloc_device(ctx, &mem_61017, bytes_60917,
                                      "mem_61017")) {
                err = 1;
                goto cleanup;
            }
            
            struct memblock_device mem_61005;
            
            mem_61005.references = NULL;
            if (memblock_alloc_device(ctx, &mem_61005, total_sizze_61115,
                                      "mem_61005")) {
                err = 1;
                goto cleanup;
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    1,
                                                    sizeof(ctx->failure_is_an_option),
                                                    &ctx->failure_is_an_option));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    3, sizeof(xdim_48112),
                                                    &xdim_48112));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    4, sizeof(ydim_48113),
                                                    &ydim_48113));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    5, sizeof(zzdim_48114),
                                                    &zzdim_48114));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    6, sizeof(y_48310),
                                                    &y_48310));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    7, sizeof(m_48377),
                                                    &m_48377));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    8, sizeof(num_groups_54169),
                                                    &num_groups_54169));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    9, sizeof(mem_60981.mem),
                                                    &mem_60981.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    10, sizeof(mem_60985.mem),
                                                    &mem_60985.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    11, sizeof(mem_60994.mem),
                                                    &mem_60994.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    12, sizeof(mem_61005.mem),
                                                    &mem_61005.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_52110,
                                                    13, sizeof(mem_61017.mem),
                                                    &mem_61017.mem));
            if (1 * ((size_t) num_groups_54169 *
                     (size_t) segmap_group_sizze_54168) != 0) {
                const size_t global_work_sizze_61361[1] =
                             {(size_t) num_groups_54169 *
                             (size_t) segmap_group_sizze_54168};
                const size_t local_work_sizze_61365[1] =
                             {segmap_group_sizze_54168};
                int64_t time_start_61362 = 0, time_end_61363 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "integrate_tke.segmap_52110");
                    fprintf(stderr, "%zu", global_work_sizze_61361[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_61365[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_61362 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->integrate_tkezisegmap_52110,
                                                                1, NULL,
                                                                global_work_sizze_61361,
                                                                local_work_sizze_61365,
                                                                0, NULL,
                                                                ctx->profiling_paused ||
                                                                !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                          &ctx->integrate_tkezisegmap_52110_runs,
                                                                                                          &ctx->integrate_tkezisegmap_52110_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_61363 = get_wall_time();
                    
                    long time_diff_61364 = time_end_61363 - time_start_61362;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "integrate_tke.segmap_52110", time_diff_61364);
                }
            }
            ctx->failure_is_an_option = 1;
            if (memblock_unref_device(ctx, &mem_60981, "mem_60981") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60985, "mem_60985") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60994, "mem_60994") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_61005, "mem_61005") != 0)
                return 1;
            
            struct memblock_device mem_61021;
            
            mem_61021.references = NULL;
            if (memblock_alloc_device(ctx, &mem_61021, bytes_60728,
                                      "mem_61021")) {
                err = 1;
                goto cleanup;
            }
            if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_61021, 0,
                                                      mem_61017, 0, 1,
                                                      xdim_48112 * ydim_48113,
                                                      zzdim_48114) != 0) {
                err = 1;
                goto cleanup;
            }
            if (memblock_unref_device(ctx, &mem_61017, "mem_61017") != 0)
                return 1;
            if (memblock_set_device(ctx, &lifted_11_map_res_mem_61023,
                                    &mem_61021, "mem_61021") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_61021, "mem_61021") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_61005, "mem_61005") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_61017, "mem_61017") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60994, "mem_60994") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60989, "mem_60989") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60964, "mem_60964") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60959, "mem_60959") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60985, "mem_60985") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60981, "mem_60981") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60940, "mem_60940") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60936, "mem_60936") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60932, "mem_60932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60928, "mem_60928") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60924, "mem_60924") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60920, "mem_60920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60916, "mem_60916") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60912, "mem_60912") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60907, "mem_60907") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60903, "mem_60903") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60899, "mem_60899") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_60895, "mem_60895") != 0)
                return 1;
        }
        if (memblock_set_device(ctx, &lifted_11_map_res_mem_61024,
                                &lifted_11_map_res_mem_61023,
                                "lifted_11_map_res_mem_61023") != 0)
            return 1;
        if (memblock_unref_device(ctx, &lifted_11_map_res_mem_61023,
                                  "lifted_11_map_res_mem_61023") != 0)
            return 1;
    }
    if (memblock_unref_device(ctx, &mem_60731, "mem_60731") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_60735, "mem_60735") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_60739, "mem_60739") != 0)
        return 1;
    
    int64_t segmap_group_sizze_54595;
    
    segmap_group_sizze_54595 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_54474;
    
    int64_t segmap_usable_groups_54596 = sdiv_up64(binop_x_60729,
                                                   segmap_group_sizze_54595);
    int64_t bytes_61026 = 4 * binop_x_60729;
    struct memblock_device mem_61028;
    
    mem_61028.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61028, bytes_61026, "mem_61028")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61030;
    
    mem_61030.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61030, binop_x_60729, "mem_61030")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61032;
    
    mem_61032.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61032, binop_x_60729, "mem_61032")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 1,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 2,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 3,
                                            sizeof(ydim_48157), &ydim_48157));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 4,
                                            sizeof(y_48308), &y_48308));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 5,
                                            sizeof(y_48309), &y_48309));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 6,
                                            sizeof(kbot_mem_60722.mem),
                                            &kbot_mem_60722.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 7,
                                            sizeof(mem_61028.mem),
                                            &mem_61028.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 8,
                                            sizeof(mem_61030.mem),
                                            &mem_61030.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54471, 9,
                                            sizeof(mem_61032.mem),
                                            &mem_61032.mem));
    if (1 * ((size_t) segmap_usable_groups_54596 *
             (size_t) segmap_group_sizze_54595) != 0) {
        const size_t global_work_sizze_61366[1] =
                     {(size_t) segmap_usable_groups_54596 *
                     (size_t) segmap_group_sizze_54595};
        const size_t local_work_sizze_61370[1] = {segmap_group_sizze_54595};
        int64_t time_start_61367 = 0, time_end_61368 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_54471");
            fprintf(stderr, "%zu", global_work_sizze_61366[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61370[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61367 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_54471,
                                                        1, NULL,
                                                        global_work_sizze_61366,
                                                        local_work_sizze_61370,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_54471_runs,
                                                                                                  &ctx->integrate_tkezisegmap_54471_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61368 = get_wall_time();
            
            long time_diff_61369 = time_end_61368 - time_start_61367;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_54471", time_diff_61369);
        }
    }
    if (memblock_unref_device(ctx, &mem_61032, "mem_61032") != 0)
        return 1;
    
    int64_t segmap_group_sizze_54629;
    
    segmap_group_sizze_54629 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_54406;
    
    int64_t segmap_usable_groups_54630 = sdiv_up64(nest_sizze_49903,
                                                   segmap_group_sizze_54629);
    struct memblock_device mem_61037;
    
    mem_61037.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61037, bytes_60728, "mem_61037")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 1,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 2,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 3,
                                            sizeof(zzdim_48114), &zzdim_48114));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 4,
                                            sizeof(ydim_48116), &ydim_48116));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 5,
                                            sizeof(zzdim_48117), &zzdim_48117));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 6,
                                            sizeof(y_48308), &y_48308));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 7,
                                            sizeof(y_48309), &y_48309));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 8,
                                            sizeof(tketaup1_mem_60703.mem),
                                            &tketaup1_mem_60703.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402, 9,
                                            sizeof(lifted_11_map_res_mem_61024.mem),
                                            &lifted_11_map_res_mem_61024.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402,
                                            10, sizeof(mem_61028.mem),
                                            &mem_61028.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402,
                                            11, sizeof(mem_61030.mem),
                                            &mem_61030.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_54402,
                                            12, sizeof(mem_61037.mem),
                                            &mem_61037.mem));
    if (1 * ((size_t) segmap_usable_groups_54630 *
             (size_t) segmap_group_sizze_54629) != 0) {
        const size_t global_work_sizze_61371[1] =
                     {(size_t) segmap_usable_groups_54630 *
                     (size_t) segmap_group_sizze_54629};
        const size_t local_work_sizze_61375[1] = {segmap_group_sizze_54629};
        int64_t time_start_61372 = 0, time_end_61373 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_54402");
            fprintf(stderr, "%zu", global_work_sizze_61371[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61375[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61372 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_54402,
                                                        1, NULL,
                                                        global_work_sizze_61371,
                                                        local_work_sizze_61375,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_54402_runs,
                                                                                                  &ctx->integrate_tkezisegmap_54402_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61373 = get_wall_time();
            
            long time_diff_61374 = time_end_61373 - time_start_61372;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_54402", time_diff_61374);
        }
    }
    if (memblock_unref_device(ctx, &lifted_11_map_res_mem_61024,
                              "lifted_11_map_res_mem_61024") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61028, "mem_61028") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61030, "mem_61030") != 0)
        return 1;
    
    int64_t y_48726 = sub64(xdim_48112, 1);
    int64_t y_48727 = sub64(ydim_48113, 1);
    int64_t segmap_group_sizze_55655;
    
    segmap_group_sizze_55655 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_55435;
    
    int64_t segmap_usable_groups_55656 = sdiv_up64(binop_x_60729,
                                                   segmap_group_sizze_55655);
    struct memblock_device mem_61041;
    
    mem_61041.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61041, bytes_60917, "mem_61041")) {
        err = 1;
        goto cleanup;
    }
    if (futrts_builtinzhgpu_map_transpose_f64(ctx, mem_61041, 0, mem_61037, 0,
                                              1, zzdim_48114, xdim_48112 *
                                              ydim_48113) != 0) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61044;
    
    mem_61044.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61044, binop_x_60729, "mem_61044")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61046;
    
    mem_61046.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61046, binop_x_60729, "mem_61046")) {
        err = 1;
        goto cleanup;
    }
    
    int64_t bytes_61047 = 8 * binop_x_60729;
    struct memblock_device mem_61049;
    
    mem_61049.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61049, bytes_61047, "mem_61049")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 3,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 4,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 5,
                                            sizeof(zzdim_48114), &zzdim_48114));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 6,
                                            sizeof(y_48308), &y_48308));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 7,
                                            sizeof(y_48309), &y_48309));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 8,
                                            sizeof(y_48727), &y_48727));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432, 9,
                                            sizeof(dzzw_mem_60719.mem),
                                            &dzzw_mem_60719.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432,
                                            10, sizeof(mem_61041.mem),
                                            &mem_61041.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432,
                                            11, sizeof(mem_61044.mem),
                                            &mem_61044.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432,
                                            12, sizeof(mem_61046.mem),
                                            &mem_61046.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55432,
                                            13, sizeof(mem_61049.mem),
                                            &mem_61049.mem));
    if (1 * ((size_t) segmap_usable_groups_55656 *
             (size_t) segmap_group_sizze_55655) != 0) {
        const size_t global_work_sizze_61376[1] =
                     {(size_t) segmap_usable_groups_55656 *
                     (size_t) segmap_group_sizze_55655};
        const size_t local_work_sizze_61380[1] = {segmap_group_sizze_55655};
        int64_t time_start_61377 = 0, time_end_61378 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_55432");
            fprintf(stderr, "%zu", global_work_sizze_61376[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61380[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61377 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_55432,
                                                        1, NULL,
                                                        global_work_sizze_61376,
                                                        local_work_sizze_61380,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_55432_runs,
                                                                                                  &ctx->integrate_tkezisegmap_55432_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61378 = get_wall_time();
            
            long time_diff_61379 = time_end_61378 - time_start_61377;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_55432", time_diff_61379);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_61041, "mem_61041") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61044, "mem_61044") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61046, "mem_61046") != 0)
        return 1;
    
    int64_t segmap_group_sizze_55702;
    
    segmap_group_sizze_55702 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_55221;
    
    int64_t segmap_usable_groups_55703 = sdiv_up64(nest_sizze_49903,
                                                   segmap_group_sizze_55702);
    struct memblock_device mem_61054;
    
    mem_61054.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61054, bytes_60728, "mem_61054")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61058;
    
    mem_61058.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61058, bytes_60728, "mem_61058")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61062;
    
    mem_61062.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61062, bytes_60728, "mem_61062")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 3,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 4,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 5,
                                            sizeof(zzdim_48114), &zzdim_48114));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 6,
                                            sizeof(ydim_48140), &ydim_48140));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 7,
                                            sizeof(zzdim_48141), &zzdim_48141));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 8,
                                            sizeof(ydim_48143), &ydim_48143));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217, 9,
                                            sizeof(zzdim_48144), &zzdim_48144));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            10, sizeof(y_48308), &y_48308));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            11, sizeof(y_48309), &y_48309));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            12, sizeof(y_48310), &y_48310));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            13, sizeof(y_48726), &y_48726));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            14, sizeof(y_48727), &y_48727));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            15, sizeof(tketau_mem_60702.mem),
                                            &tketau_mem_60702.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            16, sizeof(maskU_mem_60711.mem),
                                            &maskU_mem_60711.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            17, sizeof(maskV_mem_60712.mem),
                                            &maskV_mem_60712.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            18, sizeof(dxu_mem_60715.mem),
                                            &dxu_mem_60715.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            19, sizeof(dyu_mem_60717.mem),
                                            &dyu_mem_60717.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            20, sizeof(cost_mem_60720.mem),
                                            &cost_mem_60720.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            21, sizeof(cosu_mem_60721.mem),
                                            &cosu_mem_60721.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            22, sizeof(mem_61037.mem),
                                            &mem_61037.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            23, sizeof(mem_61054.mem),
                                            &mem_61054.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            24, sizeof(mem_61058.mem),
                                            &mem_61058.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_55217,
                                            25, sizeof(mem_61062.mem),
                                            &mem_61062.mem));
    if (1 * ((size_t) segmap_usable_groups_55703 *
             (size_t) segmap_group_sizze_55702) != 0) {
        const size_t global_work_sizze_61381[1] =
                     {(size_t) segmap_usable_groups_55703 *
                     (size_t) segmap_group_sizze_55702};
        const size_t local_work_sizze_61385[1] = {segmap_group_sizze_55702};
        int64_t time_start_61382 = 0, time_end_61383 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_55217");
            fprintf(stderr, "%zu", global_work_sizze_61381[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61385[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61382 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_55217,
                                                        1, NULL,
                                                        global_work_sizze_61381,
                                                        local_work_sizze_61385,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_55217_runs,
                                                                                                  &ctx->integrate_tkezisegmap_55217_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61383 = get_wall_time();
            
            long time_diff_61384 = time_end_61383 - time_start_61382;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_55217", time_diff_61384);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_61037, "mem_61037") != 0)
        return 1;
    
    int64_t segmap_group_sizze_58435;
    
    segmap_group_sizze_58435 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_57271;
    
    int64_t segmap_usable_groups_58436 = sdiv_up64(nest_sizze_49903,
                                                   segmap_group_sizze_58435);
    struct memblock_device mem_61067;
    
    mem_61067.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61067, bytes_60728, "mem_61067")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61071;
    
    mem_61071.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61071, bytes_60728, "mem_61071")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61075;
    
    mem_61075.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61075, bytes_60728, "mem_61075")) {
        err = 1;
        goto cleanup;
    }
    
    struct memblock_device mem_61079;
    
    mem_61079.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61079, bytes_60728, "mem_61079")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 3,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 4,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 5,
                                            sizeof(zzdim_48114), &zzdim_48114));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 6,
                                            sizeof(ydim_48131), &ydim_48131));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 7,
                                            sizeof(zzdim_48132), &zzdim_48132));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 8,
                                            sizeof(ydim_48134), &ydim_48134));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267, 9,
                                            sizeof(zzdim_48135), &zzdim_48135));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            10, sizeof(ydim_48137),
                                            &ydim_48137));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            11, sizeof(zzdim_48138),
                                            &zzdim_48138));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            12, sizeof(ydim_48146),
                                            &ydim_48146));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            13, sizeof(zzdim_48147),
                                            &zzdim_48147));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            14, sizeof(y_48308), &y_48308));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            15, sizeof(y_48309), &y_48309));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            16, sizeof(y_48310), &y_48310));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            17, sizeof(tketau_mem_60702.mem),
                                            &tketau_mem_60702.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            18, sizeof(utau_mem_60708.mem),
                                            &utau_mem_60708.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            19, sizeof(vtau_mem_60709.mem),
                                            &vtau_mem_60709.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            20, sizeof(wtau_mem_60710.mem),
                                            &wtau_mem_60710.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            21, sizeof(maskW_mem_60713.mem),
                                            &maskW_mem_60713.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            22, sizeof(dxt_mem_60714.mem),
                                            &dxt_mem_60714.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            23, sizeof(dyt_mem_60716.mem),
                                            &dyt_mem_60716.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            24, sizeof(dzzw_mem_60719.mem),
                                            &dzzw_mem_60719.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            25, sizeof(cost_mem_60720.mem),
                                            &cost_mem_60720.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            26, sizeof(cosu_mem_60721.mem),
                                            &cosu_mem_60721.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            27, sizeof(mem_61054.mem),
                                            &mem_61054.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            28, sizeof(mem_61058.mem),
                                            &mem_61058.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            29, sizeof(mem_61062.mem),
                                            &mem_61062.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            30, sizeof(mem_61067.mem),
                                            &mem_61067.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            31, sizeof(mem_61071.mem),
                                            &mem_61071.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            32, sizeof(mem_61075.mem),
                                            &mem_61075.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_57267,
                                            33, sizeof(mem_61079.mem),
                                            &mem_61079.mem));
    if (1 * ((size_t) segmap_usable_groups_58436 *
             (size_t) segmap_group_sizze_58435) != 0) {
        const size_t global_work_sizze_61386[1] =
                     {(size_t) segmap_usable_groups_58436 *
                     (size_t) segmap_group_sizze_58435};
        const size_t local_work_sizze_61390[1] = {segmap_group_sizze_58435};
        int64_t time_start_61387 = 0, time_end_61388 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_57267");
            fprintf(stderr, "%zu", global_work_sizze_61386[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61390[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61387 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_57267,
                                                        1, NULL,
                                                        global_work_sizze_61386,
                                                        local_work_sizze_61390,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_57267_runs,
                                                                                                  &ctx->integrate_tkezisegmap_57267_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61388 = get_wall_time();
            
            long time_diff_61389 = time_end_61388 - time_start_61387;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_57267", time_diff_61389);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_61054, "mem_61054") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61058, "mem_61058") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61062, "mem_61062") != 0)
        return 1;
    
    int64_t segmap_group_sizze_59921;
    
    segmap_group_sizze_59921 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_59408;
    
    int64_t segmap_usable_groups_59922 = sdiv_up64(nest_sizze_49903,
                                                   segmap_group_sizze_59921);
    struct memblock_device mem_61084;
    
    mem_61084.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61084, bytes_60728, "mem_61084")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 1,
                                            sizeof(ctx->failure_is_an_option),
                                            &ctx->failure_is_an_option));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 3,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 4,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 5,
                                            sizeof(zzdim_48114), &zzdim_48114));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 6,
                                            sizeof(ydim_48122), &ydim_48122));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 7,
                                            sizeof(zzdim_48123), &zzdim_48123));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 8,
                                            sizeof(ydim_48146), &ydim_48146));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404, 9,
                                            sizeof(zzdim_48147), &zzdim_48147));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            10, sizeof(y_48308), &y_48308));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            11, sizeof(y_48309), &y_48309));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            12, sizeof(y_48310), &y_48310));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            13, sizeof(dtketau_mem_60705.mem),
                                            &dtketau_mem_60705.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            14, sizeof(maskW_mem_60713.mem),
                                            &maskW_mem_60713.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            15, sizeof(dxt_mem_60714.mem),
                                            &dxt_mem_60714.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            16, sizeof(dyt_mem_60716.mem),
                                            &dyt_mem_60716.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            17, sizeof(dzzw_mem_60719.mem),
                                            &dzzw_mem_60719.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            18, sizeof(cost_mem_60720.mem),
                                            &cost_mem_60720.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            19, sizeof(mem_61067.mem),
                                            &mem_61067.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            20, sizeof(mem_61071.mem),
                                            &mem_61071.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            21, sizeof(mem_61075.mem),
                                            &mem_61075.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_59404,
                                            22, sizeof(mem_61084.mem),
                                            &mem_61084.mem));
    if (1 * ((size_t) segmap_usable_groups_59922 *
             (size_t) segmap_group_sizze_59921) != 0) {
        const size_t global_work_sizze_61391[1] =
                     {(size_t) segmap_usable_groups_59922 *
                     (size_t) segmap_group_sizze_59921};
        const size_t local_work_sizze_61395[1] = {segmap_group_sizze_59921};
        int64_t time_start_61392 = 0, time_end_61393 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_59404");
            fprintf(stderr, "%zu", global_work_sizze_61391[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61395[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61392 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_59404,
                                                        1, NULL,
                                                        global_work_sizze_61391,
                                                        local_work_sizze_61395,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_59404_runs,
                                                                                                  &ctx->integrate_tkezisegmap_59404_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61393 = get_wall_time();
            
            long time_diff_61394 = time_end_61393 - time_start_61392;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_59404", time_diff_61394);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_61067, "mem_61067") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61071, "mem_61071") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61075, "mem_61075") != 0)
        return 1;
    
    int64_t segmap_group_sizze_60352;
    
    segmap_group_sizze_60352 =
        ctx->sizes.integrate_tkezisegmap_group_sizze_60205;
    
    int64_t segmap_usable_groups_60353 = sdiv_up64(nest_sizze_49903,
                                                   segmap_group_sizze_60352);
    struct memblock_device mem_61089;
    
    mem_61089.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61089, bytes_60728, "mem_61089")) {
        err = 1;
        goto cleanup;
    }
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 1,
                                            sizeof(xdim_48112), &xdim_48112));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 2,
                                            sizeof(ydim_48113), &ydim_48113));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 3,
                                            sizeof(zzdim_48114), &zzdim_48114));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 4,
                                            sizeof(ydim_48128), &ydim_48128));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 5,
                                            sizeof(zzdim_48129), &zzdim_48129));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 6,
                                            sizeof(dtketaum1_mem_60707.mem),
                                            &dtketaum1_mem_60707.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 7,
                                            sizeof(mem_61079.mem),
                                            &mem_61079.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 8,
                                            sizeof(mem_61084.mem),
                                            &mem_61084.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->integrate_tkezisegmap_60201, 9,
                                            sizeof(mem_61089.mem),
                                            &mem_61089.mem));
    if (1 * ((size_t) segmap_usable_groups_60353 *
             (size_t) segmap_group_sizze_60352) != 0) {
        const size_t global_work_sizze_61396[1] =
                     {(size_t) segmap_usable_groups_60353 *
                     (size_t) segmap_group_sizze_60352};
        const size_t local_work_sizze_61400[1] = {segmap_group_sizze_60352};
        int64_t time_start_61397 = 0, time_end_61398 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "integrate_tke.segmap_60201");
            fprintf(stderr, "%zu", global_work_sizze_61396[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_61400[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_61397 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->integrate_tkezisegmap_60201,
                                                        1, NULL,
                                                        global_work_sizze_61396,
                                                        local_work_sizze_61400,
                                                        0, NULL,
                                                        ctx->profiling_paused ||
                                                        !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                  &ctx->integrate_tkezisegmap_60201_runs,
                                                                                                  &ctx->integrate_tkezisegmap_60201_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_61398 = get_wall_time();
            
            long time_diff_61399 = time_end_61398 - time_start_61397;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "integrate_tke.segmap_60201", time_diff_61399);
        }
    }
    if (memblock_unref_device(ctx, &mem_61079, "mem_61079") != 0)
        return 1;
    
    struct memblock_device mem_61093;
    
    mem_61093.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61093, bytes_60728, "mem_61093")) {
        err = 1;
        goto cleanup;
    }
    if (xdim_48112 * ydim_48113 * zzdim_48114 * (int64_t) sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     tketaum1_mem_60704.mem,
                                                     mem_61093.mem, 0, 0,
                                                     xdim_48112 * ydim_48113 *
                                                     zzdim_48114 *
                                                     (int64_t) sizeof(double),
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_61098;
    
    mem_61098.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61098, bytes_60728, "mem_61098")) {
        err = 1;
        goto cleanup;
    }
    if (xdim_48112 * ydim_48113 * zzdim_48114 * (int64_t) sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     dtketaup1_mem_60706.mem,
                                                     mem_61098.mem, 0, 0,
                                                     xdim_48112 * ydim_48113 *
                                                     zzdim_48114 *
                                                     (int64_t) sizeof(double),
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    
    struct memblock_device mem_61103;
    
    mem_61103.references = NULL;
    if (memblock_alloc_device(ctx, &mem_61103, bytes_60728, "mem_61103")) {
        err = 1;
        goto cleanup;
    }
    if (xdim_48112 * ydim_48113 * zzdim_48114 * (int64_t) sizeof(double) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                     dtketaum1_mem_60707.mem,
                                                     mem_61103.mem, 0, 0,
                                                     xdim_48112 * ydim_48113 *
                                                     zzdim_48114 *
                                                     (int64_t) sizeof(double),
                                                     0, NULL,
                                                     ctx->profiling_paused ||
                                                     !ctx->profiling ? NULL : opencl_get_event(&ctx->opencl,
                                                                                               &ctx->copy_dev_to_dev_runs,
                                                                                               &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    out_arrsizze_61124 = xdim_48112;
    out_arrsizze_61125 = ydim_48113;
    out_arrsizze_61126 = zzdim_48114;
    out_arrsizze_61128 = xdim_48112;
    out_arrsizze_61129 = ydim_48113;
    out_arrsizze_61130 = zzdim_48114;
    out_arrsizze_61132 = xdim_48112;
    out_arrsizze_61133 = ydim_48113;
    out_arrsizze_61134 = zzdim_48114;
    out_arrsizze_61136 = xdim_48112;
    out_arrsizze_61137 = ydim_48113;
    out_arrsizze_61138 = zzdim_48114;
    out_arrsizze_61140 = xdim_48112;
    out_arrsizze_61141 = ydim_48113;
    out_arrsizze_61142 = zzdim_48114;
    out_arrsizze_61144 = xdim_48112;
    out_arrsizze_61145 = ydim_48113;
    out_arrsizze_61146 = zzdim_48114;
    out_arrsizze_61148 = xdim_48112;
    out_arrsizze_61149 = ydim_48113;
    if (memblock_set_device(ctx, &out_mem_61123, &tketau_mem_60702,
                            "tketau_mem_60702") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_61127, &mem_61089, "mem_61089") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_61131, &mem_61093, "mem_61093") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_61135, &mem_61084, "mem_61084") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_61139, &mem_61098, "mem_61098") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_61143, &mem_61103, "mem_61103") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_61147, &mem_61049, "mem_61049") != 0)
        return 1;
    (*out_mem_p_61299).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_61299, &out_mem_61123,
                            "out_mem_61123") != 0)
        return 1;
    *out_out_arrsizze_61300 = out_arrsizze_61124;
    *out_out_arrsizze_61301 = out_arrsizze_61125;
    *out_out_arrsizze_61302 = out_arrsizze_61126;
    (*out_mem_p_61303).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_61303, &out_mem_61127,
                            "out_mem_61127") != 0)
        return 1;
    *out_out_arrsizze_61304 = out_arrsizze_61128;
    *out_out_arrsizze_61305 = out_arrsizze_61129;
    *out_out_arrsizze_61306 = out_arrsizze_61130;
    (*out_mem_p_61307).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_61307, &out_mem_61131,
                            "out_mem_61131") != 0)
        return 1;
    *out_out_arrsizze_61308 = out_arrsizze_61132;
    *out_out_arrsizze_61309 = out_arrsizze_61133;
    *out_out_arrsizze_61310 = out_arrsizze_61134;
    (*out_mem_p_61311).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_61311, &out_mem_61135,
                            "out_mem_61135") != 0)
        return 1;
    *out_out_arrsizze_61312 = out_arrsizze_61136;
    *out_out_arrsizze_61313 = out_arrsizze_61137;
    *out_out_arrsizze_61314 = out_arrsizze_61138;
    (*out_mem_p_61315).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_61315, &out_mem_61139,
                            "out_mem_61139") != 0)
        return 1;
    *out_out_arrsizze_61316 = out_arrsizze_61140;
    *out_out_arrsizze_61317 = out_arrsizze_61141;
    *out_out_arrsizze_61318 = out_arrsizze_61142;
    (*out_mem_p_61319).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_61319, &out_mem_61143,
                            "out_mem_61143") != 0)
        return 1;
    *out_out_arrsizze_61320 = out_arrsizze_61144;
    *out_out_arrsizze_61321 = out_arrsizze_61145;
    *out_out_arrsizze_61322 = out_arrsizze_61146;
    (*out_mem_p_61323).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_61323, &out_mem_61147,
                            "out_mem_61147") != 0)
        return 1;
    *out_out_arrsizze_61324 = out_arrsizze_61148;
    *out_out_arrsizze_61325 = out_arrsizze_61149;
    if (memblock_unref_device(ctx, &mem_61103, "mem_61103") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61098, "mem_61098") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61093, "mem_61093") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61089, "mem_61089") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61084, "mem_61084") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61079, "mem_61079") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61075, "mem_61075") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61071, "mem_61071") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61067, "mem_61067") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61062, "mem_61062") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61058, "mem_61058") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61054, "mem_61054") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61049, "mem_61049") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61046, "mem_61046") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61044, "mem_61044") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61041, "mem_61041") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61037, "mem_61037") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61032, "mem_61032") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61030, "mem_61030") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_61028, "mem_61028") != 0)
        return 1;
    if (memblock_unref_device(ctx, &lifted_11_map_res_mem_61024,
                              "lifted_11_map_res_mem_61024") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_60739, "mem_60739") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_60735, "mem_60735") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_60731, "mem_60731") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_61147, "out_mem_61147") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_61143, "out_mem_61143") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_61139, "out_mem_61139") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_61135, "out_mem_61135") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_61131, "out_mem_61131") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_61127, "out_mem_61127") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_61123, "out_mem_61123") != 0)
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
    struct memblock_device tketau_mem_60702;
    
    tketau_mem_60702.references = NULL;
    
    struct memblock_device tketaup1_mem_60703;
    
    tketaup1_mem_60703.references = NULL;
    
    struct memblock_device tketaum1_mem_60704;
    
    tketaum1_mem_60704.references = NULL;
    
    struct memblock_device dtketau_mem_60705;
    
    dtketau_mem_60705.references = NULL;
    
    struct memblock_device dtketaup1_mem_60706;
    
    dtketaup1_mem_60706.references = NULL;
    
    struct memblock_device dtketaum1_mem_60707;
    
    dtketaum1_mem_60707.references = NULL;
    
    struct memblock_device utau_mem_60708;
    
    utau_mem_60708.references = NULL;
    
    struct memblock_device vtau_mem_60709;
    
    vtau_mem_60709.references = NULL;
    
    struct memblock_device wtau_mem_60710;
    
    wtau_mem_60710.references = NULL;
    
    struct memblock_device maskU_mem_60711;
    
    maskU_mem_60711.references = NULL;
    
    struct memblock_device maskV_mem_60712;
    
    maskV_mem_60712.references = NULL;
    
    struct memblock_device maskW_mem_60713;
    
    maskW_mem_60713.references = NULL;
    
    struct memblock_device dxt_mem_60714;
    
    dxt_mem_60714.references = NULL;
    
    struct memblock_device dxu_mem_60715;
    
    dxu_mem_60715.references = NULL;
    
    struct memblock_device dyt_mem_60716;
    
    dyt_mem_60716.references = NULL;
    
    struct memblock_device dyu_mem_60717;
    
    dyu_mem_60717.references = NULL;
    
    struct memblock_device dzzt_mem_60718;
    
    dzzt_mem_60718.references = NULL;
    
    struct memblock_device dzzw_mem_60719;
    
    dzzw_mem_60719.references = NULL;
    
    struct memblock_device cost_mem_60720;
    
    cost_mem_60720.references = NULL;
    
    struct memblock_device cosu_mem_60721;
    
    cosu_mem_60721.references = NULL;
    
    struct memblock_device kbot_mem_60722;
    
    kbot_mem_60722.references = NULL;
    
    struct memblock_device kappaM_mem_60723;
    
    kappaM_mem_60723.references = NULL;
    
    struct memblock_device mxl_mem_60724;
    
    mxl_mem_60724.references = NULL;
    
    struct memblock_device forc_mem_60725;
    
    forc_mem_60725.references = NULL;
    
    struct memblock_device forc_tke_surface_mem_60726;
    
    forc_tke_surface_mem_60726.references = NULL;
    
    int64_t xdim_48112;
    int64_t ydim_48113;
    int64_t zzdim_48114;
    int64_t xdim_48115;
    int64_t ydim_48116;
    int64_t zzdim_48117;
    int64_t xdim_48118;
    int64_t ydim_48119;
    int64_t zzdim_48120;
    int64_t xdim_48121;
    int64_t ydim_48122;
    int64_t zzdim_48123;
    int64_t xdim_48124;
    int64_t ydim_48125;
    int64_t zzdim_48126;
    int64_t xdim_48127;
    int64_t ydim_48128;
    int64_t zzdim_48129;
    int64_t xdim_48130;
    int64_t ydim_48131;
    int64_t zzdim_48132;
    int64_t xdim_48133;
    int64_t ydim_48134;
    int64_t zzdim_48135;
    int64_t xdim_48136;
    int64_t ydim_48137;
    int64_t zzdim_48138;
    int64_t xdim_48139;
    int64_t ydim_48140;
    int64_t zzdim_48141;
    int64_t xdim_48142;
    int64_t ydim_48143;
    int64_t zzdim_48144;
    int64_t xdim_48145;
    int64_t ydim_48146;
    int64_t zzdim_48147;
    int64_t xdim_48148;
    int64_t xdim_48149;
    int64_t ydim_48150;
    int64_t ydim_48151;
    int64_t zzdim_48152;
    int64_t zzdim_48153;
    int64_t ydim_48154;
    int64_t ydim_48155;
    int64_t xdim_48156;
    int64_t ydim_48157;
    int64_t xdim_48158;
    int64_t ydim_48159;
    int64_t zzdim_48160;
    int64_t xdim_48161;
    int64_t ydim_48162;
    int64_t zzdim_48163;
    int64_t xdim_48164;
    int64_t ydim_48165;
    int64_t zzdim_48166;
    int64_t xdim_48167;
    int64_t ydim_48168;
    struct memblock_device out_mem_61123;
    
    out_mem_61123.references = NULL;
    
    int64_t out_arrsizze_61124;
    int64_t out_arrsizze_61125;
    int64_t out_arrsizze_61126;
    struct memblock_device out_mem_61127;
    
    out_mem_61127.references = NULL;
    
    int64_t out_arrsizze_61128;
    int64_t out_arrsizze_61129;
    int64_t out_arrsizze_61130;
    struct memblock_device out_mem_61131;
    
    out_mem_61131.references = NULL;
    
    int64_t out_arrsizze_61132;
    int64_t out_arrsizze_61133;
    int64_t out_arrsizze_61134;
    struct memblock_device out_mem_61135;
    
    out_mem_61135.references = NULL;
    
    int64_t out_arrsizze_61136;
    int64_t out_arrsizze_61137;
    int64_t out_arrsizze_61138;
    struct memblock_device out_mem_61139;
    
    out_mem_61139.references = NULL;
    
    int64_t out_arrsizze_61140;
    int64_t out_arrsizze_61141;
    int64_t out_arrsizze_61142;
    struct memblock_device out_mem_61143;
    
    out_mem_61143.references = NULL;
    
    int64_t out_arrsizze_61144;
    int64_t out_arrsizze_61145;
    int64_t out_arrsizze_61146;
    struct memblock_device out_mem_61147;
    
    out_mem_61147.references = NULL;
    
    int64_t out_arrsizze_61148;
    int64_t out_arrsizze_61149;
    
    lock_lock(&ctx->lock);
    tketau_mem_60702 = in0->mem;
    xdim_48112 = in0->shape[0];
    ydim_48113 = in0->shape[1];
    zzdim_48114 = in0->shape[2];
    tketaup1_mem_60703 = in1->mem;
    xdim_48115 = in1->shape[0];
    ydim_48116 = in1->shape[1];
    zzdim_48117 = in1->shape[2];
    tketaum1_mem_60704 = in2->mem;
    xdim_48118 = in2->shape[0];
    ydim_48119 = in2->shape[1];
    zzdim_48120 = in2->shape[2];
    dtketau_mem_60705 = in3->mem;
    xdim_48121 = in3->shape[0];
    ydim_48122 = in3->shape[1];
    zzdim_48123 = in3->shape[2];
    dtketaup1_mem_60706 = in4->mem;
    xdim_48124 = in4->shape[0];
    ydim_48125 = in4->shape[1];
    zzdim_48126 = in4->shape[2];
    dtketaum1_mem_60707 = in5->mem;
    xdim_48127 = in5->shape[0];
    ydim_48128 = in5->shape[1];
    zzdim_48129 = in5->shape[2];
    utau_mem_60708 = in6->mem;
    xdim_48130 = in6->shape[0];
    ydim_48131 = in6->shape[1];
    zzdim_48132 = in6->shape[2];
    vtau_mem_60709 = in7->mem;
    xdim_48133 = in7->shape[0];
    ydim_48134 = in7->shape[1];
    zzdim_48135 = in7->shape[2];
    wtau_mem_60710 = in8->mem;
    xdim_48136 = in8->shape[0];
    ydim_48137 = in8->shape[1];
    zzdim_48138 = in8->shape[2];
    maskU_mem_60711 = in9->mem;
    xdim_48139 = in9->shape[0];
    ydim_48140 = in9->shape[1];
    zzdim_48141 = in9->shape[2];
    maskV_mem_60712 = in10->mem;
    xdim_48142 = in10->shape[0];
    ydim_48143 = in10->shape[1];
    zzdim_48144 = in10->shape[2];
    maskW_mem_60713 = in11->mem;
    xdim_48145 = in11->shape[0];
    ydim_48146 = in11->shape[1];
    zzdim_48147 = in11->shape[2];
    dxt_mem_60714 = in12->mem;
    xdim_48148 = in12->shape[0];
    dxu_mem_60715 = in13->mem;
    xdim_48149 = in13->shape[0];
    dyt_mem_60716 = in14->mem;
    ydim_48150 = in14->shape[0];
    dyu_mem_60717 = in15->mem;
    ydim_48151 = in15->shape[0];
    dzzt_mem_60718 = in16->mem;
    zzdim_48152 = in16->shape[0];
    dzzw_mem_60719 = in17->mem;
    zzdim_48153 = in17->shape[0];
    cost_mem_60720 = in18->mem;
    ydim_48154 = in18->shape[0];
    cosu_mem_60721 = in19->mem;
    ydim_48155 = in19->shape[0];
    kbot_mem_60722 = in20->mem;
    xdim_48156 = in20->shape[0];
    ydim_48157 = in20->shape[1];
    kappaM_mem_60723 = in21->mem;
    xdim_48158 = in21->shape[0];
    ydim_48159 = in21->shape[1];
    zzdim_48160 = in21->shape[2];
    mxl_mem_60724 = in22->mem;
    xdim_48161 = in22->shape[0];
    ydim_48162 = in22->shape[1];
    zzdim_48163 = in22->shape[2];
    forc_mem_60725 = in23->mem;
    xdim_48164 = in23->shape[0];
    ydim_48165 = in23->shape[1];
    zzdim_48166 = in23->shape[2];
    forc_tke_surface_mem_60726 = in24->mem;
    xdim_48167 = in24->shape[0];
    ydim_48168 = in24->shape[1];
    
    int ret = futrts_integrate_tke(ctx, &out_mem_61123, &out_arrsizze_61124,
                                   &out_arrsizze_61125, &out_arrsizze_61126,
                                   &out_mem_61127, &out_arrsizze_61128,
                                   &out_arrsizze_61129, &out_arrsizze_61130,
                                   &out_mem_61131, &out_arrsizze_61132,
                                   &out_arrsizze_61133, &out_arrsizze_61134,
                                   &out_mem_61135, &out_arrsizze_61136,
                                   &out_arrsizze_61137, &out_arrsizze_61138,
                                   &out_mem_61139, &out_arrsizze_61140,
                                   &out_arrsizze_61141, &out_arrsizze_61142,
                                   &out_mem_61143, &out_arrsizze_61144,
                                   &out_arrsizze_61145, &out_arrsizze_61146,
                                   &out_mem_61147, &out_arrsizze_61148,
                                   &out_arrsizze_61149, tketau_mem_60702,
                                   tketaup1_mem_60703, tketaum1_mem_60704,
                                   dtketau_mem_60705, dtketaup1_mem_60706,
                                   dtketaum1_mem_60707, utau_mem_60708,
                                   vtau_mem_60709, wtau_mem_60710,
                                   maskU_mem_60711, maskV_mem_60712,
                                   maskW_mem_60713, dxt_mem_60714,
                                   dxu_mem_60715, dyt_mem_60716, dyu_mem_60717,
                                   dzzt_mem_60718, dzzw_mem_60719,
                                   cost_mem_60720, cosu_mem_60721,
                                   kbot_mem_60722, kappaM_mem_60723,
                                   mxl_mem_60724, forc_mem_60725,
                                   forc_tke_surface_mem_60726, xdim_48112,
                                   ydim_48113, zzdim_48114, xdim_48115,
                                   ydim_48116, zzdim_48117, xdim_48118,
                                   ydim_48119, zzdim_48120, xdim_48121,
                                   ydim_48122, zzdim_48123, xdim_48124,
                                   ydim_48125, zzdim_48126, xdim_48127,
                                   ydim_48128, zzdim_48129, xdim_48130,
                                   ydim_48131, zzdim_48132, xdim_48133,
                                   ydim_48134, zzdim_48135, xdim_48136,
                                   ydim_48137, zzdim_48138, xdim_48139,
                                   ydim_48140, zzdim_48141, xdim_48142,
                                   ydim_48143, zzdim_48144, xdim_48145,
                                   ydim_48146, zzdim_48147, xdim_48148,
                                   xdim_48149, ydim_48150, ydim_48151,
                                   zzdim_48152, zzdim_48153, ydim_48154,
                                   ydim_48155, xdim_48156, ydim_48157,
                                   xdim_48158, ydim_48159, zzdim_48160,
                                   xdim_48161, ydim_48162, zzdim_48163,
                                   xdim_48164, ydim_48165, zzdim_48166,
                                   xdim_48167, ydim_48168);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out0)->mem = out_mem_61123;
        (*out0)->shape[0] = out_arrsizze_61124;
        (*out0)->shape[1] = out_arrsizze_61125;
        (*out0)->shape[2] = out_arrsizze_61126;
        assert((*out1 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out1)->mem = out_mem_61127;
        (*out1)->shape[0] = out_arrsizze_61128;
        (*out1)->shape[1] = out_arrsizze_61129;
        (*out1)->shape[2] = out_arrsizze_61130;
        assert((*out2 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out2)->mem = out_mem_61131;
        (*out2)->shape[0] = out_arrsizze_61132;
        (*out2)->shape[1] = out_arrsizze_61133;
        (*out2)->shape[2] = out_arrsizze_61134;
        assert((*out3 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out3)->mem = out_mem_61135;
        (*out3)->shape[0] = out_arrsizze_61136;
        (*out3)->shape[1] = out_arrsizze_61137;
        (*out3)->shape[2] = out_arrsizze_61138;
        assert((*out4 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out4)->mem = out_mem_61139;
        (*out4)->shape[0] = out_arrsizze_61140;
        (*out4)->shape[1] = out_arrsizze_61141;
        (*out4)->shape[2] = out_arrsizze_61142;
        assert((*out5 =
                (struct futhark_f64_3d *) malloc(sizeof(struct futhark_f64_3d))) !=
            NULL);
        (*out5)->mem = out_mem_61143;
        (*out5)->shape[0] = out_arrsizze_61144;
        (*out5)->shape[1] = out_arrsizze_61145;
        (*out5)->shape[2] = out_arrsizze_61146;
        assert((*out6 =
                (struct futhark_f64_2d *) malloc(sizeof(struct futhark_f64_2d))) !=
            NULL);
        (*out6)->mem = out_mem_61147;
        (*out6)->shape[0] = out_arrsizze_61148;
        (*out6)->shape[1] = out_arrsizze_61149;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
