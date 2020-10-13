import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, valuef) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and (self.device.type & device_type) == device_type:
               sizes[size] = valuef(self.device)
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0

    self.max_local_memory = int(self.device.local_mem_size)

    # Futhark reserves 4 bytes of local memory for its own purposes.
    self.max_local_memory -= 4

    # See comment in rts/c/opencl.h.
    if self.platform.name.find('NVIDIA CUDA') >= 0:
        self.max_local_memory -= 12

    self.free_list = {}

    self.global_failure = self.pool.allocate(np.int32().itemsize)
    cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)
    self.global_failure_args = self.pool.allocate(np.int32().itemsize *
                                                  (self.global_failure_args_max+1))
    self.failure_is_an_option = np.int32(0)

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            # Bespoke sizes have no limit or default.
            max_value = None
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi').replace('#', 'zh'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()

def sync(self):
    failure = np.empty(1, dtype=np.int32)
    cl.enqueue_copy(self.queue, failure, self.global_failure, is_blocking=True)
    self.failure_is_an_option = np.int32(0)
    if failure[0] >= 0:
        # Reset failure information.
        cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)

        # Read failure args.
        failure_args = np.empty(self.global_failure_args_max+1, dtype=np.int32)
        cl.enqueue_copy(self.queue, failure_args, self.global_failure_args, is_blocking=True)

        raise Exception(self.failure_msgs[failure[0]].format(*failure_args))
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
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
// Start of atomics.h

inline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline float atomic_fadd_f32_global(volatile __global float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline float atomic_fadd_f32_local(volatile __local float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

inline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

// End of atomics.h




__kernel void map_transpose_f64(__local volatile
                                int64_t *block_9_backing_aligned_0,
                                int32_t destoffset_1, int32_t srcoffset_3,
                                int32_t num_arrays_4, int32_t x_elems_5,
                                int32_t y_elems_6, int32_t mulx_7,
                                int32_t muly_8, __global
                                unsigned char *destmem_0, __global
                                unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {
                ((__local double *) block_9)[(get_local_id_1_39 + j_43 * 8) *
                                             33 + get_local_id_0_38] =
                    ((__global double *) srcmem_2)[idata_offset_34 +
                                                   index_in_35];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {
                ((__global double *) destmem_0)[odata_offset_33 +
                                                index_out_36] = ((__local
                                                                  double *) block_9)[get_local_id_0_38 *
                                                                                     33 +
                                                                                     get_local_id_1_39 +
                                                                                     j_43 *
                                                                                     8];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f64_low_height(__local volatile
                                           int64_t *block_9_backing_aligned_0,
                                           int32_t destoffset_1,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t mulx_7, int32_t muly_8,
                                           __global unsigned char *destmem_0,
                                           __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_7) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_7);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local double *) block_9)[get_local_id_1_39 * 17 +
                                     get_local_id_0_38] = ((__global
                                                            double *) srcmem_2)[idata_offset_34 +
                                                                                index_in_35];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);
    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_7) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global double *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__local double *) block_9)[get_local_id_0_38 * 17 +
                                         get_local_id_1_39];
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f64_low_width(__local volatile
                                          int64_t *block_9_backing_aligned_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t mulx_7, int32_t muly_8,
                                          __global unsigned char *destmem_0,
                                          __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_8);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_8) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local double *) block_9)[get_local_id_1_39 * 17 +
                                     get_local_id_0_38] = ((__global
                                                            double *) srcmem_2)[idata_offset_34 +
                                                                                index_in_35];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_8) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_8);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global double *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__local double *) block_9)[get_local_id_0_38 * 17 +
                                         get_local_id_1_39];
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f64_small(__local volatile
                                      int64_t *block_9_backing_aligned_0,
                                      int32_t destoffset_1, int32_t srcoffset_3,
                                      int32_t num_arrays_4, int32_t x_elems_5,
                                      int32_t y_elems_6, int32_t mulx_7,
                                      int32_t muly_8, __global
                                      unsigned char *destmem_0, __global
                                      unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 8) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 8) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {
        ((__global double *) destmem_0)[odata_offset_33 + index_out_36] =
            ((__global double *) srcmem_2)[idata_offset_34 + index_in_35];
    }
    
  error_0:
    return;
}
__kernel void tridagNestedziscan_stage1_12884(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_17080_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_17078_backing_aligned_1,
                                              int32_t n_11293, int32_t m_11294,
                                              int32_t m_11298, __global
                                              unsigned char *c_mem_16586,
                                              __global unsigned char *mem_16663,
                                              __global unsigned char *mem_16680,
                                              __global unsigned char *mem_16690,
                                              __global unsigned char *mem_16695,
                                              int32_t num_threads_17072)
{
    #define segscan_group_sizze_13620 (tridagNestedzisegscan_group_sizze_12878)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17080_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17080_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17078_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17078_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17073;
    int32_t local_tid_17074;
    int32_t group_sizze_17077;
    int32_t wave_sizze_17076;
    int32_t group_tid_17075;
    
    global_tid_17073 = get_global_id(0);
    local_tid_17074 = get_local_id(0);
    group_sizze_17077 = get_local_size(0);
    wave_sizze_17076 = LOCKSTEP_WIDTH;
    group_tid_17075 = get_group_id(0);
    
    int32_t phys_tid_12884;
    
    phys_tid_12884 = global_tid_17073;
    
    __local char *scan_arr_mem_17078;
    __local char *scan_arr_mem_17080;
    
    scan_arr_mem_17078 = (__local char *) scan_arr_mem_17078_backing_0;
    scan_arr_mem_17080 = (__local char *) scan_arr_mem_17080_backing_1;
    
    double x_13625;
    double x_13626;
    double x_13627;
    double x_13628;
    
    x_13625 = 0.0;
    x_13626 = 1.0;
    for (int32_t j_17082 = 0; j_17082 < sdiv_up32(n_11293 * m_11294,
                                                  num_threads_17072);
         j_17082++) {
        int32_t chunk_offset_17083 = segscan_group_sizze_13620 * j_17082 +
                group_tid_17075 * (segscan_group_sizze_13620 *
                                   sdiv_up32(n_11293 * m_11294,
                                             num_threads_17072));
        int32_t flat_idx_17084 = chunk_offset_17083 + local_tid_17074;
        int32_t gtid_12873 = squot32(flat_idx_17084, m_11294);
        int32_t gtid_12883 = flat_idx_17084 - squot32(flat_idx_17084, m_11294) *
                m_11294;
        
        // threads in bounds read input
        {
            if (slt32(gtid_12873, n_11293) && slt32(gtid_12883, m_11294)) {
                int32_t x_13637 = sub32(m_11294, gtid_12883);
                int32_t i_13638 = sub32(x_13637, 1);
                bool cond_13639 = slt32(0, gtid_12883);
                double res_13640;
                double res_13641;
                
                if (cond_13639) {
                    double x_13642 = ((__global
                                       double *) mem_16680)[gtid_12873 *
                                                            m_11294 + i_13638];
                    double y_13643 = ((__global
                                       double *) mem_16663)[gtid_12873 *
                                                            m_11294 + i_13638];
                    double res_13644 = x_13642 / y_13643;
                    double x_13645 = ((__global
                                       double *) c_mem_16586)[gtid_12873 *
                                                              m_11298 +
                                                              i_13638];
                    double y_13646 = x_13645 / y_13643;
                    double res_13647 = 0.0 - y_13646;
                    
                    res_13640 = res_13644;
                    res_13641 = res_13647;
                } else {
                    res_13640 = 0.0;
                    res_13641 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_13627 = res_13640;
                    x_13628 = res_13641;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_12873, n_11293) && slt32(gtid_12883,
                                                          m_11294))) {
                    x_13627 = 0.0;
                    x_13628 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_13629 = x_13625 * x_13628;
                double res_13630 = x_13627 + y_13629;
                double res_13631 = x_13626 * x_13628;
                
                ((__local double *) scan_arr_mem_17078)[local_tid_17074] =
                    res_13630;
                ((__local double *) scan_arr_mem_17080)[local_tid_17074] =
                    res_13631;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_17085;
            double x_17086;
            double x_17087;
            double x_17088;
            double x_17092;
            double x_17093;
            double x_17094;
            double x_17095;
            int32_t skip_threads_17099;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_17074, segscan_group_sizze_13620)) {
                    x_17087 = ((volatile __local
                                double *) scan_arr_mem_17078)[local_tid_17074];
                    x_17088 = ((volatile __local
                                double *) scan_arr_mem_17080)[local_tid_17074];
                    if ((local_tid_17074 - squot32(local_tid_17074, 32) * 32) ==
                        0) {
                        x_17085 = x_17087;
                        x_17086 = x_17088;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17099 = 1;
                while (slt32(skip_threads_17099, 32)) {
                    if (sle32(skip_threads_17099, local_tid_17074 -
                              squot32(local_tid_17074, 32) * 32) &&
                        slt32(local_tid_17074, segscan_group_sizze_13620)) {
                        // read operands
                        {
                            x_17085 = ((volatile __local
                                        double *) scan_arr_mem_17078)[local_tid_17074 -
                                                                      skip_threads_17099];
                            x_17086 = ((volatile __local
                                        double *) scan_arr_mem_17080)[local_tid_17074 -
                                                                      skip_threads_17099];
                        }
                        // perform operation
                        {
                            bool inactive_17100 = slt32(srem32(local_tid_17074 +
                                                               chunk_offset_17083,
                                                               m_11294),
                                                        local_tid_17074 +
                                                        chunk_offset_17083 -
                                                        (local_tid_17074 -
                                                         skip_threads_17099 +
                                                         chunk_offset_17083));
                            
                            if (inactive_17100) {
                                x_17085 = x_17087;
                                x_17086 = x_17088;
                            }
                            if (!inactive_17100) {
                                double y_17089 = x_17085 * x_17088;
                                double res_17090 = x_17087 + y_17089;
                                double res_17091 = x_17086 * x_17088;
                                
                                x_17085 = res_17090;
                                x_17086 = res_17091;
                            }
                        }
                    }
                    if (sle32(wave_sizze_17076, skip_threads_17099)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17099, local_tid_17074 -
                              squot32(local_tid_17074, 32) * 32) &&
                        slt32(local_tid_17074, segscan_group_sizze_13620)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_17078)[local_tid_17074] =
                                x_17085;
                            x_17087 = x_17085;
                            ((volatile __local
                              double *) scan_arr_mem_17080)[local_tid_17074] =
                                x_17086;
                            x_17088 = x_17086;
                        }
                    }
                    if (sle32(wave_sizze_17076, skip_threads_17099)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17099 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17074 - squot32(local_tid_17074, 32) * 32) ==
                    31 && slt32(local_tid_17074, segscan_group_sizze_13620)) {
                    ((volatile __local
                      double *) scan_arr_mem_17078)[squot32(local_tid_17074,
                                                            32)] = x_17085;
                    ((volatile __local
                      double *) scan_arr_mem_17080)[squot32(local_tid_17074,
                                                            32)] = x_17086;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17101;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17074, 32) == 0 &&
                        slt32(local_tid_17074, segscan_group_sizze_13620)) {
                        x_17094 = ((volatile __local
                                    double *) scan_arr_mem_17078)[local_tid_17074];
                        x_17095 = ((volatile __local
                                    double *) scan_arr_mem_17080)[local_tid_17074];
                        if ((local_tid_17074 - squot32(local_tid_17074, 32) *
                             32) == 0) {
                            x_17092 = x_17094;
                            x_17093 = x_17095;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17101 = 1;
                    while (slt32(skip_threads_17101, 32)) {
                        if (sle32(skip_threads_17101, local_tid_17074 -
                                  squot32(local_tid_17074, 32) * 32) &&
                            (squot32(local_tid_17074, 32) == 0 &&
                             slt32(local_tid_17074,
                                   segscan_group_sizze_13620))) {
                            // read operands
                            {
                                x_17092 = ((volatile __local
                                            double *) scan_arr_mem_17078)[local_tid_17074 -
                                                                          skip_threads_17101];
                                x_17093 = ((volatile __local
                                            double *) scan_arr_mem_17080)[local_tid_17074 -
                                                                          skip_threads_17101];
                            }
                            // perform operation
                            {
                                bool inactive_17102 =
                                     slt32(srem32(local_tid_17074 * 32 + 32 -
                                                  1 + chunk_offset_17083,
                                                  m_11294), local_tid_17074 *
                                           32 + 32 - 1 + chunk_offset_17083 -
                                           ((local_tid_17074 -
                                             skip_threads_17101) * 32 + 32 - 1 +
                                            chunk_offset_17083));
                                
                                if (inactive_17102) {
                                    x_17092 = x_17094;
                                    x_17093 = x_17095;
                                }
                                if (!inactive_17102) {
                                    double y_17096 = x_17092 * x_17095;
                                    double res_17097 = x_17094 + y_17096;
                                    double res_17098 = x_17093 * x_17095;
                                    
                                    x_17092 = res_17097;
                                    x_17093 = res_17098;
                                }
                            }
                        }
                        if (sle32(wave_sizze_17076, skip_threads_17101)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17101, local_tid_17074 -
                                  squot32(local_tid_17074, 32) * 32) &&
                            (squot32(local_tid_17074, 32) == 0 &&
                             slt32(local_tid_17074,
                                   segscan_group_sizze_13620))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_17078)[local_tid_17074] =
                                    x_17092;
                                x_17094 = x_17092;
                                ((volatile __local
                                  double *) scan_arr_mem_17080)[local_tid_17074] =
                                    x_17093;
                                x_17095 = x_17093;
                            }
                        }
                        if (sle32(wave_sizze_17076, skip_threads_17101)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17101 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17074, 32) == 0 ||
                      !slt32(local_tid_17074, segscan_group_sizze_13620))) {
                    // read operands
                    {
                        x_17087 = x_17085;
                        x_17088 = x_17086;
                        x_17085 = ((__local
                                    double *) scan_arr_mem_17078)[squot32(local_tid_17074,
                                                                          32) -
                                                                  1];
                        x_17086 = ((__local
                                    double *) scan_arr_mem_17080)[squot32(local_tid_17074,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_17103 = slt32(srem32(local_tid_17074 +
                                                           chunk_offset_17083,
                                                           m_11294),
                                                    local_tid_17074 +
                                                    chunk_offset_17083 -
                                                    (squot32(local_tid_17074,
                                                             32) * 32 - 1 +
                                                     chunk_offset_17083));
                        
                        if (inactive_17103) {
                            x_17085 = x_17087;
                            x_17086 = x_17088;
                        }
                        if (!inactive_17103) {
                            double y_17089 = x_17085 * x_17088;
                            double res_17090 = x_17087 + y_17089;
                            double res_17091 = x_17086 * x_17088;
                            
                            x_17085 = res_17090;
                            x_17086 = res_17091;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_17078)[local_tid_17074] =
                            x_17085;
                        ((__local
                          double *) scan_arr_mem_17080)[local_tid_17074] =
                            x_17086;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17074, 32) == 0) {
                    ((__local double *) scan_arr_mem_17078)[local_tid_17074] =
                        x_17087;
                    ((__local double *) scan_arr_mem_17080)[local_tid_17074] =
                        x_17088;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_12873, n_11293) && slt32(gtid_12883, m_11294)) {
                    ((__global double *) mem_16690)[gtid_12873 * m_11294 +
                                                    gtid_12883] = ((__local
                                                                    double *) scan_arr_mem_17078)[local_tid_17074];
                    ((__global double *) mem_16695)[gtid_12873 * m_11294 +
                                                    gtid_12883] = ((__local
                                                                    double *) scan_arr_mem_17080)[local_tid_17074];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17104 = slt32(srem32(chunk_offset_17083 +
                                                          segscan_group_sizze_13620,
                                                          m_11294),
                                                   chunk_offset_17083 +
                                                   segscan_group_sizze_13620 -
                                                   (chunk_offset_17083 +
                                                    segscan_group_sizze_13620 -
                                                    1));
                bool should_load_carry_17105 = local_tid_17074 == 0 &&
                     !crosses_segment_17104;
                
                if (should_load_carry_17105) {
                    x_13625 = ((__local
                                double *) scan_arr_mem_17078)[segscan_group_sizze_13620 -
                                                              1];
                    x_13626 = ((__local
                                double *) scan_arr_mem_17080)[segscan_group_sizze_13620 -
                                                              1];
                }
                if (!should_load_carry_17105) {
                    x_13625 = 0.0;
                    x_13626 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_13620
}
__kernel void tridagNestedziscan_stage1_13039(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16997_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16995_backing_aligned_1,
                                              int32_t n_11293, int32_t m_11294,
                                              int32_t m_11300, __global
                                              unsigned char *a_mem_16584,
                                              __global
                                              unsigned char *y_mem_16587,
                                              __global unsigned char *mem_16663,
                                              __global unsigned char *mem_16669,
                                              __global unsigned char *mem_16674,
                                              int32_t num_threads_16989)
{
    #define segscan_group_sizze_13515 (tridagNestedzisegscan_group_sizze_13033)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16997_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16997_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16995_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16995_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16990;
    int32_t local_tid_16991;
    int32_t group_sizze_16994;
    int32_t wave_sizze_16993;
    int32_t group_tid_16992;
    
    global_tid_16990 = get_global_id(0);
    local_tid_16991 = get_local_id(0);
    group_sizze_16994 = get_local_size(0);
    wave_sizze_16993 = LOCKSTEP_WIDTH;
    group_tid_16992 = get_group_id(0);
    
    int32_t phys_tid_13039;
    
    phys_tid_13039 = global_tid_16990;
    
    __local char *scan_arr_mem_16995;
    __local char *scan_arr_mem_16997;
    
    scan_arr_mem_16995 = (__local char *) scan_arr_mem_16995_backing_0;
    scan_arr_mem_16997 = (__local char *) scan_arr_mem_16997_backing_1;
    
    double x_13520;
    double x_13521;
    double x_13522;
    double x_13523;
    
    x_13520 = 0.0;
    x_13521 = 1.0;
    for (int32_t j_16999 = 0; j_16999 < sdiv_up32(n_11293 * m_11294,
                                                  num_threads_16989);
         j_16999++) {
        int32_t chunk_offset_17000 = segscan_group_sizze_13515 * j_16999 +
                group_tid_16992 * (segscan_group_sizze_13515 *
                                   sdiv_up32(n_11293 * m_11294,
                                             num_threads_16989));
        int32_t flat_idx_17001 = chunk_offset_17000 + local_tid_16991;
        int32_t gtid_13028 = squot32(flat_idx_17001, m_11294);
        int32_t gtid_13038 = flat_idx_17001 - squot32(flat_idx_17001, m_11294) *
                m_11294;
        
        // threads in bounds read input
        {
            if (slt32(gtid_13028, n_11293) && slt32(gtid_13038, m_11294)) {
                bool cond_13534 = slt32(0, gtid_13038);
                double res_13535;
                
                if (cond_13534) {
                    double x_elem_13532 = ((__global
                                            double *) y_mem_16587)[gtid_13028 *
                                                                   m_11300 +
                                                                   gtid_13038];
                    
                    res_13535 = x_elem_13532;
                } else {
                    res_13535 = 0.0;
                }
                
                double res_13536;
                
                if (cond_13534) {
                    double x_elem_13533 = ((__global
                                            double *) a_mem_16584)[gtid_13028 *
                                                                   m_11294 +
                                                                   gtid_13038];
                    int32_t i_13537 = sub32(gtid_13038, 1);
                    double y_13538 = ((__global
                                       double *) mem_16663)[gtid_13028 *
                                                            m_11294 + i_13537];
                    double y_13539 = x_elem_13533 / y_13538;
                    double res_13540 = 0.0 - y_13539;
                    
                    res_13536 = res_13540;
                } else {
                    res_13536 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_13522 = res_13535;
                    x_13523 = res_13536;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_13028, n_11293) && slt32(gtid_13038,
                                                          m_11294))) {
                    x_13522 = 0.0;
                    x_13523 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_13524 = x_13520 * x_13523;
                double res_13525 = x_13522 + y_13524;
                double res_13526 = x_13521 * x_13523;
                
                ((__local double *) scan_arr_mem_16995)[local_tid_16991] =
                    res_13525;
                ((__local double *) scan_arr_mem_16997)[local_tid_16991] =
                    res_13526;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_17002;
            double x_17003;
            double x_17004;
            double x_17005;
            double x_17009;
            double x_17010;
            double x_17011;
            double x_17012;
            int32_t skip_threads_17016;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16991, segscan_group_sizze_13515)) {
                    x_17004 = ((volatile __local
                                double *) scan_arr_mem_16995)[local_tid_16991];
                    x_17005 = ((volatile __local
                                double *) scan_arr_mem_16997)[local_tid_16991];
                    if ((local_tid_16991 - squot32(local_tid_16991, 32) * 32) ==
                        0) {
                        x_17002 = x_17004;
                        x_17003 = x_17005;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17016 = 1;
                while (slt32(skip_threads_17016, 32)) {
                    if (sle32(skip_threads_17016, local_tid_16991 -
                              squot32(local_tid_16991, 32) * 32) &&
                        slt32(local_tid_16991, segscan_group_sizze_13515)) {
                        // read operands
                        {
                            x_17002 = ((volatile __local
                                        double *) scan_arr_mem_16995)[local_tid_16991 -
                                                                      skip_threads_17016];
                            x_17003 = ((volatile __local
                                        double *) scan_arr_mem_16997)[local_tid_16991 -
                                                                      skip_threads_17016];
                        }
                        // perform operation
                        {
                            bool inactive_17017 = slt32(srem32(local_tid_16991 +
                                                               chunk_offset_17000,
                                                               m_11294),
                                                        local_tid_16991 +
                                                        chunk_offset_17000 -
                                                        (local_tid_16991 -
                                                         skip_threads_17016 +
                                                         chunk_offset_17000));
                            
                            if (inactive_17017) {
                                x_17002 = x_17004;
                                x_17003 = x_17005;
                            }
                            if (!inactive_17017) {
                                double y_17006 = x_17002 * x_17005;
                                double res_17007 = x_17004 + y_17006;
                                double res_17008 = x_17003 * x_17005;
                                
                                x_17002 = res_17007;
                                x_17003 = res_17008;
                            }
                        }
                    }
                    if (sle32(wave_sizze_16993, skip_threads_17016)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17016, local_tid_16991 -
                              squot32(local_tid_16991, 32) * 32) &&
                        slt32(local_tid_16991, segscan_group_sizze_13515)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_16995)[local_tid_16991] =
                                x_17002;
                            x_17004 = x_17002;
                            ((volatile __local
                              double *) scan_arr_mem_16997)[local_tid_16991] =
                                x_17003;
                            x_17005 = x_17003;
                        }
                    }
                    if (sle32(wave_sizze_16993, skip_threads_17016)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17016 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16991 - squot32(local_tid_16991, 32) * 32) ==
                    31 && slt32(local_tid_16991, segscan_group_sizze_13515)) {
                    ((volatile __local
                      double *) scan_arr_mem_16995)[squot32(local_tid_16991,
                                                            32)] = x_17002;
                    ((volatile __local
                      double *) scan_arr_mem_16997)[squot32(local_tid_16991,
                                                            32)] = x_17003;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17018;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16991, 32) == 0 &&
                        slt32(local_tid_16991, segscan_group_sizze_13515)) {
                        x_17011 = ((volatile __local
                                    double *) scan_arr_mem_16995)[local_tid_16991];
                        x_17012 = ((volatile __local
                                    double *) scan_arr_mem_16997)[local_tid_16991];
                        if ((local_tid_16991 - squot32(local_tid_16991, 32) *
                             32) == 0) {
                            x_17009 = x_17011;
                            x_17010 = x_17012;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17018 = 1;
                    while (slt32(skip_threads_17018, 32)) {
                        if (sle32(skip_threads_17018, local_tid_16991 -
                                  squot32(local_tid_16991, 32) * 32) &&
                            (squot32(local_tid_16991, 32) == 0 &&
                             slt32(local_tid_16991,
                                   segscan_group_sizze_13515))) {
                            // read operands
                            {
                                x_17009 = ((volatile __local
                                            double *) scan_arr_mem_16995)[local_tid_16991 -
                                                                          skip_threads_17018];
                                x_17010 = ((volatile __local
                                            double *) scan_arr_mem_16997)[local_tid_16991 -
                                                                          skip_threads_17018];
                            }
                            // perform operation
                            {
                                bool inactive_17019 =
                                     slt32(srem32(local_tid_16991 * 32 + 32 -
                                                  1 + chunk_offset_17000,
                                                  m_11294), local_tid_16991 *
                                           32 + 32 - 1 + chunk_offset_17000 -
                                           ((local_tid_16991 -
                                             skip_threads_17018) * 32 + 32 - 1 +
                                            chunk_offset_17000));
                                
                                if (inactive_17019) {
                                    x_17009 = x_17011;
                                    x_17010 = x_17012;
                                }
                                if (!inactive_17019) {
                                    double y_17013 = x_17009 * x_17012;
                                    double res_17014 = x_17011 + y_17013;
                                    double res_17015 = x_17010 * x_17012;
                                    
                                    x_17009 = res_17014;
                                    x_17010 = res_17015;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16993, skip_threads_17018)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17018, local_tid_16991 -
                                  squot32(local_tid_16991, 32) * 32) &&
                            (squot32(local_tid_16991, 32) == 0 &&
                             slt32(local_tid_16991,
                                   segscan_group_sizze_13515))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_16995)[local_tid_16991] =
                                    x_17009;
                                x_17011 = x_17009;
                                ((volatile __local
                                  double *) scan_arr_mem_16997)[local_tid_16991] =
                                    x_17010;
                                x_17012 = x_17010;
                            }
                        }
                        if (sle32(wave_sizze_16993, skip_threads_17018)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17018 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16991, 32) == 0 ||
                      !slt32(local_tid_16991, segscan_group_sizze_13515))) {
                    // read operands
                    {
                        x_17004 = x_17002;
                        x_17005 = x_17003;
                        x_17002 = ((__local
                                    double *) scan_arr_mem_16995)[squot32(local_tid_16991,
                                                                          32) -
                                                                  1];
                        x_17003 = ((__local
                                    double *) scan_arr_mem_16997)[squot32(local_tid_16991,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_17020 = slt32(srem32(local_tid_16991 +
                                                           chunk_offset_17000,
                                                           m_11294),
                                                    local_tid_16991 +
                                                    chunk_offset_17000 -
                                                    (squot32(local_tid_16991,
                                                             32) * 32 - 1 +
                                                     chunk_offset_17000));
                        
                        if (inactive_17020) {
                            x_17002 = x_17004;
                            x_17003 = x_17005;
                        }
                        if (!inactive_17020) {
                            double y_17006 = x_17002 * x_17005;
                            double res_17007 = x_17004 + y_17006;
                            double res_17008 = x_17003 * x_17005;
                            
                            x_17002 = res_17007;
                            x_17003 = res_17008;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_16995)[local_tid_16991] =
                            x_17002;
                        ((__local
                          double *) scan_arr_mem_16997)[local_tid_16991] =
                            x_17003;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16991, 32) == 0) {
                    ((__local double *) scan_arr_mem_16995)[local_tid_16991] =
                        x_17004;
                    ((__local double *) scan_arr_mem_16997)[local_tid_16991] =
                        x_17005;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_13028, n_11293) && slt32(gtid_13038, m_11294)) {
                    ((__global double *) mem_16669)[gtid_13028 * m_11294 +
                                                    gtid_13038] = ((__local
                                                                    double *) scan_arr_mem_16995)[local_tid_16991];
                    ((__global double *) mem_16674)[gtid_13028 * m_11294 +
                                                    gtid_13038] = ((__local
                                                                    double *) scan_arr_mem_16997)[local_tid_16991];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17021 = slt32(srem32(chunk_offset_17000 +
                                                          segscan_group_sizze_13515,
                                                          m_11294),
                                                   chunk_offset_17000 +
                                                   segscan_group_sizze_13515 -
                                                   (chunk_offset_17000 +
                                                    segscan_group_sizze_13515 -
                                                    1));
                bool should_load_carry_17022 = local_tid_16991 == 0 &&
                     !crosses_segment_17021;
                
                if (should_load_carry_17022) {
                    x_13520 = ((__local
                                double *) scan_arr_mem_16995)[segscan_group_sizze_13515 -
                                                              1];
                    x_13521 = ((__local
                                double *) scan_arr_mem_16997)[segscan_group_sizze_13515 -
                                                              1];
                }
                if (!should_load_carry_17022) {
                    x_13520 = 0.0;
                    x_13521 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_13515
}
__kernel void tridagNestedziscan_stage1_13272(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16864_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16862_backing_aligned_1,
                                              __local volatile
                                              int64_t *scan_arr_mem_16860_backing_aligned_2,
                                              __local volatile
                                              int64_t *scan_arr_mem_16858_backing_aligned_3,
                                              int32_t n_11293, int32_t m_11294,
                                              int32_t m_11296, int32_t m_11298,
                                              __global
                                              unsigned char *a_mem_16584,
                                              __global
                                              unsigned char *b_mem_16585,
                                              __global
                                              unsigned char *c_mem_16586,
                                              __global unsigned char *mem_16642,
                                              __global unsigned char *mem_16647,
                                              __global unsigned char *mem_16652,
                                              __global unsigned char *mem_16657,
                                              int32_t num_threads_16852)
{
    #define segscan_group_sizze_13347 (tridagNestedzisegscan_group_sizze_13266)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16864_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16864_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16862_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16862_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16860_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16860_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16858_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16858_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16853;
    int32_t local_tid_16854;
    int32_t group_sizze_16857;
    int32_t wave_sizze_16856;
    int32_t group_tid_16855;
    
    global_tid_16853 = get_global_id(0);
    local_tid_16854 = get_local_id(0);
    group_sizze_16857 = get_local_size(0);
    wave_sizze_16856 = LOCKSTEP_WIDTH;
    group_tid_16855 = get_group_id(0);
    
    int32_t phys_tid_13272;
    
    phys_tid_13272 = global_tid_16853;
    
    __local char *scan_arr_mem_16858;
    __local char *scan_arr_mem_16860;
    __local char *scan_arr_mem_16862;
    __local char *scan_arr_mem_16864;
    
    scan_arr_mem_16858 = (__local char *) scan_arr_mem_16858_backing_0;
    scan_arr_mem_16860 = (__local char *) scan_arr_mem_16860_backing_1;
    scan_arr_mem_16862 = (__local char *) scan_arr_mem_16862_backing_2;
    scan_arr_mem_16864 = (__local char *) scan_arr_mem_16864_backing_3;
    
    double x_13354;
    double x_13355;
    double x_13356;
    double x_13357;
    double x_13358;
    double x_13359;
    double x_13360;
    double x_13361;
    
    x_13354 = 1.0;
    x_13355 = 0.0;
    x_13356 = 0.0;
    x_13357 = 1.0;
    for (int32_t j_16866 = 0; j_16866 < sdiv_up32(n_11293 * m_11294,
                                                  num_threads_16852);
         j_16866++) {
        int32_t chunk_offset_16867 = segscan_group_sizze_13347 * j_16866 +
                group_tid_16855 * (segscan_group_sizze_13347 *
                                   sdiv_up32(n_11293 * m_11294,
                                             num_threads_16852));
        int32_t flat_idx_16868 = chunk_offset_16867 + local_tid_16854;
        int32_t gtid_13261 = squot32(flat_idx_16868, m_11294);
        int32_t gtid_13271 = flat_idx_16868 - squot32(flat_idx_16868, m_11294) *
                m_11294;
        
        // threads in bounds read input
        {
            if (slt32(gtid_13261, n_11293) && slt32(gtid_13271, m_11294)) {
                bool cond_13386 = slt32(0, gtid_13271);
                double res_13387;
                
                if (cond_13386) {
                    res_13387 = 1.0;
                } else {
                    res_13387 = 0.0;
                }
                
                double res_13388;
                
                if (cond_13386) {
                    res_13388 = 0.0;
                } else {
                    res_13388 = 1.0;
                }
                
                double res_13389;
                
                if (cond_13386) {
                    double x_elem_13384 = ((__global
                                            double *) b_mem_16585)[gtid_13261 *
                                                                   m_11296 +
                                                                   gtid_13271];
                    
                    res_13389 = x_elem_13384;
                } else {
                    res_13389 = 1.0;
                }
                
                double res_13390;
                
                if (cond_13386) {
                    double x_elem_13385 = ((__global
                                            double *) a_mem_16584)[gtid_13261 *
                                                                   m_11294 +
                                                                   gtid_13271];
                    int32_t i_13391 = sub32(gtid_13271, 1);
                    double y_13392 = ((__global
                                       double *) c_mem_16586)[gtid_13261 *
                                                              m_11298 +
                                                              i_13391];
                    double y_13393 = x_elem_13385 * y_13392;
                    double res_13394 = 0.0 - y_13393;
                    
                    res_13390 = res_13394;
                } else {
                    res_13390 = 0.0;
                }
                // write to-scan values to parameters
                {
                    x_13358 = res_13389;
                    x_13359 = res_13390;
                    x_13360 = res_13387;
                    x_13361 = res_13388;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_13261, n_11293) && slt32(gtid_13271,
                                                          m_11294))) {
                    x_13358 = 1.0;
                    x_13359 = 0.0;
                    x_13360 = 0.0;
                    x_13361 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_13362 = x_13354 * x_13358;
                double value_13363 = 1.0 / y_13362;
                double y_13364 = x_13356 * x_13359;
                double x_13365 = y_13362 + y_13364;
                double res_13366 = value_13363 * x_13365;
                double x_13367 = x_13355 * x_13358;
                double y_13368 = x_13357 * x_13359;
                double x_13369 = x_13367 + y_13368;
                double res_13370 = value_13363 * x_13369;
                double x_13371 = x_13354 * x_13360;
                double y_13372 = x_13356 * x_13361;
                double x_13373 = x_13371 + y_13372;
                double res_13374 = value_13363 * x_13373;
                double x_13375 = x_13355 * x_13360;
                double y_13376 = x_13357 * x_13361;
                double x_13377 = x_13375 + y_13376;
                double res_13378 = value_13363 * x_13377;
                
                ((__local double *) scan_arr_mem_16858)[local_tid_16854] =
                    res_13366;
                ((__local double *) scan_arr_mem_16860)[local_tid_16854] =
                    res_13370;
                ((__local double *) scan_arr_mem_16862)[local_tid_16854] =
                    res_13374;
                ((__local double *) scan_arr_mem_16864)[local_tid_16854] =
                    res_13378;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_16869;
            double x_16870;
            double x_16871;
            double x_16872;
            double x_16873;
            double x_16874;
            double x_16875;
            double x_16876;
            double x_16894;
            double x_16895;
            double x_16896;
            double x_16897;
            double x_16898;
            double x_16899;
            double x_16900;
            double x_16901;
            int32_t skip_threads_16919;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16854, segscan_group_sizze_13347)) {
                    x_16873 = ((volatile __local
                                double *) scan_arr_mem_16858)[local_tid_16854];
                    x_16874 = ((volatile __local
                                double *) scan_arr_mem_16860)[local_tid_16854];
                    x_16875 = ((volatile __local
                                double *) scan_arr_mem_16862)[local_tid_16854];
                    x_16876 = ((volatile __local
                                double *) scan_arr_mem_16864)[local_tid_16854];
                    if ((local_tid_16854 - squot32(local_tid_16854, 32) * 32) ==
                        0) {
                        x_16869 = x_16873;
                        x_16870 = x_16874;
                        x_16871 = x_16875;
                        x_16872 = x_16876;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16919 = 1;
                while (slt32(skip_threads_16919, 32)) {
                    if (sle32(skip_threads_16919, local_tid_16854 -
                              squot32(local_tid_16854, 32) * 32) &&
                        slt32(local_tid_16854, segscan_group_sizze_13347)) {
                        // read operands
                        {
                            x_16869 = ((volatile __local
                                        double *) scan_arr_mem_16858)[local_tid_16854 -
                                                                      skip_threads_16919];
                            x_16870 = ((volatile __local
                                        double *) scan_arr_mem_16860)[local_tid_16854 -
                                                                      skip_threads_16919];
                            x_16871 = ((volatile __local
                                        double *) scan_arr_mem_16862)[local_tid_16854 -
                                                                      skip_threads_16919];
                            x_16872 = ((volatile __local
                                        double *) scan_arr_mem_16864)[local_tid_16854 -
                                                                      skip_threads_16919];
                        }
                        // perform operation
                        {
                            bool inactive_16920 = slt32(srem32(local_tid_16854 +
                                                               chunk_offset_16867,
                                                               m_11294),
                                                        local_tid_16854 +
                                                        chunk_offset_16867 -
                                                        (local_tid_16854 -
                                                         skip_threads_16919 +
                                                         chunk_offset_16867));
                            
                            if (inactive_16920) {
                                x_16869 = x_16873;
                                x_16870 = x_16874;
                                x_16871 = x_16875;
                                x_16872 = x_16876;
                            }
                            if (!inactive_16920) {
                                double y_16877 = x_16869 * x_16873;
                                double value_16878 = 1.0 / y_16877;
                                double y_16879 = x_16871 * x_16874;
                                double x_16880 = y_16877 + y_16879;
                                double res_16881 = value_16878 * x_16880;
                                double x_16882 = x_16870 * x_16873;
                                double y_16883 = x_16872 * x_16874;
                                double x_16884 = x_16882 + y_16883;
                                double res_16885 = value_16878 * x_16884;
                                double x_16886 = x_16869 * x_16875;
                                double y_16887 = x_16871 * x_16876;
                                double x_16888 = x_16886 + y_16887;
                                double res_16889 = value_16878 * x_16888;
                                double x_16890 = x_16870 * x_16875;
                                double y_16891 = x_16872 * x_16876;
                                double x_16892 = x_16890 + y_16891;
                                double res_16893 = value_16878 * x_16892;
                                
                                x_16869 = res_16881;
                                x_16870 = res_16885;
                                x_16871 = res_16889;
                                x_16872 = res_16893;
                            }
                        }
                    }
                    if (sle32(wave_sizze_16856, skip_threads_16919)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16919, local_tid_16854 -
                              squot32(local_tid_16854, 32) * 32) &&
                        slt32(local_tid_16854, segscan_group_sizze_13347)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_16858)[local_tid_16854] =
                                x_16869;
                            x_16873 = x_16869;
                            ((volatile __local
                              double *) scan_arr_mem_16860)[local_tid_16854] =
                                x_16870;
                            x_16874 = x_16870;
                            ((volatile __local
                              double *) scan_arr_mem_16862)[local_tid_16854] =
                                x_16871;
                            x_16875 = x_16871;
                            ((volatile __local
                              double *) scan_arr_mem_16864)[local_tid_16854] =
                                x_16872;
                            x_16876 = x_16872;
                        }
                    }
                    if (sle32(wave_sizze_16856, skip_threads_16919)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16919 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16854 - squot32(local_tid_16854, 32) * 32) ==
                    31 && slt32(local_tid_16854, segscan_group_sizze_13347)) {
                    ((volatile __local
                      double *) scan_arr_mem_16858)[squot32(local_tid_16854,
                                                            32)] = x_16869;
                    ((volatile __local
                      double *) scan_arr_mem_16860)[squot32(local_tid_16854,
                                                            32)] = x_16870;
                    ((volatile __local
                      double *) scan_arr_mem_16862)[squot32(local_tid_16854,
                                                            32)] = x_16871;
                    ((volatile __local
                      double *) scan_arr_mem_16864)[squot32(local_tid_16854,
                                                            32)] = x_16872;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16921;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16854, 32) == 0 &&
                        slt32(local_tid_16854, segscan_group_sizze_13347)) {
                        x_16898 = ((volatile __local
                                    double *) scan_arr_mem_16858)[local_tid_16854];
                        x_16899 = ((volatile __local
                                    double *) scan_arr_mem_16860)[local_tid_16854];
                        x_16900 = ((volatile __local
                                    double *) scan_arr_mem_16862)[local_tid_16854];
                        x_16901 = ((volatile __local
                                    double *) scan_arr_mem_16864)[local_tid_16854];
                        if ((local_tid_16854 - squot32(local_tid_16854, 32) *
                             32) == 0) {
                            x_16894 = x_16898;
                            x_16895 = x_16899;
                            x_16896 = x_16900;
                            x_16897 = x_16901;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16921 = 1;
                    while (slt32(skip_threads_16921, 32)) {
                        if (sle32(skip_threads_16921, local_tid_16854 -
                                  squot32(local_tid_16854, 32) * 32) &&
                            (squot32(local_tid_16854, 32) == 0 &&
                             slt32(local_tid_16854,
                                   segscan_group_sizze_13347))) {
                            // read operands
                            {
                                x_16894 = ((volatile __local
                                            double *) scan_arr_mem_16858)[local_tid_16854 -
                                                                          skip_threads_16921];
                                x_16895 = ((volatile __local
                                            double *) scan_arr_mem_16860)[local_tid_16854 -
                                                                          skip_threads_16921];
                                x_16896 = ((volatile __local
                                            double *) scan_arr_mem_16862)[local_tid_16854 -
                                                                          skip_threads_16921];
                                x_16897 = ((volatile __local
                                            double *) scan_arr_mem_16864)[local_tid_16854 -
                                                                          skip_threads_16921];
                            }
                            // perform operation
                            {
                                bool inactive_16922 =
                                     slt32(srem32(local_tid_16854 * 32 + 32 -
                                                  1 + chunk_offset_16867,
                                                  m_11294), local_tid_16854 *
                                           32 + 32 - 1 + chunk_offset_16867 -
                                           ((local_tid_16854 -
                                             skip_threads_16921) * 32 + 32 - 1 +
                                            chunk_offset_16867));
                                
                                if (inactive_16922) {
                                    x_16894 = x_16898;
                                    x_16895 = x_16899;
                                    x_16896 = x_16900;
                                    x_16897 = x_16901;
                                }
                                if (!inactive_16922) {
                                    double y_16902 = x_16894 * x_16898;
                                    double value_16903 = 1.0 / y_16902;
                                    double y_16904 = x_16896 * x_16899;
                                    double x_16905 = y_16902 + y_16904;
                                    double res_16906 = value_16903 * x_16905;
                                    double x_16907 = x_16895 * x_16898;
                                    double y_16908 = x_16897 * x_16899;
                                    double x_16909 = x_16907 + y_16908;
                                    double res_16910 = value_16903 * x_16909;
                                    double x_16911 = x_16894 * x_16900;
                                    double y_16912 = x_16896 * x_16901;
                                    double x_16913 = x_16911 + y_16912;
                                    double res_16914 = value_16903 * x_16913;
                                    double x_16915 = x_16895 * x_16900;
                                    double y_16916 = x_16897 * x_16901;
                                    double x_16917 = x_16915 + y_16916;
                                    double res_16918 = value_16903 * x_16917;
                                    
                                    x_16894 = res_16906;
                                    x_16895 = res_16910;
                                    x_16896 = res_16914;
                                    x_16897 = res_16918;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16856, skip_threads_16921)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16921, local_tid_16854 -
                                  squot32(local_tid_16854, 32) * 32) &&
                            (squot32(local_tid_16854, 32) == 0 &&
                             slt32(local_tid_16854,
                                   segscan_group_sizze_13347))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_16858)[local_tid_16854] =
                                    x_16894;
                                x_16898 = x_16894;
                                ((volatile __local
                                  double *) scan_arr_mem_16860)[local_tid_16854] =
                                    x_16895;
                                x_16899 = x_16895;
                                ((volatile __local
                                  double *) scan_arr_mem_16862)[local_tid_16854] =
                                    x_16896;
                                x_16900 = x_16896;
                                ((volatile __local
                                  double *) scan_arr_mem_16864)[local_tid_16854] =
                                    x_16897;
                                x_16901 = x_16897;
                            }
                        }
                        if (sle32(wave_sizze_16856, skip_threads_16921)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16921 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16854, 32) == 0 ||
                      !slt32(local_tid_16854, segscan_group_sizze_13347))) {
                    // read operands
                    {
                        x_16873 = x_16869;
                        x_16874 = x_16870;
                        x_16875 = x_16871;
                        x_16876 = x_16872;
                        x_16869 = ((__local
                                    double *) scan_arr_mem_16858)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                        x_16870 = ((__local
                                    double *) scan_arr_mem_16860)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                        x_16871 = ((__local
                                    double *) scan_arr_mem_16862)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                        x_16872 = ((__local
                                    double *) scan_arr_mem_16864)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_16923 = slt32(srem32(local_tid_16854 +
                                                           chunk_offset_16867,
                                                           m_11294),
                                                    local_tid_16854 +
                                                    chunk_offset_16867 -
                                                    (squot32(local_tid_16854,
                                                             32) * 32 - 1 +
                                                     chunk_offset_16867));
                        
                        if (inactive_16923) {
                            x_16869 = x_16873;
                            x_16870 = x_16874;
                            x_16871 = x_16875;
                            x_16872 = x_16876;
                        }
                        if (!inactive_16923) {
                            double y_16877 = x_16869 * x_16873;
                            double value_16878 = 1.0 / y_16877;
                            double y_16879 = x_16871 * x_16874;
                            double x_16880 = y_16877 + y_16879;
                            double res_16881 = value_16878 * x_16880;
                            double x_16882 = x_16870 * x_16873;
                            double y_16883 = x_16872 * x_16874;
                            double x_16884 = x_16882 + y_16883;
                            double res_16885 = value_16878 * x_16884;
                            double x_16886 = x_16869 * x_16875;
                            double y_16887 = x_16871 * x_16876;
                            double x_16888 = x_16886 + y_16887;
                            double res_16889 = value_16878 * x_16888;
                            double x_16890 = x_16870 * x_16875;
                            double y_16891 = x_16872 * x_16876;
                            double x_16892 = x_16890 + y_16891;
                            double res_16893 = value_16878 * x_16892;
                            
                            x_16869 = res_16881;
                            x_16870 = res_16885;
                            x_16871 = res_16889;
                            x_16872 = res_16893;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_16858)[local_tid_16854] =
                            x_16869;
                        ((__local
                          double *) scan_arr_mem_16860)[local_tid_16854] =
                            x_16870;
                        ((__local
                          double *) scan_arr_mem_16862)[local_tid_16854] =
                            x_16871;
                        ((__local
                          double *) scan_arr_mem_16864)[local_tid_16854] =
                            x_16872;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16854, 32) == 0) {
                    ((__local double *) scan_arr_mem_16858)[local_tid_16854] =
                        x_16873;
                    ((__local double *) scan_arr_mem_16860)[local_tid_16854] =
                        x_16874;
                    ((__local double *) scan_arr_mem_16862)[local_tid_16854] =
                        x_16875;
                    ((__local double *) scan_arr_mem_16864)[local_tid_16854] =
                        x_16876;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_13261, n_11293) && slt32(gtid_13271, m_11294)) {
                    ((__global double *) mem_16642)[gtid_13261 * m_11294 +
                                                    gtid_13271] = ((__local
                                                                    double *) scan_arr_mem_16858)[local_tid_16854];
                    ((__global double *) mem_16647)[gtid_13261 * m_11294 +
                                                    gtid_13271] = ((__local
                                                                    double *) scan_arr_mem_16860)[local_tid_16854];
                    ((__global double *) mem_16652)[gtid_13261 * m_11294 +
                                                    gtid_13271] = ((__local
                                                                    double *) scan_arr_mem_16862)[local_tid_16854];
                    ((__global double *) mem_16657)[gtid_13261 * m_11294 +
                                                    gtid_13271] = ((__local
                                                                    double *) scan_arr_mem_16864)[local_tid_16854];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16924 = slt32(srem32(chunk_offset_16867 +
                                                          segscan_group_sizze_13347,
                                                          m_11294),
                                                   chunk_offset_16867 +
                                                   segscan_group_sizze_13347 -
                                                   (chunk_offset_16867 +
                                                    segscan_group_sizze_13347 -
                                                    1));
                bool should_load_carry_16925 = local_tid_16854 == 0 &&
                     !crosses_segment_16924;
                
                if (should_load_carry_16925) {
                    x_13354 = ((__local
                                double *) scan_arr_mem_16858)[segscan_group_sizze_13347 -
                                                              1];
                    x_13355 = ((__local
                                double *) scan_arr_mem_16860)[segscan_group_sizze_13347 -
                                                              1];
                    x_13356 = ((__local
                                double *) scan_arr_mem_16862)[segscan_group_sizze_13347 -
                                                              1];
                    x_13357 = ((__local
                                double *) scan_arr_mem_16864)[segscan_group_sizze_13347 -
                                                              1];
                }
                if (!should_load_carry_16925) {
                    x_13354 = 1.0;
                    x_13355 = 0.0;
                    x_13356 = 0.0;
                    x_13357 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_13347
}
__kernel void tridagNestedziscan_stage2_12884(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_17113_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_17111_backing_aligned_1,
                                              int32_t n_11293, int32_t m_11294,
                                              __global unsigned char *mem_16690,
                                              __global unsigned char *mem_16695,
                                              int32_t stage1_num_groups_17071,
                                              int32_t num_threads_17072)
{
    #define segscan_group_sizze_13620 (tridagNestedzisegscan_group_sizze_12878)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17113_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17113_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17111_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17111_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17106;
    int32_t local_tid_17107;
    int32_t group_sizze_17110;
    int32_t wave_sizze_17109;
    int32_t group_tid_17108;
    
    global_tid_17106 = get_global_id(0);
    local_tid_17107 = get_local_id(0);
    group_sizze_17110 = get_local_size(0);
    wave_sizze_17109 = LOCKSTEP_WIDTH;
    group_tid_17108 = get_group_id(0);
    
    int32_t phys_tid_12884;
    
    phys_tid_12884 = global_tid_17106;
    
    __local char *scan_arr_mem_17111;
    __local char *scan_arr_mem_17113;
    
    scan_arr_mem_17111 = (__local char *) scan_arr_mem_17111_backing_0;
    scan_arr_mem_17113 = (__local char *) scan_arr_mem_17113_backing_1;
    
    int32_t flat_idx_17115;
    
    flat_idx_17115 = (local_tid_17107 + 1) * (segscan_group_sizze_13620 *
                                              sdiv_up32(n_11293 * m_11294,
                                                        num_threads_17072)) - 1;
    
    int32_t gtid_12873;
    
    gtid_12873 = squot32(flat_idx_17115, m_11294);
    
    int32_t gtid_12883;
    
    gtid_12883 = flat_idx_17115 - squot32(flat_idx_17115, m_11294) * m_11294;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_12873, n_11293) && slt32(gtid_12883, m_11294)) {
            ((__local double *) scan_arr_mem_17111)[local_tid_17107] =
                ((__global double *) mem_16690)[gtid_12873 * m_11294 +
                                                gtid_12883];
            ((__local double *) scan_arr_mem_17113)[local_tid_17107] =
                ((__global double *) mem_16695)[gtid_12873 * m_11294 +
                                                gtid_12883];
        } else {
            ((__local double *) scan_arr_mem_17111)[local_tid_17107] = 0.0;
            ((__local double *) scan_arr_mem_17113)[local_tid_17107] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_13625;
    double x_13626;
    double x_13627;
    double x_13628;
    double x_17116;
    double x_17117;
    double x_17118;
    double x_17119;
    int32_t skip_threads_17123;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_17107, stage1_num_groups_17071)) {
            x_13627 = ((volatile __local
                        double *) scan_arr_mem_17111)[local_tid_17107];
            x_13628 = ((volatile __local
                        double *) scan_arr_mem_17113)[local_tid_17107];
            if ((local_tid_17107 - squot32(local_tid_17107, 32) * 32) == 0) {
                x_13625 = x_13627;
                x_13626 = x_13628;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17123 = 1;
        while (slt32(skip_threads_17123, 32)) {
            if (sle32(skip_threads_17123, local_tid_17107 -
                      squot32(local_tid_17107, 32) * 32) &&
                slt32(local_tid_17107, stage1_num_groups_17071)) {
                // read operands
                {
                    x_13625 = ((volatile __local
                                double *) scan_arr_mem_17111)[local_tid_17107 -
                                                              skip_threads_17123];
                    x_13626 = ((volatile __local
                                double *) scan_arr_mem_17113)[local_tid_17107 -
                                                              skip_threads_17123];
                }
                // perform operation
                {
                    bool inactive_17124 = slt32(srem32((local_tid_17107 + 1) *
                                                       (segscan_group_sizze_13620 *
                                                        sdiv_up32(n_11293 *
                                                                  m_11294,
                                                                  num_threads_17072)) -
                                                       1, m_11294),
                                                (local_tid_17107 + 1) *
                                                (segscan_group_sizze_13620 *
                                                 sdiv_up32(n_11293 * m_11294,
                                                           num_threads_17072)) -
                                                1 - ((local_tid_17107 -
                                                      skip_threads_17123 + 1) *
                                                     (segscan_group_sizze_13620 *
                                                      sdiv_up32(n_11293 *
                                                                m_11294,
                                                                num_threads_17072)) -
                                                     1));
                    
                    if (inactive_17124) {
                        x_13625 = x_13627;
                        x_13626 = x_13628;
                    }
                    if (!inactive_17124) {
                        double y_13629 = x_13625 * x_13628;
                        double res_13630 = x_13627 + y_13629;
                        double res_13631 = x_13626 * x_13628;
                        
                        x_13625 = res_13630;
                        x_13626 = res_13631;
                    }
                }
            }
            if (sle32(wave_sizze_17109, skip_threads_17123)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17123, local_tid_17107 -
                      squot32(local_tid_17107, 32) * 32) &&
                slt32(local_tid_17107, stage1_num_groups_17071)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_17111)[local_tid_17107] = x_13625;
                    x_13627 = x_13625;
                    ((volatile __local
                      double *) scan_arr_mem_17113)[local_tid_17107] = x_13626;
                    x_13628 = x_13626;
                }
            }
            if (sle32(wave_sizze_17109, skip_threads_17123)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17123 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17107 - squot32(local_tid_17107, 32) * 32) == 31 &&
            slt32(local_tid_17107, stage1_num_groups_17071)) {
            ((volatile __local
              double *) scan_arr_mem_17111)[squot32(local_tid_17107, 32)] =
                x_13625;
            ((volatile __local
              double *) scan_arr_mem_17113)[squot32(local_tid_17107, 32)] =
                x_13626;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17125;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17107, 32) == 0 && slt32(local_tid_17107,
                                                           stage1_num_groups_17071)) {
                x_17118 = ((volatile __local
                            double *) scan_arr_mem_17111)[local_tid_17107];
                x_17119 = ((volatile __local
                            double *) scan_arr_mem_17113)[local_tid_17107];
                if ((local_tid_17107 - squot32(local_tid_17107, 32) * 32) ==
                    0) {
                    x_17116 = x_17118;
                    x_17117 = x_17119;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17125 = 1;
            while (slt32(skip_threads_17125, 32)) {
                if (sle32(skip_threads_17125, local_tid_17107 -
                          squot32(local_tid_17107, 32) * 32) &&
                    (squot32(local_tid_17107, 32) == 0 && slt32(local_tid_17107,
                                                                stage1_num_groups_17071))) {
                    // read operands
                    {
                        x_17116 = ((volatile __local
                                    double *) scan_arr_mem_17111)[local_tid_17107 -
                                                                  skip_threads_17125];
                        x_17117 = ((volatile __local
                                    double *) scan_arr_mem_17113)[local_tid_17107 -
                                                                  skip_threads_17125];
                    }
                    // perform operation
                    {
                        bool inactive_17126 = slt32(srem32((local_tid_17107 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_13620 *
                                                            sdiv_up32(n_11293 *
                                                                      m_11294,
                                                                      num_threads_17072)) -
                                                           1, m_11294),
                                                    (local_tid_17107 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_13620 *
                                                     sdiv_up32(n_11293 *
                                                               m_11294,
                                                               num_threads_17072)) -
                                                    1 - (((local_tid_17107 -
                                                           skip_threads_17125) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_13620 *
                                                          sdiv_up32(n_11293 *
                                                                    m_11294,
                                                                    num_threads_17072)) -
                                                         1));
                        
                        if (inactive_17126) {
                            x_17116 = x_17118;
                            x_17117 = x_17119;
                        }
                        if (!inactive_17126) {
                            double y_17120 = x_17116 * x_17119;
                            double res_17121 = x_17118 + y_17120;
                            double res_17122 = x_17117 * x_17119;
                            
                            x_17116 = res_17121;
                            x_17117 = res_17122;
                        }
                    }
                }
                if (sle32(wave_sizze_17109, skip_threads_17125)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17125, local_tid_17107 -
                          squot32(local_tid_17107, 32) * 32) &&
                    (squot32(local_tid_17107, 32) == 0 && slt32(local_tid_17107,
                                                                stage1_num_groups_17071))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_17111)[local_tid_17107] =
                            x_17116;
                        x_17118 = x_17116;
                        ((volatile __local
                          double *) scan_arr_mem_17113)[local_tid_17107] =
                            x_17117;
                        x_17119 = x_17117;
                    }
                }
                if (sle32(wave_sizze_17109, skip_threads_17125)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17125 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17107, 32) == 0 || !slt32(local_tid_17107,
                                                          stage1_num_groups_17071))) {
            // read operands
            {
                x_13627 = x_13625;
                x_13628 = x_13626;
                x_13625 = ((__local
                            double *) scan_arr_mem_17111)[squot32(local_tid_17107,
                                                                  32) - 1];
                x_13626 = ((__local
                            double *) scan_arr_mem_17113)[squot32(local_tid_17107,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_17127 = slt32(srem32((local_tid_17107 + 1) *
                                                   (segscan_group_sizze_13620 *
                                                    sdiv_up32(n_11293 * m_11294,
                                                              num_threads_17072)) -
                                                   1, m_11294),
                                            (local_tid_17107 + 1) *
                                            (segscan_group_sizze_13620 *
                                             sdiv_up32(n_11293 * m_11294,
                                                       num_threads_17072)) - 1 -
                                            ((squot32(local_tid_17107, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_13620 *
                                              sdiv_up32(n_11293 * m_11294,
                                                        num_threads_17072)) -
                                             1));
                
                if (inactive_17127) {
                    x_13625 = x_13627;
                    x_13626 = x_13628;
                }
                if (!inactive_17127) {
                    double y_13629 = x_13625 * x_13628;
                    double res_13630 = x_13627 + y_13629;
                    double res_13631 = x_13626 * x_13628;
                    
                    x_13625 = res_13630;
                    x_13626 = res_13631;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_17111)[local_tid_17107] =
                    x_13625;
                ((__local double *) scan_arr_mem_17113)[local_tid_17107] =
                    x_13626;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17107, 32) == 0) {
            ((__local double *) scan_arr_mem_17111)[local_tid_17107] = x_13627;
            ((__local double *) scan_arr_mem_17113)[local_tid_17107] = x_13628;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_12873, n_11293) && slt32(gtid_12883, m_11294)) {
            ((__global double *) mem_16690)[gtid_12873 * m_11294 + gtid_12883] =
                ((__local double *) scan_arr_mem_17111)[local_tid_17107];
            ((__global double *) mem_16695)[gtid_12873 * m_11294 + gtid_12883] =
                ((__local double *) scan_arr_mem_17113)[local_tid_17107];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_13620
}
__kernel void tridagNestedziscan_stage2_13039(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_17030_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_17028_backing_aligned_1,
                                              int32_t n_11293, int32_t m_11294,
                                              __global unsigned char *mem_16669,
                                              __global unsigned char *mem_16674,
                                              int32_t stage1_num_groups_16988,
                                              int32_t num_threads_16989)
{
    #define segscan_group_sizze_13515 (tridagNestedzisegscan_group_sizze_13033)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17030_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17030_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17028_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17028_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17023;
    int32_t local_tid_17024;
    int32_t group_sizze_17027;
    int32_t wave_sizze_17026;
    int32_t group_tid_17025;
    
    global_tid_17023 = get_global_id(0);
    local_tid_17024 = get_local_id(0);
    group_sizze_17027 = get_local_size(0);
    wave_sizze_17026 = LOCKSTEP_WIDTH;
    group_tid_17025 = get_group_id(0);
    
    int32_t phys_tid_13039;
    
    phys_tid_13039 = global_tid_17023;
    
    __local char *scan_arr_mem_17028;
    __local char *scan_arr_mem_17030;
    
    scan_arr_mem_17028 = (__local char *) scan_arr_mem_17028_backing_0;
    scan_arr_mem_17030 = (__local char *) scan_arr_mem_17030_backing_1;
    
    int32_t flat_idx_17032;
    
    flat_idx_17032 = (local_tid_17024 + 1) * (segscan_group_sizze_13515 *
                                              sdiv_up32(n_11293 * m_11294,
                                                        num_threads_16989)) - 1;
    
    int32_t gtid_13028;
    
    gtid_13028 = squot32(flat_idx_17032, m_11294);
    
    int32_t gtid_13038;
    
    gtid_13038 = flat_idx_17032 - squot32(flat_idx_17032, m_11294) * m_11294;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_13028, n_11293) && slt32(gtid_13038, m_11294)) {
            ((__local double *) scan_arr_mem_17028)[local_tid_17024] =
                ((__global double *) mem_16669)[gtid_13028 * m_11294 +
                                                gtid_13038];
            ((__local double *) scan_arr_mem_17030)[local_tid_17024] =
                ((__global double *) mem_16674)[gtid_13028 * m_11294 +
                                                gtid_13038];
        } else {
            ((__local double *) scan_arr_mem_17028)[local_tid_17024] = 0.0;
            ((__local double *) scan_arr_mem_17030)[local_tid_17024] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_13520;
    double x_13521;
    double x_13522;
    double x_13523;
    double x_17033;
    double x_17034;
    double x_17035;
    double x_17036;
    int32_t skip_threads_17040;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_17024, stage1_num_groups_16988)) {
            x_13522 = ((volatile __local
                        double *) scan_arr_mem_17028)[local_tid_17024];
            x_13523 = ((volatile __local
                        double *) scan_arr_mem_17030)[local_tid_17024];
            if ((local_tid_17024 - squot32(local_tid_17024, 32) * 32) == 0) {
                x_13520 = x_13522;
                x_13521 = x_13523;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17040 = 1;
        while (slt32(skip_threads_17040, 32)) {
            if (sle32(skip_threads_17040, local_tid_17024 -
                      squot32(local_tid_17024, 32) * 32) &&
                slt32(local_tid_17024, stage1_num_groups_16988)) {
                // read operands
                {
                    x_13520 = ((volatile __local
                                double *) scan_arr_mem_17028)[local_tid_17024 -
                                                              skip_threads_17040];
                    x_13521 = ((volatile __local
                                double *) scan_arr_mem_17030)[local_tid_17024 -
                                                              skip_threads_17040];
                }
                // perform operation
                {
                    bool inactive_17041 = slt32(srem32((local_tid_17024 + 1) *
                                                       (segscan_group_sizze_13515 *
                                                        sdiv_up32(n_11293 *
                                                                  m_11294,
                                                                  num_threads_16989)) -
                                                       1, m_11294),
                                                (local_tid_17024 + 1) *
                                                (segscan_group_sizze_13515 *
                                                 sdiv_up32(n_11293 * m_11294,
                                                           num_threads_16989)) -
                                                1 - ((local_tid_17024 -
                                                      skip_threads_17040 + 1) *
                                                     (segscan_group_sizze_13515 *
                                                      sdiv_up32(n_11293 *
                                                                m_11294,
                                                                num_threads_16989)) -
                                                     1));
                    
                    if (inactive_17041) {
                        x_13520 = x_13522;
                        x_13521 = x_13523;
                    }
                    if (!inactive_17041) {
                        double y_13524 = x_13520 * x_13523;
                        double res_13525 = x_13522 + y_13524;
                        double res_13526 = x_13521 * x_13523;
                        
                        x_13520 = res_13525;
                        x_13521 = res_13526;
                    }
                }
            }
            if (sle32(wave_sizze_17026, skip_threads_17040)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17040, local_tid_17024 -
                      squot32(local_tid_17024, 32) * 32) &&
                slt32(local_tid_17024, stage1_num_groups_16988)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_17028)[local_tid_17024] = x_13520;
                    x_13522 = x_13520;
                    ((volatile __local
                      double *) scan_arr_mem_17030)[local_tid_17024] = x_13521;
                    x_13523 = x_13521;
                }
            }
            if (sle32(wave_sizze_17026, skip_threads_17040)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17040 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17024 - squot32(local_tid_17024, 32) * 32) == 31 &&
            slt32(local_tid_17024, stage1_num_groups_16988)) {
            ((volatile __local
              double *) scan_arr_mem_17028)[squot32(local_tid_17024, 32)] =
                x_13520;
            ((volatile __local
              double *) scan_arr_mem_17030)[squot32(local_tid_17024, 32)] =
                x_13521;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17042;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17024, 32) == 0 && slt32(local_tid_17024,
                                                           stage1_num_groups_16988)) {
                x_17035 = ((volatile __local
                            double *) scan_arr_mem_17028)[local_tid_17024];
                x_17036 = ((volatile __local
                            double *) scan_arr_mem_17030)[local_tid_17024];
                if ((local_tid_17024 - squot32(local_tid_17024, 32) * 32) ==
                    0) {
                    x_17033 = x_17035;
                    x_17034 = x_17036;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17042 = 1;
            while (slt32(skip_threads_17042, 32)) {
                if (sle32(skip_threads_17042, local_tid_17024 -
                          squot32(local_tid_17024, 32) * 32) &&
                    (squot32(local_tid_17024, 32) == 0 && slt32(local_tid_17024,
                                                                stage1_num_groups_16988))) {
                    // read operands
                    {
                        x_17033 = ((volatile __local
                                    double *) scan_arr_mem_17028)[local_tid_17024 -
                                                                  skip_threads_17042];
                        x_17034 = ((volatile __local
                                    double *) scan_arr_mem_17030)[local_tid_17024 -
                                                                  skip_threads_17042];
                    }
                    // perform operation
                    {
                        bool inactive_17043 = slt32(srem32((local_tid_17024 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_13515 *
                                                            sdiv_up32(n_11293 *
                                                                      m_11294,
                                                                      num_threads_16989)) -
                                                           1, m_11294),
                                                    (local_tid_17024 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_13515 *
                                                     sdiv_up32(n_11293 *
                                                               m_11294,
                                                               num_threads_16989)) -
                                                    1 - (((local_tid_17024 -
                                                           skip_threads_17042) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_13515 *
                                                          sdiv_up32(n_11293 *
                                                                    m_11294,
                                                                    num_threads_16989)) -
                                                         1));
                        
                        if (inactive_17043) {
                            x_17033 = x_17035;
                            x_17034 = x_17036;
                        }
                        if (!inactive_17043) {
                            double y_17037 = x_17033 * x_17036;
                            double res_17038 = x_17035 + y_17037;
                            double res_17039 = x_17034 * x_17036;
                            
                            x_17033 = res_17038;
                            x_17034 = res_17039;
                        }
                    }
                }
                if (sle32(wave_sizze_17026, skip_threads_17042)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17042, local_tid_17024 -
                          squot32(local_tid_17024, 32) * 32) &&
                    (squot32(local_tid_17024, 32) == 0 && slt32(local_tid_17024,
                                                                stage1_num_groups_16988))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_17028)[local_tid_17024] =
                            x_17033;
                        x_17035 = x_17033;
                        ((volatile __local
                          double *) scan_arr_mem_17030)[local_tid_17024] =
                            x_17034;
                        x_17036 = x_17034;
                    }
                }
                if (sle32(wave_sizze_17026, skip_threads_17042)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17042 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17024, 32) == 0 || !slt32(local_tid_17024,
                                                          stage1_num_groups_16988))) {
            // read operands
            {
                x_13522 = x_13520;
                x_13523 = x_13521;
                x_13520 = ((__local
                            double *) scan_arr_mem_17028)[squot32(local_tid_17024,
                                                                  32) - 1];
                x_13521 = ((__local
                            double *) scan_arr_mem_17030)[squot32(local_tid_17024,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_17044 = slt32(srem32((local_tid_17024 + 1) *
                                                   (segscan_group_sizze_13515 *
                                                    sdiv_up32(n_11293 * m_11294,
                                                              num_threads_16989)) -
                                                   1, m_11294),
                                            (local_tid_17024 + 1) *
                                            (segscan_group_sizze_13515 *
                                             sdiv_up32(n_11293 * m_11294,
                                                       num_threads_16989)) - 1 -
                                            ((squot32(local_tid_17024, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_13515 *
                                              sdiv_up32(n_11293 * m_11294,
                                                        num_threads_16989)) -
                                             1));
                
                if (inactive_17044) {
                    x_13520 = x_13522;
                    x_13521 = x_13523;
                }
                if (!inactive_17044) {
                    double y_13524 = x_13520 * x_13523;
                    double res_13525 = x_13522 + y_13524;
                    double res_13526 = x_13521 * x_13523;
                    
                    x_13520 = res_13525;
                    x_13521 = res_13526;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_17028)[local_tid_17024] =
                    x_13520;
                ((__local double *) scan_arr_mem_17030)[local_tid_17024] =
                    x_13521;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17024, 32) == 0) {
            ((__local double *) scan_arr_mem_17028)[local_tid_17024] = x_13522;
            ((__local double *) scan_arr_mem_17030)[local_tid_17024] = x_13523;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_13028, n_11293) && slt32(gtid_13038, m_11294)) {
            ((__global double *) mem_16669)[gtid_13028 * m_11294 + gtid_13038] =
                ((__local double *) scan_arr_mem_17028)[local_tid_17024];
            ((__global double *) mem_16674)[gtid_13028 * m_11294 + gtid_13038] =
                ((__local double *) scan_arr_mem_17030)[local_tid_17024];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_13515
}
__kernel void tridagNestedziscan_stage2_13272(__global int *global_failure,
                                              __local volatile
                                              int64_t *scan_arr_mem_16937_backing_aligned_0,
                                              __local volatile
                                              int64_t *scan_arr_mem_16935_backing_aligned_1,
                                              __local volatile
                                              int64_t *scan_arr_mem_16933_backing_aligned_2,
                                              __local volatile
                                              int64_t *scan_arr_mem_16931_backing_aligned_3,
                                              int32_t n_11293, int32_t m_11294,
                                              __global unsigned char *mem_16642,
                                              __global unsigned char *mem_16647,
                                              __global unsigned char *mem_16652,
                                              __global unsigned char *mem_16657,
                                              int32_t stage1_num_groups_16851,
                                              int32_t num_threads_16852)
{
    #define segscan_group_sizze_13347 (tridagNestedzisegscan_group_sizze_13266)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16937_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16937_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16935_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16935_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16933_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16933_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16931_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16931_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16926;
    int32_t local_tid_16927;
    int32_t group_sizze_16930;
    int32_t wave_sizze_16929;
    int32_t group_tid_16928;
    
    global_tid_16926 = get_global_id(0);
    local_tid_16927 = get_local_id(0);
    group_sizze_16930 = get_local_size(0);
    wave_sizze_16929 = LOCKSTEP_WIDTH;
    group_tid_16928 = get_group_id(0);
    
    int32_t phys_tid_13272;
    
    phys_tid_13272 = global_tid_16926;
    
    __local char *scan_arr_mem_16931;
    __local char *scan_arr_mem_16933;
    __local char *scan_arr_mem_16935;
    __local char *scan_arr_mem_16937;
    
    scan_arr_mem_16931 = (__local char *) scan_arr_mem_16931_backing_0;
    scan_arr_mem_16933 = (__local char *) scan_arr_mem_16933_backing_1;
    scan_arr_mem_16935 = (__local char *) scan_arr_mem_16935_backing_2;
    scan_arr_mem_16937 = (__local char *) scan_arr_mem_16937_backing_3;
    
    int32_t flat_idx_16939;
    
    flat_idx_16939 = (local_tid_16927 + 1) * (segscan_group_sizze_13347 *
                                              sdiv_up32(n_11293 * m_11294,
                                                        num_threads_16852)) - 1;
    
    int32_t gtid_13261;
    
    gtid_13261 = squot32(flat_idx_16939, m_11294);
    
    int32_t gtid_13271;
    
    gtid_13271 = flat_idx_16939 - squot32(flat_idx_16939, m_11294) * m_11294;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_13261, n_11293) && slt32(gtid_13271, m_11294)) {
            ((__local double *) scan_arr_mem_16931)[local_tid_16927] =
                ((__global double *) mem_16642)[gtid_13261 * m_11294 +
                                                gtid_13271];
            ((__local double *) scan_arr_mem_16933)[local_tid_16927] =
                ((__global double *) mem_16647)[gtid_13261 * m_11294 +
                                                gtid_13271];
            ((__local double *) scan_arr_mem_16935)[local_tid_16927] =
                ((__global double *) mem_16652)[gtid_13261 * m_11294 +
                                                gtid_13271];
            ((__local double *) scan_arr_mem_16937)[local_tid_16927] =
                ((__global double *) mem_16657)[gtid_13261 * m_11294 +
                                                gtid_13271];
        } else {
            ((__local double *) scan_arr_mem_16931)[local_tid_16927] = 1.0;
            ((__local double *) scan_arr_mem_16933)[local_tid_16927] = 0.0;
            ((__local double *) scan_arr_mem_16935)[local_tid_16927] = 0.0;
            ((__local double *) scan_arr_mem_16937)[local_tid_16927] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_13354;
    double x_13355;
    double x_13356;
    double x_13357;
    double x_13358;
    double x_13359;
    double x_13360;
    double x_13361;
    double x_16940;
    double x_16941;
    double x_16942;
    double x_16943;
    double x_16944;
    double x_16945;
    double x_16946;
    double x_16947;
    int32_t skip_threads_16965;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16927, stage1_num_groups_16851)) {
            x_13358 = ((volatile __local
                        double *) scan_arr_mem_16931)[local_tid_16927];
            x_13359 = ((volatile __local
                        double *) scan_arr_mem_16933)[local_tid_16927];
            x_13360 = ((volatile __local
                        double *) scan_arr_mem_16935)[local_tid_16927];
            x_13361 = ((volatile __local
                        double *) scan_arr_mem_16937)[local_tid_16927];
            if ((local_tid_16927 - squot32(local_tid_16927, 32) * 32) == 0) {
                x_13354 = x_13358;
                x_13355 = x_13359;
                x_13356 = x_13360;
                x_13357 = x_13361;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16965 = 1;
        while (slt32(skip_threads_16965, 32)) {
            if (sle32(skip_threads_16965, local_tid_16927 -
                      squot32(local_tid_16927, 32) * 32) &&
                slt32(local_tid_16927, stage1_num_groups_16851)) {
                // read operands
                {
                    x_13354 = ((volatile __local
                                double *) scan_arr_mem_16931)[local_tid_16927 -
                                                              skip_threads_16965];
                    x_13355 = ((volatile __local
                                double *) scan_arr_mem_16933)[local_tid_16927 -
                                                              skip_threads_16965];
                    x_13356 = ((volatile __local
                                double *) scan_arr_mem_16935)[local_tid_16927 -
                                                              skip_threads_16965];
                    x_13357 = ((volatile __local
                                double *) scan_arr_mem_16937)[local_tid_16927 -
                                                              skip_threads_16965];
                }
                // perform operation
                {
                    bool inactive_16966 = slt32(srem32((local_tid_16927 + 1) *
                                                       (segscan_group_sizze_13347 *
                                                        sdiv_up32(n_11293 *
                                                                  m_11294,
                                                                  num_threads_16852)) -
                                                       1, m_11294),
                                                (local_tid_16927 + 1) *
                                                (segscan_group_sizze_13347 *
                                                 sdiv_up32(n_11293 * m_11294,
                                                           num_threads_16852)) -
                                                1 - ((local_tid_16927 -
                                                      skip_threads_16965 + 1) *
                                                     (segscan_group_sizze_13347 *
                                                      sdiv_up32(n_11293 *
                                                                m_11294,
                                                                num_threads_16852)) -
                                                     1));
                    
                    if (inactive_16966) {
                        x_13354 = x_13358;
                        x_13355 = x_13359;
                        x_13356 = x_13360;
                        x_13357 = x_13361;
                    }
                    if (!inactive_16966) {
                        double y_13362 = x_13354 * x_13358;
                        double value_13363 = 1.0 / y_13362;
                        double y_13364 = x_13356 * x_13359;
                        double x_13365 = y_13362 + y_13364;
                        double res_13366 = value_13363 * x_13365;
                        double x_13367 = x_13355 * x_13358;
                        double y_13368 = x_13357 * x_13359;
                        double x_13369 = x_13367 + y_13368;
                        double res_13370 = value_13363 * x_13369;
                        double x_13371 = x_13354 * x_13360;
                        double y_13372 = x_13356 * x_13361;
                        double x_13373 = x_13371 + y_13372;
                        double res_13374 = value_13363 * x_13373;
                        double x_13375 = x_13355 * x_13360;
                        double y_13376 = x_13357 * x_13361;
                        double x_13377 = x_13375 + y_13376;
                        double res_13378 = value_13363 * x_13377;
                        
                        x_13354 = res_13366;
                        x_13355 = res_13370;
                        x_13356 = res_13374;
                        x_13357 = res_13378;
                    }
                }
            }
            if (sle32(wave_sizze_16929, skip_threads_16965)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16965, local_tid_16927 -
                      squot32(local_tid_16927, 32) * 32) &&
                slt32(local_tid_16927, stage1_num_groups_16851)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_16931)[local_tid_16927] = x_13354;
                    x_13358 = x_13354;
                    ((volatile __local
                      double *) scan_arr_mem_16933)[local_tid_16927] = x_13355;
                    x_13359 = x_13355;
                    ((volatile __local
                      double *) scan_arr_mem_16935)[local_tid_16927] = x_13356;
                    x_13360 = x_13356;
                    ((volatile __local
                      double *) scan_arr_mem_16937)[local_tid_16927] = x_13357;
                    x_13361 = x_13357;
                }
            }
            if (sle32(wave_sizze_16929, skip_threads_16965)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16965 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16927 - squot32(local_tid_16927, 32) * 32) == 31 &&
            slt32(local_tid_16927, stage1_num_groups_16851)) {
            ((volatile __local
              double *) scan_arr_mem_16931)[squot32(local_tid_16927, 32)] =
                x_13354;
            ((volatile __local
              double *) scan_arr_mem_16933)[squot32(local_tid_16927, 32)] =
                x_13355;
            ((volatile __local
              double *) scan_arr_mem_16935)[squot32(local_tid_16927, 32)] =
                x_13356;
            ((volatile __local
              double *) scan_arr_mem_16937)[squot32(local_tid_16927, 32)] =
                x_13357;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16967;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16927, 32) == 0 && slt32(local_tid_16927,
                                                           stage1_num_groups_16851)) {
                x_16944 = ((volatile __local
                            double *) scan_arr_mem_16931)[local_tid_16927];
                x_16945 = ((volatile __local
                            double *) scan_arr_mem_16933)[local_tid_16927];
                x_16946 = ((volatile __local
                            double *) scan_arr_mem_16935)[local_tid_16927];
                x_16947 = ((volatile __local
                            double *) scan_arr_mem_16937)[local_tid_16927];
                if ((local_tid_16927 - squot32(local_tid_16927, 32) * 32) ==
                    0) {
                    x_16940 = x_16944;
                    x_16941 = x_16945;
                    x_16942 = x_16946;
                    x_16943 = x_16947;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16967 = 1;
            while (slt32(skip_threads_16967, 32)) {
                if (sle32(skip_threads_16967, local_tid_16927 -
                          squot32(local_tid_16927, 32) * 32) &&
                    (squot32(local_tid_16927, 32) == 0 && slt32(local_tid_16927,
                                                                stage1_num_groups_16851))) {
                    // read operands
                    {
                        x_16940 = ((volatile __local
                                    double *) scan_arr_mem_16931)[local_tid_16927 -
                                                                  skip_threads_16967];
                        x_16941 = ((volatile __local
                                    double *) scan_arr_mem_16933)[local_tid_16927 -
                                                                  skip_threads_16967];
                        x_16942 = ((volatile __local
                                    double *) scan_arr_mem_16935)[local_tid_16927 -
                                                                  skip_threads_16967];
                        x_16943 = ((volatile __local
                                    double *) scan_arr_mem_16937)[local_tid_16927 -
                                                                  skip_threads_16967];
                    }
                    // perform operation
                    {
                        bool inactive_16968 = slt32(srem32((local_tid_16927 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_13347 *
                                                            sdiv_up32(n_11293 *
                                                                      m_11294,
                                                                      num_threads_16852)) -
                                                           1, m_11294),
                                                    (local_tid_16927 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_13347 *
                                                     sdiv_up32(n_11293 *
                                                               m_11294,
                                                               num_threads_16852)) -
                                                    1 - (((local_tid_16927 -
                                                           skip_threads_16967) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_13347 *
                                                          sdiv_up32(n_11293 *
                                                                    m_11294,
                                                                    num_threads_16852)) -
                                                         1));
                        
                        if (inactive_16968) {
                            x_16940 = x_16944;
                            x_16941 = x_16945;
                            x_16942 = x_16946;
                            x_16943 = x_16947;
                        }
                        if (!inactive_16968) {
                            double y_16948 = x_16940 * x_16944;
                            double value_16949 = 1.0 / y_16948;
                            double y_16950 = x_16942 * x_16945;
                            double x_16951 = y_16948 + y_16950;
                            double res_16952 = value_16949 * x_16951;
                            double x_16953 = x_16941 * x_16944;
                            double y_16954 = x_16943 * x_16945;
                            double x_16955 = x_16953 + y_16954;
                            double res_16956 = value_16949 * x_16955;
                            double x_16957 = x_16940 * x_16946;
                            double y_16958 = x_16942 * x_16947;
                            double x_16959 = x_16957 + y_16958;
                            double res_16960 = value_16949 * x_16959;
                            double x_16961 = x_16941 * x_16946;
                            double y_16962 = x_16943 * x_16947;
                            double x_16963 = x_16961 + y_16962;
                            double res_16964 = value_16949 * x_16963;
                            
                            x_16940 = res_16952;
                            x_16941 = res_16956;
                            x_16942 = res_16960;
                            x_16943 = res_16964;
                        }
                    }
                }
                if (sle32(wave_sizze_16929, skip_threads_16967)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16967, local_tid_16927 -
                          squot32(local_tid_16927, 32) * 32) &&
                    (squot32(local_tid_16927, 32) == 0 && slt32(local_tid_16927,
                                                                stage1_num_groups_16851))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_16931)[local_tid_16927] =
                            x_16940;
                        x_16944 = x_16940;
                        ((volatile __local
                          double *) scan_arr_mem_16933)[local_tid_16927] =
                            x_16941;
                        x_16945 = x_16941;
                        ((volatile __local
                          double *) scan_arr_mem_16935)[local_tid_16927] =
                            x_16942;
                        x_16946 = x_16942;
                        ((volatile __local
                          double *) scan_arr_mem_16937)[local_tid_16927] =
                            x_16943;
                        x_16947 = x_16943;
                    }
                }
                if (sle32(wave_sizze_16929, skip_threads_16967)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16967 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16927, 32) == 0 || !slt32(local_tid_16927,
                                                          stage1_num_groups_16851))) {
            // read operands
            {
                x_13358 = x_13354;
                x_13359 = x_13355;
                x_13360 = x_13356;
                x_13361 = x_13357;
                x_13354 = ((__local
                            double *) scan_arr_mem_16931)[squot32(local_tid_16927,
                                                                  32) - 1];
                x_13355 = ((__local
                            double *) scan_arr_mem_16933)[squot32(local_tid_16927,
                                                                  32) - 1];
                x_13356 = ((__local
                            double *) scan_arr_mem_16935)[squot32(local_tid_16927,
                                                                  32) - 1];
                x_13357 = ((__local
                            double *) scan_arr_mem_16937)[squot32(local_tid_16927,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_16969 = slt32(srem32((local_tid_16927 + 1) *
                                                   (segscan_group_sizze_13347 *
                                                    sdiv_up32(n_11293 * m_11294,
                                                              num_threads_16852)) -
                                                   1, m_11294),
                                            (local_tid_16927 + 1) *
                                            (segscan_group_sizze_13347 *
                                             sdiv_up32(n_11293 * m_11294,
                                                       num_threads_16852)) - 1 -
                                            ((squot32(local_tid_16927, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_13347 *
                                              sdiv_up32(n_11293 * m_11294,
                                                        num_threads_16852)) -
                                             1));
                
                if (inactive_16969) {
                    x_13354 = x_13358;
                    x_13355 = x_13359;
                    x_13356 = x_13360;
                    x_13357 = x_13361;
                }
                if (!inactive_16969) {
                    double y_13362 = x_13354 * x_13358;
                    double value_13363 = 1.0 / y_13362;
                    double y_13364 = x_13356 * x_13359;
                    double x_13365 = y_13362 + y_13364;
                    double res_13366 = value_13363 * x_13365;
                    double x_13367 = x_13355 * x_13358;
                    double y_13368 = x_13357 * x_13359;
                    double x_13369 = x_13367 + y_13368;
                    double res_13370 = value_13363 * x_13369;
                    double x_13371 = x_13354 * x_13360;
                    double y_13372 = x_13356 * x_13361;
                    double x_13373 = x_13371 + y_13372;
                    double res_13374 = value_13363 * x_13373;
                    double x_13375 = x_13355 * x_13360;
                    double y_13376 = x_13357 * x_13361;
                    double x_13377 = x_13375 + y_13376;
                    double res_13378 = value_13363 * x_13377;
                    
                    x_13354 = res_13366;
                    x_13355 = res_13370;
                    x_13356 = res_13374;
                    x_13357 = res_13378;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_16931)[local_tid_16927] =
                    x_13354;
                ((__local double *) scan_arr_mem_16933)[local_tid_16927] =
                    x_13355;
                ((__local double *) scan_arr_mem_16935)[local_tid_16927] =
                    x_13356;
                ((__local double *) scan_arr_mem_16937)[local_tid_16927] =
                    x_13357;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16927, 32) == 0) {
            ((__local double *) scan_arr_mem_16931)[local_tid_16927] = x_13358;
            ((__local double *) scan_arr_mem_16933)[local_tid_16927] = x_13359;
            ((__local double *) scan_arr_mem_16935)[local_tid_16927] = x_13360;
            ((__local double *) scan_arr_mem_16937)[local_tid_16927] = x_13361;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_13261, n_11293) && slt32(gtid_13271, m_11294)) {
            ((__global double *) mem_16642)[gtid_13261 * m_11294 + gtid_13271] =
                ((__local double *) scan_arr_mem_16931)[local_tid_16927];
            ((__global double *) mem_16647)[gtid_13261 * m_11294 + gtid_13271] =
                ((__local double *) scan_arr_mem_16933)[local_tid_16927];
            ((__global double *) mem_16652)[gtid_13261 * m_11294 + gtid_13271] =
                ((__local double *) scan_arr_mem_16935)[local_tid_16927];
            ((__global double *) mem_16657)[gtid_13261 * m_11294 + gtid_13271] =
                ((__local double *) scan_arr_mem_16937)[local_tid_16927];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_13347
}
__kernel void tridagNestedziscan_stage3_12884(__global int *global_failure,
                                              int32_t n_11293, int32_t m_11294,
                                              int32_t num_groups_13621, __global
                                              unsigned char *mem_16690, __global
                                              unsigned char *mem_16695,
                                              int32_t num_threads_17072,
                                              int32_t required_groups_17128)
{
    #define segscan_group_sizze_13620 (tridagNestedzisegscan_group_sizze_12878)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17129;
    int32_t local_tid_17130;
    int32_t group_sizze_17133;
    int32_t wave_sizze_17132;
    int32_t group_tid_17131;
    
    global_tid_17129 = get_global_id(0);
    local_tid_17130 = get_local_id(0);
    group_sizze_17133 = get_local_size(0);
    wave_sizze_17132 = LOCKSTEP_WIDTH;
    group_tid_17131 = get_group_id(0);
    
    int32_t phys_tid_12884;
    
    phys_tid_12884 = global_tid_17129;
    
    int32_t phys_group_id_17134;
    
    phys_group_id_17134 = get_group_id(0);
    for (int32_t i_17135 = 0; i_17135 < sdiv_up32(required_groups_17128 -
                                                  phys_group_id_17134,
                                                  num_groups_13621);
         i_17135++) {
        int32_t virt_group_id_17136 = phys_group_id_17134 + i_17135 *
                num_groups_13621;
        int32_t flat_idx_17137 = virt_group_id_17136 *
                segscan_group_sizze_13620 + local_tid_17130;
        int32_t gtid_12873 = squot32(flat_idx_17137, m_11294);
        int32_t gtid_12883 = flat_idx_17137 - squot32(flat_idx_17137, m_11294) *
                m_11294;
        int32_t orig_group_17138 = squot32(flat_idx_17137,
                                           segscan_group_sizze_13620 *
                                           sdiv_up32(n_11293 * m_11294,
                                                     num_threads_17072));
        int32_t carry_in_flat_idx_17139 = orig_group_17138 *
                (segscan_group_sizze_13620 * sdiv_up32(n_11293 * m_11294,
                                                       num_threads_17072)) - 1;
        
        if (slt32(gtid_12873, n_11293) && slt32(gtid_12883, m_11294)) {
            if (!(orig_group_17138 == 0 || (flat_idx_17137 ==
                                            (orig_group_17138 + 1) *
                                            (segscan_group_sizze_13620 *
                                             sdiv_up32(n_11293 * m_11294,
                                                       num_threads_17072)) -
                                            1 || slt32(srem32(flat_idx_17137,
                                                              m_11294),
                                                       flat_idx_17137 -
                                                       carry_in_flat_idx_17139)))) {
                double x_13625;
                double x_13626;
                double x_13627;
                double x_13628;
                
                x_13625 = ((__global
                            double *) mem_16690)[squot32(carry_in_flat_idx_17139,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_17139 -
                                                  squot32(carry_in_flat_idx_17139,
                                                          m_11294) * m_11294)];
                x_13626 = ((__global
                            double *) mem_16695)[squot32(carry_in_flat_idx_17139,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_17139 -
                                                  squot32(carry_in_flat_idx_17139,
                                                          m_11294) * m_11294)];
                x_13627 = ((__global double *) mem_16690)[gtid_12873 * m_11294 +
                                                          gtid_12883];
                x_13628 = ((__global double *) mem_16695)[gtid_12873 * m_11294 +
                                                          gtid_12883];
                
                double y_13629;
                
                y_13629 = x_13625 * x_13628;
                
                double res_13630 = x_13627 + y_13629;
                double res_13631 = x_13626 * x_13628;
                
                x_13625 = res_13630;
                x_13626 = res_13631;
                ((__global double *) mem_16690)[gtid_12873 * m_11294 +
                                                gtid_12883] = x_13625;
                ((__global double *) mem_16695)[gtid_12873 * m_11294 +
                                                gtid_12883] = x_13626;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_13620
}
__kernel void tridagNestedziscan_stage3_13039(__global int *global_failure,
                                              int32_t n_11293, int32_t m_11294,
                                              int32_t num_groups_13516, __global
                                              unsigned char *mem_16669, __global
                                              unsigned char *mem_16674,
                                              int32_t num_threads_16989,
                                              int32_t required_groups_17045)
{
    #define segscan_group_sizze_13515 (tridagNestedzisegscan_group_sizze_13033)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17046;
    int32_t local_tid_17047;
    int32_t group_sizze_17050;
    int32_t wave_sizze_17049;
    int32_t group_tid_17048;
    
    global_tid_17046 = get_global_id(0);
    local_tid_17047 = get_local_id(0);
    group_sizze_17050 = get_local_size(0);
    wave_sizze_17049 = LOCKSTEP_WIDTH;
    group_tid_17048 = get_group_id(0);
    
    int32_t phys_tid_13039;
    
    phys_tid_13039 = global_tid_17046;
    
    int32_t phys_group_id_17051;
    
    phys_group_id_17051 = get_group_id(0);
    for (int32_t i_17052 = 0; i_17052 < sdiv_up32(required_groups_17045 -
                                                  phys_group_id_17051,
                                                  num_groups_13516);
         i_17052++) {
        int32_t virt_group_id_17053 = phys_group_id_17051 + i_17052 *
                num_groups_13516;
        int32_t flat_idx_17054 = virt_group_id_17053 *
                segscan_group_sizze_13515 + local_tid_17047;
        int32_t gtid_13028 = squot32(flat_idx_17054, m_11294);
        int32_t gtid_13038 = flat_idx_17054 - squot32(flat_idx_17054, m_11294) *
                m_11294;
        int32_t orig_group_17055 = squot32(flat_idx_17054,
                                           segscan_group_sizze_13515 *
                                           sdiv_up32(n_11293 * m_11294,
                                                     num_threads_16989));
        int32_t carry_in_flat_idx_17056 = orig_group_17055 *
                (segscan_group_sizze_13515 * sdiv_up32(n_11293 * m_11294,
                                                       num_threads_16989)) - 1;
        
        if (slt32(gtid_13028, n_11293) && slt32(gtid_13038, m_11294)) {
            if (!(orig_group_17055 == 0 || (flat_idx_17054 ==
                                            (orig_group_17055 + 1) *
                                            (segscan_group_sizze_13515 *
                                             sdiv_up32(n_11293 * m_11294,
                                                       num_threads_16989)) -
                                            1 || slt32(srem32(flat_idx_17054,
                                                              m_11294),
                                                       flat_idx_17054 -
                                                       carry_in_flat_idx_17056)))) {
                double x_13520;
                double x_13521;
                double x_13522;
                double x_13523;
                
                x_13520 = ((__global
                            double *) mem_16669)[squot32(carry_in_flat_idx_17056,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_17056 -
                                                  squot32(carry_in_flat_idx_17056,
                                                          m_11294) * m_11294)];
                x_13521 = ((__global
                            double *) mem_16674)[squot32(carry_in_flat_idx_17056,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_17056 -
                                                  squot32(carry_in_flat_idx_17056,
                                                          m_11294) * m_11294)];
                x_13522 = ((__global double *) mem_16669)[gtid_13028 * m_11294 +
                                                          gtid_13038];
                x_13523 = ((__global double *) mem_16674)[gtid_13028 * m_11294 +
                                                          gtid_13038];
                
                double y_13524;
                
                y_13524 = x_13520 * x_13523;
                
                double res_13525 = x_13522 + y_13524;
                double res_13526 = x_13521 * x_13523;
                
                x_13520 = res_13525;
                x_13521 = res_13526;
                ((__global double *) mem_16669)[gtid_13028 * m_11294 +
                                                gtid_13038] = x_13520;
                ((__global double *) mem_16674)[gtid_13028 * m_11294 +
                                                gtid_13038] = x_13521;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_13515
}
__kernel void tridagNestedziscan_stage3_13272(__global int *global_failure,
                                              int32_t n_11293, int32_t m_11294,
                                              int32_t num_groups_13348, __global
                                              unsigned char *mem_16642, __global
                                              unsigned char *mem_16647, __global
                                              unsigned char *mem_16652, __global
                                              unsigned char *mem_16657,
                                              int32_t num_threads_16852,
                                              int32_t required_groups_16970)
{
    #define segscan_group_sizze_13347 (tridagNestedzisegscan_group_sizze_13266)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16971;
    int32_t local_tid_16972;
    int32_t group_sizze_16975;
    int32_t wave_sizze_16974;
    int32_t group_tid_16973;
    
    global_tid_16971 = get_global_id(0);
    local_tid_16972 = get_local_id(0);
    group_sizze_16975 = get_local_size(0);
    wave_sizze_16974 = LOCKSTEP_WIDTH;
    group_tid_16973 = get_group_id(0);
    
    int32_t phys_tid_13272;
    
    phys_tid_13272 = global_tid_16971;
    
    int32_t phys_group_id_16976;
    
    phys_group_id_16976 = get_group_id(0);
    for (int32_t i_16977 = 0; i_16977 < sdiv_up32(required_groups_16970 -
                                                  phys_group_id_16976,
                                                  num_groups_13348);
         i_16977++) {
        int32_t virt_group_id_16978 = phys_group_id_16976 + i_16977 *
                num_groups_13348;
        int32_t flat_idx_16979 = virt_group_id_16978 *
                segscan_group_sizze_13347 + local_tid_16972;
        int32_t gtid_13261 = squot32(flat_idx_16979, m_11294);
        int32_t gtid_13271 = flat_idx_16979 - squot32(flat_idx_16979, m_11294) *
                m_11294;
        int32_t orig_group_16980 = squot32(flat_idx_16979,
                                           segscan_group_sizze_13347 *
                                           sdiv_up32(n_11293 * m_11294,
                                                     num_threads_16852));
        int32_t carry_in_flat_idx_16981 = orig_group_16980 *
                (segscan_group_sizze_13347 * sdiv_up32(n_11293 * m_11294,
                                                       num_threads_16852)) - 1;
        
        if (slt32(gtid_13261, n_11293) && slt32(gtid_13271, m_11294)) {
            if (!(orig_group_16980 == 0 || (flat_idx_16979 ==
                                            (orig_group_16980 + 1) *
                                            (segscan_group_sizze_13347 *
                                             sdiv_up32(n_11293 * m_11294,
                                                       num_threads_16852)) -
                                            1 || slt32(srem32(flat_idx_16979,
                                                              m_11294),
                                                       flat_idx_16979 -
                                                       carry_in_flat_idx_16981)))) {
                double x_13354;
                double x_13355;
                double x_13356;
                double x_13357;
                double x_13358;
                double x_13359;
                double x_13360;
                double x_13361;
                
                x_13354 = ((__global
                            double *) mem_16642)[squot32(carry_in_flat_idx_16981,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          m_11294) * m_11294)];
                x_13355 = ((__global
                            double *) mem_16647)[squot32(carry_in_flat_idx_16981,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          m_11294) * m_11294)];
                x_13356 = ((__global
                            double *) mem_16652)[squot32(carry_in_flat_idx_16981,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          m_11294) * m_11294)];
                x_13357 = ((__global
                            double *) mem_16657)[squot32(carry_in_flat_idx_16981,
                                                         m_11294) * m_11294 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          m_11294) * m_11294)];
                x_13358 = ((__global double *) mem_16642)[gtid_13261 * m_11294 +
                                                          gtid_13271];
                x_13359 = ((__global double *) mem_16647)[gtid_13261 * m_11294 +
                                                          gtid_13271];
                x_13360 = ((__global double *) mem_16652)[gtid_13261 * m_11294 +
                                                          gtid_13271];
                x_13361 = ((__global double *) mem_16657)[gtid_13261 * m_11294 +
                                                          gtid_13271];
                
                double y_13362;
                
                y_13362 = x_13354 * x_13358;
                
                double value_13363 = 1.0 / y_13362;
                double y_13364 = x_13356 * x_13359;
                double x_13365 = y_13362 + y_13364;
                double res_13366 = value_13363 * x_13365;
                double x_13367 = x_13355 * x_13358;
                double y_13368 = x_13357 * x_13359;
                double x_13369 = x_13367 + y_13368;
                double res_13370 = value_13363 * x_13369;
                double x_13371 = x_13354 * x_13360;
                double y_13372 = x_13356 * x_13361;
                double x_13373 = x_13371 + y_13372;
                double res_13374 = value_13363 * x_13373;
                double x_13375 = x_13355 * x_13360;
                double y_13376 = x_13357 * x_13361;
                double x_13377 = x_13375 + y_13376;
                double res_13378 = value_13363 * x_13377;
                
                x_13354 = res_13366;
                x_13355 = res_13370;
                x_13356 = res_13374;
                x_13357 = res_13378;
                ((__global double *) mem_16642)[gtid_13261 * m_11294 +
                                                gtid_13271] = x_13354;
                ((__global double *) mem_16647)[gtid_13261 * m_11294 +
                                                gtid_13271] = x_13355;
                ((__global double *) mem_16652)[gtid_13261 * m_11294 +
                                                gtid_13271] = x_13356;
                ((__global double *) mem_16657)[gtid_13261 * m_11294 +
                                                gtid_13271] = x_13357;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_13347
}
__kernel void tridagNestedzisegmap_12741(__global int *global_failure,
                                         int32_t n_11293, int32_t m_11294,
                                         __global unsigned char *mem_16701,
                                         __global unsigned char *mem_16707)
{
    #define segmap_group_sizze_13722 (tridagNestedzisegmap_group_sizze_12746)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17145;
    int32_t local_tid_17146;
    int32_t group_sizze_17149;
    int32_t wave_sizze_17148;
    int32_t group_tid_17147;
    
    global_tid_17145 = get_global_id(0);
    local_tid_17146 = get_local_id(0);
    group_sizze_17149 = get_local_size(0);
    wave_sizze_17148 = LOCKSTEP_WIDTH;
    group_tid_17147 = get_group_id(0);
    
    int32_t phys_tid_12741;
    
    phys_tid_12741 = global_tid_17145;
    
    int32_t gtid_12739;
    
    gtid_12739 = squot32(group_tid_17147 * segmap_group_sizze_13722 +
                         local_tid_17146, m_11294);
    
    int32_t gtid_12740;
    
    gtid_12740 = group_tid_17147 * segmap_group_sizze_13722 + local_tid_17146 -
        squot32(group_tid_17147 * segmap_group_sizze_13722 + local_tid_17146,
                m_11294) * m_11294;
    if (slt32(gtid_12739, n_11293) && slt32(gtid_12740, m_11294)) {
        int32_t x_13729 = sub32(m_11294, gtid_12740);
        int32_t i_13730 = sub32(x_13729, 1);
        double res_13731 = ((__global double *) mem_16701)[gtid_12739 *
                                                           m_11294 + i_13730];
        
        ((__global double *) mem_16707)[gtid_12739 * m_11294 + gtid_12740] =
            res_13731;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_13722
}
__kernel void tridagNestedzisegmap_12815(__global int *global_failure,
                                         int32_t n_11293, int32_t m_11294,
                                         __global unsigned char *mem_16684,
                                         __global unsigned char *mem_16690,
                                         __global unsigned char *mem_16695,
                                         __global unsigned char *mem_16701)
{
    #define segmap_group_sizze_13682 (tridagNestedzisegmap_group_sizze_12820)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17140;
    int32_t local_tid_17141;
    int32_t group_sizze_17144;
    int32_t wave_sizze_17143;
    int32_t group_tid_17142;
    
    global_tid_17140 = get_global_id(0);
    local_tid_17141 = get_local_id(0);
    group_sizze_17144 = get_local_size(0);
    wave_sizze_17143 = LOCKSTEP_WIDTH;
    group_tid_17142 = get_group_id(0);
    
    int32_t phys_tid_12815;
    
    phys_tid_12815 = global_tid_17140;
    
    int32_t gtid_12813;
    
    gtid_12813 = squot32(group_tid_17142 * segmap_group_sizze_13682 +
                         local_tid_17141, m_11294);
    
    int32_t gtid_12814;
    
    gtid_12814 = group_tid_17142 * segmap_group_sizze_13682 + local_tid_17141 -
        squot32(group_tid_17142 * segmap_group_sizze_13682 + local_tid_17141,
                m_11294) * m_11294;
    if (slt32(gtid_12813, n_11293) && slt32(gtid_12814, m_11294)) {
        double yn_13687 = ((__global double *) mem_16684)[gtid_12813];
        double x_13688 = ((__global double *) mem_16690)[gtid_12813 * m_11294 +
                                                         gtid_12814];
        double x_13689 = ((__global double *) mem_16695)[gtid_12813 * m_11294 +
                                                         gtid_12814];
        double y_13693 = yn_13687 * x_13689;
        double res_13694 = x_13688 + y_13693;
        
        ((__global double *) mem_16701)[gtid_12813 * m_11294 + gtid_12814] =
            res_13694;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_13682
}
__kernel void tridagNestedzisegmap_12909(__global int *global_failure,
                                         int32_t n_11293, int32_t m_11294,
                                         int32_t i_11321,
                                         int32_t num_groups_13603, __global
                                         unsigned char *mem_16663, __global
                                         unsigned char *mem_16680, __global
                                         unsigned char *mem_16684)
{
    #define segmap_group_sizze_13602 (tridagNestedzisegmap_group_sizze_12912)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17062;
    int32_t local_tid_17063;
    int32_t group_sizze_17066;
    int32_t wave_sizze_17065;
    int32_t group_tid_17064;
    
    global_tid_17062 = get_global_id(0);
    local_tid_17063 = get_local_id(0);
    group_sizze_17066 = get_local_size(0);
    wave_sizze_17065 = LOCKSTEP_WIDTH;
    group_tid_17064 = get_group_id(0);
    
    int32_t phys_tid_12909;
    
    phys_tid_12909 = global_tid_17062;
    
    int32_t phys_group_id_17067;
    
    phys_group_id_17067 = get_group_id(0);
    for (int32_t i_17068 = 0; i_17068 < sdiv_up32(sdiv_up32(n_11293,
                                                            segmap_group_sizze_13602) -
                                                  phys_group_id_17067,
                                                  num_groups_13603);
         i_17068++) {
        int32_t virt_group_id_17069 = phys_group_id_17067 + i_17068 *
                num_groups_13603;
        int32_t gtid_12908 = virt_group_id_17069 * segmap_group_sizze_13602 +
                local_tid_17063;
        
        if (slt32(gtid_12908, n_11293)) {
            double x_13609 = ((__global double *) mem_16680)[gtid_12908 *
                                                             m_11294 + i_11321];
            double y_13610 = ((__global double *) mem_16663)[gtid_12908 *
                                                             m_11294 + i_11321];
            double yn_13611 = x_13609 / y_13610;
            
            ((__global double *) mem_16684)[gtid_12908] = yn_13611;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_13602
}
__kernel void tridagNestedzisegmap_12970(__global int *global_failure,
                                         int32_t n_11293, int32_t m_11294,
                                         int32_t m_11300, __global
                                         unsigned char *y_mem_16587, __global
                                         unsigned char *mem_16669, __global
                                         unsigned char *mem_16674, __global
                                         unsigned char *mem_16680)
{
    #define segmap_group_sizze_13575 (tridagNestedzisegmap_group_sizze_12975)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17057;
    int32_t local_tid_17058;
    int32_t group_sizze_17061;
    int32_t wave_sizze_17060;
    int32_t group_tid_17059;
    
    global_tid_17057 = get_global_id(0);
    local_tid_17058 = get_local_id(0);
    group_sizze_17061 = get_local_size(0);
    wave_sizze_17060 = LOCKSTEP_WIDTH;
    group_tid_17059 = get_group_id(0);
    
    int32_t phys_tid_12970;
    
    phys_tid_12970 = global_tid_17057;
    
    int32_t gtid_12968;
    
    gtid_12968 = squot32(group_tid_17059 * segmap_group_sizze_13575 +
                         local_tid_17058, m_11294);
    
    int32_t gtid_12969;
    
    gtid_12969 = group_tid_17059 * segmap_group_sizze_13575 + local_tid_17058 -
        squot32(group_tid_17059 * segmap_group_sizze_13575 + local_tid_17058,
                m_11294) * m_11294;
    if (slt32(gtid_12968, n_11293) && slt32(gtid_12969, m_11294)) {
        double as_transformed_row_13580 = ((__global
                                            double *) y_mem_16587)[gtid_12968 *
                                                                   m_11300];
        double x_13581 = ((__global double *) mem_16669)[gtid_12968 * m_11294 +
                                                         gtid_12969];
        double x_13582 = ((__global double *) mem_16674)[gtid_12968 * m_11294 +
                                                         gtid_12969];
        double y_13586 = as_transformed_row_13580 * x_13582;
        double res_13587 = x_13581 + y_13586;
        
        ((__global double *) mem_16680)[gtid_12968 * m_11294 + gtid_12969] =
            res_13587;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_13575
}
__kernel void tridagNestedzisegmap_13145(__global int *global_failure,
                                         int32_t n_11293, int32_t m_11294,
                                         int32_t m_11296, __global
                                         unsigned char *b_mem_16585, __global
                                         unsigned char *mem_16642, __global
                                         unsigned char *mem_16647, __global
                                         unsigned char *mem_16652, __global
                                         unsigned char *mem_16657, __global
                                         unsigned char *mem_16663)
{
    #define segmap_group_sizze_13451 (tridagNestedzisegmap_group_sizze_13150)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16982;
    int32_t local_tid_16983;
    int32_t group_sizze_16986;
    int32_t wave_sizze_16985;
    int32_t group_tid_16984;
    
    global_tid_16982 = get_global_id(0);
    local_tid_16983 = get_local_id(0);
    group_sizze_16986 = get_local_size(0);
    wave_sizze_16985 = LOCKSTEP_WIDTH;
    group_tid_16984 = get_group_id(0);
    
    int32_t phys_tid_13145;
    
    phys_tid_13145 = global_tid_16982;
    
    int32_t gtid_13143;
    
    gtid_13143 = squot32(group_tid_16984 * segmap_group_sizze_13451 +
                         local_tid_16983, m_11294);
    
    int32_t gtid_13144;
    
    gtid_13144 = group_tid_16984 * segmap_group_sizze_13451 + local_tid_16983 -
        squot32(group_tid_16984 * segmap_group_sizze_13451 + local_tid_16983,
                m_11294) * m_11294;
    if (slt32(gtid_13143, n_11293) && slt32(gtid_13144, m_11294)) {
        double as_transformed_row_13456 = ((__global
                                            double *) b_mem_16585)[gtid_13143 *
                                                                   m_11296];
        double x_13457 = ((__global double *) mem_16642)[gtid_13143 * m_11294 +
                                                         gtid_13144];
        double x_13458 = ((__global double *) mem_16647)[gtid_13143 * m_11294 +
                                                         gtid_13144];
        double x_13459 = ((__global double *) mem_16652)[gtid_13143 * m_11294 +
                                                         gtid_13144];
        double x_13460 = ((__global double *) mem_16657)[gtid_13143 * m_11294 +
                                                         gtid_13144];
        double value_13462 = 1.0 / x_13457;
        double res_13465 = x_13457 * value_13462;
        double res_13469 = x_13458 * value_13462;
        double res_13473 = x_13459 * value_13462;
        double res_13477 = x_13460 * value_13462;
        double x_13478 = as_transformed_row_13456 * res_13465;
        double x_13479 = res_13469 + x_13478;
        double x_13480 = as_transformed_row_13456 * res_13473;
        double y_13481 = res_13477 + x_13480;
        double res_13482 = x_13479 / y_13481;
        
        ((__global double *) mem_16663)[gtid_13143 * m_11294 + gtid_13144] =
            res_13482;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_13451
}
__kernel void tridagNestedzisegmap_intragroup_12059(__global
                                                    int *global_failure,
                                                    __local volatile
                                                    int64_t *mem_16631_backing_aligned_0,
                                                    __local volatile
                                                    int64_t *mem_16627_backing_aligned_1,
                                                    __local volatile
                                                    int64_t *mem_16623_backing_aligned_2,
                                                    __local volatile
                                                    int64_t *mem_16620_backing_aligned_3,
                                                    __local volatile
                                                    int64_t *mem_16616_backing_aligned_4,
                                                    __local volatile
                                                    int64_t *mem_16612_backing_aligned_5,
                                                    __local volatile
                                                    int64_t *mem_16609_backing_aligned_6,
                                                    __local volatile
                                                    int64_t *mem_16605_backing_aligned_7,
                                                    __local volatile
                                                    int64_t *mem_16601_backing_aligned_8,
                                                    __local volatile
                                                    int64_t *mem_16598_backing_aligned_9,
                                                    __local volatile
                                                    int64_t *mem_16595_backing_aligned_10,
                                                    __local volatile
                                                    int64_t *mem_16592_backing_aligned_11,
                                                    int32_t m_11294,
                                                    int32_t m_11296,
                                                    int32_t m_11298,
                                                    int32_t m_11300,
                                                    int32_t i_11321, __global
                                                    unsigned char *a_mem_16584,
                                                    __global
                                                    unsigned char *b_mem_16585,
                                                    __global
                                                    unsigned char *c_mem_16586,
                                                    __global
                                                    unsigned char *y_mem_16587,
                                                    __global
                                                    unsigned char *mem_16636)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_16631_backing_11 = (__local volatile
                                                            char *) mem_16631_backing_aligned_0;
    __local volatile char *restrict mem_16627_backing_10 = (__local volatile
                                                            char *) mem_16627_backing_aligned_1;
    __local volatile char *restrict mem_16623_backing_9 = (__local volatile
                                                           char *) mem_16623_backing_aligned_2;
    __local volatile char *restrict mem_16620_backing_8 = (__local volatile
                                                           char *) mem_16620_backing_aligned_3;
    __local volatile char *restrict mem_16616_backing_7 = (__local volatile
                                                           char *) mem_16616_backing_aligned_4;
    __local volatile char *restrict mem_16612_backing_6 = (__local volatile
                                                           char *) mem_16612_backing_aligned_5;
    __local volatile char *restrict mem_16609_backing_5 = (__local volatile
                                                           char *) mem_16609_backing_aligned_6;
    __local volatile char *restrict mem_16605_backing_4 = (__local volatile
                                                           char *) mem_16605_backing_aligned_7;
    __local volatile char *restrict mem_16601_backing_3 = (__local volatile
                                                           char *) mem_16601_backing_aligned_8;
    __local volatile char *restrict mem_16598_backing_2 = (__local volatile
                                                           char *) mem_16598_backing_aligned_9;
    __local volatile char *restrict mem_16595_backing_1 = (__local volatile
                                                           char *) mem_16595_backing_aligned_10;
    __local volatile char *restrict mem_16592_backing_0 = (__local volatile
                                                           char *) mem_16592_backing_aligned_11;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16779;
    int32_t local_tid_16780;
    int32_t group_sizze_16783;
    int32_t wave_sizze_16782;
    int32_t group_tid_16781;
    
    global_tid_16779 = get_global_id(0);
    local_tid_16780 = get_local_id(0);
    group_sizze_16783 = get_local_size(0);
    wave_sizze_16782 = LOCKSTEP_WIDTH;
    group_tid_16781 = get_group_id(0);
    
    int32_t phys_tid_12059;
    
    phys_tid_12059 = group_tid_16781;
    
    int32_t ltid_pre_16784;
    
    ltid_pre_16784 = local_tid_16780;
    
    int32_t gtid_11998;
    
    gtid_11998 = group_tid_16781;
    
    double as_transformed_row_12509;
    
    as_transformed_row_12509 = ((__global double *) b_mem_16585)[gtid_11998 *
                                                                 m_11296];
    
    double as_transformed_row_12510 = ((__global
                                        double *) y_mem_16587)[gtid_11998 *
                                                               m_11300];
    __local char *mem_16592;
    
    mem_16592 = (__local char *) mem_16592_backing_0;
    
    __local char *mem_16595;
    
    mem_16595 = (__local char *) mem_16595_backing_1;
    
    __local char *mem_16598;
    
    mem_16598 = (__local char *) mem_16598_backing_2;
    
    __local char *mem_16601;
    
    mem_16601 = (__local char *) mem_16601_backing_3;
    
    int32_t gtid_12001 = ltid_pre_16784;
    int32_t phys_tid_12002 = local_tid_16780;
    
    if (slt32(gtid_12001, m_11294)) {
        bool cond_12551 = slt32(0, gtid_12001);
        double res_12552;
        
        if (cond_12551) {
            res_12552 = 1.0;
        } else {
            res_12552 = 0.0;
        }
        
        double res_12553;
        
        if (cond_12551) {
            res_12553 = 0.0;
        } else {
            res_12553 = 1.0;
        }
        
        double res_12554;
        
        if (cond_12551) {
            double x_elem_12549 = ((__global double *) b_mem_16585)[gtid_11998 *
                                                                    m_11296 +
                                                                    gtid_12001];
            
            res_12554 = x_elem_12549;
        } else {
            res_12554 = 1.0;
        }
        
        double res_12555;
        
        if (cond_12551) {
            double x_elem_12550 = ((__global double *) a_mem_16584)[gtid_11998 *
                                                                    m_11294 +
                                                                    gtid_12001];
            int32_t i_12556 = sub32(gtid_12001, 1);
            double y_12557 = ((__global double *) c_mem_16586)[gtid_11998 *
                                                               m_11298 +
                                                               i_12556];
            double y_12558 = x_elem_12550 * y_12557;
            double res_12559 = 0.0 - y_12558;
            
            res_12555 = res_12559;
        } else {
            res_12555 = 0.0;
        }
        ((__local double *) mem_16592)[gtid_12001] = res_12554;
        ((__local double *) mem_16595)[gtid_12001] = res_12555;
        ((__local double *) mem_16598)[gtid_12001] = res_12552;
        ((__local double *) mem_16601)[gtid_12001] = res_12553;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16785;
    
    dims_flat_16785 = m_11294;
    
    double x_12523;
    double x_12524;
    double x_12525;
    double x_12526;
    double x_12527;
    double x_12528;
    double x_12529;
    double x_12530;
    double x_16790;
    double x_16791;
    double x_16792;
    double x_16793;
    double x_16794;
    double x_16795;
    double x_16796;
    double x_16797;
    int32_t skip_threads_16815;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16780, m_11294)) {
            x_12527 = ((volatile __local double *) mem_16592)[local_tid_16780];
            x_12528 = ((volatile __local double *) mem_16595)[local_tid_16780];
            x_12529 = ((volatile __local double *) mem_16598)[local_tid_16780];
            x_12530 = ((volatile __local double *) mem_16601)[local_tid_16780];
            if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 0) {
                x_12523 = x_12527;
                x_12524 = x_12528;
                x_12525 = x_12529;
                x_12526 = x_12530;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16815 = 1;
        while (slt32(skip_threads_16815, 32)) {
            if (sle32(skip_threads_16815, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, m_11294)) {
                // read operands
                {
                    x_12523 = ((volatile __local
                                double *) mem_16592)[local_tid_16780 -
                                                     skip_threads_16815];
                    x_12524 = ((volatile __local
                                double *) mem_16595)[local_tid_16780 -
                                                     skip_threads_16815];
                    x_12525 = ((volatile __local
                                double *) mem_16598)[local_tid_16780 -
                                                     skip_threads_16815];
                    x_12526 = ((volatile __local
                                double *) mem_16601)[local_tid_16780 -
                                                     skip_threads_16815];
                }
                // perform operation
                {
                    bool inactive_16816 = slt32(srem32(local_tid_16780,
                                                       m_11294),
                                                local_tid_16780 -
                                                (local_tid_16780 -
                                                 skip_threads_16815));
                    
                    if (inactive_16816) {
                        x_12523 = x_12527;
                        x_12524 = x_12528;
                        x_12525 = x_12529;
                        x_12526 = x_12530;
                    }
                    if (!inactive_16816) {
                        double y_12531 = x_12523 * x_12527;
                        double value_12532 = 1.0 / y_12531;
                        double y_12533 = x_12525 * x_12528;
                        double x_12534 = y_12531 + y_12533;
                        double res_12535 = value_12532 * x_12534;
                        double x_12536 = x_12524 * x_12527;
                        double y_12537 = x_12526 * x_12528;
                        double x_12538 = x_12536 + y_12537;
                        double res_12539 = value_12532 * x_12538;
                        double x_12540 = x_12523 * x_12529;
                        double y_12541 = x_12525 * x_12530;
                        double x_12542 = x_12540 + y_12541;
                        double res_12543 = value_12532 * x_12542;
                        double x_12544 = x_12524 * x_12529;
                        double y_12545 = x_12526 * x_12530;
                        double x_12546 = x_12544 + y_12545;
                        double res_12547 = value_12532 * x_12546;
                        
                        x_12523 = res_12535;
                        x_12524 = res_12539;
                        x_12525 = res_12543;
                        x_12526 = res_12547;
                    }
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16815)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16815, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, m_11294)) {
                // write result
                {
                    ((volatile __local double *) mem_16592)[local_tid_16780] =
                        x_12523;
                    x_12527 = x_12523;
                    ((volatile __local double *) mem_16595)[local_tid_16780] =
                        x_12524;
                    x_12528 = x_12524;
                    ((volatile __local double *) mem_16598)[local_tid_16780] =
                        x_12525;
                    x_12529 = x_12525;
                    ((volatile __local double *) mem_16601)[local_tid_16780] =
                        x_12526;
                    x_12530 = x_12526;
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16815)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16815 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 31 &&
            slt32(local_tid_16780, m_11294)) {
            ((volatile __local double *) mem_16592)[squot32(local_tid_16780,
                                                            32)] = x_12523;
            ((volatile __local double *) mem_16595)[squot32(local_tid_16780,
                                                            32)] = x_12524;
            ((volatile __local double *) mem_16598)[squot32(local_tid_16780,
                                                            32)] = x_12525;
            ((volatile __local double *) mem_16601)[squot32(local_tid_16780,
                                                            32)] = x_12526;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16817;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                           m_11294)) {
                x_16794 = ((volatile __local
                            double *) mem_16592)[local_tid_16780];
                x_16795 = ((volatile __local
                            double *) mem_16595)[local_tid_16780];
                x_16796 = ((volatile __local
                            double *) mem_16598)[local_tid_16780];
                x_16797 = ((volatile __local
                            double *) mem_16601)[local_tid_16780];
                if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) ==
                    0) {
                    x_16790 = x_16794;
                    x_16791 = x_16795;
                    x_16792 = x_16796;
                    x_16793 = x_16797;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16817 = 1;
            while (slt32(skip_threads_16817, 32)) {
                if (sle32(skip_threads_16817, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                m_11294))) {
                    // read operands
                    {
                        x_16790 = ((volatile __local
                                    double *) mem_16592)[local_tid_16780 -
                                                         skip_threads_16817];
                        x_16791 = ((volatile __local
                                    double *) mem_16595)[local_tid_16780 -
                                                         skip_threads_16817];
                        x_16792 = ((volatile __local
                                    double *) mem_16598)[local_tid_16780 -
                                                         skip_threads_16817];
                        x_16793 = ((volatile __local
                                    double *) mem_16601)[local_tid_16780 -
                                                         skip_threads_16817];
                    }
                    // perform operation
                    {
                        bool inactive_16818 = slt32(srem32(local_tid_16780 *
                                                           32 + 32 - 1,
                                                           m_11294),
                                                    local_tid_16780 * 32 + 32 -
                                                    1 - ((local_tid_16780 -
                                                          skip_threads_16817) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16818) {
                            x_16790 = x_16794;
                            x_16791 = x_16795;
                            x_16792 = x_16796;
                            x_16793 = x_16797;
                        }
                        if (!inactive_16818) {
                            double y_16798 = x_16790 * x_16794;
                            double value_16799 = 1.0 / y_16798;
                            double y_16800 = x_16792 * x_16795;
                            double x_16801 = y_16798 + y_16800;
                            double res_16802 = value_16799 * x_16801;
                            double x_16803 = x_16791 * x_16794;
                            double y_16804 = x_16793 * x_16795;
                            double x_16805 = x_16803 + y_16804;
                            double res_16806 = value_16799 * x_16805;
                            double x_16807 = x_16790 * x_16796;
                            double y_16808 = x_16792 * x_16797;
                            double x_16809 = x_16807 + y_16808;
                            double res_16810 = value_16799 * x_16809;
                            double x_16811 = x_16791 * x_16796;
                            double y_16812 = x_16793 * x_16797;
                            double x_16813 = x_16811 + y_16812;
                            double res_16814 = value_16799 * x_16813;
                            
                            x_16790 = res_16802;
                            x_16791 = res_16806;
                            x_16792 = res_16810;
                            x_16793 = res_16814;
                        }
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16817)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16817, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                m_11294))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16592)[local_tid_16780] = x_16790;
                        x_16794 = x_16790;
                        ((volatile __local
                          double *) mem_16595)[local_tid_16780] = x_16791;
                        x_16795 = x_16791;
                        ((volatile __local
                          double *) mem_16598)[local_tid_16780] = x_16792;
                        x_16796 = x_16792;
                        ((volatile __local
                          double *) mem_16601)[local_tid_16780] = x_16793;
                        x_16797 = x_16793;
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16817)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16817 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16780, 32) == 0 || !slt32(local_tid_16780,
                                                          m_11294))) {
            // read operands
            {
                x_12527 = x_12523;
                x_12528 = x_12524;
                x_12529 = x_12525;
                x_12530 = x_12526;
                x_12523 = ((__local
                            double *) mem_16592)[squot32(local_tid_16780, 32) -
                                                 1];
                x_12524 = ((__local
                            double *) mem_16595)[squot32(local_tid_16780, 32) -
                                                 1];
                x_12525 = ((__local
                            double *) mem_16598)[squot32(local_tid_16780, 32) -
                                                 1];
                x_12526 = ((__local
                            double *) mem_16601)[squot32(local_tid_16780, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16819 = slt32(srem32(local_tid_16780, m_11294),
                                            local_tid_16780 -
                                            (squot32(local_tid_16780, 32) * 32 -
                                             1));
                
                if (inactive_16819) {
                    x_12523 = x_12527;
                    x_12524 = x_12528;
                    x_12525 = x_12529;
                    x_12526 = x_12530;
                }
                if (!inactive_16819) {
                    double y_12531 = x_12523 * x_12527;
                    double value_12532 = 1.0 / y_12531;
                    double y_12533 = x_12525 * x_12528;
                    double x_12534 = y_12531 + y_12533;
                    double res_12535 = value_12532 * x_12534;
                    double x_12536 = x_12524 * x_12527;
                    double y_12537 = x_12526 * x_12528;
                    double x_12538 = x_12536 + y_12537;
                    double res_12539 = value_12532 * x_12538;
                    double x_12540 = x_12523 * x_12529;
                    double y_12541 = x_12525 * x_12530;
                    double x_12542 = x_12540 + y_12541;
                    double res_12543 = value_12532 * x_12542;
                    double x_12544 = x_12524 * x_12529;
                    double y_12545 = x_12526 * x_12530;
                    double x_12546 = x_12544 + y_12545;
                    double res_12547 = value_12532 * x_12546;
                    
                    x_12523 = res_12535;
                    x_12524 = res_12539;
                    x_12525 = res_12543;
                    x_12526 = res_12547;
                }
            }
            // write final result
            {
                ((__local double *) mem_16592)[local_tid_16780] = x_12523;
                ((__local double *) mem_16595)[local_tid_16780] = x_12524;
                ((__local double *) mem_16598)[local_tid_16780] = x_12525;
                ((__local double *) mem_16601)[local_tid_16780] = x_12526;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16780, 32) == 0) {
            ((__local double *) mem_16592)[local_tid_16780] = x_12527;
            ((__local double *) mem_16595)[local_tid_16780] = x_12528;
            ((__local double *) mem_16598)[local_tid_16780] = x_12529;
            ((__local double *) mem_16601)[local_tid_16780] = x_12530;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16605;
    
    mem_16605 = (__local char *) mem_16605_backing_4;
    
    int32_t gtid_12003 = ltid_pre_16784;
    int32_t phys_tid_12004 = local_tid_16780;
    
    if (slt32(gtid_12003, m_11294)) {
        double x_12588 = ((__local double *) mem_16592)[gtid_12003];
        double x_12589 = ((__local double *) mem_16595)[gtid_12003];
        double x_12590 = ((__local double *) mem_16598)[gtid_12003];
        double x_12591 = ((__local double *) mem_16601)[gtid_12003];
        double value_12593 = 1.0 / x_12588;
        double res_12596 = x_12588 * value_12593;
        double res_12600 = x_12589 * value_12593;
        double res_12604 = x_12590 * value_12593;
        double res_12608 = x_12591 * value_12593;
        double x_12609 = as_transformed_row_12509 * res_12596;
        double x_12610 = res_12600 + x_12609;
        double x_12611 = as_transformed_row_12509 * res_12604;
        double y_12612 = res_12608 + x_12611;
        double res_12613 = x_12610 / y_12612;
        
        ((__local double *) mem_16605)[gtid_12003] = res_12613;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16609;
    
    mem_16609 = (__local char *) mem_16609_backing_5;
    
    __local char *mem_16612;
    
    mem_16612 = (__local char *) mem_16612_backing_6;
    
    int32_t gtid_12031 = ltid_pre_16784;
    int32_t phys_tid_12032 = local_tid_16780;
    
    if (slt32(gtid_12031, m_11294)) {
        bool cond_12637 = slt32(0, gtid_12031);
        double res_12638;
        
        if (cond_12637) {
            double x_elem_12635 = ((__global double *) y_mem_16587)[gtid_11998 *
                                                                    m_11300 +
                                                                    gtid_12031];
            
            res_12638 = x_elem_12635;
        } else {
            res_12638 = 0.0;
        }
        
        double res_12639;
        
        if (cond_12637) {
            double x_elem_12636 = ((__global double *) a_mem_16584)[gtid_11998 *
                                                                    m_11294 +
                                                                    gtid_12031];
            int32_t i_12640 = sub32(gtid_12031, 1);
            double y_12641 = ((__local double *) mem_16605)[i_12640];
            double y_12642 = x_elem_12636 / y_12641;
            double res_12643 = 0.0 - y_12642;
            
            res_12639 = res_12643;
        } else {
            res_12639 = 1.0;
        }
        ((__local double *) mem_16609)[gtid_12031] = res_12638;
        ((__local double *) mem_16612)[gtid_12031] = res_12639;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16820;
    
    dims_flat_16820 = m_11294;
    
    double x_12627;
    double x_12628;
    double x_12629;
    double x_12630;
    double x_16823;
    double x_16824;
    double x_16825;
    double x_16826;
    int32_t skip_threads_16830;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16780, m_11294)) {
            x_12629 = ((volatile __local double *) mem_16609)[local_tid_16780];
            x_12630 = ((volatile __local double *) mem_16612)[local_tid_16780];
            if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 0) {
                x_12627 = x_12629;
                x_12628 = x_12630;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16830 = 1;
        while (slt32(skip_threads_16830, 32)) {
            if (sle32(skip_threads_16830, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, m_11294)) {
                // read operands
                {
                    x_12627 = ((volatile __local
                                double *) mem_16609)[local_tid_16780 -
                                                     skip_threads_16830];
                    x_12628 = ((volatile __local
                                double *) mem_16612)[local_tid_16780 -
                                                     skip_threads_16830];
                }
                // perform operation
                {
                    bool inactive_16831 = slt32(srem32(local_tid_16780,
                                                       m_11294),
                                                local_tid_16780 -
                                                (local_tid_16780 -
                                                 skip_threads_16830));
                    
                    if (inactive_16831) {
                        x_12627 = x_12629;
                        x_12628 = x_12630;
                    }
                    if (!inactive_16831) {
                        double y_12631 = x_12627 * x_12630;
                        double res_12632 = x_12629 + y_12631;
                        double res_12633 = x_12628 * x_12630;
                        
                        x_12627 = res_12632;
                        x_12628 = res_12633;
                    }
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16830)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16830, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, m_11294)) {
                // write result
                {
                    ((volatile __local double *) mem_16609)[local_tid_16780] =
                        x_12627;
                    x_12629 = x_12627;
                    ((volatile __local double *) mem_16612)[local_tid_16780] =
                        x_12628;
                    x_12630 = x_12628;
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16830)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16830 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 31 &&
            slt32(local_tid_16780, m_11294)) {
            ((volatile __local double *) mem_16609)[squot32(local_tid_16780,
                                                            32)] = x_12627;
            ((volatile __local double *) mem_16612)[squot32(local_tid_16780,
                                                            32)] = x_12628;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16832;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                           m_11294)) {
                x_16825 = ((volatile __local
                            double *) mem_16609)[local_tid_16780];
                x_16826 = ((volatile __local
                            double *) mem_16612)[local_tid_16780];
                if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) ==
                    0) {
                    x_16823 = x_16825;
                    x_16824 = x_16826;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16832 = 1;
            while (slt32(skip_threads_16832, 32)) {
                if (sle32(skip_threads_16832, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                m_11294))) {
                    // read operands
                    {
                        x_16823 = ((volatile __local
                                    double *) mem_16609)[local_tid_16780 -
                                                         skip_threads_16832];
                        x_16824 = ((volatile __local
                                    double *) mem_16612)[local_tid_16780 -
                                                         skip_threads_16832];
                    }
                    // perform operation
                    {
                        bool inactive_16833 = slt32(srem32(local_tid_16780 *
                                                           32 + 32 - 1,
                                                           m_11294),
                                                    local_tid_16780 * 32 + 32 -
                                                    1 - ((local_tid_16780 -
                                                          skip_threads_16832) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16833) {
                            x_16823 = x_16825;
                            x_16824 = x_16826;
                        }
                        if (!inactive_16833) {
                            double y_16827 = x_16823 * x_16826;
                            double res_16828 = x_16825 + y_16827;
                            double res_16829 = x_16824 * x_16826;
                            
                            x_16823 = res_16828;
                            x_16824 = res_16829;
                        }
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16832)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16832, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                m_11294))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16609)[local_tid_16780] = x_16823;
                        x_16825 = x_16823;
                        ((volatile __local
                          double *) mem_16612)[local_tid_16780] = x_16824;
                        x_16826 = x_16824;
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16832)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16832 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16780, 32) == 0 || !slt32(local_tid_16780,
                                                          m_11294))) {
            // read operands
            {
                x_12629 = x_12627;
                x_12630 = x_12628;
                x_12627 = ((__local
                            double *) mem_16609)[squot32(local_tid_16780, 32) -
                                                 1];
                x_12628 = ((__local
                            double *) mem_16612)[squot32(local_tid_16780, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16834 = slt32(srem32(local_tid_16780, m_11294),
                                            local_tid_16780 -
                                            (squot32(local_tid_16780, 32) * 32 -
                                             1));
                
                if (inactive_16834) {
                    x_12627 = x_12629;
                    x_12628 = x_12630;
                }
                if (!inactive_16834) {
                    double y_12631 = x_12627 * x_12630;
                    double res_12632 = x_12629 + y_12631;
                    double res_12633 = x_12628 * x_12630;
                    
                    x_12627 = res_12632;
                    x_12628 = res_12633;
                }
            }
            // write final result
            {
                ((__local double *) mem_16609)[local_tid_16780] = x_12627;
                ((__local double *) mem_16612)[local_tid_16780] = x_12628;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16780, 32) == 0) {
            ((__local double *) mem_16609)[local_tid_16780] = x_12629;
            ((__local double *) mem_16612)[local_tid_16780] = x_12630;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16616;
    
    mem_16616 = (__local char *) mem_16616_backing_7;
    
    int32_t gtid_12033 = ltid_pre_16784;
    int32_t phys_tid_12034 = local_tid_16780;
    
    if (slt32(gtid_12033, m_11294)) {
        double x_12654 = ((__local double *) mem_16609)[gtid_12033];
        double x_12655 = ((__local double *) mem_16612)[gtid_12033];
        double y_12659 = as_transformed_row_12510 * x_12655;
        double res_12660 = x_12654 + y_12659;
        
        ((__local double *) mem_16616)[gtid_12033] = res_12660;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_12664 = ((__local double *) mem_16616)[i_11321];
    double y_12665 = ((__local double *) mem_16605)[i_11321];
    double yn_12666 = x_12664 / y_12665;
    __local char *mem_16620;
    
    mem_16620 = (__local char *) mem_16620_backing_8;
    
    __local char *mem_16623;
    
    mem_16623 = (__local char *) mem_16623_backing_9;
    
    int32_t gtid_12042 = ltid_pre_16784;
    int32_t phys_tid_12043 = local_tid_16780;
    
    if (slt32(gtid_12042, m_11294)) {
        int32_t x_12681 = sub32(m_11294, gtid_12042);
        int32_t i_12682 = sub32(x_12681, 1);
        bool cond_12683 = slt32(0, gtid_12042);
        double res_12684;
        double res_12685;
        
        if (cond_12683) {
            double x_12686 = ((__local double *) mem_16616)[i_12682];
            double y_12687 = ((__local double *) mem_16605)[i_12682];
            double res_12688 = x_12686 / y_12687;
            double x_12689 = ((__global double *) c_mem_16586)[gtid_11998 *
                                                               m_11298 +
                                                               i_12682];
            double y_12690 = x_12689 / y_12687;
            double res_12691 = 0.0 - y_12690;
            
            res_12684 = res_12688;
            res_12685 = res_12691;
        } else {
            res_12684 = 0.0;
            res_12685 = 1.0;
        }
        ((__local double *) mem_16620)[gtid_12042] = res_12684;
        ((__local double *) mem_16623)[gtid_12042] = res_12685;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16835;
    
    dims_flat_16835 = m_11294;
    
    double x_12673;
    double x_12674;
    double x_12675;
    double x_12676;
    double x_16838;
    double x_16839;
    double x_16840;
    double x_16841;
    int32_t skip_threads_16845;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16780, m_11294)) {
            x_12675 = ((volatile __local double *) mem_16620)[local_tid_16780];
            x_12676 = ((volatile __local double *) mem_16623)[local_tid_16780];
            if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 0) {
                x_12673 = x_12675;
                x_12674 = x_12676;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16845 = 1;
        while (slt32(skip_threads_16845, 32)) {
            if (sle32(skip_threads_16845, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, m_11294)) {
                // read operands
                {
                    x_12673 = ((volatile __local
                                double *) mem_16620)[local_tid_16780 -
                                                     skip_threads_16845];
                    x_12674 = ((volatile __local
                                double *) mem_16623)[local_tid_16780 -
                                                     skip_threads_16845];
                }
                // perform operation
                {
                    bool inactive_16846 = slt32(srem32(local_tid_16780,
                                                       m_11294),
                                                local_tid_16780 -
                                                (local_tid_16780 -
                                                 skip_threads_16845));
                    
                    if (inactive_16846) {
                        x_12673 = x_12675;
                        x_12674 = x_12676;
                    }
                    if (!inactive_16846) {
                        double y_12677 = x_12673 * x_12676;
                        double res_12678 = x_12675 + y_12677;
                        double res_12679 = x_12674 * x_12676;
                        
                        x_12673 = res_12678;
                        x_12674 = res_12679;
                    }
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16845)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16845, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, m_11294)) {
                // write result
                {
                    ((volatile __local double *) mem_16620)[local_tid_16780] =
                        x_12673;
                    x_12675 = x_12673;
                    ((volatile __local double *) mem_16623)[local_tid_16780] =
                        x_12674;
                    x_12676 = x_12674;
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16845)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16845 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 31 &&
            slt32(local_tid_16780, m_11294)) {
            ((volatile __local double *) mem_16620)[squot32(local_tid_16780,
                                                            32)] = x_12673;
            ((volatile __local double *) mem_16623)[squot32(local_tid_16780,
                                                            32)] = x_12674;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16847;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                           m_11294)) {
                x_16840 = ((volatile __local
                            double *) mem_16620)[local_tid_16780];
                x_16841 = ((volatile __local
                            double *) mem_16623)[local_tid_16780];
                if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) ==
                    0) {
                    x_16838 = x_16840;
                    x_16839 = x_16841;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16847 = 1;
            while (slt32(skip_threads_16847, 32)) {
                if (sle32(skip_threads_16847, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                m_11294))) {
                    // read operands
                    {
                        x_16838 = ((volatile __local
                                    double *) mem_16620)[local_tid_16780 -
                                                         skip_threads_16847];
                        x_16839 = ((volatile __local
                                    double *) mem_16623)[local_tid_16780 -
                                                         skip_threads_16847];
                    }
                    // perform operation
                    {
                        bool inactive_16848 = slt32(srem32(local_tid_16780 *
                                                           32 + 32 - 1,
                                                           m_11294),
                                                    local_tid_16780 * 32 + 32 -
                                                    1 - ((local_tid_16780 -
                                                          skip_threads_16847) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16848) {
                            x_16838 = x_16840;
                            x_16839 = x_16841;
                        }
                        if (!inactive_16848) {
                            double y_16842 = x_16838 * x_16841;
                            double res_16843 = x_16840 + y_16842;
                            double res_16844 = x_16839 * x_16841;
                            
                            x_16838 = res_16843;
                            x_16839 = res_16844;
                        }
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16847)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16847, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                m_11294))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16620)[local_tid_16780] = x_16838;
                        x_16840 = x_16838;
                        ((volatile __local
                          double *) mem_16623)[local_tid_16780] = x_16839;
                        x_16841 = x_16839;
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16847)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16847 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16780, 32) == 0 || !slt32(local_tid_16780,
                                                          m_11294))) {
            // read operands
            {
                x_12675 = x_12673;
                x_12676 = x_12674;
                x_12673 = ((__local
                            double *) mem_16620)[squot32(local_tid_16780, 32) -
                                                 1];
                x_12674 = ((__local
                            double *) mem_16623)[squot32(local_tid_16780, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16849 = slt32(srem32(local_tid_16780, m_11294),
                                            local_tid_16780 -
                                            (squot32(local_tid_16780, 32) * 32 -
                                             1));
                
                if (inactive_16849) {
                    x_12673 = x_12675;
                    x_12674 = x_12676;
                }
                if (!inactive_16849) {
                    double y_12677 = x_12673 * x_12676;
                    double res_12678 = x_12675 + y_12677;
                    double res_12679 = x_12674 * x_12676;
                    
                    x_12673 = res_12678;
                    x_12674 = res_12679;
                }
            }
            // write final result
            {
                ((__local double *) mem_16620)[local_tid_16780] = x_12673;
                ((__local double *) mem_16623)[local_tid_16780] = x_12674;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16780, 32) == 0) {
            ((__local double *) mem_16620)[local_tid_16780] = x_12675;
            ((__local double *) mem_16623)[local_tid_16780] = x_12676;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16627;
    
    mem_16627 = (__local char *) mem_16627_backing_10;
    
    int32_t gtid_12044 = ltid_pre_16784;
    int32_t phys_tid_12045 = local_tid_16780;
    
    if (slt32(gtid_12044, m_11294)) {
        double x_12702 = ((__local double *) mem_16620)[gtid_12044];
        double x_12703 = ((__local double *) mem_16623)[gtid_12044];
        double y_12707 = yn_12666 * x_12703;
        double res_12708 = x_12702 + y_12707;
        
        ((__local double *) mem_16627)[gtid_12044] = res_12708;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16631;
    
    mem_16631 = (__local char *) mem_16631_backing_11;
    
    int32_t gtid_12053 = ltid_pre_16784;
    int32_t phys_tid_12054 = local_tid_16780;
    
    if (slt32(gtid_12053, m_11294)) {
        int32_t x_12714 = sub32(m_11294, gtid_12053);
        int32_t i_12715 = sub32(x_12714, 1);
        double res_12716 = ((__local double *) mem_16627)[i_12715];
        
        ((__local double *) mem_16631)[gtid_12053] = res_12716;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global double *) mem_16636)[gtid_11998 * m_11294 + local_tid_16780] =
        ((__local double *) mem_16631)[local_tid_16780];
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_7:
    return;
}
__kernel void tridagNestedConstziscan_stage1_14622(__global int *global_failure,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_17080_backing_aligned_0,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_17078_backing_aligned_1,
                                                   int32_t n_11537,
                                                   int32_t INNER_DIM_11542,
                                                   __global
                                                   unsigned char *c_mem_16586,
                                                   __global
                                                   unsigned char *mem_16645,
                                                   __global
                                                   unsigned char *mem_16659,
                                                   __global
                                                   unsigned char *mem_16668,
                                                   __global
                                                   unsigned char *mem_16672,
                                                   int32_t num_threads_17072)
{
    #define segscan_group_sizze_15358 (tridagNestedConstzisegscan_group_sizze_14616)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17080_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17080_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17078_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17078_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17073;
    int32_t local_tid_17074;
    int32_t group_sizze_17077;
    int32_t wave_sizze_17076;
    int32_t group_tid_17075;
    
    global_tid_17073 = get_global_id(0);
    local_tid_17074 = get_local_id(0);
    group_sizze_17077 = get_local_size(0);
    wave_sizze_17076 = LOCKSTEP_WIDTH;
    group_tid_17075 = get_group_id(0);
    
    int32_t phys_tid_14622;
    
    phys_tid_14622 = global_tid_17073;
    
    __local char *scan_arr_mem_17078;
    __local char *scan_arr_mem_17080;
    
    scan_arr_mem_17078 = (__local char *) scan_arr_mem_17078_backing_0;
    scan_arr_mem_17080 = (__local char *) scan_arr_mem_17080_backing_1;
    
    double x_15363;
    double x_15364;
    double x_15365;
    double x_15366;
    
    x_15363 = 0.0;
    x_15364 = 1.0;
    for (int32_t j_17082 = 0; j_17082 < sdiv_up32(n_11537 * 115,
                                                  num_threads_17072);
         j_17082++) {
        int32_t chunk_offset_17083 = segscan_group_sizze_15358 * j_17082 +
                group_tid_17075 * (segscan_group_sizze_15358 *
                                   sdiv_up32(n_11537 * 115, num_threads_17072));
        int32_t flat_idx_17084 = chunk_offset_17083 + local_tid_17074;
        int32_t gtid_14611 = squot32(flat_idx_17084, 115);
        int32_t gtid_14621 = flat_idx_17084 - squot32(flat_idx_17084, 115) *
                115;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14611, n_11537) && slt32(gtid_14621, 115)) {
                int32_t x_15375 = sub32(115, gtid_14621);
                int32_t i_15376 = sub32(x_15375, 1);
                bool cond_15377 = slt32(0, gtid_14621);
                double res_15378;
                double res_15379;
                
                if (cond_15377) {
                    double x_15380 = ((__global
                                       double *) mem_16659)[gtid_14611 * 115 +
                                                            i_15376];
                    double y_15381 = ((__global
                                       double *) mem_16645)[gtid_14611 * 115 +
                                                            i_15376];
                    double res_15382 = x_15380 / y_15381;
                    double x_15383 = ((__global
                                       double *) c_mem_16586)[gtid_14611 *
                                                              INNER_DIM_11542 +
                                                              i_15376];
                    double y_15384 = x_15383 / y_15381;
                    double res_15385 = 0.0 - y_15384;
                    
                    res_15378 = res_15382;
                    res_15379 = res_15385;
                } else {
                    res_15378 = 0.0;
                    res_15379 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_15365 = res_15378;
                    x_15366 = res_15379;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_14611, n_11537) && slt32(gtid_14621, 115))) {
                    x_15365 = 0.0;
                    x_15366 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_15367 = x_15363 * x_15366;
                double res_15368 = x_15365 + y_15367;
                double res_15369 = x_15364 * x_15366;
                
                ((__local double *) scan_arr_mem_17078)[local_tid_17074] =
                    res_15368;
                ((__local double *) scan_arr_mem_17080)[local_tid_17074] =
                    res_15369;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_17085;
            double x_17086;
            double x_17087;
            double x_17088;
            double x_17092;
            double x_17093;
            double x_17094;
            double x_17095;
            int32_t skip_threads_17099;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_17074, segscan_group_sizze_15358)) {
                    x_17087 = ((volatile __local
                                double *) scan_arr_mem_17078)[local_tid_17074];
                    x_17088 = ((volatile __local
                                double *) scan_arr_mem_17080)[local_tid_17074];
                    if ((local_tid_17074 - squot32(local_tid_17074, 32) * 32) ==
                        0) {
                        x_17085 = x_17087;
                        x_17086 = x_17088;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17099 = 1;
                while (slt32(skip_threads_17099, 32)) {
                    if (sle32(skip_threads_17099, local_tid_17074 -
                              squot32(local_tid_17074, 32) * 32) &&
                        slt32(local_tid_17074, segscan_group_sizze_15358)) {
                        // read operands
                        {
                            x_17085 = ((volatile __local
                                        double *) scan_arr_mem_17078)[local_tid_17074 -
                                                                      skip_threads_17099];
                            x_17086 = ((volatile __local
                                        double *) scan_arr_mem_17080)[local_tid_17074 -
                                                                      skip_threads_17099];
                        }
                        // perform operation
                        {
                            bool inactive_17100 = slt32(srem32(local_tid_17074 +
                                                               chunk_offset_17083,
                                                               115),
                                                        local_tid_17074 +
                                                        chunk_offset_17083 -
                                                        (local_tid_17074 -
                                                         skip_threads_17099 +
                                                         chunk_offset_17083));
                            
                            if (inactive_17100) {
                                x_17085 = x_17087;
                                x_17086 = x_17088;
                            }
                            if (!inactive_17100) {
                                double y_17089 = x_17085 * x_17088;
                                double res_17090 = x_17087 + y_17089;
                                double res_17091 = x_17086 * x_17088;
                                
                                x_17085 = res_17090;
                                x_17086 = res_17091;
                            }
                        }
                    }
                    if (sle32(wave_sizze_17076, skip_threads_17099)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17099, local_tid_17074 -
                              squot32(local_tid_17074, 32) * 32) &&
                        slt32(local_tid_17074, segscan_group_sizze_15358)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_17078)[local_tid_17074] =
                                x_17085;
                            x_17087 = x_17085;
                            ((volatile __local
                              double *) scan_arr_mem_17080)[local_tid_17074] =
                                x_17086;
                            x_17088 = x_17086;
                        }
                    }
                    if (sle32(wave_sizze_17076, skip_threads_17099)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17099 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_17074 - squot32(local_tid_17074, 32) * 32) ==
                    31 && slt32(local_tid_17074, segscan_group_sizze_15358)) {
                    ((volatile __local
                      double *) scan_arr_mem_17078)[squot32(local_tid_17074,
                                                            32)] = x_17085;
                    ((volatile __local
                      double *) scan_arr_mem_17080)[squot32(local_tid_17074,
                                                            32)] = x_17086;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17101;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_17074, 32) == 0 &&
                        slt32(local_tid_17074, segscan_group_sizze_15358)) {
                        x_17094 = ((volatile __local
                                    double *) scan_arr_mem_17078)[local_tid_17074];
                        x_17095 = ((volatile __local
                                    double *) scan_arr_mem_17080)[local_tid_17074];
                        if ((local_tid_17074 - squot32(local_tid_17074, 32) *
                             32) == 0) {
                            x_17092 = x_17094;
                            x_17093 = x_17095;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17101 = 1;
                    while (slt32(skip_threads_17101, 32)) {
                        if (sle32(skip_threads_17101, local_tid_17074 -
                                  squot32(local_tid_17074, 32) * 32) &&
                            (squot32(local_tid_17074, 32) == 0 &&
                             slt32(local_tid_17074,
                                   segscan_group_sizze_15358))) {
                            // read operands
                            {
                                x_17092 = ((volatile __local
                                            double *) scan_arr_mem_17078)[local_tid_17074 -
                                                                          skip_threads_17101];
                                x_17093 = ((volatile __local
                                            double *) scan_arr_mem_17080)[local_tid_17074 -
                                                                          skip_threads_17101];
                            }
                            // perform operation
                            {
                                bool inactive_17102 =
                                     slt32(srem32(local_tid_17074 * 32 + 32 -
                                                  1 + chunk_offset_17083, 115),
                                           local_tid_17074 * 32 + 32 - 1 +
                                           chunk_offset_17083 -
                                           ((local_tid_17074 -
                                             skip_threads_17101) * 32 + 32 - 1 +
                                            chunk_offset_17083));
                                
                                if (inactive_17102) {
                                    x_17092 = x_17094;
                                    x_17093 = x_17095;
                                }
                                if (!inactive_17102) {
                                    double y_17096 = x_17092 * x_17095;
                                    double res_17097 = x_17094 + y_17096;
                                    double res_17098 = x_17093 * x_17095;
                                    
                                    x_17092 = res_17097;
                                    x_17093 = res_17098;
                                }
                            }
                        }
                        if (sle32(wave_sizze_17076, skip_threads_17101)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17101, local_tid_17074 -
                                  squot32(local_tid_17074, 32) * 32) &&
                            (squot32(local_tid_17074, 32) == 0 &&
                             slt32(local_tid_17074,
                                   segscan_group_sizze_15358))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_17078)[local_tid_17074] =
                                    x_17092;
                                x_17094 = x_17092;
                                ((volatile __local
                                  double *) scan_arr_mem_17080)[local_tid_17074] =
                                    x_17093;
                                x_17095 = x_17093;
                            }
                        }
                        if (sle32(wave_sizze_17076, skip_threads_17101)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17101 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_17074, 32) == 0 ||
                      !slt32(local_tid_17074, segscan_group_sizze_15358))) {
                    // read operands
                    {
                        x_17087 = x_17085;
                        x_17088 = x_17086;
                        x_17085 = ((__local
                                    double *) scan_arr_mem_17078)[squot32(local_tid_17074,
                                                                          32) -
                                                                  1];
                        x_17086 = ((__local
                                    double *) scan_arr_mem_17080)[squot32(local_tid_17074,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_17103 = slt32(srem32(local_tid_17074 +
                                                           chunk_offset_17083,
                                                           115),
                                                    local_tid_17074 +
                                                    chunk_offset_17083 -
                                                    (squot32(local_tid_17074,
                                                             32) * 32 - 1 +
                                                     chunk_offset_17083));
                        
                        if (inactive_17103) {
                            x_17085 = x_17087;
                            x_17086 = x_17088;
                        }
                        if (!inactive_17103) {
                            double y_17089 = x_17085 * x_17088;
                            double res_17090 = x_17087 + y_17089;
                            double res_17091 = x_17086 * x_17088;
                            
                            x_17085 = res_17090;
                            x_17086 = res_17091;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_17078)[local_tid_17074] =
                            x_17085;
                        ((__local
                          double *) scan_arr_mem_17080)[local_tid_17074] =
                            x_17086;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_17074, 32) == 0) {
                    ((__local double *) scan_arr_mem_17078)[local_tid_17074] =
                        x_17087;
                    ((__local double *) scan_arr_mem_17080)[local_tid_17074] =
                        x_17088;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14611, n_11537) && slt32(gtid_14621, 115)) {
                    ((__global double *) mem_16668)[gtid_14611 * 115 +
                                                    gtid_14621] = ((__local
                                                                    double *) scan_arr_mem_17078)[local_tid_17074];
                    ((__global double *) mem_16672)[gtid_14611 * 115 +
                                                    gtid_14621] = ((__local
                                                                    double *) scan_arr_mem_17080)[local_tid_17074];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17104 = slt32(srem32(chunk_offset_17083 +
                                                          segscan_group_sizze_15358,
                                                          115),
                                                   chunk_offset_17083 +
                                                   segscan_group_sizze_15358 -
                                                   (chunk_offset_17083 +
                                                    segscan_group_sizze_15358 -
                                                    1));
                bool should_load_carry_17105 = local_tid_17074 == 0 &&
                     !crosses_segment_17104;
                
                if (should_load_carry_17105) {
                    x_15363 = ((__local
                                double *) scan_arr_mem_17078)[segscan_group_sizze_15358 -
                                                              1];
                    x_15364 = ((__local
                                double *) scan_arr_mem_17080)[segscan_group_sizze_15358 -
                                                              1];
                }
                if (!should_load_carry_17105) {
                    x_15363 = 0.0;
                    x_15364 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_15358
}
__kernel void tridagNestedConstziscan_stage1_14777(__global int *global_failure,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16997_backing_aligned_0,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16995_backing_aligned_1,
                                                   int32_t n_11537,
                                                   int32_t INNER_DIM_11538,
                                                   int32_t INNER_DIM_11544,
                                                   __global
                                                   unsigned char *a_mem_16584,
                                                   __global
                                                   unsigned char *y_mem_16587,
                                                   __global
                                                   unsigned char *mem_16645,
                                                   __global
                                                   unsigned char *mem_16650,
                                                   __global
                                                   unsigned char *mem_16654,
                                                   int32_t num_threads_16989)
{
    #define segscan_group_sizze_15253 (tridagNestedConstzisegscan_group_sizze_14771)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16997_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16997_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16995_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16995_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16990;
    int32_t local_tid_16991;
    int32_t group_sizze_16994;
    int32_t wave_sizze_16993;
    int32_t group_tid_16992;
    
    global_tid_16990 = get_global_id(0);
    local_tid_16991 = get_local_id(0);
    group_sizze_16994 = get_local_size(0);
    wave_sizze_16993 = LOCKSTEP_WIDTH;
    group_tid_16992 = get_group_id(0);
    
    int32_t phys_tid_14777;
    
    phys_tid_14777 = global_tid_16990;
    
    __local char *scan_arr_mem_16995;
    __local char *scan_arr_mem_16997;
    
    scan_arr_mem_16995 = (__local char *) scan_arr_mem_16995_backing_0;
    scan_arr_mem_16997 = (__local char *) scan_arr_mem_16997_backing_1;
    
    double x_15258;
    double x_15259;
    double x_15260;
    double x_15261;
    
    x_15258 = 0.0;
    x_15259 = 1.0;
    for (int32_t j_16999 = 0; j_16999 < sdiv_up32(n_11537 * 115,
                                                  num_threads_16989);
         j_16999++) {
        int32_t chunk_offset_17000 = segscan_group_sizze_15253 * j_16999 +
                group_tid_16992 * (segscan_group_sizze_15253 *
                                   sdiv_up32(n_11537 * 115, num_threads_16989));
        int32_t flat_idx_17001 = chunk_offset_17000 + local_tid_16991;
        int32_t gtid_14766 = squot32(flat_idx_17001, 115);
        int32_t gtid_14776 = flat_idx_17001 - squot32(flat_idx_17001, 115) *
                115;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14766, n_11537) && slt32(gtid_14776, 115)) {
                bool cond_15272 = slt32(0, gtid_14776);
                double res_15273;
                
                if (cond_15272) {
                    double x_elem_15270 = ((__global
                                            double *) y_mem_16587)[gtid_14766 *
                                                                   INNER_DIM_11544 +
                                                                   gtid_14776];
                    
                    res_15273 = x_elem_15270;
                } else {
                    res_15273 = 0.0;
                }
                
                double res_15274;
                
                if (cond_15272) {
                    double x_elem_15271 = ((__global
                                            double *) a_mem_16584)[gtid_14766 *
                                                                   INNER_DIM_11538 +
                                                                   gtid_14776];
                    int32_t i_15275 = sub32(gtid_14776, 1);
                    double y_15276 = ((__global
                                       double *) mem_16645)[gtid_14766 * 115 +
                                                            i_15275];
                    double y_15277 = x_elem_15271 / y_15276;
                    double res_15278 = 0.0 - y_15277;
                    
                    res_15274 = res_15278;
                } else {
                    res_15274 = 1.0;
                }
                // write to-scan values to parameters
                {
                    x_15260 = res_15273;
                    x_15261 = res_15274;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_14766, n_11537) && slt32(gtid_14776, 115))) {
                    x_15260 = 0.0;
                    x_15261 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_15262 = x_15258 * x_15261;
                double res_15263 = x_15260 + y_15262;
                double res_15264 = x_15259 * x_15261;
                
                ((__local double *) scan_arr_mem_16995)[local_tid_16991] =
                    res_15263;
                ((__local double *) scan_arr_mem_16997)[local_tid_16991] =
                    res_15264;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_17002;
            double x_17003;
            double x_17004;
            double x_17005;
            double x_17009;
            double x_17010;
            double x_17011;
            double x_17012;
            int32_t skip_threads_17016;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16991, segscan_group_sizze_15253)) {
                    x_17004 = ((volatile __local
                                double *) scan_arr_mem_16995)[local_tid_16991];
                    x_17005 = ((volatile __local
                                double *) scan_arr_mem_16997)[local_tid_16991];
                    if ((local_tid_16991 - squot32(local_tid_16991, 32) * 32) ==
                        0) {
                        x_17002 = x_17004;
                        x_17003 = x_17005;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_17016 = 1;
                while (slt32(skip_threads_17016, 32)) {
                    if (sle32(skip_threads_17016, local_tid_16991 -
                              squot32(local_tid_16991, 32) * 32) &&
                        slt32(local_tid_16991, segscan_group_sizze_15253)) {
                        // read operands
                        {
                            x_17002 = ((volatile __local
                                        double *) scan_arr_mem_16995)[local_tid_16991 -
                                                                      skip_threads_17016];
                            x_17003 = ((volatile __local
                                        double *) scan_arr_mem_16997)[local_tid_16991 -
                                                                      skip_threads_17016];
                        }
                        // perform operation
                        {
                            bool inactive_17017 = slt32(srem32(local_tid_16991 +
                                                               chunk_offset_17000,
                                                               115),
                                                        local_tid_16991 +
                                                        chunk_offset_17000 -
                                                        (local_tid_16991 -
                                                         skip_threads_17016 +
                                                         chunk_offset_17000));
                            
                            if (inactive_17017) {
                                x_17002 = x_17004;
                                x_17003 = x_17005;
                            }
                            if (!inactive_17017) {
                                double y_17006 = x_17002 * x_17005;
                                double res_17007 = x_17004 + y_17006;
                                double res_17008 = x_17003 * x_17005;
                                
                                x_17002 = res_17007;
                                x_17003 = res_17008;
                            }
                        }
                    }
                    if (sle32(wave_sizze_16993, skip_threads_17016)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_17016, local_tid_16991 -
                              squot32(local_tid_16991, 32) * 32) &&
                        slt32(local_tid_16991, segscan_group_sizze_15253)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_16995)[local_tid_16991] =
                                x_17002;
                            x_17004 = x_17002;
                            ((volatile __local
                              double *) scan_arr_mem_16997)[local_tid_16991] =
                                x_17003;
                            x_17005 = x_17003;
                        }
                    }
                    if (sle32(wave_sizze_16993, skip_threads_17016)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_17016 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16991 - squot32(local_tid_16991, 32) * 32) ==
                    31 && slt32(local_tid_16991, segscan_group_sizze_15253)) {
                    ((volatile __local
                      double *) scan_arr_mem_16995)[squot32(local_tid_16991,
                                                            32)] = x_17002;
                    ((volatile __local
                      double *) scan_arr_mem_16997)[squot32(local_tid_16991,
                                                            32)] = x_17003;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_17018;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16991, 32) == 0 &&
                        slt32(local_tid_16991, segscan_group_sizze_15253)) {
                        x_17011 = ((volatile __local
                                    double *) scan_arr_mem_16995)[local_tid_16991];
                        x_17012 = ((volatile __local
                                    double *) scan_arr_mem_16997)[local_tid_16991];
                        if ((local_tid_16991 - squot32(local_tid_16991, 32) *
                             32) == 0) {
                            x_17009 = x_17011;
                            x_17010 = x_17012;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_17018 = 1;
                    while (slt32(skip_threads_17018, 32)) {
                        if (sle32(skip_threads_17018, local_tid_16991 -
                                  squot32(local_tid_16991, 32) * 32) &&
                            (squot32(local_tid_16991, 32) == 0 &&
                             slt32(local_tid_16991,
                                   segscan_group_sizze_15253))) {
                            // read operands
                            {
                                x_17009 = ((volatile __local
                                            double *) scan_arr_mem_16995)[local_tid_16991 -
                                                                          skip_threads_17018];
                                x_17010 = ((volatile __local
                                            double *) scan_arr_mem_16997)[local_tid_16991 -
                                                                          skip_threads_17018];
                            }
                            // perform operation
                            {
                                bool inactive_17019 =
                                     slt32(srem32(local_tid_16991 * 32 + 32 -
                                                  1 + chunk_offset_17000, 115),
                                           local_tid_16991 * 32 + 32 - 1 +
                                           chunk_offset_17000 -
                                           ((local_tid_16991 -
                                             skip_threads_17018) * 32 + 32 - 1 +
                                            chunk_offset_17000));
                                
                                if (inactive_17019) {
                                    x_17009 = x_17011;
                                    x_17010 = x_17012;
                                }
                                if (!inactive_17019) {
                                    double y_17013 = x_17009 * x_17012;
                                    double res_17014 = x_17011 + y_17013;
                                    double res_17015 = x_17010 * x_17012;
                                    
                                    x_17009 = res_17014;
                                    x_17010 = res_17015;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16993, skip_threads_17018)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_17018, local_tid_16991 -
                                  squot32(local_tid_16991, 32) * 32) &&
                            (squot32(local_tid_16991, 32) == 0 &&
                             slt32(local_tid_16991,
                                   segscan_group_sizze_15253))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_16995)[local_tid_16991] =
                                    x_17009;
                                x_17011 = x_17009;
                                ((volatile __local
                                  double *) scan_arr_mem_16997)[local_tid_16991] =
                                    x_17010;
                                x_17012 = x_17010;
                            }
                        }
                        if (sle32(wave_sizze_16993, skip_threads_17018)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_17018 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16991, 32) == 0 ||
                      !slt32(local_tid_16991, segscan_group_sizze_15253))) {
                    // read operands
                    {
                        x_17004 = x_17002;
                        x_17005 = x_17003;
                        x_17002 = ((__local
                                    double *) scan_arr_mem_16995)[squot32(local_tid_16991,
                                                                          32) -
                                                                  1];
                        x_17003 = ((__local
                                    double *) scan_arr_mem_16997)[squot32(local_tid_16991,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_17020 = slt32(srem32(local_tid_16991 +
                                                           chunk_offset_17000,
                                                           115),
                                                    local_tid_16991 +
                                                    chunk_offset_17000 -
                                                    (squot32(local_tid_16991,
                                                             32) * 32 - 1 +
                                                     chunk_offset_17000));
                        
                        if (inactive_17020) {
                            x_17002 = x_17004;
                            x_17003 = x_17005;
                        }
                        if (!inactive_17020) {
                            double y_17006 = x_17002 * x_17005;
                            double res_17007 = x_17004 + y_17006;
                            double res_17008 = x_17003 * x_17005;
                            
                            x_17002 = res_17007;
                            x_17003 = res_17008;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_16995)[local_tid_16991] =
                            x_17002;
                        ((__local
                          double *) scan_arr_mem_16997)[local_tid_16991] =
                            x_17003;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16991, 32) == 0) {
                    ((__local double *) scan_arr_mem_16995)[local_tid_16991] =
                        x_17004;
                    ((__local double *) scan_arr_mem_16997)[local_tid_16991] =
                        x_17005;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14766, n_11537) && slt32(gtid_14776, 115)) {
                    ((__global double *) mem_16650)[gtid_14766 * 115 +
                                                    gtid_14776] = ((__local
                                                                    double *) scan_arr_mem_16995)[local_tid_16991];
                    ((__global double *) mem_16654)[gtid_14766 * 115 +
                                                    gtid_14776] = ((__local
                                                                    double *) scan_arr_mem_16997)[local_tid_16991];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_17021 = slt32(srem32(chunk_offset_17000 +
                                                          segscan_group_sizze_15253,
                                                          115),
                                                   chunk_offset_17000 +
                                                   segscan_group_sizze_15253 -
                                                   (chunk_offset_17000 +
                                                    segscan_group_sizze_15253 -
                                                    1));
                bool should_load_carry_17022 = local_tid_16991 == 0 &&
                     !crosses_segment_17021;
                
                if (should_load_carry_17022) {
                    x_15258 = ((__local
                                double *) scan_arr_mem_16995)[segscan_group_sizze_15253 -
                                                              1];
                    x_15259 = ((__local
                                double *) scan_arr_mem_16997)[segscan_group_sizze_15253 -
                                                              1];
                }
                if (!should_load_carry_17022) {
                    x_15258 = 0.0;
                    x_15259 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_15253
}
__kernel void tridagNestedConstziscan_stage1_15010(__global int *global_failure,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16864_backing_aligned_0,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16862_backing_aligned_1,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16860_backing_aligned_2,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16858_backing_aligned_3,
                                                   int32_t n_11537,
                                                   int32_t INNER_DIM_11538,
                                                   int32_t INNER_DIM_11540,
                                                   int32_t INNER_DIM_11542,
                                                   __global
                                                   unsigned char *a_mem_16584,
                                                   __global
                                                   unsigned char *b_mem_16585,
                                                   __global
                                                   unsigned char *c_mem_16586,
                                                   __global
                                                   unsigned char *mem_16628,
                                                   __global
                                                   unsigned char *mem_16632,
                                                   __global
                                                   unsigned char *mem_16636,
                                                   __global
                                                   unsigned char *mem_16640,
                                                   int32_t num_threads_16852)
{
    #define segscan_group_sizze_15085 (tridagNestedConstzisegscan_group_sizze_15004)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16864_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16864_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16862_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16862_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16860_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16860_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16858_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16858_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16853;
    int32_t local_tid_16854;
    int32_t group_sizze_16857;
    int32_t wave_sizze_16856;
    int32_t group_tid_16855;
    
    global_tid_16853 = get_global_id(0);
    local_tid_16854 = get_local_id(0);
    group_sizze_16857 = get_local_size(0);
    wave_sizze_16856 = LOCKSTEP_WIDTH;
    group_tid_16855 = get_group_id(0);
    
    int32_t phys_tid_15010;
    
    phys_tid_15010 = global_tid_16853;
    
    __local char *scan_arr_mem_16858;
    __local char *scan_arr_mem_16860;
    __local char *scan_arr_mem_16862;
    __local char *scan_arr_mem_16864;
    
    scan_arr_mem_16858 = (__local char *) scan_arr_mem_16858_backing_0;
    scan_arr_mem_16860 = (__local char *) scan_arr_mem_16860_backing_1;
    scan_arr_mem_16862 = (__local char *) scan_arr_mem_16862_backing_2;
    scan_arr_mem_16864 = (__local char *) scan_arr_mem_16864_backing_3;
    
    double x_15092;
    double x_15093;
    double x_15094;
    double x_15095;
    double x_15096;
    double x_15097;
    double x_15098;
    double x_15099;
    
    x_15092 = 1.0;
    x_15093 = 0.0;
    x_15094 = 0.0;
    x_15095 = 1.0;
    for (int32_t j_16866 = 0; j_16866 < sdiv_up32(n_11537 * 115,
                                                  num_threads_16852);
         j_16866++) {
        int32_t chunk_offset_16867 = segscan_group_sizze_15085 * j_16866 +
                group_tid_16855 * (segscan_group_sizze_15085 *
                                   sdiv_up32(n_11537 * 115, num_threads_16852));
        int32_t flat_idx_16868 = chunk_offset_16867 + local_tid_16854;
        int32_t gtid_14999 = squot32(flat_idx_16868, 115);
        int32_t gtid_15009 = flat_idx_16868 - squot32(flat_idx_16868, 115) *
                115;
        
        // threads in bounds read input
        {
            if (slt32(gtid_14999, n_11537) && slt32(gtid_15009, 115)) {
                bool cond_15124 = slt32(0, gtid_15009);
                double res_15125;
                
                if (cond_15124) {
                    res_15125 = 1.0;
                } else {
                    res_15125 = 0.0;
                }
                
                double res_15126;
                
                if (cond_15124) {
                    res_15126 = 0.0;
                } else {
                    res_15126 = 1.0;
                }
                
                double res_15127;
                
                if (cond_15124) {
                    double x_elem_15122 = ((__global
                                            double *) b_mem_16585)[gtid_14999 *
                                                                   INNER_DIM_11540 +
                                                                   gtid_15009];
                    
                    res_15127 = x_elem_15122;
                } else {
                    res_15127 = 1.0;
                }
                
                double res_15128;
                
                if (cond_15124) {
                    double x_elem_15123 = ((__global
                                            double *) a_mem_16584)[gtid_14999 *
                                                                   INNER_DIM_11538 +
                                                                   gtid_15009];
                    int32_t i_15129 = sub32(gtid_15009, 1);
                    double y_15130 = ((__global
                                       double *) c_mem_16586)[gtid_14999 *
                                                              INNER_DIM_11542 +
                                                              i_15129];
                    double y_15131 = x_elem_15123 * y_15130;
                    double res_15132 = 0.0 - y_15131;
                    
                    res_15128 = res_15132;
                } else {
                    res_15128 = 0.0;
                }
                // write to-scan values to parameters
                {
                    x_15096 = res_15127;
                    x_15097 = res_15128;
                    x_15098 = res_15125;
                    x_15099 = res_15126;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_14999, n_11537) && slt32(gtid_15009, 115))) {
                    x_15096 = 1.0;
                    x_15097 = 0.0;
                    x_15098 = 0.0;
                    x_15099 = 1.0;
                }
            }
            // combine with carry and write to local memory
            {
                double y_15100 = x_15092 * x_15096;
                double value_15101 = 1.0 / y_15100;
                double y_15102 = x_15094 * x_15097;
                double x_15103 = y_15100 + y_15102;
                double res_15104 = value_15101 * x_15103;
                double x_15105 = x_15093 * x_15096;
                double y_15106 = x_15095 * x_15097;
                double x_15107 = x_15105 + y_15106;
                double res_15108 = value_15101 * x_15107;
                double x_15109 = x_15092 * x_15098;
                double y_15110 = x_15094 * x_15099;
                double x_15111 = x_15109 + y_15110;
                double res_15112 = value_15101 * x_15111;
                double x_15113 = x_15093 * x_15098;
                double y_15114 = x_15095 * x_15099;
                double x_15115 = x_15113 + y_15114;
                double res_15116 = value_15101 * x_15115;
                
                ((__local double *) scan_arr_mem_16858)[local_tid_16854] =
                    res_15104;
                ((__local double *) scan_arr_mem_16860)[local_tid_16854] =
                    res_15108;
                ((__local double *) scan_arr_mem_16862)[local_tid_16854] =
                    res_15112;
                ((__local double *) scan_arr_mem_16864)[local_tid_16854] =
                    res_15116;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            double x_16869;
            double x_16870;
            double x_16871;
            double x_16872;
            double x_16873;
            double x_16874;
            double x_16875;
            double x_16876;
            double x_16894;
            double x_16895;
            double x_16896;
            double x_16897;
            double x_16898;
            double x_16899;
            double x_16900;
            double x_16901;
            int32_t skip_threads_16919;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_16854, segscan_group_sizze_15085)) {
                    x_16873 = ((volatile __local
                                double *) scan_arr_mem_16858)[local_tid_16854];
                    x_16874 = ((volatile __local
                                double *) scan_arr_mem_16860)[local_tid_16854];
                    x_16875 = ((volatile __local
                                double *) scan_arr_mem_16862)[local_tid_16854];
                    x_16876 = ((volatile __local
                                double *) scan_arr_mem_16864)[local_tid_16854];
                    if ((local_tid_16854 - squot32(local_tid_16854, 32) * 32) ==
                        0) {
                        x_16869 = x_16873;
                        x_16870 = x_16874;
                        x_16871 = x_16875;
                        x_16872 = x_16876;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_16919 = 1;
                while (slt32(skip_threads_16919, 32)) {
                    if (sle32(skip_threads_16919, local_tid_16854 -
                              squot32(local_tid_16854, 32) * 32) &&
                        slt32(local_tid_16854, segscan_group_sizze_15085)) {
                        // read operands
                        {
                            x_16869 = ((volatile __local
                                        double *) scan_arr_mem_16858)[local_tid_16854 -
                                                                      skip_threads_16919];
                            x_16870 = ((volatile __local
                                        double *) scan_arr_mem_16860)[local_tid_16854 -
                                                                      skip_threads_16919];
                            x_16871 = ((volatile __local
                                        double *) scan_arr_mem_16862)[local_tid_16854 -
                                                                      skip_threads_16919];
                            x_16872 = ((volatile __local
                                        double *) scan_arr_mem_16864)[local_tid_16854 -
                                                                      skip_threads_16919];
                        }
                        // perform operation
                        {
                            bool inactive_16920 = slt32(srem32(local_tid_16854 +
                                                               chunk_offset_16867,
                                                               115),
                                                        local_tid_16854 +
                                                        chunk_offset_16867 -
                                                        (local_tid_16854 -
                                                         skip_threads_16919 +
                                                         chunk_offset_16867));
                            
                            if (inactive_16920) {
                                x_16869 = x_16873;
                                x_16870 = x_16874;
                                x_16871 = x_16875;
                                x_16872 = x_16876;
                            }
                            if (!inactive_16920) {
                                double y_16877 = x_16869 * x_16873;
                                double value_16878 = 1.0 / y_16877;
                                double y_16879 = x_16871 * x_16874;
                                double x_16880 = y_16877 + y_16879;
                                double res_16881 = value_16878 * x_16880;
                                double x_16882 = x_16870 * x_16873;
                                double y_16883 = x_16872 * x_16874;
                                double x_16884 = x_16882 + y_16883;
                                double res_16885 = value_16878 * x_16884;
                                double x_16886 = x_16869 * x_16875;
                                double y_16887 = x_16871 * x_16876;
                                double x_16888 = x_16886 + y_16887;
                                double res_16889 = value_16878 * x_16888;
                                double x_16890 = x_16870 * x_16875;
                                double y_16891 = x_16872 * x_16876;
                                double x_16892 = x_16890 + y_16891;
                                double res_16893 = value_16878 * x_16892;
                                
                                x_16869 = res_16881;
                                x_16870 = res_16885;
                                x_16871 = res_16889;
                                x_16872 = res_16893;
                            }
                        }
                    }
                    if (sle32(wave_sizze_16856, skip_threads_16919)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_16919, local_tid_16854 -
                              squot32(local_tid_16854, 32) * 32) &&
                        slt32(local_tid_16854, segscan_group_sizze_15085)) {
                        // write result
                        {
                            ((volatile __local
                              double *) scan_arr_mem_16858)[local_tid_16854] =
                                x_16869;
                            x_16873 = x_16869;
                            ((volatile __local
                              double *) scan_arr_mem_16860)[local_tid_16854] =
                                x_16870;
                            x_16874 = x_16870;
                            ((volatile __local
                              double *) scan_arr_mem_16862)[local_tid_16854] =
                                x_16871;
                            x_16875 = x_16871;
                            ((volatile __local
                              double *) scan_arr_mem_16864)[local_tid_16854] =
                                x_16872;
                            x_16876 = x_16872;
                        }
                    }
                    if (sle32(wave_sizze_16856, skip_threads_16919)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_16919 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_16854 - squot32(local_tid_16854, 32) * 32) ==
                    31 && slt32(local_tid_16854, segscan_group_sizze_15085)) {
                    ((volatile __local
                      double *) scan_arr_mem_16858)[squot32(local_tid_16854,
                                                            32)] = x_16869;
                    ((volatile __local
                      double *) scan_arr_mem_16860)[squot32(local_tid_16854,
                                                            32)] = x_16870;
                    ((volatile __local
                      double *) scan_arr_mem_16862)[squot32(local_tid_16854,
                                                            32)] = x_16871;
                    ((volatile __local
                      double *) scan_arr_mem_16864)[squot32(local_tid_16854,
                                                            32)] = x_16872;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_16921;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_16854, 32) == 0 &&
                        slt32(local_tid_16854, segscan_group_sizze_15085)) {
                        x_16898 = ((volatile __local
                                    double *) scan_arr_mem_16858)[local_tid_16854];
                        x_16899 = ((volatile __local
                                    double *) scan_arr_mem_16860)[local_tid_16854];
                        x_16900 = ((volatile __local
                                    double *) scan_arr_mem_16862)[local_tid_16854];
                        x_16901 = ((volatile __local
                                    double *) scan_arr_mem_16864)[local_tid_16854];
                        if ((local_tid_16854 - squot32(local_tid_16854, 32) *
                             32) == 0) {
                            x_16894 = x_16898;
                            x_16895 = x_16899;
                            x_16896 = x_16900;
                            x_16897 = x_16901;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_16921 = 1;
                    while (slt32(skip_threads_16921, 32)) {
                        if (sle32(skip_threads_16921, local_tid_16854 -
                                  squot32(local_tid_16854, 32) * 32) &&
                            (squot32(local_tid_16854, 32) == 0 &&
                             slt32(local_tid_16854,
                                   segscan_group_sizze_15085))) {
                            // read operands
                            {
                                x_16894 = ((volatile __local
                                            double *) scan_arr_mem_16858)[local_tid_16854 -
                                                                          skip_threads_16921];
                                x_16895 = ((volatile __local
                                            double *) scan_arr_mem_16860)[local_tid_16854 -
                                                                          skip_threads_16921];
                                x_16896 = ((volatile __local
                                            double *) scan_arr_mem_16862)[local_tid_16854 -
                                                                          skip_threads_16921];
                                x_16897 = ((volatile __local
                                            double *) scan_arr_mem_16864)[local_tid_16854 -
                                                                          skip_threads_16921];
                            }
                            // perform operation
                            {
                                bool inactive_16922 =
                                     slt32(srem32(local_tid_16854 * 32 + 32 -
                                                  1 + chunk_offset_16867, 115),
                                           local_tid_16854 * 32 + 32 - 1 +
                                           chunk_offset_16867 -
                                           ((local_tid_16854 -
                                             skip_threads_16921) * 32 + 32 - 1 +
                                            chunk_offset_16867));
                                
                                if (inactive_16922) {
                                    x_16894 = x_16898;
                                    x_16895 = x_16899;
                                    x_16896 = x_16900;
                                    x_16897 = x_16901;
                                }
                                if (!inactive_16922) {
                                    double y_16902 = x_16894 * x_16898;
                                    double value_16903 = 1.0 / y_16902;
                                    double y_16904 = x_16896 * x_16899;
                                    double x_16905 = y_16902 + y_16904;
                                    double res_16906 = value_16903 * x_16905;
                                    double x_16907 = x_16895 * x_16898;
                                    double y_16908 = x_16897 * x_16899;
                                    double x_16909 = x_16907 + y_16908;
                                    double res_16910 = value_16903 * x_16909;
                                    double x_16911 = x_16894 * x_16900;
                                    double y_16912 = x_16896 * x_16901;
                                    double x_16913 = x_16911 + y_16912;
                                    double res_16914 = value_16903 * x_16913;
                                    double x_16915 = x_16895 * x_16900;
                                    double y_16916 = x_16897 * x_16901;
                                    double x_16917 = x_16915 + y_16916;
                                    double res_16918 = value_16903 * x_16917;
                                    
                                    x_16894 = res_16906;
                                    x_16895 = res_16910;
                                    x_16896 = res_16914;
                                    x_16897 = res_16918;
                                }
                            }
                        }
                        if (sle32(wave_sizze_16856, skip_threads_16921)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_16921, local_tid_16854 -
                                  squot32(local_tid_16854, 32) * 32) &&
                            (squot32(local_tid_16854, 32) == 0 &&
                             slt32(local_tid_16854,
                                   segscan_group_sizze_15085))) {
                            // write result
                            {
                                ((volatile __local
                                  double *) scan_arr_mem_16858)[local_tid_16854] =
                                    x_16894;
                                x_16898 = x_16894;
                                ((volatile __local
                                  double *) scan_arr_mem_16860)[local_tid_16854] =
                                    x_16895;
                                x_16899 = x_16895;
                                ((volatile __local
                                  double *) scan_arr_mem_16862)[local_tid_16854] =
                                    x_16896;
                                x_16900 = x_16896;
                                ((volatile __local
                                  double *) scan_arr_mem_16864)[local_tid_16854] =
                                    x_16897;
                                x_16901 = x_16897;
                            }
                        }
                        if (sle32(wave_sizze_16856, skip_threads_16921)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_16921 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_16854, 32) == 0 ||
                      !slt32(local_tid_16854, segscan_group_sizze_15085))) {
                    // read operands
                    {
                        x_16873 = x_16869;
                        x_16874 = x_16870;
                        x_16875 = x_16871;
                        x_16876 = x_16872;
                        x_16869 = ((__local
                                    double *) scan_arr_mem_16858)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                        x_16870 = ((__local
                                    double *) scan_arr_mem_16860)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                        x_16871 = ((__local
                                    double *) scan_arr_mem_16862)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                        x_16872 = ((__local
                                    double *) scan_arr_mem_16864)[squot32(local_tid_16854,
                                                                          32) -
                                                                  1];
                    }
                    // perform operation
                    {
                        bool inactive_16923 = slt32(srem32(local_tid_16854 +
                                                           chunk_offset_16867,
                                                           115),
                                                    local_tid_16854 +
                                                    chunk_offset_16867 -
                                                    (squot32(local_tid_16854,
                                                             32) * 32 - 1 +
                                                     chunk_offset_16867));
                        
                        if (inactive_16923) {
                            x_16869 = x_16873;
                            x_16870 = x_16874;
                            x_16871 = x_16875;
                            x_16872 = x_16876;
                        }
                        if (!inactive_16923) {
                            double y_16877 = x_16869 * x_16873;
                            double value_16878 = 1.0 / y_16877;
                            double y_16879 = x_16871 * x_16874;
                            double x_16880 = y_16877 + y_16879;
                            double res_16881 = value_16878 * x_16880;
                            double x_16882 = x_16870 * x_16873;
                            double y_16883 = x_16872 * x_16874;
                            double x_16884 = x_16882 + y_16883;
                            double res_16885 = value_16878 * x_16884;
                            double x_16886 = x_16869 * x_16875;
                            double y_16887 = x_16871 * x_16876;
                            double x_16888 = x_16886 + y_16887;
                            double res_16889 = value_16878 * x_16888;
                            double x_16890 = x_16870 * x_16875;
                            double y_16891 = x_16872 * x_16876;
                            double x_16892 = x_16890 + y_16891;
                            double res_16893 = value_16878 * x_16892;
                            
                            x_16869 = res_16881;
                            x_16870 = res_16885;
                            x_16871 = res_16889;
                            x_16872 = res_16893;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          double *) scan_arr_mem_16858)[local_tid_16854] =
                            x_16869;
                        ((__local
                          double *) scan_arr_mem_16860)[local_tid_16854] =
                            x_16870;
                        ((__local
                          double *) scan_arr_mem_16862)[local_tid_16854] =
                            x_16871;
                        ((__local
                          double *) scan_arr_mem_16864)[local_tid_16854] =
                            x_16872;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_16854, 32) == 0) {
                    ((__local double *) scan_arr_mem_16858)[local_tid_16854] =
                        x_16873;
                    ((__local double *) scan_arr_mem_16860)[local_tid_16854] =
                        x_16874;
                    ((__local double *) scan_arr_mem_16862)[local_tid_16854] =
                        x_16875;
                    ((__local double *) scan_arr_mem_16864)[local_tid_16854] =
                        x_16876;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_14999, n_11537) && slt32(gtid_15009, 115)) {
                    ((__global double *) mem_16628)[gtid_14999 * 115 +
                                                    gtid_15009] = ((__local
                                                                    double *) scan_arr_mem_16858)[local_tid_16854];
                    ((__global double *) mem_16632)[gtid_14999 * 115 +
                                                    gtid_15009] = ((__local
                                                                    double *) scan_arr_mem_16860)[local_tid_16854];
                    ((__global double *) mem_16636)[gtid_14999 * 115 +
                                                    gtid_15009] = ((__local
                                                                    double *) scan_arr_mem_16862)[local_tid_16854];
                    ((__global double *) mem_16640)[gtid_14999 * 115 +
                                                    gtid_15009] = ((__local
                                                                    double *) scan_arr_mem_16864)[local_tid_16854];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_16924 = slt32(srem32(chunk_offset_16867 +
                                                          segscan_group_sizze_15085,
                                                          115),
                                                   chunk_offset_16867 +
                                                   segscan_group_sizze_15085 -
                                                   (chunk_offset_16867 +
                                                    segscan_group_sizze_15085 -
                                                    1));
                bool should_load_carry_16925 = local_tid_16854 == 0 &&
                     !crosses_segment_16924;
                
                if (should_load_carry_16925) {
                    x_15092 = ((__local
                                double *) scan_arr_mem_16858)[segscan_group_sizze_15085 -
                                                              1];
                    x_15093 = ((__local
                                double *) scan_arr_mem_16860)[segscan_group_sizze_15085 -
                                                              1];
                    x_15094 = ((__local
                                double *) scan_arr_mem_16862)[segscan_group_sizze_15085 -
                                                              1];
                    x_15095 = ((__local
                                double *) scan_arr_mem_16864)[segscan_group_sizze_15085 -
                                                              1];
                }
                if (!should_load_carry_16925) {
                    x_15092 = 1.0;
                    x_15093 = 0.0;
                    x_15094 = 0.0;
                    x_15095 = 1.0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_15085
}
__kernel void tridagNestedConstziscan_stage2_14622(__global int *global_failure,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_17113_backing_aligned_0,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_17111_backing_aligned_1,
                                                   int32_t n_11537, __global
                                                   unsigned char *mem_16668,
                                                   __global
                                                   unsigned char *mem_16672,
                                                   int32_t stage1_num_groups_17071,
                                                   int32_t num_threads_17072)
{
    #define segscan_group_sizze_15358 (tridagNestedConstzisegscan_group_sizze_14616)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17113_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17113_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17111_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17111_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17106;
    int32_t local_tid_17107;
    int32_t group_sizze_17110;
    int32_t wave_sizze_17109;
    int32_t group_tid_17108;
    
    global_tid_17106 = get_global_id(0);
    local_tid_17107 = get_local_id(0);
    group_sizze_17110 = get_local_size(0);
    wave_sizze_17109 = LOCKSTEP_WIDTH;
    group_tid_17108 = get_group_id(0);
    
    int32_t phys_tid_14622;
    
    phys_tid_14622 = global_tid_17106;
    
    __local char *scan_arr_mem_17111;
    __local char *scan_arr_mem_17113;
    
    scan_arr_mem_17111 = (__local char *) scan_arr_mem_17111_backing_0;
    scan_arr_mem_17113 = (__local char *) scan_arr_mem_17113_backing_1;
    
    int32_t flat_idx_17115;
    
    flat_idx_17115 = (local_tid_17107 + 1) * (segscan_group_sizze_15358 *
                                              sdiv_up32(n_11537 * 115,
                                                        num_threads_17072)) - 1;
    
    int32_t gtid_14611;
    
    gtid_14611 = squot32(flat_idx_17115, 115);
    
    int32_t gtid_14621;
    
    gtid_14621 = flat_idx_17115 - squot32(flat_idx_17115, 115) * 115;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14611, n_11537) && slt32(gtid_14621, 115)) {
            ((__local double *) scan_arr_mem_17111)[local_tid_17107] =
                ((__global double *) mem_16668)[gtid_14611 * 115 + gtid_14621];
            ((__local double *) scan_arr_mem_17113)[local_tid_17107] =
                ((__global double *) mem_16672)[gtid_14611 * 115 + gtid_14621];
        } else {
            ((__local double *) scan_arr_mem_17111)[local_tid_17107] = 0.0;
            ((__local double *) scan_arr_mem_17113)[local_tid_17107] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_15363;
    double x_15364;
    double x_15365;
    double x_15366;
    double x_17116;
    double x_17117;
    double x_17118;
    double x_17119;
    int32_t skip_threads_17123;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_17107, stage1_num_groups_17071)) {
            x_15365 = ((volatile __local
                        double *) scan_arr_mem_17111)[local_tid_17107];
            x_15366 = ((volatile __local
                        double *) scan_arr_mem_17113)[local_tid_17107];
            if ((local_tid_17107 - squot32(local_tid_17107, 32) * 32) == 0) {
                x_15363 = x_15365;
                x_15364 = x_15366;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17123 = 1;
        while (slt32(skip_threads_17123, 32)) {
            if (sle32(skip_threads_17123, local_tid_17107 -
                      squot32(local_tid_17107, 32) * 32) &&
                slt32(local_tid_17107, stage1_num_groups_17071)) {
                // read operands
                {
                    x_15363 = ((volatile __local
                                double *) scan_arr_mem_17111)[local_tid_17107 -
                                                              skip_threads_17123];
                    x_15364 = ((volatile __local
                                double *) scan_arr_mem_17113)[local_tid_17107 -
                                                              skip_threads_17123];
                }
                // perform operation
                {
                    bool inactive_17124 = slt32(srem32((local_tid_17107 + 1) *
                                                       (segscan_group_sizze_15358 *
                                                        sdiv_up32(n_11537 * 115,
                                                                  num_threads_17072)) -
                                                       1, 115),
                                                (local_tid_17107 + 1) *
                                                (segscan_group_sizze_15358 *
                                                 sdiv_up32(n_11537 * 115,
                                                           num_threads_17072)) -
                                                1 - ((local_tid_17107 -
                                                      skip_threads_17123 + 1) *
                                                     (segscan_group_sizze_15358 *
                                                      sdiv_up32(n_11537 * 115,
                                                                num_threads_17072)) -
                                                     1));
                    
                    if (inactive_17124) {
                        x_15363 = x_15365;
                        x_15364 = x_15366;
                    }
                    if (!inactive_17124) {
                        double y_15367 = x_15363 * x_15366;
                        double res_15368 = x_15365 + y_15367;
                        double res_15369 = x_15364 * x_15366;
                        
                        x_15363 = res_15368;
                        x_15364 = res_15369;
                    }
                }
            }
            if (sle32(wave_sizze_17109, skip_threads_17123)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17123, local_tid_17107 -
                      squot32(local_tid_17107, 32) * 32) &&
                slt32(local_tid_17107, stage1_num_groups_17071)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_17111)[local_tid_17107] = x_15363;
                    x_15365 = x_15363;
                    ((volatile __local
                      double *) scan_arr_mem_17113)[local_tid_17107] = x_15364;
                    x_15366 = x_15364;
                }
            }
            if (sle32(wave_sizze_17109, skip_threads_17123)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17123 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17107 - squot32(local_tid_17107, 32) * 32) == 31 &&
            slt32(local_tid_17107, stage1_num_groups_17071)) {
            ((volatile __local
              double *) scan_arr_mem_17111)[squot32(local_tid_17107, 32)] =
                x_15363;
            ((volatile __local
              double *) scan_arr_mem_17113)[squot32(local_tid_17107, 32)] =
                x_15364;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17125;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17107, 32) == 0 && slt32(local_tid_17107,
                                                           stage1_num_groups_17071)) {
                x_17118 = ((volatile __local
                            double *) scan_arr_mem_17111)[local_tid_17107];
                x_17119 = ((volatile __local
                            double *) scan_arr_mem_17113)[local_tid_17107];
                if ((local_tid_17107 - squot32(local_tid_17107, 32) * 32) ==
                    0) {
                    x_17116 = x_17118;
                    x_17117 = x_17119;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17125 = 1;
            while (slt32(skip_threads_17125, 32)) {
                if (sle32(skip_threads_17125, local_tid_17107 -
                          squot32(local_tid_17107, 32) * 32) &&
                    (squot32(local_tid_17107, 32) == 0 && slt32(local_tid_17107,
                                                                stage1_num_groups_17071))) {
                    // read operands
                    {
                        x_17116 = ((volatile __local
                                    double *) scan_arr_mem_17111)[local_tid_17107 -
                                                                  skip_threads_17125];
                        x_17117 = ((volatile __local
                                    double *) scan_arr_mem_17113)[local_tid_17107 -
                                                                  skip_threads_17125];
                    }
                    // perform operation
                    {
                        bool inactive_17126 = slt32(srem32((local_tid_17107 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_15358 *
                                                            sdiv_up32(n_11537 *
                                                                      115,
                                                                      num_threads_17072)) -
                                                           1, 115),
                                                    (local_tid_17107 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_15358 *
                                                     sdiv_up32(n_11537 * 115,
                                                               num_threads_17072)) -
                                                    1 - (((local_tid_17107 -
                                                           skip_threads_17125) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_15358 *
                                                          sdiv_up32(n_11537 *
                                                                    115,
                                                                    num_threads_17072)) -
                                                         1));
                        
                        if (inactive_17126) {
                            x_17116 = x_17118;
                            x_17117 = x_17119;
                        }
                        if (!inactive_17126) {
                            double y_17120 = x_17116 * x_17119;
                            double res_17121 = x_17118 + y_17120;
                            double res_17122 = x_17117 * x_17119;
                            
                            x_17116 = res_17121;
                            x_17117 = res_17122;
                        }
                    }
                }
                if (sle32(wave_sizze_17109, skip_threads_17125)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17125, local_tid_17107 -
                          squot32(local_tid_17107, 32) * 32) &&
                    (squot32(local_tid_17107, 32) == 0 && slt32(local_tid_17107,
                                                                stage1_num_groups_17071))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_17111)[local_tid_17107] =
                            x_17116;
                        x_17118 = x_17116;
                        ((volatile __local
                          double *) scan_arr_mem_17113)[local_tid_17107] =
                            x_17117;
                        x_17119 = x_17117;
                    }
                }
                if (sle32(wave_sizze_17109, skip_threads_17125)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17125 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17107, 32) == 0 || !slt32(local_tid_17107,
                                                          stage1_num_groups_17071))) {
            // read operands
            {
                x_15365 = x_15363;
                x_15366 = x_15364;
                x_15363 = ((__local
                            double *) scan_arr_mem_17111)[squot32(local_tid_17107,
                                                                  32) - 1];
                x_15364 = ((__local
                            double *) scan_arr_mem_17113)[squot32(local_tid_17107,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_17127 = slt32(srem32((local_tid_17107 + 1) *
                                                   (segscan_group_sizze_15358 *
                                                    sdiv_up32(n_11537 * 115,
                                                              num_threads_17072)) -
                                                   1, 115), (local_tid_17107 +
                                                             1) *
                                            (segscan_group_sizze_15358 *
                                             sdiv_up32(n_11537 * 115,
                                                       num_threads_17072)) - 1 -
                                            ((squot32(local_tid_17107, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_15358 *
                                              sdiv_up32(n_11537 * 115,
                                                        num_threads_17072)) -
                                             1));
                
                if (inactive_17127) {
                    x_15363 = x_15365;
                    x_15364 = x_15366;
                }
                if (!inactive_17127) {
                    double y_15367 = x_15363 * x_15366;
                    double res_15368 = x_15365 + y_15367;
                    double res_15369 = x_15364 * x_15366;
                    
                    x_15363 = res_15368;
                    x_15364 = res_15369;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_17111)[local_tid_17107] =
                    x_15363;
                ((__local double *) scan_arr_mem_17113)[local_tid_17107] =
                    x_15364;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17107, 32) == 0) {
            ((__local double *) scan_arr_mem_17111)[local_tid_17107] = x_15365;
            ((__local double *) scan_arr_mem_17113)[local_tid_17107] = x_15366;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14611, n_11537) && slt32(gtid_14621, 115)) {
            ((__global double *) mem_16668)[gtid_14611 * 115 + gtid_14621] =
                ((__local double *) scan_arr_mem_17111)[local_tid_17107];
            ((__global double *) mem_16672)[gtid_14611 * 115 + gtid_14621] =
                ((__local double *) scan_arr_mem_17113)[local_tid_17107];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_15358
}
__kernel void tridagNestedConstziscan_stage2_14777(__global int *global_failure,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_17030_backing_aligned_0,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_17028_backing_aligned_1,
                                                   int32_t n_11537, __global
                                                   unsigned char *mem_16650,
                                                   __global
                                                   unsigned char *mem_16654,
                                                   int32_t stage1_num_groups_16988,
                                                   int32_t num_threads_16989)
{
    #define segscan_group_sizze_15253 (tridagNestedConstzisegscan_group_sizze_14771)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_17030_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_17030_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_17028_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_17028_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17023;
    int32_t local_tid_17024;
    int32_t group_sizze_17027;
    int32_t wave_sizze_17026;
    int32_t group_tid_17025;
    
    global_tid_17023 = get_global_id(0);
    local_tid_17024 = get_local_id(0);
    group_sizze_17027 = get_local_size(0);
    wave_sizze_17026 = LOCKSTEP_WIDTH;
    group_tid_17025 = get_group_id(0);
    
    int32_t phys_tid_14777;
    
    phys_tid_14777 = global_tid_17023;
    
    __local char *scan_arr_mem_17028;
    __local char *scan_arr_mem_17030;
    
    scan_arr_mem_17028 = (__local char *) scan_arr_mem_17028_backing_0;
    scan_arr_mem_17030 = (__local char *) scan_arr_mem_17030_backing_1;
    
    int32_t flat_idx_17032;
    
    flat_idx_17032 = (local_tid_17024 + 1) * (segscan_group_sizze_15253 *
                                              sdiv_up32(n_11537 * 115,
                                                        num_threads_16989)) - 1;
    
    int32_t gtid_14766;
    
    gtid_14766 = squot32(flat_idx_17032, 115);
    
    int32_t gtid_14776;
    
    gtid_14776 = flat_idx_17032 - squot32(flat_idx_17032, 115) * 115;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14766, n_11537) && slt32(gtid_14776, 115)) {
            ((__local double *) scan_arr_mem_17028)[local_tid_17024] =
                ((__global double *) mem_16650)[gtid_14766 * 115 + gtid_14776];
            ((__local double *) scan_arr_mem_17030)[local_tid_17024] =
                ((__global double *) mem_16654)[gtid_14766 * 115 + gtid_14776];
        } else {
            ((__local double *) scan_arr_mem_17028)[local_tid_17024] = 0.0;
            ((__local double *) scan_arr_mem_17030)[local_tid_17024] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_15258;
    double x_15259;
    double x_15260;
    double x_15261;
    double x_17033;
    double x_17034;
    double x_17035;
    double x_17036;
    int32_t skip_threads_17040;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_17024, stage1_num_groups_16988)) {
            x_15260 = ((volatile __local
                        double *) scan_arr_mem_17028)[local_tid_17024];
            x_15261 = ((volatile __local
                        double *) scan_arr_mem_17030)[local_tid_17024];
            if ((local_tid_17024 - squot32(local_tid_17024, 32) * 32) == 0) {
                x_15258 = x_15260;
                x_15259 = x_15261;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_17040 = 1;
        while (slt32(skip_threads_17040, 32)) {
            if (sle32(skip_threads_17040, local_tid_17024 -
                      squot32(local_tid_17024, 32) * 32) &&
                slt32(local_tid_17024, stage1_num_groups_16988)) {
                // read operands
                {
                    x_15258 = ((volatile __local
                                double *) scan_arr_mem_17028)[local_tid_17024 -
                                                              skip_threads_17040];
                    x_15259 = ((volatile __local
                                double *) scan_arr_mem_17030)[local_tid_17024 -
                                                              skip_threads_17040];
                }
                // perform operation
                {
                    bool inactive_17041 = slt32(srem32((local_tid_17024 + 1) *
                                                       (segscan_group_sizze_15253 *
                                                        sdiv_up32(n_11537 * 115,
                                                                  num_threads_16989)) -
                                                       1, 115),
                                                (local_tid_17024 + 1) *
                                                (segscan_group_sizze_15253 *
                                                 sdiv_up32(n_11537 * 115,
                                                           num_threads_16989)) -
                                                1 - ((local_tid_17024 -
                                                      skip_threads_17040 + 1) *
                                                     (segscan_group_sizze_15253 *
                                                      sdiv_up32(n_11537 * 115,
                                                                num_threads_16989)) -
                                                     1));
                    
                    if (inactive_17041) {
                        x_15258 = x_15260;
                        x_15259 = x_15261;
                    }
                    if (!inactive_17041) {
                        double y_15262 = x_15258 * x_15261;
                        double res_15263 = x_15260 + y_15262;
                        double res_15264 = x_15259 * x_15261;
                        
                        x_15258 = res_15263;
                        x_15259 = res_15264;
                    }
                }
            }
            if (sle32(wave_sizze_17026, skip_threads_17040)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_17040, local_tid_17024 -
                      squot32(local_tid_17024, 32) * 32) &&
                slt32(local_tid_17024, stage1_num_groups_16988)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_17028)[local_tid_17024] = x_15258;
                    x_15260 = x_15258;
                    ((volatile __local
                      double *) scan_arr_mem_17030)[local_tid_17024] = x_15259;
                    x_15261 = x_15259;
                }
            }
            if (sle32(wave_sizze_17026, skip_threads_17040)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_17040 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_17024 - squot32(local_tid_17024, 32) * 32) == 31 &&
            slt32(local_tid_17024, stage1_num_groups_16988)) {
            ((volatile __local
              double *) scan_arr_mem_17028)[squot32(local_tid_17024, 32)] =
                x_15258;
            ((volatile __local
              double *) scan_arr_mem_17030)[squot32(local_tid_17024, 32)] =
                x_15259;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_17042;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_17024, 32) == 0 && slt32(local_tid_17024,
                                                           stage1_num_groups_16988)) {
                x_17035 = ((volatile __local
                            double *) scan_arr_mem_17028)[local_tid_17024];
                x_17036 = ((volatile __local
                            double *) scan_arr_mem_17030)[local_tid_17024];
                if ((local_tid_17024 - squot32(local_tid_17024, 32) * 32) ==
                    0) {
                    x_17033 = x_17035;
                    x_17034 = x_17036;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_17042 = 1;
            while (slt32(skip_threads_17042, 32)) {
                if (sle32(skip_threads_17042, local_tid_17024 -
                          squot32(local_tid_17024, 32) * 32) &&
                    (squot32(local_tid_17024, 32) == 0 && slt32(local_tid_17024,
                                                                stage1_num_groups_16988))) {
                    // read operands
                    {
                        x_17033 = ((volatile __local
                                    double *) scan_arr_mem_17028)[local_tid_17024 -
                                                                  skip_threads_17042];
                        x_17034 = ((volatile __local
                                    double *) scan_arr_mem_17030)[local_tid_17024 -
                                                                  skip_threads_17042];
                    }
                    // perform operation
                    {
                        bool inactive_17043 = slt32(srem32((local_tid_17024 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_15253 *
                                                            sdiv_up32(n_11537 *
                                                                      115,
                                                                      num_threads_16989)) -
                                                           1, 115),
                                                    (local_tid_17024 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_15253 *
                                                     sdiv_up32(n_11537 * 115,
                                                               num_threads_16989)) -
                                                    1 - (((local_tid_17024 -
                                                           skip_threads_17042) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_15253 *
                                                          sdiv_up32(n_11537 *
                                                                    115,
                                                                    num_threads_16989)) -
                                                         1));
                        
                        if (inactive_17043) {
                            x_17033 = x_17035;
                            x_17034 = x_17036;
                        }
                        if (!inactive_17043) {
                            double y_17037 = x_17033 * x_17036;
                            double res_17038 = x_17035 + y_17037;
                            double res_17039 = x_17034 * x_17036;
                            
                            x_17033 = res_17038;
                            x_17034 = res_17039;
                        }
                    }
                }
                if (sle32(wave_sizze_17026, skip_threads_17042)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_17042, local_tid_17024 -
                          squot32(local_tid_17024, 32) * 32) &&
                    (squot32(local_tid_17024, 32) == 0 && slt32(local_tid_17024,
                                                                stage1_num_groups_16988))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_17028)[local_tid_17024] =
                            x_17033;
                        x_17035 = x_17033;
                        ((volatile __local
                          double *) scan_arr_mem_17030)[local_tid_17024] =
                            x_17034;
                        x_17036 = x_17034;
                    }
                }
                if (sle32(wave_sizze_17026, skip_threads_17042)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_17042 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_17024, 32) == 0 || !slt32(local_tid_17024,
                                                          stage1_num_groups_16988))) {
            // read operands
            {
                x_15260 = x_15258;
                x_15261 = x_15259;
                x_15258 = ((__local
                            double *) scan_arr_mem_17028)[squot32(local_tid_17024,
                                                                  32) - 1];
                x_15259 = ((__local
                            double *) scan_arr_mem_17030)[squot32(local_tid_17024,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_17044 = slt32(srem32((local_tid_17024 + 1) *
                                                   (segscan_group_sizze_15253 *
                                                    sdiv_up32(n_11537 * 115,
                                                              num_threads_16989)) -
                                                   1, 115), (local_tid_17024 +
                                                             1) *
                                            (segscan_group_sizze_15253 *
                                             sdiv_up32(n_11537 * 115,
                                                       num_threads_16989)) - 1 -
                                            ((squot32(local_tid_17024, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_15253 *
                                              sdiv_up32(n_11537 * 115,
                                                        num_threads_16989)) -
                                             1));
                
                if (inactive_17044) {
                    x_15258 = x_15260;
                    x_15259 = x_15261;
                }
                if (!inactive_17044) {
                    double y_15262 = x_15258 * x_15261;
                    double res_15263 = x_15260 + y_15262;
                    double res_15264 = x_15259 * x_15261;
                    
                    x_15258 = res_15263;
                    x_15259 = res_15264;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_17028)[local_tid_17024] =
                    x_15258;
                ((__local double *) scan_arr_mem_17030)[local_tid_17024] =
                    x_15259;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_17024, 32) == 0) {
            ((__local double *) scan_arr_mem_17028)[local_tid_17024] = x_15260;
            ((__local double *) scan_arr_mem_17030)[local_tid_17024] = x_15261;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14766, n_11537) && slt32(gtid_14776, 115)) {
            ((__global double *) mem_16650)[gtid_14766 * 115 + gtid_14776] =
                ((__local double *) scan_arr_mem_17028)[local_tid_17024];
            ((__global double *) mem_16654)[gtid_14766 * 115 + gtid_14776] =
                ((__local double *) scan_arr_mem_17030)[local_tid_17024];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_15253
}
__kernel void tridagNestedConstziscan_stage2_15010(__global int *global_failure,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16937_backing_aligned_0,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16935_backing_aligned_1,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16933_backing_aligned_2,
                                                   __local volatile
                                                   int64_t *scan_arr_mem_16931_backing_aligned_3,
                                                   int32_t n_11537, __global
                                                   unsigned char *mem_16628,
                                                   __global
                                                   unsigned char *mem_16632,
                                                   __global
                                                   unsigned char *mem_16636,
                                                   __global
                                                   unsigned char *mem_16640,
                                                   int32_t stage1_num_groups_16851,
                                                   int32_t num_threads_16852)
{
    #define segscan_group_sizze_15085 (tridagNestedConstzisegscan_group_sizze_15004)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_16937_backing_3 =
                          (__local volatile
                           char *) scan_arr_mem_16937_backing_aligned_0;
    __local volatile char *restrict scan_arr_mem_16935_backing_2 =
                          (__local volatile
                           char *) scan_arr_mem_16935_backing_aligned_1;
    __local volatile char *restrict scan_arr_mem_16933_backing_1 =
                          (__local volatile
                           char *) scan_arr_mem_16933_backing_aligned_2;
    __local volatile char *restrict scan_arr_mem_16931_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_16931_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16926;
    int32_t local_tid_16927;
    int32_t group_sizze_16930;
    int32_t wave_sizze_16929;
    int32_t group_tid_16928;
    
    global_tid_16926 = get_global_id(0);
    local_tid_16927 = get_local_id(0);
    group_sizze_16930 = get_local_size(0);
    wave_sizze_16929 = LOCKSTEP_WIDTH;
    group_tid_16928 = get_group_id(0);
    
    int32_t phys_tid_15010;
    
    phys_tid_15010 = global_tid_16926;
    
    __local char *scan_arr_mem_16931;
    __local char *scan_arr_mem_16933;
    __local char *scan_arr_mem_16935;
    __local char *scan_arr_mem_16937;
    
    scan_arr_mem_16931 = (__local char *) scan_arr_mem_16931_backing_0;
    scan_arr_mem_16933 = (__local char *) scan_arr_mem_16933_backing_1;
    scan_arr_mem_16935 = (__local char *) scan_arr_mem_16935_backing_2;
    scan_arr_mem_16937 = (__local char *) scan_arr_mem_16937_backing_3;
    
    int32_t flat_idx_16939;
    
    flat_idx_16939 = (local_tid_16927 + 1) * (segscan_group_sizze_15085 *
                                              sdiv_up32(n_11537 * 115,
                                                        num_threads_16852)) - 1;
    
    int32_t gtid_14999;
    
    gtid_14999 = squot32(flat_idx_16939, 115);
    
    int32_t gtid_15009;
    
    gtid_15009 = flat_idx_16939 - squot32(flat_idx_16939, 115) * 115;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_14999, n_11537) && slt32(gtid_15009, 115)) {
            ((__local double *) scan_arr_mem_16931)[local_tid_16927] =
                ((__global double *) mem_16628)[gtid_14999 * 115 + gtid_15009];
            ((__local double *) scan_arr_mem_16933)[local_tid_16927] =
                ((__global double *) mem_16632)[gtid_14999 * 115 + gtid_15009];
            ((__local double *) scan_arr_mem_16935)[local_tid_16927] =
                ((__global double *) mem_16636)[gtid_14999 * 115 + gtid_15009];
            ((__local double *) scan_arr_mem_16937)[local_tid_16927] =
                ((__global double *) mem_16640)[gtid_14999 * 115 + gtid_15009];
        } else {
            ((__local double *) scan_arr_mem_16931)[local_tid_16927] = 1.0;
            ((__local double *) scan_arr_mem_16933)[local_tid_16927] = 0.0;
            ((__local double *) scan_arr_mem_16935)[local_tid_16927] = 0.0;
            ((__local double *) scan_arr_mem_16937)[local_tid_16927] = 1.0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_15092;
    double x_15093;
    double x_15094;
    double x_15095;
    double x_15096;
    double x_15097;
    double x_15098;
    double x_15099;
    double x_16940;
    double x_16941;
    double x_16942;
    double x_16943;
    double x_16944;
    double x_16945;
    double x_16946;
    double x_16947;
    int32_t skip_threads_16965;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16927, stage1_num_groups_16851)) {
            x_15096 = ((volatile __local
                        double *) scan_arr_mem_16931)[local_tid_16927];
            x_15097 = ((volatile __local
                        double *) scan_arr_mem_16933)[local_tid_16927];
            x_15098 = ((volatile __local
                        double *) scan_arr_mem_16935)[local_tid_16927];
            x_15099 = ((volatile __local
                        double *) scan_arr_mem_16937)[local_tid_16927];
            if ((local_tid_16927 - squot32(local_tid_16927, 32) * 32) == 0) {
                x_15092 = x_15096;
                x_15093 = x_15097;
                x_15094 = x_15098;
                x_15095 = x_15099;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16965 = 1;
        while (slt32(skip_threads_16965, 32)) {
            if (sle32(skip_threads_16965, local_tid_16927 -
                      squot32(local_tid_16927, 32) * 32) &&
                slt32(local_tid_16927, stage1_num_groups_16851)) {
                // read operands
                {
                    x_15092 = ((volatile __local
                                double *) scan_arr_mem_16931)[local_tid_16927 -
                                                              skip_threads_16965];
                    x_15093 = ((volatile __local
                                double *) scan_arr_mem_16933)[local_tid_16927 -
                                                              skip_threads_16965];
                    x_15094 = ((volatile __local
                                double *) scan_arr_mem_16935)[local_tid_16927 -
                                                              skip_threads_16965];
                    x_15095 = ((volatile __local
                                double *) scan_arr_mem_16937)[local_tid_16927 -
                                                              skip_threads_16965];
                }
                // perform operation
                {
                    bool inactive_16966 = slt32(srem32((local_tid_16927 + 1) *
                                                       (segscan_group_sizze_15085 *
                                                        sdiv_up32(n_11537 * 115,
                                                                  num_threads_16852)) -
                                                       1, 115),
                                                (local_tid_16927 + 1) *
                                                (segscan_group_sizze_15085 *
                                                 sdiv_up32(n_11537 * 115,
                                                           num_threads_16852)) -
                                                1 - ((local_tid_16927 -
                                                      skip_threads_16965 + 1) *
                                                     (segscan_group_sizze_15085 *
                                                      sdiv_up32(n_11537 * 115,
                                                                num_threads_16852)) -
                                                     1));
                    
                    if (inactive_16966) {
                        x_15092 = x_15096;
                        x_15093 = x_15097;
                        x_15094 = x_15098;
                        x_15095 = x_15099;
                    }
                    if (!inactive_16966) {
                        double y_15100 = x_15092 * x_15096;
                        double value_15101 = 1.0 / y_15100;
                        double y_15102 = x_15094 * x_15097;
                        double x_15103 = y_15100 + y_15102;
                        double res_15104 = value_15101 * x_15103;
                        double x_15105 = x_15093 * x_15096;
                        double y_15106 = x_15095 * x_15097;
                        double x_15107 = x_15105 + y_15106;
                        double res_15108 = value_15101 * x_15107;
                        double x_15109 = x_15092 * x_15098;
                        double y_15110 = x_15094 * x_15099;
                        double x_15111 = x_15109 + y_15110;
                        double res_15112 = value_15101 * x_15111;
                        double x_15113 = x_15093 * x_15098;
                        double y_15114 = x_15095 * x_15099;
                        double x_15115 = x_15113 + y_15114;
                        double res_15116 = value_15101 * x_15115;
                        
                        x_15092 = res_15104;
                        x_15093 = res_15108;
                        x_15094 = res_15112;
                        x_15095 = res_15116;
                    }
                }
            }
            if (sle32(wave_sizze_16929, skip_threads_16965)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16965, local_tid_16927 -
                      squot32(local_tid_16927, 32) * 32) &&
                slt32(local_tid_16927, stage1_num_groups_16851)) {
                // write result
                {
                    ((volatile __local
                      double *) scan_arr_mem_16931)[local_tid_16927] = x_15092;
                    x_15096 = x_15092;
                    ((volatile __local
                      double *) scan_arr_mem_16933)[local_tid_16927] = x_15093;
                    x_15097 = x_15093;
                    ((volatile __local
                      double *) scan_arr_mem_16935)[local_tid_16927] = x_15094;
                    x_15098 = x_15094;
                    ((volatile __local
                      double *) scan_arr_mem_16937)[local_tid_16927] = x_15095;
                    x_15099 = x_15095;
                }
            }
            if (sle32(wave_sizze_16929, skip_threads_16965)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16965 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16927 - squot32(local_tid_16927, 32) * 32) == 31 &&
            slt32(local_tid_16927, stage1_num_groups_16851)) {
            ((volatile __local
              double *) scan_arr_mem_16931)[squot32(local_tid_16927, 32)] =
                x_15092;
            ((volatile __local
              double *) scan_arr_mem_16933)[squot32(local_tid_16927, 32)] =
                x_15093;
            ((volatile __local
              double *) scan_arr_mem_16935)[squot32(local_tid_16927, 32)] =
                x_15094;
            ((volatile __local
              double *) scan_arr_mem_16937)[squot32(local_tid_16927, 32)] =
                x_15095;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16967;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16927, 32) == 0 && slt32(local_tid_16927,
                                                           stage1_num_groups_16851)) {
                x_16944 = ((volatile __local
                            double *) scan_arr_mem_16931)[local_tid_16927];
                x_16945 = ((volatile __local
                            double *) scan_arr_mem_16933)[local_tid_16927];
                x_16946 = ((volatile __local
                            double *) scan_arr_mem_16935)[local_tid_16927];
                x_16947 = ((volatile __local
                            double *) scan_arr_mem_16937)[local_tid_16927];
                if ((local_tid_16927 - squot32(local_tid_16927, 32) * 32) ==
                    0) {
                    x_16940 = x_16944;
                    x_16941 = x_16945;
                    x_16942 = x_16946;
                    x_16943 = x_16947;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16967 = 1;
            while (slt32(skip_threads_16967, 32)) {
                if (sle32(skip_threads_16967, local_tid_16927 -
                          squot32(local_tid_16927, 32) * 32) &&
                    (squot32(local_tid_16927, 32) == 0 && slt32(local_tid_16927,
                                                                stage1_num_groups_16851))) {
                    // read operands
                    {
                        x_16940 = ((volatile __local
                                    double *) scan_arr_mem_16931)[local_tid_16927 -
                                                                  skip_threads_16967];
                        x_16941 = ((volatile __local
                                    double *) scan_arr_mem_16933)[local_tid_16927 -
                                                                  skip_threads_16967];
                        x_16942 = ((volatile __local
                                    double *) scan_arr_mem_16935)[local_tid_16927 -
                                                                  skip_threads_16967];
                        x_16943 = ((volatile __local
                                    double *) scan_arr_mem_16937)[local_tid_16927 -
                                                                  skip_threads_16967];
                    }
                    // perform operation
                    {
                        bool inactive_16968 = slt32(srem32((local_tid_16927 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_15085 *
                                                            sdiv_up32(n_11537 *
                                                                      115,
                                                                      num_threads_16852)) -
                                                           1, 115),
                                                    (local_tid_16927 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_15085 *
                                                     sdiv_up32(n_11537 * 115,
                                                               num_threads_16852)) -
                                                    1 - (((local_tid_16927 -
                                                           skip_threads_16967) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_15085 *
                                                          sdiv_up32(n_11537 *
                                                                    115,
                                                                    num_threads_16852)) -
                                                         1));
                        
                        if (inactive_16968) {
                            x_16940 = x_16944;
                            x_16941 = x_16945;
                            x_16942 = x_16946;
                            x_16943 = x_16947;
                        }
                        if (!inactive_16968) {
                            double y_16948 = x_16940 * x_16944;
                            double value_16949 = 1.0 / y_16948;
                            double y_16950 = x_16942 * x_16945;
                            double x_16951 = y_16948 + y_16950;
                            double res_16952 = value_16949 * x_16951;
                            double x_16953 = x_16941 * x_16944;
                            double y_16954 = x_16943 * x_16945;
                            double x_16955 = x_16953 + y_16954;
                            double res_16956 = value_16949 * x_16955;
                            double x_16957 = x_16940 * x_16946;
                            double y_16958 = x_16942 * x_16947;
                            double x_16959 = x_16957 + y_16958;
                            double res_16960 = value_16949 * x_16959;
                            double x_16961 = x_16941 * x_16946;
                            double y_16962 = x_16943 * x_16947;
                            double x_16963 = x_16961 + y_16962;
                            double res_16964 = value_16949 * x_16963;
                            
                            x_16940 = res_16952;
                            x_16941 = res_16956;
                            x_16942 = res_16960;
                            x_16943 = res_16964;
                        }
                    }
                }
                if (sle32(wave_sizze_16929, skip_threads_16967)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16967, local_tid_16927 -
                          squot32(local_tid_16927, 32) * 32) &&
                    (squot32(local_tid_16927, 32) == 0 && slt32(local_tid_16927,
                                                                stage1_num_groups_16851))) {
                    // write result
                    {
                        ((volatile __local
                          double *) scan_arr_mem_16931)[local_tid_16927] =
                            x_16940;
                        x_16944 = x_16940;
                        ((volatile __local
                          double *) scan_arr_mem_16933)[local_tid_16927] =
                            x_16941;
                        x_16945 = x_16941;
                        ((volatile __local
                          double *) scan_arr_mem_16935)[local_tid_16927] =
                            x_16942;
                        x_16946 = x_16942;
                        ((volatile __local
                          double *) scan_arr_mem_16937)[local_tid_16927] =
                            x_16943;
                        x_16947 = x_16943;
                    }
                }
                if (sle32(wave_sizze_16929, skip_threads_16967)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16967 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16927, 32) == 0 || !slt32(local_tid_16927,
                                                          stage1_num_groups_16851))) {
            // read operands
            {
                x_15096 = x_15092;
                x_15097 = x_15093;
                x_15098 = x_15094;
                x_15099 = x_15095;
                x_15092 = ((__local
                            double *) scan_arr_mem_16931)[squot32(local_tid_16927,
                                                                  32) - 1];
                x_15093 = ((__local
                            double *) scan_arr_mem_16933)[squot32(local_tid_16927,
                                                                  32) - 1];
                x_15094 = ((__local
                            double *) scan_arr_mem_16935)[squot32(local_tid_16927,
                                                                  32) - 1];
                x_15095 = ((__local
                            double *) scan_arr_mem_16937)[squot32(local_tid_16927,
                                                                  32) - 1];
            }
            // perform operation
            {
                bool inactive_16969 = slt32(srem32((local_tid_16927 + 1) *
                                                   (segscan_group_sizze_15085 *
                                                    sdiv_up32(n_11537 * 115,
                                                              num_threads_16852)) -
                                                   1, 115), (local_tid_16927 +
                                                             1) *
                                            (segscan_group_sizze_15085 *
                                             sdiv_up32(n_11537 * 115,
                                                       num_threads_16852)) - 1 -
                                            ((squot32(local_tid_16927, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_15085 *
                                              sdiv_up32(n_11537 * 115,
                                                        num_threads_16852)) -
                                             1));
                
                if (inactive_16969) {
                    x_15092 = x_15096;
                    x_15093 = x_15097;
                    x_15094 = x_15098;
                    x_15095 = x_15099;
                }
                if (!inactive_16969) {
                    double y_15100 = x_15092 * x_15096;
                    double value_15101 = 1.0 / y_15100;
                    double y_15102 = x_15094 * x_15097;
                    double x_15103 = y_15100 + y_15102;
                    double res_15104 = value_15101 * x_15103;
                    double x_15105 = x_15093 * x_15096;
                    double y_15106 = x_15095 * x_15097;
                    double x_15107 = x_15105 + y_15106;
                    double res_15108 = value_15101 * x_15107;
                    double x_15109 = x_15092 * x_15098;
                    double y_15110 = x_15094 * x_15099;
                    double x_15111 = x_15109 + y_15110;
                    double res_15112 = value_15101 * x_15111;
                    double x_15113 = x_15093 * x_15098;
                    double y_15114 = x_15095 * x_15099;
                    double x_15115 = x_15113 + y_15114;
                    double res_15116 = value_15101 * x_15115;
                    
                    x_15092 = res_15104;
                    x_15093 = res_15108;
                    x_15094 = res_15112;
                    x_15095 = res_15116;
                }
            }
            // write final result
            {
                ((__local double *) scan_arr_mem_16931)[local_tid_16927] =
                    x_15092;
                ((__local double *) scan_arr_mem_16933)[local_tid_16927] =
                    x_15093;
                ((__local double *) scan_arr_mem_16935)[local_tid_16927] =
                    x_15094;
                ((__local double *) scan_arr_mem_16937)[local_tid_16927] =
                    x_15095;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16927, 32) == 0) {
            ((__local double *) scan_arr_mem_16931)[local_tid_16927] = x_15096;
            ((__local double *) scan_arr_mem_16933)[local_tid_16927] = x_15097;
            ((__local double *) scan_arr_mem_16935)[local_tid_16927] = x_15098;
            ((__local double *) scan_arr_mem_16937)[local_tid_16927] = x_15099;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_14999, n_11537) && slt32(gtid_15009, 115)) {
            ((__global double *) mem_16628)[gtid_14999 * 115 + gtid_15009] =
                ((__local double *) scan_arr_mem_16931)[local_tid_16927];
            ((__global double *) mem_16632)[gtid_14999 * 115 + gtid_15009] =
                ((__local double *) scan_arr_mem_16933)[local_tid_16927];
            ((__global double *) mem_16636)[gtid_14999 * 115 + gtid_15009] =
                ((__local double *) scan_arr_mem_16935)[local_tid_16927];
            ((__global double *) mem_16640)[gtid_14999 * 115 + gtid_15009] =
                ((__local double *) scan_arr_mem_16937)[local_tid_16927];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_15085
}
__kernel void tridagNestedConstziscan_stage3_14622(__global int *global_failure,
                                                   int32_t n_11537,
                                                   int32_t num_groups_15359,
                                                   __global
                                                   unsigned char *mem_16668,
                                                   __global
                                                   unsigned char *mem_16672,
                                                   int32_t num_threads_17072,
                                                   int32_t required_groups_17128)
{
    #define segscan_group_sizze_15358 (tridagNestedConstzisegscan_group_sizze_14616)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17129;
    int32_t local_tid_17130;
    int32_t group_sizze_17133;
    int32_t wave_sizze_17132;
    int32_t group_tid_17131;
    
    global_tid_17129 = get_global_id(0);
    local_tid_17130 = get_local_id(0);
    group_sizze_17133 = get_local_size(0);
    wave_sizze_17132 = LOCKSTEP_WIDTH;
    group_tid_17131 = get_group_id(0);
    
    int32_t phys_tid_14622;
    
    phys_tid_14622 = global_tid_17129;
    
    int32_t phys_group_id_17134;
    
    phys_group_id_17134 = get_group_id(0);
    for (int32_t i_17135 = 0; i_17135 < sdiv_up32(required_groups_17128 -
                                                  phys_group_id_17134,
                                                  num_groups_15359);
         i_17135++) {
        int32_t virt_group_id_17136 = phys_group_id_17134 + i_17135 *
                num_groups_15359;
        int32_t flat_idx_17137 = virt_group_id_17136 *
                segscan_group_sizze_15358 + local_tid_17130;
        int32_t gtid_14611 = squot32(flat_idx_17137, 115);
        int32_t gtid_14621 = flat_idx_17137 - squot32(flat_idx_17137, 115) *
                115;
        int32_t orig_group_17138 = squot32(flat_idx_17137,
                                           segscan_group_sizze_15358 *
                                           sdiv_up32(n_11537 * 115,
                                                     num_threads_17072));
        int32_t carry_in_flat_idx_17139 = orig_group_17138 *
                (segscan_group_sizze_15358 * sdiv_up32(n_11537 * 115,
                                                       num_threads_17072)) - 1;
        
        if (slt32(gtid_14611, n_11537) && slt32(gtid_14621, 115)) {
            if (!(orig_group_17138 == 0 || (flat_idx_17137 ==
                                            (orig_group_17138 + 1) *
                                            (segscan_group_sizze_15358 *
                                             sdiv_up32(n_11537 * 115,
                                                       num_threads_17072)) -
                                            1 || slt32(srem32(flat_idx_17137,
                                                              115),
                                                       flat_idx_17137 -
                                                       carry_in_flat_idx_17139)))) {
                double x_15363;
                double x_15364;
                double x_15365;
                double x_15366;
                
                x_15363 = ((__global
                            double *) mem_16668)[squot32(carry_in_flat_idx_17139,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_17139 -
                                                  squot32(carry_in_flat_idx_17139,
                                                          115) * 115)];
                x_15364 = ((__global
                            double *) mem_16672)[squot32(carry_in_flat_idx_17139,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_17139 -
                                                  squot32(carry_in_flat_idx_17139,
                                                          115) * 115)];
                x_15365 = ((__global double *) mem_16668)[gtid_14611 * 115 +
                                                          gtid_14621];
                x_15366 = ((__global double *) mem_16672)[gtid_14611 * 115 +
                                                          gtid_14621];
                
                double y_15367;
                
                y_15367 = x_15363 * x_15366;
                
                double res_15368 = x_15365 + y_15367;
                double res_15369 = x_15364 * x_15366;
                
                x_15363 = res_15368;
                x_15364 = res_15369;
                ((__global double *) mem_16668)[gtid_14611 * 115 + gtid_14621] =
                    x_15363;
                ((__global double *) mem_16672)[gtid_14611 * 115 + gtid_14621] =
                    x_15364;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_15358
}
__kernel void tridagNestedConstziscan_stage3_14777(__global int *global_failure,
                                                   int32_t n_11537,
                                                   int32_t num_groups_15254,
                                                   __global
                                                   unsigned char *mem_16650,
                                                   __global
                                                   unsigned char *mem_16654,
                                                   int32_t num_threads_16989,
                                                   int32_t required_groups_17045)
{
    #define segscan_group_sizze_15253 (tridagNestedConstzisegscan_group_sizze_14771)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17046;
    int32_t local_tid_17047;
    int32_t group_sizze_17050;
    int32_t wave_sizze_17049;
    int32_t group_tid_17048;
    
    global_tid_17046 = get_global_id(0);
    local_tid_17047 = get_local_id(0);
    group_sizze_17050 = get_local_size(0);
    wave_sizze_17049 = LOCKSTEP_WIDTH;
    group_tid_17048 = get_group_id(0);
    
    int32_t phys_tid_14777;
    
    phys_tid_14777 = global_tid_17046;
    
    int32_t phys_group_id_17051;
    
    phys_group_id_17051 = get_group_id(0);
    for (int32_t i_17052 = 0; i_17052 < sdiv_up32(required_groups_17045 -
                                                  phys_group_id_17051,
                                                  num_groups_15254);
         i_17052++) {
        int32_t virt_group_id_17053 = phys_group_id_17051 + i_17052 *
                num_groups_15254;
        int32_t flat_idx_17054 = virt_group_id_17053 *
                segscan_group_sizze_15253 + local_tid_17047;
        int32_t gtid_14766 = squot32(flat_idx_17054, 115);
        int32_t gtid_14776 = flat_idx_17054 - squot32(flat_idx_17054, 115) *
                115;
        int32_t orig_group_17055 = squot32(flat_idx_17054,
                                           segscan_group_sizze_15253 *
                                           sdiv_up32(n_11537 * 115,
                                                     num_threads_16989));
        int32_t carry_in_flat_idx_17056 = orig_group_17055 *
                (segscan_group_sizze_15253 * sdiv_up32(n_11537 * 115,
                                                       num_threads_16989)) - 1;
        
        if (slt32(gtid_14766, n_11537) && slt32(gtid_14776, 115)) {
            if (!(orig_group_17055 == 0 || (flat_idx_17054 ==
                                            (orig_group_17055 + 1) *
                                            (segscan_group_sizze_15253 *
                                             sdiv_up32(n_11537 * 115,
                                                       num_threads_16989)) -
                                            1 || slt32(srem32(flat_idx_17054,
                                                              115),
                                                       flat_idx_17054 -
                                                       carry_in_flat_idx_17056)))) {
                double x_15258;
                double x_15259;
                double x_15260;
                double x_15261;
                
                x_15258 = ((__global
                            double *) mem_16650)[squot32(carry_in_flat_idx_17056,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_17056 -
                                                  squot32(carry_in_flat_idx_17056,
                                                          115) * 115)];
                x_15259 = ((__global
                            double *) mem_16654)[squot32(carry_in_flat_idx_17056,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_17056 -
                                                  squot32(carry_in_flat_idx_17056,
                                                          115) * 115)];
                x_15260 = ((__global double *) mem_16650)[gtid_14766 * 115 +
                                                          gtid_14776];
                x_15261 = ((__global double *) mem_16654)[gtid_14766 * 115 +
                                                          gtid_14776];
                
                double y_15262;
                
                y_15262 = x_15258 * x_15261;
                
                double res_15263 = x_15260 + y_15262;
                double res_15264 = x_15259 * x_15261;
                
                x_15258 = res_15263;
                x_15259 = res_15264;
                ((__global double *) mem_16650)[gtid_14766 * 115 + gtid_14776] =
                    x_15258;
                ((__global double *) mem_16654)[gtid_14766 * 115 + gtid_14776] =
                    x_15259;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_15253
}
__kernel void tridagNestedConstziscan_stage3_15010(__global int *global_failure,
                                                   int32_t n_11537,
                                                   int32_t num_groups_15086,
                                                   __global
                                                   unsigned char *mem_16628,
                                                   __global
                                                   unsigned char *mem_16632,
                                                   __global
                                                   unsigned char *mem_16636,
                                                   __global
                                                   unsigned char *mem_16640,
                                                   int32_t num_threads_16852,
                                                   int32_t required_groups_16970)
{
    #define segscan_group_sizze_15085 (tridagNestedConstzisegscan_group_sizze_15004)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16971;
    int32_t local_tid_16972;
    int32_t group_sizze_16975;
    int32_t wave_sizze_16974;
    int32_t group_tid_16973;
    
    global_tid_16971 = get_global_id(0);
    local_tid_16972 = get_local_id(0);
    group_sizze_16975 = get_local_size(0);
    wave_sizze_16974 = LOCKSTEP_WIDTH;
    group_tid_16973 = get_group_id(0);
    
    int32_t phys_tid_15010;
    
    phys_tid_15010 = global_tid_16971;
    
    int32_t phys_group_id_16976;
    
    phys_group_id_16976 = get_group_id(0);
    for (int32_t i_16977 = 0; i_16977 < sdiv_up32(required_groups_16970 -
                                                  phys_group_id_16976,
                                                  num_groups_15086);
         i_16977++) {
        int32_t virt_group_id_16978 = phys_group_id_16976 + i_16977 *
                num_groups_15086;
        int32_t flat_idx_16979 = virt_group_id_16978 *
                segscan_group_sizze_15085 + local_tid_16972;
        int32_t gtid_14999 = squot32(flat_idx_16979, 115);
        int32_t gtid_15009 = flat_idx_16979 - squot32(flat_idx_16979, 115) *
                115;
        int32_t orig_group_16980 = squot32(flat_idx_16979,
                                           segscan_group_sizze_15085 *
                                           sdiv_up32(n_11537 * 115,
                                                     num_threads_16852));
        int32_t carry_in_flat_idx_16981 = orig_group_16980 *
                (segscan_group_sizze_15085 * sdiv_up32(n_11537 * 115,
                                                       num_threads_16852)) - 1;
        
        if (slt32(gtid_14999, n_11537) && slt32(gtid_15009, 115)) {
            if (!(orig_group_16980 == 0 || (flat_idx_16979 ==
                                            (orig_group_16980 + 1) *
                                            (segscan_group_sizze_15085 *
                                             sdiv_up32(n_11537 * 115,
                                                       num_threads_16852)) -
                                            1 || slt32(srem32(flat_idx_16979,
                                                              115),
                                                       flat_idx_16979 -
                                                       carry_in_flat_idx_16981)))) {
                double x_15092;
                double x_15093;
                double x_15094;
                double x_15095;
                double x_15096;
                double x_15097;
                double x_15098;
                double x_15099;
                
                x_15092 = ((__global
                            double *) mem_16628)[squot32(carry_in_flat_idx_16981,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          115) * 115)];
                x_15093 = ((__global
                            double *) mem_16632)[squot32(carry_in_flat_idx_16981,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          115) * 115)];
                x_15094 = ((__global
                            double *) mem_16636)[squot32(carry_in_flat_idx_16981,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          115) * 115)];
                x_15095 = ((__global
                            double *) mem_16640)[squot32(carry_in_flat_idx_16981,
                                                         115) * 115 +
                                                 (carry_in_flat_idx_16981 -
                                                  squot32(carry_in_flat_idx_16981,
                                                          115) * 115)];
                x_15096 = ((__global double *) mem_16628)[gtid_14999 * 115 +
                                                          gtid_15009];
                x_15097 = ((__global double *) mem_16632)[gtid_14999 * 115 +
                                                          gtid_15009];
                x_15098 = ((__global double *) mem_16636)[gtid_14999 * 115 +
                                                          gtid_15009];
                x_15099 = ((__global double *) mem_16640)[gtid_14999 * 115 +
                                                          gtid_15009];
                
                double y_15100;
                
                y_15100 = x_15092 * x_15096;
                
                double value_15101 = 1.0 / y_15100;
                double y_15102 = x_15094 * x_15097;
                double x_15103 = y_15100 + y_15102;
                double res_15104 = value_15101 * x_15103;
                double x_15105 = x_15093 * x_15096;
                double y_15106 = x_15095 * x_15097;
                double x_15107 = x_15105 + y_15106;
                double res_15108 = value_15101 * x_15107;
                double x_15109 = x_15092 * x_15098;
                double y_15110 = x_15094 * x_15099;
                double x_15111 = x_15109 + y_15110;
                double res_15112 = value_15101 * x_15111;
                double x_15113 = x_15093 * x_15098;
                double y_15114 = x_15095 * x_15099;
                double x_15115 = x_15113 + y_15114;
                double res_15116 = value_15101 * x_15115;
                
                x_15092 = res_15104;
                x_15093 = res_15108;
                x_15094 = res_15112;
                x_15095 = res_15116;
                ((__global double *) mem_16628)[gtid_14999 * 115 + gtid_15009] =
                    x_15092;
                ((__global double *) mem_16632)[gtid_14999 * 115 + gtid_15009] =
                    x_15093;
                ((__global double *) mem_16636)[gtid_14999 * 115 + gtid_15009] =
                    x_15094;
                ((__global double *) mem_16640)[gtid_14999 * 115 + gtid_15009] =
                    x_15095;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_15085
}
__kernel void tridagNestedConstzisegmap_14479(__global int *global_failure,
                                              int32_t n_11537, __global
                                              unsigned char *mem_16677, __global
                                              unsigned char *mem_16682)
{
    #define segmap_group_sizze_15460 (tridagNestedConstzisegmap_group_sizze_14484)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17145;
    int32_t local_tid_17146;
    int32_t group_sizze_17149;
    int32_t wave_sizze_17148;
    int32_t group_tid_17147;
    
    global_tid_17145 = get_global_id(0);
    local_tid_17146 = get_local_id(0);
    group_sizze_17149 = get_local_size(0);
    wave_sizze_17148 = LOCKSTEP_WIDTH;
    group_tid_17147 = get_group_id(0);
    
    int32_t phys_tid_14479;
    
    phys_tid_14479 = global_tid_17145;
    
    int32_t gtid_14477;
    
    gtid_14477 = squot32(group_tid_17147 * segmap_group_sizze_15460 +
                         local_tid_17146, 115);
    
    int32_t gtid_14478;
    
    gtid_14478 = group_tid_17147 * segmap_group_sizze_15460 + local_tid_17146 -
        squot32(group_tid_17147 * segmap_group_sizze_15460 + local_tid_17146,
                115) * 115;
    if (slt32(gtid_14477, n_11537) && slt32(gtid_14478, 115)) {
        int32_t x_15467 = sub32(115, gtid_14478);
        int32_t i_15468 = sub32(x_15467, 1);
        double res_15469 = ((__global double *) mem_16677)[gtid_14477 * 115 +
                                                           i_15468];
        
        ((__global double *) mem_16682)[gtid_14477 * 115 + gtid_14478] =
            res_15469;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_15460
}
__kernel void tridagNestedConstzisegmap_14553(__global int *global_failure,
                                              int32_t n_11537, __global
                                              unsigned char *mem_16663, __global
                                              unsigned char *mem_16668, __global
                                              unsigned char *mem_16672, __global
                                              unsigned char *mem_16677)
{
    #define segmap_group_sizze_15420 (tridagNestedConstzisegmap_group_sizze_14558)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17140;
    int32_t local_tid_17141;
    int32_t group_sizze_17144;
    int32_t wave_sizze_17143;
    int32_t group_tid_17142;
    
    global_tid_17140 = get_global_id(0);
    local_tid_17141 = get_local_id(0);
    group_sizze_17144 = get_local_size(0);
    wave_sizze_17143 = LOCKSTEP_WIDTH;
    group_tid_17142 = get_group_id(0);
    
    int32_t phys_tid_14553;
    
    phys_tid_14553 = global_tid_17140;
    
    int32_t gtid_14551;
    
    gtid_14551 = squot32(group_tid_17142 * segmap_group_sizze_15420 +
                         local_tid_17141, 115);
    
    int32_t gtid_14552;
    
    gtid_14552 = group_tid_17142 * segmap_group_sizze_15420 + local_tid_17141 -
        squot32(group_tid_17142 * segmap_group_sizze_15420 + local_tid_17141,
                115) * 115;
    if (slt32(gtid_14551, n_11537) && slt32(gtid_14552, 115)) {
        double yn_15425 = ((__global double *) mem_16663)[gtid_14551];
        double x_15426 = ((__global double *) mem_16668)[gtid_14551 * 115 +
                                                         gtid_14552];
        double x_15427 = ((__global double *) mem_16672)[gtid_14551 * 115 +
                                                         gtid_14552];
        double y_15431 = yn_15425 * x_15427;
        double res_15432 = x_15426 + y_15431;
        
        ((__global double *) mem_16677)[gtid_14551 * 115 + gtid_14552] =
            res_15432;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_15420
}
__kernel void tridagNestedConstzisegmap_14647(__global int *global_failure,
                                              int32_t n_11537,
                                              int32_t num_groups_15341, __global
                                              unsigned char *mem_16645, __global
                                              unsigned char *mem_16659, __global
                                              unsigned char *mem_16663)
{
    #define segmap_group_sizze_15340 (tridagNestedConstzisegmap_group_sizze_14650)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17062;
    int32_t local_tid_17063;
    int32_t group_sizze_17066;
    int32_t wave_sizze_17065;
    int32_t group_tid_17064;
    
    global_tid_17062 = get_global_id(0);
    local_tid_17063 = get_local_id(0);
    group_sizze_17066 = get_local_size(0);
    wave_sizze_17065 = LOCKSTEP_WIDTH;
    group_tid_17064 = get_group_id(0);
    
    int32_t phys_tid_14647;
    
    phys_tid_14647 = global_tid_17062;
    
    int32_t phys_group_id_17067;
    
    phys_group_id_17067 = get_group_id(0);
    for (int32_t i_17068 = 0; i_17068 < sdiv_up32(sdiv_up32(n_11537,
                                                            segmap_group_sizze_15340) -
                                                  phys_group_id_17067,
                                                  num_groups_15341);
         i_17068++) {
        int32_t virt_group_id_17069 = phys_group_id_17067 + i_17068 *
                num_groups_15341;
        int32_t gtid_14646 = virt_group_id_17069 * segmap_group_sizze_15340 +
                local_tid_17063;
        
        if (slt32(gtid_14646, n_11537)) {
            double x_15347 = ((__global double *) mem_16659)[gtid_14646 * 115 +
                                                             114];
            double y_15348 = ((__global double *) mem_16645)[gtid_14646 * 115 +
                                                             114];
            double yn_15349 = x_15347 / y_15348;
            
            ((__global double *) mem_16663)[gtid_14646] = yn_15349;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_15340
}
__kernel void tridagNestedConstzisegmap_14708(__global int *global_failure,
                                              int32_t n_11537,
                                              int32_t INNER_DIM_11544, __global
                                              unsigned char *y_mem_16587,
                                              __global unsigned char *mem_16650,
                                              __global unsigned char *mem_16654,
                                              __global unsigned char *mem_16659)
{
    #define segmap_group_sizze_15313 (tridagNestedConstzisegmap_group_sizze_14713)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_17057;
    int32_t local_tid_17058;
    int32_t group_sizze_17061;
    int32_t wave_sizze_17060;
    int32_t group_tid_17059;
    
    global_tid_17057 = get_global_id(0);
    local_tid_17058 = get_local_id(0);
    group_sizze_17061 = get_local_size(0);
    wave_sizze_17060 = LOCKSTEP_WIDTH;
    group_tid_17059 = get_group_id(0);
    
    int32_t phys_tid_14708;
    
    phys_tid_14708 = global_tid_17057;
    
    int32_t gtid_14706;
    
    gtid_14706 = squot32(group_tid_17059 * segmap_group_sizze_15313 +
                         local_tid_17058, 115);
    
    int32_t gtid_14707;
    
    gtid_14707 = group_tid_17059 * segmap_group_sizze_15313 + local_tid_17058 -
        squot32(group_tid_17059 * segmap_group_sizze_15313 + local_tid_17058,
                115) * 115;
    if (slt32(gtid_14706, n_11537) && slt32(gtid_14707, 115)) {
        double as_transformed_row_15318 = ((__global
                                            double *) y_mem_16587)[gtid_14706 *
                                                                   INNER_DIM_11544];
        double x_15319 = ((__global double *) mem_16650)[gtid_14706 * 115 +
                                                         gtid_14707];
        double x_15320 = ((__global double *) mem_16654)[gtid_14706 * 115 +
                                                         gtid_14707];
        double y_15324 = as_transformed_row_15318 * x_15320;
        double res_15325 = x_15319 + y_15324;
        
        ((__global double *) mem_16659)[gtid_14706 * 115 + gtid_14707] =
            res_15325;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_15313
}
__kernel void tridagNestedConstzisegmap_14883(__global int *global_failure,
                                              int32_t n_11537,
                                              int32_t INNER_DIM_11540, __global
                                              unsigned char *b_mem_16585,
                                              __global unsigned char *mem_16628,
                                              __global unsigned char *mem_16632,
                                              __global unsigned char *mem_16636,
                                              __global unsigned char *mem_16640,
                                              __global unsigned char *mem_16645)
{
    #define segmap_group_sizze_15189 (tridagNestedConstzisegmap_group_sizze_14888)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16982;
    int32_t local_tid_16983;
    int32_t group_sizze_16986;
    int32_t wave_sizze_16985;
    int32_t group_tid_16984;
    
    global_tid_16982 = get_global_id(0);
    local_tid_16983 = get_local_id(0);
    group_sizze_16986 = get_local_size(0);
    wave_sizze_16985 = LOCKSTEP_WIDTH;
    group_tid_16984 = get_group_id(0);
    
    int32_t phys_tid_14883;
    
    phys_tid_14883 = global_tid_16982;
    
    int32_t gtid_14881;
    
    gtid_14881 = squot32(group_tid_16984 * segmap_group_sizze_15189 +
                         local_tid_16983, 115);
    
    int32_t gtid_14882;
    
    gtid_14882 = group_tid_16984 * segmap_group_sizze_15189 + local_tid_16983 -
        squot32(group_tid_16984 * segmap_group_sizze_15189 + local_tid_16983,
                115) * 115;
    if (slt32(gtid_14881, n_11537) && slt32(gtid_14882, 115)) {
        double as_transformed_row_15194 = ((__global
                                            double *) b_mem_16585)[gtid_14881 *
                                                                   INNER_DIM_11540];
        double x_15195 = ((__global double *) mem_16628)[gtid_14881 * 115 +
                                                         gtid_14882];
        double x_15196 = ((__global double *) mem_16632)[gtid_14881 * 115 +
                                                         gtid_14882];
        double x_15197 = ((__global double *) mem_16636)[gtid_14881 * 115 +
                                                         gtid_14882];
        double x_15198 = ((__global double *) mem_16640)[gtid_14881 * 115 +
                                                         gtid_14882];
        double value_15200 = 1.0 / x_15195;
        double res_15203 = x_15195 * value_15200;
        double res_15207 = x_15196 * value_15200;
        double res_15211 = x_15197 * value_15200;
        double res_15215 = x_15198 * value_15200;
        double x_15216 = as_transformed_row_15194 * res_15203;
        double x_15217 = res_15207 + x_15216;
        double x_15218 = as_transformed_row_15194 * res_15211;
        double y_15219 = res_15215 + x_15218;
        double res_15220 = x_15217 / y_15219;
        
        ((__global double *) mem_16645)[gtid_14881 * 115 + gtid_14882] =
            res_15220;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_15189
}
__kernel void tridagNestedConstzisegmap_intragroup_13797(__global
                                                         int *global_failure,
                                                         __local volatile
                                                         int64_t *mem_16619_backing_aligned_0,
                                                         __local volatile
                                                         int64_t *mem_16616_backing_aligned_1,
                                                         __local volatile
                                                         int64_t *mem_16613_backing_aligned_2,
                                                         __local volatile
                                                         int64_t *mem_16611_backing_aligned_3,
                                                         __local volatile
                                                         int64_t *mem_16608_backing_aligned_4,
                                                         __local volatile
                                                         int64_t *mem_16605_backing_aligned_5,
                                                         __local volatile
                                                         int64_t *mem_16603_backing_aligned_6,
                                                         __local volatile
                                                         int64_t *mem_16600_backing_aligned_7,
                                                         __local volatile
                                                         int64_t *mem_16597_backing_aligned_8,
                                                         __local volatile
                                                         int64_t *mem_16595_backing_aligned_9,
                                                         __local volatile
                                                         int64_t *mem_16593_backing_aligned_10,
                                                         __local volatile
                                                         int64_t *mem_16591_backing_aligned_11,
                                                         int32_t INNER_DIM_11538,
                                                         int32_t INNER_DIM_11540,
                                                         int32_t INNER_DIM_11542,
                                                         int32_t INNER_DIM_11544,
                                                         __global
                                                         unsigned char *a_mem_16584,
                                                         __global
                                                         unsigned char *b_mem_16585,
                                                         __global
                                                         unsigned char *c_mem_16586,
                                                         __global
                                                         unsigned char *y_mem_16587,
                                                         __global
                                                         unsigned char *mem_16623)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_16619_backing_11 = (__local volatile
                                                            char *) mem_16619_backing_aligned_0;
    __local volatile char *restrict mem_16616_backing_10 = (__local volatile
                                                            char *) mem_16616_backing_aligned_1;
    __local volatile char *restrict mem_16613_backing_9 = (__local volatile
                                                           char *) mem_16613_backing_aligned_2;
    __local volatile char *restrict mem_16611_backing_8 = (__local volatile
                                                           char *) mem_16611_backing_aligned_3;
    __local volatile char *restrict mem_16608_backing_7 = (__local volatile
                                                           char *) mem_16608_backing_aligned_4;
    __local volatile char *restrict mem_16605_backing_6 = (__local volatile
                                                           char *) mem_16605_backing_aligned_5;
    __local volatile char *restrict mem_16603_backing_5 = (__local volatile
                                                           char *) mem_16603_backing_aligned_6;
    __local volatile char *restrict mem_16600_backing_4 = (__local volatile
                                                           char *) mem_16600_backing_aligned_7;
    __local volatile char *restrict mem_16597_backing_3 = (__local volatile
                                                           char *) mem_16597_backing_aligned_8;
    __local volatile char *restrict mem_16595_backing_2 = (__local volatile
                                                           char *) mem_16595_backing_aligned_9;
    __local volatile char *restrict mem_16593_backing_1 = (__local volatile
                                                           char *) mem_16593_backing_aligned_10;
    __local volatile char *restrict mem_16591_backing_0 = (__local volatile
                                                           char *) mem_16591_backing_aligned_11;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_16779;
    int32_t local_tid_16780;
    int32_t group_sizze_16783;
    int32_t wave_sizze_16782;
    int32_t group_tid_16781;
    
    global_tid_16779 = get_global_id(0);
    local_tid_16780 = get_local_id(0);
    group_sizze_16783 = get_local_size(0);
    wave_sizze_16782 = LOCKSTEP_WIDTH;
    group_tid_16781 = get_group_id(0);
    
    int32_t phys_tid_13797;
    
    phys_tid_13797 = group_tid_16781;
    
    int32_t ltid_pre_16784;
    
    ltid_pre_16784 = local_tid_16780;
    
    int32_t gtid_13736;
    
    gtid_13736 = group_tid_16781;
    
    double as_transformed_row_14247;
    
    as_transformed_row_14247 = ((__global double *) b_mem_16585)[gtid_13736 *
                                                                 INNER_DIM_11540];
    
    double as_transformed_row_14248 = ((__global
                                        double *) y_mem_16587)[gtid_13736 *
                                                               INNER_DIM_11544];
    __local char *mem_16591;
    
    mem_16591 = (__local char *) mem_16591_backing_0;
    
    __local char *mem_16593;
    
    mem_16593 = (__local char *) mem_16593_backing_1;
    
    __local char *mem_16595;
    
    mem_16595 = (__local char *) mem_16595_backing_2;
    
    __local char *mem_16597;
    
    mem_16597 = (__local char *) mem_16597_backing_3;
    
    int32_t gtid_13739 = ltid_pre_16784;
    int32_t phys_tid_13740 = local_tid_16780;
    
    if (slt32(gtid_13739, 115)) {
        bool cond_14289 = slt32(0, gtid_13739);
        double res_14290;
        
        if (cond_14289) {
            res_14290 = 1.0;
        } else {
            res_14290 = 0.0;
        }
        
        double res_14291;
        
        if (cond_14289) {
            res_14291 = 0.0;
        } else {
            res_14291 = 1.0;
        }
        
        double res_14292;
        
        if (cond_14289) {
            double x_elem_14287 = ((__global double *) b_mem_16585)[gtid_13736 *
                                                                    INNER_DIM_11540 +
                                                                    gtid_13739];
            
            res_14292 = x_elem_14287;
        } else {
            res_14292 = 1.0;
        }
        
        double res_14293;
        
        if (cond_14289) {
            double x_elem_14288 = ((__global double *) a_mem_16584)[gtid_13736 *
                                                                    INNER_DIM_11538 +
                                                                    gtid_13739];
            int32_t i_14294 = sub32(gtid_13739, 1);
            double y_14295 = ((__global double *) c_mem_16586)[gtid_13736 *
                                                               INNER_DIM_11542 +
                                                               i_14294];
            double y_14296 = x_elem_14288 * y_14295;
            double res_14297 = 0.0 - y_14296;
            
            res_14293 = res_14297;
        } else {
            res_14293 = 0.0;
        }
        ((__local double *) mem_16591)[gtid_13739] = res_14292;
        ((__local double *) mem_16593)[gtid_13739] = res_14293;
        ((__local double *) mem_16595)[gtid_13739] = res_14290;
        ((__local double *) mem_16597)[gtid_13739] = res_14291;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16785;
    
    dims_flat_16785 = 115;
    
    double x_14261;
    double x_14262;
    double x_14263;
    double x_14264;
    double x_14265;
    double x_14266;
    double x_14267;
    double x_14268;
    double x_16790;
    double x_16791;
    double x_16792;
    double x_16793;
    double x_16794;
    double x_16795;
    double x_16796;
    double x_16797;
    int32_t skip_threads_16815;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16780, 115)) {
            x_14265 = ((volatile __local double *) mem_16591)[local_tid_16780];
            x_14266 = ((volatile __local double *) mem_16593)[local_tid_16780];
            x_14267 = ((volatile __local double *) mem_16595)[local_tid_16780];
            x_14268 = ((volatile __local double *) mem_16597)[local_tid_16780];
            if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 0) {
                x_14261 = x_14265;
                x_14262 = x_14266;
                x_14263 = x_14267;
                x_14264 = x_14268;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16815 = 1;
        while (slt32(skip_threads_16815, 32)) {
            if (sle32(skip_threads_16815, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, 115)) {
                // read operands
                {
                    x_14261 = ((volatile __local
                                double *) mem_16591)[local_tid_16780 -
                                                     skip_threads_16815];
                    x_14262 = ((volatile __local
                                double *) mem_16593)[local_tid_16780 -
                                                     skip_threads_16815];
                    x_14263 = ((volatile __local
                                double *) mem_16595)[local_tid_16780 -
                                                     skip_threads_16815];
                    x_14264 = ((volatile __local
                                double *) mem_16597)[local_tid_16780 -
                                                     skip_threads_16815];
                }
                // perform operation
                {
                    bool inactive_16816 = slt32(srem32(local_tid_16780, 115),
                                                local_tid_16780 -
                                                (local_tid_16780 -
                                                 skip_threads_16815));
                    
                    if (inactive_16816) {
                        x_14261 = x_14265;
                        x_14262 = x_14266;
                        x_14263 = x_14267;
                        x_14264 = x_14268;
                    }
                    if (!inactive_16816) {
                        double y_14269 = x_14261 * x_14265;
                        double value_14270 = 1.0 / y_14269;
                        double y_14271 = x_14263 * x_14266;
                        double x_14272 = y_14269 + y_14271;
                        double res_14273 = value_14270 * x_14272;
                        double x_14274 = x_14262 * x_14265;
                        double y_14275 = x_14264 * x_14266;
                        double x_14276 = x_14274 + y_14275;
                        double res_14277 = value_14270 * x_14276;
                        double x_14278 = x_14261 * x_14267;
                        double y_14279 = x_14263 * x_14268;
                        double x_14280 = x_14278 + y_14279;
                        double res_14281 = value_14270 * x_14280;
                        double x_14282 = x_14262 * x_14267;
                        double y_14283 = x_14264 * x_14268;
                        double x_14284 = x_14282 + y_14283;
                        double res_14285 = value_14270 * x_14284;
                        
                        x_14261 = res_14273;
                        x_14262 = res_14277;
                        x_14263 = res_14281;
                        x_14264 = res_14285;
                    }
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16815)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16815, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, 115)) {
                // write result
                {
                    ((volatile __local double *) mem_16591)[local_tid_16780] =
                        x_14261;
                    x_14265 = x_14261;
                    ((volatile __local double *) mem_16593)[local_tid_16780] =
                        x_14262;
                    x_14266 = x_14262;
                    ((volatile __local double *) mem_16595)[local_tid_16780] =
                        x_14263;
                    x_14267 = x_14263;
                    ((volatile __local double *) mem_16597)[local_tid_16780] =
                        x_14264;
                    x_14268 = x_14264;
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16815)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16815 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 31 &&
            slt32(local_tid_16780, 115)) {
            ((volatile __local double *) mem_16591)[squot32(local_tid_16780,
                                                            32)] = x_14261;
            ((volatile __local double *) mem_16593)[squot32(local_tid_16780,
                                                            32)] = x_14262;
            ((volatile __local double *) mem_16595)[squot32(local_tid_16780,
                                                            32)] = x_14263;
            ((volatile __local double *) mem_16597)[squot32(local_tid_16780,
                                                            32)] = x_14264;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16817;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                           115)) {
                x_16794 = ((volatile __local
                            double *) mem_16591)[local_tid_16780];
                x_16795 = ((volatile __local
                            double *) mem_16593)[local_tid_16780];
                x_16796 = ((volatile __local
                            double *) mem_16595)[local_tid_16780];
                x_16797 = ((volatile __local
                            double *) mem_16597)[local_tid_16780];
                if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) ==
                    0) {
                    x_16790 = x_16794;
                    x_16791 = x_16795;
                    x_16792 = x_16796;
                    x_16793 = x_16797;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16817 = 1;
            while (slt32(skip_threads_16817, 32)) {
                if (sle32(skip_threads_16817, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                115))) {
                    // read operands
                    {
                        x_16790 = ((volatile __local
                                    double *) mem_16591)[local_tid_16780 -
                                                         skip_threads_16817];
                        x_16791 = ((volatile __local
                                    double *) mem_16593)[local_tid_16780 -
                                                         skip_threads_16817];
                        x_16792 = ((volatile __local
                                    double *) mem_16595)[local_tid_16780 -
                                                         skip_threads_16817];
                        x_16793 = ((volatile __local
                                    double *) mem_16597)[local_tid_16780 -
                                                         skip_threads_16817];
                    }
                    // perform operation
                    {
                        bool inactive_16818 = slt32(srem32(local_tid_16780 *
                                                           32 + 32 - 1, 115),
                                                    local_tid_16780 * 32 + 32 -
                                                    1 - ((local_tid_16780 -
                                                          skip_threads_16817) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16818) {
                            x_16790 = x_16794;
                            x_16791 = x_16795;
                            x_16792 = x_16796;
                            x_16793 = x_16797;
                        }
                        if (!inactive_16818) {
                            double y_16798 = x_16790 * x_16794;
                            double value_16799 = 1.0 / y_16798;
                            double y_16800 = x_16792 * x_16795;
                            double x_16801 = y_16798 + y_16800;
                            double res_16802 = value_16799 * x_16801;
                            double x_16803 = x_16791 * x_16794;
                            double y_16804 = x_16793 * x_16795;
                            double x_16805 = x_16803 + y_16804;
                            double res_16806 = value_16799 * x_16805;
                            double x_16807 = x_16790 * x_16796;
                            double y_16808 = x_16792 * x_16797;
                            double x_16809 = x_16807 + y_16808;
                            double res_16810 = value_16799 * x_16809;
                            double x_16811 = x_16791 * x_16796;
                            double y_16812 = x_16793 * x_16797;
                            double x_16813 = x_16811 + y_16812;
                            double res_16814 = value_16799 * x_16813;
                            
                            x_16790 = res_16802;
                            x_16791 = res_16806;
                            x_16792 = res_16810;
                            x_16793 = res_16814;
                        }
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16817)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16817, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                115))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16591)[local_tid_16780] = x_16790;
                        x_16794 = x_16790;
                        ((volatile __local
                          double *) mem_16593)[local_tid_16780] = x_16791;
                        x_16795 = x_16791;
                        ((volatile __local
                          double *) mem_16595)[local_tid_16780] = x_16792;
                        x_16796 = x_16792;
                        ((volatile __local
                          double *) mem_16597)[local_tid_16780] = x_16793;
                        x_16797 = x_16793;
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16817)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16817 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16780, 32) == 0 || !slt32(local_tid_16780,
                                                          115))) {
            // read operands
            {
                x_14265 = x_14261;
                x_14266 = x_14262;
                x_14267 = x_14263;
                x_14268 = x_14264;
                x_14261 = ((__local
                            double *) mem_16591)[squot32(local_tid_16780, 32) -
                                                 1];
                x_14262 = ((__local
                            double *) mem_16593)[squot32(local_tid_16780, 32) -
                                                 1];
                x_14263 = ((__local
                            double *) mem_16595)[squot32(local_tid_16780, 32) -
                                                 1];
                x_14264 = ((__local
                            double *) mem_16597)[squot32(local_tid_16780, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16819 = slt32(srem32(local_tid_16780, 115),
                                            local_tid_16780 -
                                            (squot32(local_tid_16780, 32) * 32 -
                                             1));
                
                if (inactive_16819) {
                    x_14261 = x_14265;
                    x_14262 = x_14266;
                    x_14263 = x_14267;
                    x_14264 = x_14268;
                }
                if (!inactive_16819) {
                    double y_14269 = x_14261 * x_14265;
                    double value_14270 = 1.0 / y_14269;
                    double y_14271 = x_14263 * x_14266;
                    double x_14272 = y_14269 + y_14271;
                    double res_14273 = value_14270 * x_14272;
                    double x_14274 = x_14262 * x_14265;
                    double y_14275 = x_14264 * x_14266;
                    double x_14276 = x_14274 + y_14275;
                    double res_14277 = value_14270 * x_14276;
                    double x_14278 = x_14261 * x_14267;
                    double y_14279 = x_14263 * x_14268;
                    double x_14280 = x_14278 + y_14279;
                    double res_14281 = value_14270 * x_14280;
                    double x_14282 = x_14262 * x_14267;
                    double y_14283 = x_14264 * x_14268;
                    double x_14284 = x_14282 + y_14283;
                    double res_14285 = value_14270 * x_14284;
                    
                    x_14261 = res_14273;
                    x_14262 = res_14277;
                    x_14263 = res_14281;
                    x_14264 = res_14285;
                }
            }
            // write final result
            {
                ((__local double *) mem_16591)[local_tid_16780] = x_14261;
                ((__local double *) mem_16593)[local_tid_16780] = x_14262;
                ((__local double *) mem_16595)[local_tid_16780] = x_14263;
                ((__local double *) mem_16597)[local_tid_16780] = x_14264;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16780, 32) == 0) {
            ((__local double *) mem_16591)[local_tid_16780] = x_14265;
            ((__local double *) mem_16593)[local_tid_16780] = x_14266;
            ((__local double *) mem_16595)[local_tid_16780] = x_14267;
            ((__local double *) mem_16597)[local_tid_16780] = x_14268;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16600;
    
    mem_16600 = (__local char *) mem_16600_backing_4;
    
    int32_t gtid_13741 = ltid_pre_16784;
    int32_t phys_tid_13742 = local_tid_16780;
    
    if (slt32(gtid_13741, 115)) {
        double x_14326 = ((__local double *) mem_16591)[gtid_13741];
        double x_14327 = ((__local double *) mem_16593)[gtid_13741];
        double x_14328 = ((__local double *) mem_16595)[gtid_13741];
        double x_14329 = ((__local double *) mem_16597)[gtid_13741];
        double value_14331 = 1.0 / x_14326;
        double res_14334 = x_14326 * value_14331;
        double res_14338 = x_14327 * value_14331;
        double res_14342 = x_14328 * value_14331;
        double res_14346 = x_14329 * value_14331;
        double x_14347 = as_transformed_row_14247 * res_14334;
        double x_14348 = res_14338 + x_14347;
        double x_14349 = as_transformed_row_14247 * res_14342;
        double y_14350 = res_14346 + x_14349;
        double res_14351 = x_14348 / y_14350;
        
        ((__local double *) mem_16600)[gtid_13741] = res_14351;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16603;
    
    mem_16603 = (__local char *) mem_16603_backing_5;
    
    __local char *mem_16605;
    
    mem_16605 = (__local char *) mem_16605_backing_6;
    
    int32_t gtid_13769 = ltid_pre_16784;
    int32_t phys_tid_13770 = local_tid_16780;
    
    if (slt32(gtid_13769, 115)) {
        bool cond_14375 = slt32(0, gtid_13769);
        double res_14376;
        
        if (cond_14375) {
            double x_elem_14373 = ((__global double *) y_mem_16587)[gtid_13736 *
                                                                    INNER_DIM_11544 +
                                                                    gtid_13769];
            
            res_14376 = x_elem_14373;
        } else {
            res_14376 = 0.0;
        }
        
        double res_14377;
        
        if (cond_14375) {
            double x_elem_14374 = ((__global double *) a_mem_16584)[gtid_13736 *
                                                                    INNER_DIM_11538 +
                                                                    gtid_13769];
            int32_t i_14378 = sub32(gtid_13769, 1);
            double y_14379 = ((__local double *) mem_16600)[i_14378];
            double y_14380 = x_elem_14374 / y_14379;
            double res_14381 = 0.0 - y_14380;
            
            res_14377 = res_14381;
        } else {
            res_14377 = 1.0;
        }
        ((__local double *) mem_16603)[gtid_13769] = res_14376;
        ((__local double *) mem_16605)[gtid_13769] = res_14377;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16820;
    
    dims_flat_16820 = 115;
    
    double x_14365;
    double x_14366;
    double x_14367;
    double x_14368;
    double x_16823;
    double x_16824;
    double x_16825;
    double x_16826;
    int32_t skip_threads_16830;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16780, 115)) {
            x_14367 = ((volatile __local double *) mem_16603)[local_tid_16780];
            x_14368 = ((volatile __local double *) mem_16605)[local_tid_16780];
            if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 0) {
                x_14365 = x_14367;
                x_14366 = x_14368;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16830 = 1;
        while (slt32(skip_threads_16830, 32)) {
            if (sle32(skip_threads_16830, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, 115)) {
                // read operands
                {
                    x_14365 = ((volatile __local
                                double *) mem_16603)[local_tid_16780 -
                                                     skip_threads_16830];
                    x_14366 = ((volatile __local
                                double *) mem_16605)[local_tid_16780 -
                                                     skip_threads_16830];
                }
                // perform operation
                {
                    bool inactive_16831 = slt32(srem32(local_tid_16780, 115),
                                                local_tid_16780 -
                                                (local_tid_16780 -
                                                 skip_threads_16830));
                    
                    if (inactive_16831) {
                        x_14365 = x_14367;
                        x_14366 = x_14368;
                    }
                    if (!inactive_16831) {
                        double y_14369 = x_14365 * x_14368;
                        double res_14370 = x_14367 + y_14369;
                        double res_14371 = x_14366 * x_14368;
                        
                        x_14365 = res_14370;
                        x_14366 = res_14371;
                    }
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16830)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16830, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, 115)) {
                // write result
                {
                    ((volatile __local double *) mem_16603)[local_tid_16780] =
                        x_14365;
                    x_14367 = x_14365;
                    ((volatile __local double *) mem_16605)[local_tid_16780] =
                        x_14366;
                    x_14368 = x_14366;
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16830)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16830 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 31 &&
            slt32(local_tid_16780, 115)) {
            ((volatile __local double *) mem_16603)[squot32(local_tid_16780,
                                                            32)] = x_14365;
            ((volatile __local double *) mem_16605)[squot32(local_tid_16780,
                                                            32)] = x_14366;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16832;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                           115)) {
                x_16825 = ((volatile __local
                            double *) mem_16603)[local_tid_16780];
                x_16826 = ((volatile __local
                            double *) mem_16605)[local_tid_16780];
                if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) ==
                    0) {
                    x_16823 = x_16825;
                    x_16824 = x_16826;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16832 = 1;
            while (slt32(skip_threads_16832, 32)) {
                if (sle32(skip_threads_16832, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                115))) {
                    // read operands
                    {
                        x_16823 = ((volatile __local
                                    double *) mem_16603)[local_tid_16780 -
                                                         skip_threads_16832];
                        x_16824 = ((volatile __local
                                    double *) mem_16605)[local_tid_16780 -
                                                         skip_threads_16832];
                    }
                    // perform operation
                    {
                        bool inactive_16833 = slt32(srem32(local_tid_16780 *
                                                           32 + 32 - 1, 115),
                                                    local_tid_16780 * 32 + 32 -
                                                    1 - ((local_tid_16780 -
                                                          skip_threads_16832) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16833) {
                            x_16823 = x_16825;
                            x_16824 = x_16826;
                        }
                        if (!inactive_16833) {
                            double y_16827 = x_16823 * x_16826;
                            double res_16828 = x_16825 + y_16827;
                            double res_16829 = x_16824 * x_16826;
                            
                            x_16823 = res_16828;
                            x_16824 = res_16829;
                        }
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16832)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16832, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                115))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16603)[local_tid_16780] = x_16823;
                        x_16825 = x_16823;
                        ((volatile __local
                          double *) mem_16605)[local_tid_16780] = x_16824;
                        x_16826 = x_16824;
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16832)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16832 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16780, 32) == 0 || !slt32(local_tid_16780,
                                                          115))) {
            // read operands
            {
                x_14367 = x_14365;
                x_14368 = x_14366;
                x_14365 = ((__local
                            double *) mem_16603)[squot32(local_tid_16780, 32) -
                                                 1];
                x_14366 = ((__local
                            double *) mem_16605)[squot32(local_tid_16780, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16834 = slt32(srem32(local_tid_16780, 115),
                                            local_tid_16780 -
                                            (squot32(local_tid_16780, 32) * 32 -
                                             1));
                
                if (inactive_16834) {
                    x_14365 = x_14367;
                    x_14366 = x_14368;
                }
                if (!inactive_16834) {
                    double y_14369 = x_14365 * x_14368;
                    double res_14370 = x_14367 + y_14369;
                    double res_14371 = x_14366 * x_14368;
                    
                    x_14365 = res_14370;
                    x_14366 = res_14371;
                }
            }
            // write final result
            {
                ((__local double *) mem_16603)[local_tid_16780] = x_14365;
                ((__local double *) mem_16605)[local_tid_16780] = x_14366;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16780, 32) == 0) {
            ((__local double *) mem_16603)[local_tid_16780] = x_14367;
            ((__local double *) mem_16605)[local_tid_16780] = x_14368;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16608;
    
    mem_16608 = (__local char *) mem_16608_backing_7;
    
    int32_t gtid_13771 = ltid_pre_16784;
    int32_t phys_tid_13772 = local_tid_16780;
    
    if (slt32(gtid_13771, 115)) {
        double x_14392 = ((__local double *) mem_16603)[gtid_13771];
        double x_14393 = ((__local double *) mem_16605)[gtid_13771];
        double y_14397 = as_transformed_row_14248 * x_14393;
        double res_14398 = x_14392 + y_14397;
        
        ((__local double *) mem_16608)[gtid_13771] = res_14398;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    double x_14402 = ((__local double *) mem_16608)[114];
    double y_14403 = ((__local double *) mem_16600)[114];
    double yn_14404 = x_14402 / y_14403;
    __local char *mem_16611;
    
    mem_16611 = (__local char *) mem_16611_backing_8;
    
    __local char *mem_16613;
    
    mem_16613 = (__local char *) mem_16613_backing_9;
    
    int32_t gtid_13780 = ltid_pre_16784;
    int32_t phys_tid_13781 = local_tid_16780;
    
    if (slt32(gtid_13780, 115)) {
        int32_t x_14419 = sub32(115, gtid_13780);
        int32_t i_14420 = sub32(x_14419, 1);
        bool cond_14421 = slt32(0, gtid_13780);
        double res_14422;
        double res_14423;
        
        if (cond_14421) {
            double x_14424 = ((__local double *) mem_16608)[i_14420];
            double y_14425 = ((__local double *) mem_16600)[i_14420];
            double res_14426 = x_14424 / y_14425;
            double x_14427 = ((__global double *) c_mem_16586)[gtid_13736 *
                                                               INNER_DIM_11542 +
                                                               i_14420];
            double y_14428 = x_14427 / y_14425;
            double res_14429 = 0.0 - y_14428;
            
            res_14422 = res_14426;
            res_14423 = res_14429;
        } else {
            res_14422 = 0.0;
            res_14423 = 1.0;
        }
        ((__local double *) mem_16611)[gtid_13780] = res_14422;
        ((__local double *) mem_16613)[gtid_13780] = res_14423;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_16835;
    
    dims_flat_16835 = 115;
    
    double x_14411;
    double x_14412;
    double x_14413;
    double x_14414;
    double x_16838;
    double x_16839;
    double x_16840;
    double x_16841;
    int32_t skip_threads_16845;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_16780, 115)) {
            x_14413 = ((volatile __local double *) mem_16611)[local_tid_16780];
            x_14414 = ((volatile __local double *) mem_16613)[local_tid_16780];
            if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 0) {
                x_14411 = x_14413;
                x_14412 = x_14414;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_16845 = 1;
        while (slt32(skip_threads_16845, 32)) {
            if (sle32(skip_threads_16845, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, 115)) {
                // read operands
                {
                    x_14411 = ((volatile __local
                                double *) mem_16611)[local_tid_16780 -
                                                     skip_threads_16845];
                    x_14412 = ((volatile __local
                                double *) mem_16613)[local_tid_16780 -
                                                     skip_threads_16845];
                }
                // perform operation
                {
                    bool inactive_16846 = slt32(srem32(local_tid_16780, 115),
                                                local_tid_16780 -
                                                (local_tid_16780 -
                                                 skip_threads_16845));
                    
                    if (inactive_16846) {
                        x_14411 = x_14413;
                        x_14412 = x_14414;
                    }
                    if (!inactive_16846) {
                        double y_14415 = x_14411 * x_14414;
                        double res_14416 = x_14413 + y_14415;
                        double res_14417 = x_14412 * x_14414;
                        
                        x_14411 = res_14416;
                        x_14412 = res_14417;
                    }
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16845)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_16845, local_tid_16780 -
                      squot32(local_tid_16780, 32) * 32) &&
                slt32(local_tid_16780, 115)) {
                // write result
                {
                    ((volatile __local double *) mem_16611)[local_tid_16780] =
                        x_14411;
                    x_14413 = x_14411;
                    ((volatile __local double *) mem_16613)[local_tid_16780] =
                        x_14412;
                    x_14414 = x_14412;
                }
            }
            if (sle32(wave_sizze_16782, skip_threads_16845)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_16845 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) == 31 &&
            slt32(local_tid_16780, 115)) {
            ((volatile __local double *) mem_16611)[squot32(local_tid_16780,
                                                            32)] = x_14411;
            ((volatile __local double *) mem_16613)[squot32(local_tid_16780,
                                                            32)] = x_14412;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_16847;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                           115)) {
                x_16840 = ((volatile __local
                            double *) mem_16611)[local_tid_16780];
                x_16841 = ((volatile __local
                            double *) mem_16613)[local_tid_16780];
                if ((local_tid_16780 - squot32(local_tid_16780, 32) * 32) ==
                    0) {
                    x_16838 = x_16840;
                    x_16839 = x_16841;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_16847 = 1;
            while (slt32(skip_threads_16847, 32)) {
                if (sle32(skip_threads_16847, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                115))) {
                    // read operands
                    {
                        x_16838 = ((volatile __local
                                    double *) mem_16611)[local_tid_16780 -
                                                         skip_threads_16847];
                        x_16839 = ((volatile __local
                                    double *) mem_16613)[local_tid_16780 -
                                                         skip_threads_16847];
                    }
                    // perform operation
                    {
                        bool inactive_16848 = slt32(srem32(local_tid_16780 *
                                                           32 + 32 - 1, 115),
                                                    local_tid_16780 * 32 + 32 -
                                                    1 - ((local_tid_16780 -
                                                          skip_threads_16847) *
                                                         32 + 32 - 1));
                        
                        if (inactive_16848) {
                            x_16838 = x_16840;
                            x_16839 = x_16841;
                        }
                        if (!inactive_16848) {
                            double y_16842 = x_16838 * x_16841;
                            double res_16843 = x_16840 + y_16842;
                            double res_16844 = x_16839 * x_16841;
                            
                            x_16838 = res_16843;
                            x_16839 = res_16844;
                        }
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16847)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_16847, local_tid_16780 -
                          squot32(local_tid_16780, 32) * 32) &&
                    (squot32(local_tid_16780, 32) == 0 && slt32(local_tid_16780,
                                                                115))) {
                    // write result
                    {
                        ((volatile __local
                          double *) mem_16611)[local_tid_16780] = x_16838;
                        x_16840 = x_16838;
                        ((volatile __local
                          double *) mem_16613)[local_tid_16780] = x_16839;
                        x_16841 = x_16839;
                    }
                }
                if (sle32(wave_sizze_16782, skip_threads_16847)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_16847 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_16780, 32) == 0 || !slt32(local_tid_16780,
                                                          115))) {
            // read operands
            {
                x_14413 = x_14411;
                x_14414 = x_14412;
                x_14411 = ((__local
                            double *) mem_16611)[squot32(local_tid_16780, 32) -
                                                 1];
                x_14412 = ((__local
                            double *) mem_16613)[squot32(local_tid_16780, 32) -
                                                 1];
            }
            // perform operation
            {
                bool inactive_16849 = slt32(srem32(local_tid_16780, 115),
                                            local_tid_16780 -
                                            (squot32(local_tid_16780, 32) * 32 -
                                             1));
                
                if (inactive_16849) {
                    x_14411 = x_14413;
                    x_14412 = x_14414;
                }
                if (!inactive_16849) {
                    double y_14415 = x_14411 * x_14414;
                    double res_14416 = x_14413 + y_14415;
                    double res_14417 = x_14412 * x_14414;
                    
                    x_14411 = res_14416;
                    x_14412 = res_14417;
                }
            }
            // write final result
            {
                ((__local double *) mem_16611)[local_tid_16780] = x_14411;
                ((__local double *) mem_16613)[local_tid_16780] = x_14412;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_16780, 32) == 0) {
            ((__local double *) mem_16611)[local_tid_16780] = x_14413;
            ((__local double *) mem_16613)[local_tid_16780] = x_14414;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16616;
    
    mem_16616 = (__local char *) mem_16616_backing_10;
    
    int32_t gtid_13782 = ltid_pre_16784;
    int32_t phys_tid_13783 = local_tid_16780;
    
    if (slt32(gtid_13782, 115)) {
        double x_14440 = ((__local double *) mem_16611)[gtid_13782];
        double x_14441 = ((__local double *) mem_16613)[gtid_13782];
        double y_14445 = yn_14404 * x_14441;
        double res_14446 = x_14440 + y_14445;
        
        ((__local double *) mem_16616)[gtid_13782] = res_14446;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_16619;
    
    mem_16619 = (__local char *) mem_16619_backing_11;
    
    int32_t gtid_13791 = ltid_pre_16784;
    int32_t phys_tid_13792 = local_tid_16780;
    
    if (slt32(gtid_13791, 115)) {
        int32_t x_14452 = sub32(115, gtid_13791);
        int32_t i_14453 = sub32(x_14452, 1);
        double res_14454 = ((__local double *) mem_16616)[i_14453];
        
        ((__local double *) mem_16619)[gtid_13791] = res_14454;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global double *) mem_16623)[gtid_13736 * 115 + local_tid_16780] =
        ((__local double *) mem_16619)[local_tid_16780];
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_7:
    return;
}
__kernel void tridagNestedSeqzisegmap_16011(__global int *global_failure,
                                            int failure_is_an_option, __global
                                            int *global_failure_args,
                                            int32_t n_11881, int32_t m_11882,
                                            int32_t n_11883, int32_t m_11884,
                                            int32_t n_11885, int32_t m_11886,
                                            int32_t n_11887, int32_t m_11888,
                                            int32_t distance_11910,
                                            int32_t m_11918,
                                            int32_t num_groups_16092, __global
                                            unsigned char *b_mem_16585, __global
                                            unsigned char *c_mem_16586, __global
                                            unsigned char *y_mem_16587, __global
                                            unsigned char *mem_16592, __global
                                            unsigned char *mem_16597, __global
                                            unsigned char *mem_16602, __global
                                            unsigned char *mem_16607, __global
                                            unsigned char *mem_16611, __global
                                            unsigned char *mem_16614, __global
                                            unsigned char *mem_16661, __global
                                            unsigned char *mem_16678)
{
    #define segmap_group_sizze_16091 (tridagNestedSeqzisegmap_group_sizze_16014)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_16776;
    int32_t local_tid_16777;
    int32_t group_sizze_16780;
    int32_t wave_sizze_16779;
    int32_t group_tid_16778;
    
    global_tid_16776 = get_global_id(0);
    local_tid_16777 = get_local_id(0);
    group_sizze_16780 = get_local_size(0);
    wave_sizze_16779 = LOCKSTEP_WIDTH;
    group_tid_16778 = get_group_id(0);
    
    int32_t phys_tid_16011;
    
    phys_tid_16011 = global_tid_16776;
    
    int32_t phys_group_id_16781;
    
    phys_group_id_16781 = get_group_id(0);
    for (int32_t i_16782 = 0; i_16782 < sdiv_up32(sdiv_up32(n_11881,
                                                            segmap_group_sizze_16091) -
                                                  phys_group_id_16781,
                                                  num_groups_16092);
         i_16782++) {
        int32_t virt_group_id_16783 = phys_group_id_16781 + i_16782 *
                num_groups_16092;
        int32_t gtid_16010 = virt_group_id_16783 * segmap_group_sizze_16091 +
                local_tid_16777;
        
        if (slt32(gtid_16010, n_11881)) {
            for (int32_t i_16581 = 0; i_16581 < m_11882; i_16581++) {
                bool cond_16102 = i_16581 == 0;
                double res_16103;
                
                if (cond_16102) {
                    bool y_16104 = slt32(0, m_11882);
                    bool index_certs_16105;
                    
                    if (!y_16104) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          0) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = m_11882;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    double x_16106 = ((__global
                                       double *) c_mem_16586)[gtid_16010 *
                                                              m_11886];
                    double y_16107 = ((__global
                                       double *) b_mem_16585)[gtid_16010 *
                                                              m_11884];
                    double res_16108 = x_16106 / y_16107;
                    
                    res_16103 = res_16108;
                } else {
                    res_16103 = 0.0;
                }
                
                double res_16109;
                
                if (cond_16102) {
                    bool y_16110 = slt32(0, m_11882);
                    bool index_certs_16111;
                    
                    if (!y_16110) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          1) == -1) {
                                global_failure_args[0] = 0;
                                global_failure_args[1] = m_11882;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    double x_16112 = ((__global
                                       double *) y_mem_16587)[gtid_16010 *
                                                              m_11888];
                    double y_16113 = ((__global
                                       double *) b_mem_16585)[gtid_16010 *
                                                              m_11884];
                    double res_16114 = x_16112 / y_16113;
                    
                    res_16109 = res_16114;
                } else {
                    res_16109 = 0.0;
                }
                ((__global double *) mem_16611)[phys_tid_16011 + i_16581 *
                                                (num_groups_16092 *
                                                 segmap_group_sizze_16091)] =
                    res_16109;
                ((__global double *) mem_16614)[phys_tid_16011 + i_16581 *
                                                (num_groups_16092 *
                                                 segmap_group_sizze_16091)] =
                    res_16103;
            }
            for (int32_t i_16117 = 0; i_16117 < distance_11910; i_16117++) {
                int32_t index_primexp_16120 = 1 + i_16117;
                bool x_16121 = sle32(0, index_primexp_16120);
                bool y_16122 = slt32(index_primexp_16120, m_11882);
                bool bounds_check_16123 = x_16121 && y_16122;
                bool index_certs_16124;
                
                if (!bounds_check_16123) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 2) ==
                            -1) {
                            global_failure_args[0] = index_primexp_16120;
                            global_failure_args[1] = m_11882;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double x_16125 = ((__global
                                   double *) mem_16592)[index_primexp_16120 *
                                                        n_11883 + gtid_16010];
                double x_16126 = ((__global
                                   double *) mem_16597)[index_primexp_16120 *
                                                        n_11881 + gtid_16010];
                bool y_16127 = slt32(i_16117, m_11882);
                bool index_certs_16128;
                
                if (!y_16127) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 3) ==
                            -1) {
                            global_failure_args[0] = i_16117;
                            global_failure_args[1] = m_11882;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double y_16129 = ((__global
                                   double *) mem_16614)[phys_tid_16011 +
                                                        i_16117 *
                                                        (num_groups_16092 *
                                                         segmap_group_sizze_16091)];
                double y_16130 = x_16126 * y_16129;
                double y_16131 = x_16125 - y_16130;
                double norm_factor_16132 = 1.0 / y_16131;
                double x_16133 = ((__global
                                   double *) mem_16602)[index_primexp_16120 *
                                                        n_11885 + gtid_16010];
                double lw_val_16134 = norm_factor_16132 * x_16133;
                
                ((__global double *) mem_16614)[phys_tid_16011 +
                                                index_primexp_16120 *
                                                (num_groups_16092 *
                                                 segmap_group_sizze_16091)] =
                    lw_val_16134;
                
                double x_16136 = ((__global
                                   double *) mem_16607)[index_primexp_16120 *
                                                        n_11887 + gtid_16010];
                double y_16137 = ((__global
                                   double *) mem_16611)[phys_tid_16011 +
                                                        i_16117 *
                                                        (num_groups_16092 *
                                                         segmap_group_sizze_16091)];
                double y_16138 = x_16126 * y_16137;
                double x_16139 = x_16136 - y_16138;
                double lw_val_16140 = norm_factor_16132 * x_16139;
                
                ((__global double *) mem_16611)[phys_tid_16011 +
                                                index_primexp_16120 *
                                                (num_groups_16092 *
                                                 segmap_group_sizze_16091)] =
                    lw_val_16140;
            }
            for (int32_t i_16788 = 0; i_16788 < m_11882; i_16788++) {
                ((__global double *) mem_16661)[phys_tid_16011 + i_16788 *
                                                (num_groups_16092 *
                                                 segmap_group_sizze_16091)] =
                    0.0;
            }
            for (int32_t i_16789 = 0; i_16789 < 1; i_16789++) {
                ((__global double *) mem_16661)[phys_tid_16011 +
                                                (distance_11910 + i_16789) *
                                                (num_groups_16092 *
                                                 segmap_group_sizze_16091)] =
                    ((__global double *) mem_16611)[phys_tid_16011 +
                                                    num_groups_16092 *
                                                    segmap_group_sizze_16091 *
                                                    distance_11910 + i_16789 *
                                                    (num_groups_16092 *
                                                     segmap_group_sizze_16091)];
            }
            for (int32_t i_16146 = 0; i_16146 < distance_11910; i_16146++) {
                int32_t binop_y_16148 = -1 * i_16146;
                int32_t index_primexp_16149 = m_11918 + binop_y_16148;
                bool x_16150 = sle32(0, index_primexp_16149);
                bool y_16151 = slt32(index_primexp_16149, m_11882);
                bool bounds_check_16152 = x_16150 && y_16151;
                bool index_certs_16153;
                
                if (!bounds_check_16152) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 4) ==
                            -1) {
                            global_failure_args[0] = index_primexp_16149;
                            global_failure_args[1] = m_11882;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double x_16154 = ((__global
                                   double *) mem_16611)[phys_tid_16011 +
                                                        index_primexp_16149 *
                                                        (num_groups_16092 *
                                                         segmap_group_sizze_16091)];
                double x_16155 = ((__global
                                   double *) mem_16614)[phys_tid_16011 +
                                                        index_primexp_16149 *
                                                        (num_groups_16092 *
                                                         segmap_group_sizze_16091)];
                int32_t i_16156 = add32(1, index_primexp_16149);
                bool x_16157 = sle32(0, i_16156);
                bool y_16158 = slt32(i_16156, m_11882);
                bool bounds_check_16159 = x_16157 && y_16158;
                bool index_certs_16160;
                
                if (!bounds_check_16159) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 5) ==
                            -1) {
                            global_failure_args[0] = i_16156;
                            global_failure_args[1] = m_11882;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double y_16161 = ((__global
                                   double *) mem_16661)[phys_tid_16011 +
                                                        i_16156 *
                                                        (num_groups_16092 *
                                                         segmap_group_sizze_16091)];
                double y_16162 = x_16155 * y_16161;
                double lw_val_16163 = x_16154 - y_16162;
                
                ((__global double *) mem_16661)[phys_tid_16011 +
                                                index_primexp_16149 *
                                                (num_groups_16092 *
                                                 segmap_group_sizze_16091)] =
                    lw_val_16163;
            }
            for (int32_t i_16791 = 0; i_16791 < m_11882; i_16791++) {
                ((__global double *) mem_16678)[i_16791 * n_11881 +
                                                gtid_16010] = ((__global
                                                                double *) mem_16661)[phys_tid_16011 +
                                                                                     i_16791 *
                                                                                     (num_groups_16092 *
                                                                                      segmap_group_sizze_16091)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_16091
}
__kernel void tridagNestedSeqConstzisegmap_15475(__global int *global_failure,
                                                 int failure_is_an_option,
                                                 __global
                                                 int *global_failure_args,
                                                 int32_t n_11783,
                                                 int32_t n_11785,
                                                 int32_t INNER_DIM_11786,
                                                 int32_t n_11787,
                                                 int32_t INNER_DIM_11788,
                                                 int32_t n_11789,
                                                 int32_t INNER_DIM_11790,
                                                 int32_t num_groups_15552,
                                                 __global
                                                 unsigned char *b_mem_16585,
                                                 __global
                                                 unsigned char *c_mem_16586,
                                                 __global
                                                 unsigned char *y_mem_16587,
                                                 __global
                                                 unsigned char *mem_16592,
                                                 __global
                                                 unsigned char *mem_16597,
                                                 __global
                                                 unsigned char *mem_16602,
                                                 __global
                                                 unsigned char *mem_16607,
                                                 __global
                                                 unsigned char *mem_16680)
{
    #define segmap_group_sizze_15551 (tridagNestedSeqConstzisegmap_group_sizze_15478)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_16776;
    int32_t local_tid_16777;
    int32_t group_sizze_16780;
    int32_t wave_sizze_16779;
    int32_t group_tid_16778;
    
    global_tid_16776 = get_global_id(0);
    local_tid_16777 = get_local_id(0);
    group_sizze_16780 = get_local_size(0);
    wave_sizze_16779 = LOCKSTEP_WIDTH;
    group_tid_16778 = get_group_id(0);
    
    int32_t phys_tid_15475;
    
    phys_tid_15475 = global_tid_16776;
    
    int32_t phys_group_id_16781;
    
    phys_group_id_16781 = get_group_id(0);
    for (int32_t i_16782 = 0; i_16782 < sdiv_up32(sdiv_up32(n_11783,
                                                            segmap_group_sizze_15551) -
                                                  phys_group_id_16781,
                                                  num_groups_15552);
         i_16782++) {
        int32_t virt_group_id_16783 = phys_group_id_16781 + i_16782 *
                num_groups_15552;
        int32_t gtid_15474 = virt_group_id_16783 * segmap_group_sizze_15551 +
                local_tid_16777;
        
        if (slt32(gtid_15474, n_11783)) {
            double mem_16610[115];
            double mem_16612[115];
            
            for (int32_t i_16581 = 0; i_16581 < 115; i_16581++) {
                bool cond_15562 = i_16581 == 0;
                double res_15563;
                
                if (cond_15562) {
                    double x_15564 = ((__global
                                       double *) c_mem_16586)[gtid_15474 *
                                                              INNER_DIM_11788];
                    double y_15565 = ((__global
                                       double *) b_mem_16585)[gtid_15474 *
                                                              INNER_DIM_11786];
                    double res_15566 = x_15564 / y_15565;
                    
                    res_15563 = res_15566;
                } else {
                    res_15563 = 0.0;
                }
                
                double res_15567;
                
                if (cond_15562) {
                    double x_15568 = ((__global
                                       double *) y_mem_16587)[gtid_15474 *
                                                              INNER_DIM_11790];
                    double y_15569 = ((__global
                                       double *) b_mem_16585)[gtid_15474 *
                                                              INNER_DIM_11786];
                    double res_15570 = x_15568 / y_15569;
                    
                    res_15567 = res_15570;
                } else {
                    res_15567 = 0.0;
                }
                mem_16610[i_16581] = res_15567;
                mem_16612[i_16581] = res_15563;
            }
            for (int32_t i_15573 = 0; i_15573 < 114; i_15573++) {
                int32_t index_primexp_15576 = 1 + i_15573;
                bool x_15577 = sle32(0, index_primexp_15576);
                bool y_15578 = slt32(index_primexp_15576, 115);
                bool bounds_check_15579 = x_15577 && y_15578;
                bool index_certs_15580;
                
                if (!bounds_check_15579) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 6) ==
                            -1) {
                            global_failure_args[0] = index_primexp_15576;
                            global_failure_args[1] = 115;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double x_15581 = ((__global
                                   double *) mem_16592)[index_primexp_15576 *
                                                        n_11785 + gtid_15474];
                double x_15582 = ((__global
                                   double *) mem_16597)[index_primexp_15576 *
                                                        n_11783 + gtid_15474];
                bool y_15583 = slt32(i_15573, 115);
                bool index_certs_15584;
                
                if (!y_15583) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 7) ==
                            -1) {
                            global_failure_args[0] = i_15573;
                            global_failure_args[1] = 115;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double y_15585 = mem_16612[i_15573];
                double y_15586 = x_15582 * y_15585;
                double y_15587 = x_15581 - y_15586;
                double norm_factor_15588 = 1.0 / y_15587;
                double x_15589 = ((__global
                                   double *) mem_16602)[index_primexp_15576 *
                                                        n_11787 + gtid_15474];
                double lw_val_15590 = norm_factor_15588 * x_15589;
                
                mem_16612[index_primexp_15576] = lw_val_15590;
                
                double x_15592 = ((__global
                                   double *) mem_16607)[index_primexp_15576 *
                                                        n_11789 + gtid_15474];
                double y_15593 = mem_16610[i_15573];
                double y_15594 = x_15582 * y_15593;
                double x_15595 = x_15592 - y_15594;
                double lw_val_15596 = norm_factor_15588 * x_15595;
                
                mem_16610[index_primexp_15576] = lw_val_15596;
            }
            
            double mem_16662[115];
            
            for (int32_t i_16788 = 0; i_16788 < 115; i_16788++) {
                mem_16662[i_16788] = 0.0;
            }
            for (int32_t i_16789 = 0; i_16789 < 1; i_16789++) {
                mem_16662[114 + i_16789] = mem_16610[114 + i_16789];
            }
            for (int32_t i_15602 = 0; i_15602 < 114; i_15602++) {
                int32_t binop_y_15604 = -1 * i_15602;
                int32_t index_primexp_15605 = 113 + binop_y_15604;
                bool x_15606 = sle32(0, index_primexp_15605);
                bool y_15607 = slt32(index_primexp_15605, 115);
                bool bounds_check_15608 = x_15606 && y_15607;
                bool index_certs_15609;
                
                if (!bounds_check_15608) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 8) ==
                            -1) {
                            global_failure_args[0] = index_primexp_15605;
                            global_failure_args[1] = 115;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double x_15610 = mem_16610[index_primexp_15605];
                double x_15611 = mem_16612[index_primexp_15605];
                int32_t i_15612 = add32(1, index_primexp_15605);
                bool x_15613 = sle32(0, i_15612);
                bool y_15614 = slt32(i_15612, 115);
                bool bounds_check_15615 = x_15613 && y_15614;
                bool index_certs_15616;
                
                if (!bounds_check_15615) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 9) ==
                            -1) {
                            global_failure_args[0] = i_15612;
                            global_failure_args[1] = 115;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                double y_15617 = mem_16662[i_15612];
                double y_15618 = x_15611 * y_15617;
                double lw_val_15619 = x_15610 - y_15618;
                
                mem_16662[index_primexp_15605] = lw_val_15619;
            }
            for (int32_t i_16791 = 0; i_16791 < 115; i_16791++) {
                ((__global double *) mem_16680)[i_16791 * n_11783 +
                                                gtid_15474] =
                    mem_16662[i_16791];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_15551
}
"""
# Start of values.py.

# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        for c in read[::-1]:
            f.unget_char(c)
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in b'01234556789ABCDEFabcdef':
            s += c
            c = f.get_char()
        elif c == b'_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16)).encode('utf8') # ugh

def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in b'xX':
        c = f.get_char() # skip X
        return parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == b'_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
        if len(s) == 0:
            raise ValueError
        return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      return c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      return parse_int(f)

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    dims = []
    for i in range(rank):
        parse_specific_string(f, '[')
        dims += [int(parse_int(f))]
        parse_specific_string(f, ']')
    if np.product(dims) != 0:
        raise ValueError
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return tuple(dims)

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if type(elems) == tuple:
        # Empty array
        return np.empty(elems, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype=FUTHARK_PRIMTYPES[bin_type_enum]['numpy_type'])
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def end_of_input(entry, f=input_reader):
    skip_spaces(f)
    if f.get_char() != b'':
        panic(1, "Expected EOF on stdin after reading input for \"%s\".", entry)

def write_value_text(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[{}]'.format(d)
                                                    for d in v.shape]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

type_strs = { np.dtype('int8'): b'  i8',
              np.dtype('int16'): b' i16',
              np.dtype('int32'): b' i32',
              np.dtype('int64'): b' i64',
              np.dtype('uint8'): b'  u8',
              np.dtype('uint16'): b' u16',
              np.dtype('uint32'): b' u32',
              np.dtype('uint64'): b' u64',
              np.dtype('float32'): b' f32',
              np.dtype('float64'): b' f64',
              np.dtype('bool'): b'bool'}

def construct_binary_value(v):
    t = v.dtype
    shape = v.shape

    elems = 1
    for d in shape:
        elems *= d

    num_bytes = 1 + 1 + 1 + 4 + len(shape) * 8 + elems * t.itemsize
    bytes = bytearray(num_bytes)
    bytes[0] = np.int8(ord('b'))
    bytes[1] = 2
    bytes[2] = np.int8(len(shape))
    bytes[3:7] = type_strs[t]

    for i in range(len(shape)):
        bytes[7+i*8:7+(i+1)*8] = np.int64(shape[i]).tostring()

    bytes[7+len(shape)*8:] = np.ascontiguousarray(v).tostring()

    return bytes

def write_value_binary(v, out=sys.stdout):
    if sys.version_info >= (3,0):
        out = out.buffer
    out.write(construct_binary_value(v))

def write_value(v, out=sys.stdout, binary=False):
    if binary:
        return write_value_binary(v, out=out)
    else:
        return write_value_text(v, out=out)

# End of values.py.
# Start of memory.py.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, shape):
  # HACK: np.ctypeslib.as_array may fail if the shape contains zeroes,
  # for some reason.
  if any(map(lambda x: x == 0, shape)):
      return np.ndarray(shape, dtype=x._type_)
  else:
      return np.ctypeslib.as_array(x, shape=shape)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset*ct.sizeof(bt), bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset)*ct.sizeof(v), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)

# End of memory.py.
# Start of panic.py.

def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.stderr.write('\n')
    sys.exit(exitcode)

# End of panic.py.
# Start of tuning.py

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

# End of tuning.py.
# Start of scalar.py.

import numpy as np
import math
import struct

def intlit(t, x):
  if t == np.int8:
    return np.int8(x)
  elif t == np.int16:
    return np.int16(x)
  elif t == np.int32:
    return np.int32(x)
  else:
    return np.int64(x)

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

# Python is so slow that we just make all the unsafe operations safe,
# always.

def sdivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x // y

def sdiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return (x+y-intlit(type(x), 1)) // y

def smodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x % y

def udivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) // unsigned(y))

def udiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed((unsigned(x)+unsigned(y)-unsigned(intlit(type(x),1))) // unsigned(y))

def umodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
sdiv_up8 = sdiv1_up6 = sdiv_up32 = sdiv_up64 = sdiv_upN
sdiv_safe8 = sdiv1_safe6 = sdiv_safe32 = sdiv_safe64 = sdivN
sdiv_up_safe8 = sdiv_up1_safe6 = sdiv_up_safe32 = sdiv_up_safe64 = sdiv_upN
smod8 = smod16 = smod32 = smod64 = smodN
smod_safe8 = smod_safe16 = smod_safe32 = smod_safe64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
udiv_up8 = udiv_up16 = udiv_up32 = udiv_up64 = udivN
udiv_safe8 = udiv_safe16 = udiv_safe32 = udiv_safe64 = udiv_upN
udiv_up_safe8 = udiv_up_safe16 = udiv_up_safe32 = udiv_up_safe64 = udiv_upN
umod8 = umod16 = umod32 = umod64 = umodN
umod_safe8 = umod_safe16 = umod_safe32 = umod_safe64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
squot_safe8 = squot_safe16 = squot_safe32 = squot_safe64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
srem_safe8 = srem_safe16 = srem_safe32 = srem_safe64 = sremN

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def clz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if x < 0:
      break
    n += 1
    x <<= np.int8(1)
  return n

def ctz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if (x & 1) == 1:
      break
    n += 1
    x >>= np.int8(1)
  return n

def popc_T(x):
  c = np.int32(0)
  while x != 0:
    x &= x - np.int8(1)
    c += np.int8(1)
  return c

futhark_popc8 = futhark_popc16 = futhark_popc32 = futhark_popc64 = popc_T
futhark_clzz8 = futhark_clzz16 = futhark_clzz32 = futhark_clzz64 = clz_T
futhark_ctzz8 = futhark_ctzz16 = futhark_ctzz32 = futhark_ctzz64 = ctz_T

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_mul_hi8(a, b):
  a = np.uint64(np.uint8(a))
  b = np.uint64(np.uint8(b))
  return np.int8((a*b) >> np.uint64(8))

def futhark_mul_hi16(a, b):
  a = np.uint64(np.uint16(a))
  b = np.uint64(np.uint16(b))
  return np.int16((a*b) >> np.uint64(16))

def futhark_mul_hi32(a, b):
  a = np.uint64(np.uint32(a))
  b = np.uint64(np.uint32(b))
  return np.int32((a*b) >> np.uint64(32))

# This one is done with arbitrary-precision integers.
def futhark_mul_hi64(a, b):
  a = int(np.uint64(a))
  b = int(np.uint64(b))
  return np.int64(np.uint64(a*b >> 64))

def futhark_mad_hi8(a, b, c):
  return futhark_mul_hi8(a,b) + c

def futhark_mad_hi16(a, b, c):
  return futhark_mul_hi16(a,b) + c

def futhark_mad_hi32(a, b, c):
  return futhark_mul_hi32(a,b) + c

def futhark_mad_hi64(a, b, c):
  return futhark_mul_hi64(a,b) + c

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_cosh64(x):
  return np.cosh(x)

def futhark_sinh64(x):
  return np.sinh(x)

def futhark_tanh64(x):
  return np.tanh(x)

def futhark_acosh64(x):
  return np.arccosh(x)

def futhark_asinh64(x):
  return np.arcsinh(x)

def futhark_atanh64(x):
  return np.arctanh(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_gamma64(x):
  return np.float64(math.gamma(x))

def futhark_lgamma64(x):
  return np.float64(math.lgamma(x))

def futhark_round64(x):
  return np.round(x)

def futhark_ceil64(x):
  return np.ceil(x)

def futhark_floor64(x):
  return np.floor(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_cosh32(x):
  return np.cosh(x)

def futhark_sinh32(x):
  return np.sinh(x)

def futhark_tanh32(x):
  return np.tanh(x)

def futhark_acosh32(x):
  return np.arccosh(x)

def futhark_asinh32(x):
  return np.arcsinh(x)

def futhark_atanh32(x):
  return np.arctanh(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_gamma32(x):
  return np.float32(math.gamma(x))

def futhark_lgamma32(x):
  return np.float32(math.lgamma(x))

def futhark_round32(x):
  return np.round(x)

def futhark_ceil32(x):
  return np.ceil(x)

def futhark_floor32(x):
  return np.floor(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])

def futhark_lerp32(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_lerp64(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_mad32(a, b, c):
  return a * b + c

def futhark_mad64(a, b, c):
  return a * b + c

def futhark_fma32(a, b, c):
  return a * b + c

def futhark_fma64(a, b, c):
  return a * b + c

# End of scalar.py.
class tke:
  entry_points = {"tridagNested": (["[][]f64", "[][]f64", "[][]f64", "[][]f64"],
                                   ["[][]f64"]),
                  "tridagNestedConst": (["[][]f64", "[][]f64", "[][]f64",
                                         "[][]f64"], ["[][]f64"]),
                  "tridagNestedSeq": (["[][]f64", "[][]f64", "[][]f64",
                                       "[][]f64"], ["[][]f64"]),
                  "tridagNestedSeqConst": (["[][]f64", "[][]f64", "[][]f64",
                                            "[][]f64"], ["[][]f64"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width",
      lambda device: np.int32(32)), ("AMD Accelerated Parallel Processing",
                                     cl.device_type.GPU, "lockstep_width",
                                     lambda device: np.int32(32)), ("",
                                                                    cl.device_type.GPU,
                                                                    "lockstep_width",
                                                                    lambda device: np.int32(1)),
     ("", cl.device_type.GPU, "num_groups",
      lambda device: (np.int32(4) * device.get_info(getattr(cl.device_info,
                                                            "MAX_COMPUTE_UNITS")))),
     ("", cl.device_type.GPU, "group_size", lambda device: np.int32(256)), ("",
                                                                            cl.device_type.GPU,
                                                                            "tile_size",
                                                                            lambda device: np.int32(32)),
     ("", cl.device_type.GPU, "threshold", lambda device: np.int32(32768)), ("",
                                                                             cl.device_type.CPU,
                                                                             "lockstep_width",
                                                                             lambda device: np.int32(1)),
     ("", cl.device_type.CPU, "num_groups",
      lambda device: device.get_info(getattr(cl.device_info, "MAX_COMPUTE_UNITS"))),
     ("", cl.device_type.CPU, "group_size", lambda device: np.int32(32)), ("",
                                                                           cl.device_type.CPU,
                                                                           "tile_size",
                                                                           lambda device: np.int32(4)),
     ("", cl.device_type.CPU, "threshold",
      lambda device: device.get_info(getattr(cl.device_info, "MAX_COMPUTE_UNITS")))]
    self.global_failure_args_max = 2
    self.failure_msgs=["Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:85:37-40\n   #1  tke.fut:85:13-65\n   #2  tke.fut:114:22-40\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:114:4-49\n   #6  tke.fut:112:1-114:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:86:37-40\n   #1  tke.fut:86:13-65\n   #2  tke.fut:114:22-40\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:114:4-49\n   #6  tke.fut:112:1-114:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:89:35-38\n   #1  tke.fut:114:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:114:4-49\n   #5  tke.fut:112:1-114:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:89:49-55\n   #1  tke.fut:114:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:114:4-49\n   #5  tke.fut:112:1-114:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:97:25-34\n   #1  tke.fut:114:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:114:4-49\n   #5  tke.fut:112:1-114:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:97:51-63\n   #1  tke.fut:114:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:114:4-49\n   #5  tke.fut:112:1-114:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:89:35-38\n   #1  tke.fut:106:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:106:4-49\n   #5  tke.fut:104:1-106:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:89:49-55\n   #1  tke.fut:106:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:106:4-49\n   #5  tke.fut:104:1-106:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:97:25-34\n   #1  tke.fut:106:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:106:4-49\n   #5  tke.fut:104:1-106:49\n",
     "Index [{}] out of bounds for array of shape [{}].\n-> #0  tke.fut:97:51-63\n   #1  tke.fut:106:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:106:4-49\n   #5  tke.fut:104:1-106:49\n"]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i32", "f64", "bool", "cert"],
                                       user_sizes=sizes,
                                       all_sizes={"tridagNested.segmap_group_size_12746": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_12820": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_12912": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_12975": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_group_size_13150": {"class": "group_size", "value": None},
                                        "tridagNested.segmap_num_groups_12914": {"class": "num_groups", "value": None},
                                        "tridagNested.segscan_group_size_12878": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagNested.segscan_group_size_13033": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagNested.segscan_group_size_13266": {"class": "group_size",
                                                                                  "value": None},
                                        "tridagNested.segscan_num_groups_12880": {"class": "num_groups",
                                                                                  "value": None},
                                        "tridagNested.segscan_num_groups_13035": {"class": "num_groups",
                                                                                  "value": None},
                                        "tridagNested.segscan_num_groups_13268": {"class": "num_groups",
                                                                                  "value": None},
                                        "tridagNested.suff_intra_par_1": {"class": "threshold ()", "value": 32},
                                        "tridagNestedConst.segmap_group_size_14484": {"class": "group_size",
                                                                                      "value": None},
                                        "tridagNestedConst.segmap_group_size_14558": {"class": "group_size",
                                                                                      "value": None},
                                        "tridagNestedConst.segmap_group_size_14650": {"class": "group_size",
                                                                                      "value": None},
                                        "tridagNestedConst.segmap_group_size_14713": {"class": "group_size",
                                                                                      "value": None},
                                        "tridagNestedConst.segmap_group_size_14888": {"class": "group_size",
                                                                                      "value": None},
                                        "tridagNestedConst.segmap_num_groups_14652": {"class": "num_groups",
                                                                                      "value": None},
                                        "tridagNestedConst.segscan_group_size_14616": {"class": "group_size",
                                                                                       "value": None},
                                        "tridagNestedConst.segscan_group_size_14771": {"class": "group_size",
                                                                                       "value": None},
                                        "tridagNestedConst.segscan_group_size_15004": {"class": "group_size",
                                                                                       "value": None},
                                        "tridagNestedConst.segscan_num_groups_14618": {"class": "num_groups",
                                                                                       "value": None},
                                        "tridagNestedConst.segscan_num_groups_14773": {"class": "num_groups",
                                                                                       "value": None},
                                        "tridagNestedConst.segscan_num_groups_15006": {"class": "num_groups",
                                                                                       "value": None},
                                        "tridagNestedConst.suff_intra_par_1": {"class": "threshold ()", "value": 32},
                                        "tridagNestedSeq.segmap_group_size_16014": {"class": "group_size",
                                                                                    "value": None},
                                        "tridagNestedSeq.segmap_num_groups_16016": {"class": "num_groups",
                                                                                    "value": None},
                                        "tridagNestedSeqConst.segmap_group_size_15478": {"class": "group_size",
                                                                                         "value": None},
                                        "tridagNestedSeqConst.segmap_num_groups_15480": {"class": "num_groups",
                                                                                         "value": None}})
    self.map_transpose_f64_var = program.map_transpose_f64
    self.map_transpose_f64_low_height_var = program.map_transpose_f64_low_height
    self.map_transpose_f64_low_width_var = program.map_transpose_f64_low_width
    self.map_transpose_f64_small_var = program.map_transpose_f64_small
    self.tridagNestedziscan_stage1_12884_var = program.tridagNestedziscan_stage1_12884
    self.tridagNestedziscan_stage1_13039_var = program.tridagNestedziscan_stage1_13039
    self.tridagNestedziscan_stage1_13272_var = program.tridagNestedziscan_stage1_13272
    self.tridagNestedziscan_stage2_12884_var = program.tridagNestedziscan_stage2_12884
    self.tridagNestedziscan_stage2_13039_var = program.tridagNestedziscan_stage2_13039
    self.tridagNestedziscan_stage2_13272_var = program.tridagNestedziscan_stage2_13272
    self.tridagNestedziscan_stage3_12884_var = program.tridagNestedziscan_stage3_12884
    self.tridagNestedziscan_stage3_13039_var = program.tridagNestedziscan_stage3_13039
    self.tridagNestedziscan_stage3_13272_var = program.tridagNestedziscan_stage3_13272
    self.tridagNestedzisegmap_12741_var = program.tridagNestedzisegmap_12741
    self.tridagNestedzisegmap_12815_var = program.tridagNestedzisegmap_12815
    self.tridagNestedzisegmap_12909_var = program.tridagNestedzisegmap_12909
    self.tridagNestedzisegmap_12970_var = program.tridagNestedzisegmap_12970
    self.tridagNestedzisegmap_13145_var = program.tridagNestedzisegmap_13145
    self.tridagNestedzisegmap_intragroup_12059_var = program.tridagNestedzisegmap_intragroup_12059
    self.tridagNestedConstziscan_stage1_14622_var = program.tridagNestedConstziscan_stage1_14622
    self.tridagNestedConstziscan_stage1_14777_var = program.tridagNestedConstziscan_stage1_14777
    self.tridagNestedConstziscan_stage1_15010_var = program.tridagNestedConstziscan_stage1_15010
    self.tridagNestedConstziscan_stage2_14622_var = program.tridagNestedConstziscan_stage2_14622
    self.tridagNestedConstziscan_stage2_14777_var = program.tridagNestedConstziscan_stage2_14777
    self.tridagNestedConstziscan_stage2_15010_var = program.tridagNestedConstziscan_stage2_15010
    self.tridagNestedConstziscan_stage3_14622_var = program.tridagNestedConstziscan_stage3_14622
    self.tridagNestedConstziscan_stage3_14777_var = program.tridagNestedConstziscan_stage3_14777
    self.tridagNestedConstziscan_stage3_15010_var = program.tridagNestedConstziscan_stage3_15010
    self.tridagNestedConstzisegmap_14479_var = program.tridagNestedConstzisegmap_14479
    self.tridagNestedConstzisegmap_14553_var = program.tridagNestedConstzisegmap_14553
    self.tridagNestedConstzisegmap_14647_var = program.tridagNestedConstzisegmap_14647
    self.tridagNestedConstzisegmap_14708_var = program.tridagNestedConstzisegmap_14708
    self.tridagNestedConstzisegmap_14883_var = program.tridagNestedConstzisegmap_14883
    self.tridagNestedConstzisegmap_intragroup_13797_var = program.tridagNestedConstzisegmap_intragroup_13797
    self.tridagNestedSeqzisegmap_16011_var = program.tridagNestedSeqzisegmap_16011
    self.tridagNestedSeqConstzisegmap_15475_var = program.tridagNestedSeqConstzisegmap_15475
    self.constants = {}
  def futhark_builtinzhmap_transpose_f64(self, destmem_0, destoffset_1,
                                         srcmem_2, srcoffset_3, num_arrays_4,
                                         x_elems_5, y_elems_6):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_8 = squot32(np.int32(16), x_elems_5)
      mulx_7 = squot32(np.int32(16), y_elems_6)
      if ((num_arrays_4 == np.int32(1)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1)))):
        if (((x_elems_5 * y_elems_6) * np.int32(8)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long(((x_elems_5 * y_elems_6) * np.int32(8))))
        if synchronous:
          sync(self)
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                                                                muly_8),
                                                                                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.map_transpose_f64_low_width_var.set_args(cl.LocalMemory(np.long(np.int32(2176))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(mulx_7),
                                                          np.int32(muly_8),
                                                          destmem_0, srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.map_transpose_f64_low_width_var,
                                       ((np.long(sdiv_up32(x_elems_5,
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                     muly_8),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              sync(self)
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(sdiv_up32(sdiv_up32(x_elems_5, mulx_7),
                                          np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.map_transpose_f64_low_height_var.set_args(cl.LocalMemory(np.long(np.int32(2176))),
                                                             np.int32(destoffset_1),
                                                             np.int32(srcoffset_3),
                                                             np.int32(num_arrays_4),
                                                             np.int32(x_elems_5),
                                                             np.int32(y_elems_6),
                                                             np.int32(mulx_7),
                                                             np.int32(muly_8),
                                                             destmem_0,
                                                             srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_transpose_f64_low_height_var,
                                         ((np.long(sdiv_up32(sdiv_up32(x_elems_5,
                                                                       mulx_7),
                                                             np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(sdiv_up32(y_elems_6,
                                                             np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                sync(self)
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                          np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.map_transpose_f64_small_var.set_args(cl.LocalMemory(np.long(np.int32(1))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(mulx_7),
                                                          np.int32(muly_8),
                                                          destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f64_small_var,
                                           ((np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                               np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  sync(self)
            else:
              if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                            np.int32(32))) * np.long(np.int32(32)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                          np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.map_transpose_f64_var.set_args(cl.LocalMemory(np.long(np.int32(8448))),
                                                    np.int32(destoffset_1),
                                                    np.int32(srcoffset_3),
                                                    np.int32(num_arrays_4),
                                                    np.int32(x_elems_5),
                                                    np.int32(y_elems_6),
                                                    np.int32(mulx_7),
                                                    np.int32(muly_8), destmem_0,
                                                    srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f64_var,
                                           ((np.long(sdiv_up32(x_elems_5,
                                                               np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(sdiv_up32(y_elems_6,
                                                               np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  sync(self)
    return ()
  def futhark_tridagNested(self, a_mem_16584, b_mem_16585, c_mem_16586,
                           y_mem_16587, n_11293, m_11294, n_11295, m_11296,
                           n_11297, m_11298, n_11299, m_11300):
    dim_match_11305 = (n_11293 == n_11295)
    dim_match_11306 = (m_11294 == m_11296)
    match_11307 = (dim_match_11305 and dim_match_11306)
    empty_or_match_cert_11308 = True
    assert match_11307, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:72:1-73:49\n" % ("function arguments of wrong shape",))
    dim_match_11310 = (n_11293 == n_11297)
    dim_match_11311 = (m_11294 == m_11298)
    match_11312 = (dim_match_11310 and dim_match_11311)
    empty_or_match_cert_11313 = True
    assert match_11312, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:72:1-73:49\n" % ("function arguments of wrong shape",))
    dim_match_11315 = (n_11293 == n_11299)
    dim_match_11316 = (m_11294 == m_11300)
    match_11317 = (dim_match_11315 and dim_match_11316)
    empty_or_match_cert_11318 = True
    assert match_11317, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:72:1-73:49\n" % ("function arguments of wrong shape",))
    i_11321 = (m_11294 - np.int32(1))
    max_group_sizze_12501 = self.max_group_size
    fits_12502 = sle32(m_11294, max_group_sizze_12501)
    suff_intra_par_12500 = (self.sizes["tridagNested.suff_intra_par_1"] <= m_11294)
    intra_suff_and_fits_12503 = (suff_intra_par_12500 and fits_12502)
    m_13343 = sext_i32_i64(m_11294)
    n_13344 = sext_i32_i64(n_11293)
    nest_sizze_13346 = (m_13343 * n_13344)
    segscan_group_sizze_13347 = self.sizes["tridagNested.segscan_group_size_13266"]
    max_num_groups_16775 = self.sizes["tridagNested.segscan_num_groups_13268"]
    num_groups_13348 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_13346,
                                                            sext_i32_i64(segscan_group_sizze_13347)),
                                                  sext_i32_i64(max_num_groups_16775))))
    segmap_group_sizze_13451 = self.sizes["tridagNested.segmap_group_size_13150"]
    segmap_group_sizze_13452 = sext_i32_i64(segmap_group_sizze_13451)
    segscan_group_sizze_13515 = self.sizes["tridagNested.segscan_group_size_13033"]
    max_num_groups_16776 = self.sizes["tridagNested.segscan_num_groups_13035"]
    num_groups_13516 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_13346,
                                                            sext_i32_i64(segscan_group_sizze_13515)),
                                                  sext_i32_i64(max_num_groups_16776))))
    segmap_group_sizze_13575 = self.sizes["tridagNested.segmap_group_size_12975"]
    segmap_group_sizze_13576 = sext_i32_i64(segmap_group_sizze_13575)
    segmap_group_sizze_13602 = self.sizes["tridagNested.segmap_group_size_12912"]
    max_num_groups_16777 = self.sizes["tridagNested.segmap_num_groups_12914"]
    num_groups_13603 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(n_13344,
                                                            sext_i32_i64(segmap_group_sizze_13602)),
                                                  sext_i32_i64(max_num_groups_16777))))
    segscan_group_sizze_13620 = self.sizes["tridagNested.segscan_group_size_12878"]
    max_num_groups_16778 = self.sizes["tridagNested.segscan_num_groups_12880"]
    num_groups_13621 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_13346,
                                                            sext_i32_i64(segscan_group_sizze_13620)),
                                                  sext_i32_i64(max_num_groups_16778))))
    segmap_group_sizze_13682 = self.sizes["tridagNested.segmap_group_size_12820"]
    segmap_group_sizze_13683 = sext_i32_i64(segmap_group_sizze_13682)
    segmap_group_sizze_13722 = self.sizes["tridagNested.segmap_group_size_12746"]
    segmap_group_sizze_13723 = sext_i32_i64(segmap_group_sizze_13722)
    binop_x_16635 = (m_13343 * n_13344)
    bytes_16632 = (np.int64(8) * binop_x_16635)
    bytes_16590 = (np.int64(8) * m_13343)
    bytes_16682 = (np.int64(8) * n_13344)
    local_memory_capacity_17150 = self.max_local_memory
    if (sle64((((((((((((bytes_16590 + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590) + bytes_16590),
              sext_i32_i64(local_memory_capacity_17150)) and intra_suff_and_fits_12503):
      mem_16636 = opencl_alloc(self, bytes_16632, "mem_16636")
      if ((1 * (np.long(n_11293) * np.long(m_11294))) != 0):
        self.tridagNestedzisegmap_intragroup_12059_var.set_args(self.global_failure,
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                cl.LocalMemory(np.long(bytes_16590)),
                                                                np.int32(m_11294),
                                                                np.int32(m_11296),
                                                                np.int32(m_11298),
                                                                np.int32(m_11300),
                                                                np.int32(i_11321),
                                                                a_mem_16584,
                                                                b_mem_16585,
                                                                c_mem_16586,
                                                                y_mem_16587,
                                                                mem_16636)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_intragroup_12059_var,
                                   ((np.long(n_11293) * np.long(m_11294)),),
                                   (np.long(m_11294),))
        if synchronous:
          sync(self)
      res_mem_16708 = mem_16636
    else:
      mem_16642 = opencl_alloc(self, bytes_16632, "mem_16642")
      mem_16647 = opencl_alloc(self, bytes_16632, "mem_16647")
      mem_16652 = opencl_alloc(self, bytes_16632, "mem_16652")
      mem_16657 = opencl_alloc(self, bytes_16632, "mem_16657")
      if slt32(np.int32(0), (n_11293 * m_11294)):
        stage1_max_num_groups_16850 = self.max_group_size
        stage1_num_groups_16851 = smin32(stage1_max_num_groups_16850,
                                         num_groups_13348)
        num_threads_16852 = (stage1_num_groups_16851 * segscan_group_sizze_13347)
        if ((1 * (np.long(stage1_num_groups_16851) * np.long(segscan_group_sizze_13347))) != 0):
          self.tridagNestedziscan_stage1_13272_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13347))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13347))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13347))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13347))))),
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            np.int32(m_11296),
                                                            np.int32(m_11298),
                                                            a_mem_16584,
                                                            b_mem_16585,
                                                            c_mem_16586,
                                                            mem_16642,
                                                            mem_16647,
                                                            mem_16652,
                                                            mem_16657,
                                                            np.int32(num_threads_16852))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage1_13272_var,
                                     ((np.long(stage1_num_groups_16851) * np.long(segscan_group_sizze_13347)),),
                                     (np.long(segscan_group_sizze_13347),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16851))) != 0):
          self.tridagNestedziscan_stage2_13272_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            mem_16642,
                                                            mem_16647,
                                                            mem_16652,
                                                            mem_16657,
                                                            np.int32(stage1_num_groups_16851),
                                                            np.int32(num_threads_16852))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage2_13272_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_16851)),),
                                     (np.long(stage1_num_groups_16851),))
          if synchronous:
            sync(self)
        required_groups_16970 = sdiv_up32((n_11293 * m_11294),
                                          segscan_group_sizze_13347)
        if ((1 * (np.long(num_groups_13348) * np.long(segscan_group_sizze_13347))) != 0):
          self.tridagNestedziscan_stage3_13272_var.set_args(self.global_failure,
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            np.int32(num_groups_13348),
                                                            mem_16642,
                                                            mem_16647,
                                                            mem_16652,
                                                            mem_16657,
                                                            np.int32(num_threads_16852),
                                                            np.int32(required_groups_16970))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage3_13272_var,
                                     ((np.long(num_groups_13348) * np.long(segscan_group_sizze_13347)),),
                                     (np.long(segscan_group_sizze_13347),))
          if synchronous:
            sync(self)
      segmap_usable_groups_64_13453 = sdiv_up64(nest_sizze_13346,
                                                segmap_group_sizze_13452)
      segmap_usable_groups_13454 = sext_i64_i32(segmap_usable_groups_64_13453)
      mem_16663 = opencl_alloc(self, bytes_16632, "mem_16663")
      if ((1 * (np.long(segmap_usable_groups_13454) * np.long(segmap_group_sizze_13451))) != 0):
        self.tridagNestedzisegmap_13145_var.set_args(self.global_failure,
                                                     np.int32(n_11293),
                                                     np.int32(m_11294),
                                                     np.int32(m_11296),
                                                     b_mem_16585, mem_16642,
                                                     mem_16647, mem_16652,
                                                     mem_16657, mem_16663)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_13145_var,
                                   ((np.long(segmap_usable_groups_13454) * np.long(segmap_group_sizze_13451)),),
                                   (np.long(segmap_group_sizze_13451),))
        if synchronous:
          sync(self)
      mem_16642 = None
      mem_16647 = None
      mem_16652 = None
      mem_16657 = None
      mem_16669 = opencl_alloc(self, bytes_16632, "mem_16669")
      mem_16674 = opencl_alloc(self, bytes_16632, "mem_16674")
      if slt32(np.int32(0), (n_11293 * m_11294)):
        stage1_max_num_groups_16987 = self.max_group_size
        stage1_num_groups_16988 = smin32(stage1_max_num_groups_16987,
                                         num_groups_13516)
        num_threads_16989 = (stage1_num_groups_16988 * segscan_group_sizze_13515)
        if ((1 * (np.long(stage1_num_groups_16988) * np.long(segscan_group_sizze_13515))) != 0):
          self.tridagNestedziscan_stage1_13039_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13515))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13515))))),
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            np.int32(m_11300),
                                                            a_mem_16584,
                                                            y_mem_16587,
                                                            mem_16663,
                                                            mem_16669,
                                                            mem_16674,
                                                            np.int32(num_threads_16989))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage1_13039_var,
                                     ((np.long(stage1_num_groups_16988) * np.long(segscan_group_sizze_13515)),),
                                     (np.long(segscan_group_sizze_13515),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16988))) != 0):
          self.tridagNestedziscan_stage2_13039_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16988))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16988))))),
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            mem_16669,
                                                            mem_16674,
                                                            np.int32(stage1_num_groups_16988),
                                                            np.int32(num_threads_16989))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage2_13039_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_16988)),),
                                     (np.long(stage1_num_groups_16988),))
          if synchronous:
            sync(self)
        required_groups_17045 = sdiv_up32((n_11293 * m_11294),
                                          segscan_group_sizze_13515)
        if ((1 * (np.long(num_groups_13516) * np.long(segscan_group_sizze_13515))) != 0):
          self.tridagNestedziscan_stage3_13039_var.set_args(self.global_failure,
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            np.int32(num_groups_13516),
                                                            mem_16669,
                                                            mem_16674,
                                                            np.int32(num_threads_16989),
                                                            np.int32(required_groups_17045))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage3_13039_var,
                                     ((np.long(num_groups_13516) * np.long(segscan_group_sizze_13515)),),
                                     (np.long(segscan_group_sizze_13515),))
          if synchronous:
            sync(self)
      segmap_usable_groups_64_13577 = sdiv_up64(nest_sizze_13346,
                                                segmap_group_sizze_13576)
      segmap_usable_groups_13578 = sext_i64_i32(segmap_usable_groups_64_13577)
      mem_16680 = opencl_alloc(self, bytes_16632, "mem_16680")
      if ((1 * (np.long(segmap_usable_groups_13578) * np.long(segmap_group_sizze_13575))) != 0):
        self.tridagNestedzisegmap_12970_var.set_args(self.global_failure,
                                                     np.int32(n_11293),
                                                     np.int32(m_11294),
                                                     np.int32(m_11300),
                                                     y_mem_16587, mem_16669,
                                                     mem_16674, mem_16680)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_12970_var,
                                   ((np.long(segmap_usable_groups_13578) * np.long(segmap_group_sizze_13575)),),
                                   (np.long(segmap_group_sizze_13575),))
        if synchronous:
          sync(self)
      mem_16669 = None
      mem_16674 = None
      mem_16684 = opencl_alloc(self, bytes_16682, "mem_16684")
      if ((1 * (np.long(num_groups_13603) * np.long(segmap_group_sizze_13602))) != 0):
        self.tridagNestedzisegmap_12909_var.set_args(self.global_failure,
                                                     np.int32(n_11293),
                                                     np.int32(m_11294),
                                                     np.int32(i_11321),
                                                     np.int32(num_groups_13603),
                                                     mem_16663, mem_16680,
                                                     mem_16684)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_12909_var,
                                   ((np.long(num_groups_13603) * np.long(segmap_group_sizze_13602)),),
                                   (np.long(segmap_group_sizze_13602),))
        if synchronous:
          sync(self)
      mem_16690 = opencl_alloc(self, bytes_16632, "mem_16690")
      mem_16695 = opencl_alloc(self, bytes_16632, "mem_16695")
      if slt32(np.int32(0), (n_11293 * m_11294)):
        stage1_max_num_groups_17070 = self.max_group_size
        stage1_num_groups_17071 = smin32(stage1_max_num_groups_17070,
                                         num_groups_13621)
        num_threads_17072 = (stage1_num_groups_17071 * segscan_group_sizze_13620)
        if ((1 * (np.long(stage1_num_groups_17071) * np.long(segscan_group_sizze_13620))) != 0):
          self.tridagNestedziscan_stage1_12884_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13620))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_13620))))),
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            np.int32(m_11298),
                                                            c_mem_16586,
                                                            mem_16663,
                                                            mem_16680,
                                                            mem_16690,
                                                            mem_16695,
                                                            np.int32(num_threads_17072))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage1_12884_var,
                                     ((np.long(stage1_num_groups_17071) * np.long(segscan_group_sizze_13620)),),
                                     (np.long(segscan_group_sizze_13620),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_17071))) != 0):
          self.tridagNestedziscan_stage2_12884_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_17071))))),
                                                            cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                          (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_17071))))),
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            mem_16690,
                                                            mem_16695,
                                                            np.int32(stage1_num_groups_17071),
                                                            np.int32(num_threads_17072))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage2_12884_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_17071)),),
                                     (np.long(stage1_num_groups_17071),))
          if synchronous:
            sync(self)
        required_groups_17128 = sdiv_up32((n_11293 * m_11294),
                                          segscan_group_sizze_13620)
        if ((1 * (np.long(num_groups_13621) * np.long(segscan_group_sizze_13620))) != 0):
          self.tridagNestedziscan_stage3_12884_var.set_args(self.global_failure,
                                                            np.int32(n_11293),
                                                            np.int32(m_11294),
                                                            np.int32(num_groups_13621),
                                                            mem_16690,
                                                            mem_16695,
                                                            np.int32(num_threads_17072),
                                                            np.int32(required_groups_17128))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedziscan_stage3_12884_var,
                                     ((np.long(num_groups_13621) * np.long(segscan_group_sizze_13620)),),
                                     (np.long(segscan_group_sizze_13620),))
          if synchronous:
            sync(self)
      mem_16663 = None
      mem_16680 = None
      segmap_usable_groups_64_13684 = sdiv_up64(nest_sizze_13346,
                                                segmap_group_sizze_13683)
      segmap_usable_groups_13685 = sext_i64_i32(segmap_usable_groups_64_13684)
      mem_16701 = opencl_alloc(self, bytes_16632, "mem_16701")
      if ((1 * (np.long(segmap_usable_groups_13685) * np.long(segmap_group_sizze_13682))) != 0):
        self.tridagNestedzisegmap_12815_var.set_args(self.global_failure,
                                                     np.int32(n_11293),
                                                     np.int32(m_11294),
                                                     mem_16684, mem_16690,
                                                     mem_16695, mem_16701)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_12815_var,
                                   ((np.long(segmap_usable_groups_13685) * np.long(segmap_group_sizze_13682)),),
                                   (np.long(segmap_group_sizze_13682),))
        if synchronous:
          sync(self)
      mem_16684 = None
      mem_16690 = None
      mem_16695 = None
      segmap_usable_groups_64_13724 = sdiv_up64(nest_sizze_13346,
                                                segmap_group_sizze_13723)
      segmap_usable_groups_13725 = sext_i64_i32(segmap_usable_groups_64_13724)
      mem_16707 = opencl_alloc(self, bytes_16632, "mem_16707")
      if ((1 * (np.long(segmap_usable_groups_13725) * np.long(segmap_group_sizze_13722))) != 0):
        self.tridagNestedzisegmap_12741_var.set_args(self.global_failure,
                                                     np.int32(n_11293),
                                                     np.int32(m_11294),
                                                     mem_16701, mem_16707)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedzisegmap_12741_var,
                                   ((np.long(segmap_usable_groups_13725) * np.long(segmap_group_sizze_13722)),),
                                   (np.long(segmap_group_sizze_13722),))
        if synchronous:
          sync(self)
      mem_16701 = None
      res_mem_16708 = mem_16707
    out_arrsizze_16773 = n_11293
    out_arrsizze_16774 = m_11294
    out_mem_16772 = res_mem_16708
    return (out_mem_16772, out_arrsizze_16773, out_arrsizze_16774)
  def futhark_tridagNestedConst(self, a_mem_16584, b_mem_16585, c_mem_16586,
                                y_mem_16587, n_11537, INNER_DIM_11538, n_11539,
                                INNER_DIM_11540, n_11541, INNER_DIM_11542,
                                n_11543, INNER_DIM_11544):
    dim_match_11549 = (np.int32(115) == INNER_DIM_11538)
    empty_or_match_cert_11550 = True
    assert dim_match_11549, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:79:1-80:49\n" % ("function arguments of wrong shape",))
    dim_match_11552 = (n_11537 == n_11539)
    dim_match_11553 = (np.int32(115) == INNER_DIM_11540)
    match_11554 = (dim_match_11552 and dim_match_11553)
    empty_or_match_cert_11555 = True
    assert match_11554, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:79:1-80:49\n" % ("function arguments of wrong shape",))
    dim_match_11557 = (n_11537 == n_11541)
    dim_match_11558 = (np.int32(115) == INNER_DIM_11542)
    match_11559 = (dim_match_11557 and dim_match_11558)
    empty_or_match_cert_11560 = True
    assert match_11559, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:79:1-80:49\n" % ("function arguments of wrong shape",))
    dim_match_11562 = (n_11537 == n_11543)
    dim_match_11563 = (np.int32(115) == INNER_DIM_11544)
    match_11564 = (dim_match_11562 and dim_match_11563)
    empty_or_match_cert_11565 = True
    assert match_11564, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:79:1-80:49\n" % ("function arguments of wrong shape",))
    max_group_sizze_14239 = self.max_group_size
    fits_14240 = sle32(np.int32(115), max_group_sizze_14239)
    suff_intra_par_14238 = (self.sizes["tridagNestedConst.suff_intra_par_1"] <= np.int32(115))
    intra_suff_and_fits_14241 = (suff_intra_par_14238 and fits_14240)
    n_15082 = sext_i32_i64(n_11537)
    nest_sizze_15084 = (np.int64(115) * n_15082)
    segscan_group_sizze_15085 = self.sizes["tridagNestedConst.segscan_group_size_15004"]
    max_num_groups_16775 = self.sizes["tridagNestedConst.segscan_num_groups_15006"]
    num_groups_15086 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_15084,
                                                            sext_i32_i64(segscan_group_sizze_15085)),
                                                  sext_i32_i64(max_num_groups_16775))))
    segmap_group_sizze_15189 = self.sizes["tridagNestedConst.segmap_group_size_14888"]
    segmap_group_sizze_15190 = sext_i32_i64(segmap_group_sizze_15189)
    segscan_group_sizze_15253 = self.sizes["tridagNestedConst.segscan_group_size_14771"]
    max_num_groups_16776 = self.sizes["tridagNestedConst.segscan_num_groups_14773"]
    num_groups_15254 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_15084,
                                                            sext_i32_i64(segscan_group_sizze_15253)),
                                                  sext_i32_i64(max_num_groups_16776))))
    segmap_group_sizze_15313 = self.sizes["tridagNestedConst.segmap_group_size_14713"]
    segmap_group_sizze_15314 = sext_i32_i64(segmap_group_sizze_15313)
    segmap_group_sizze_15340 = self.sizes["tridagNestedConst.segmap_group_size_14650"]
    max_num_groups_16777 = self.sizes["tridagNestedConst.segmap_num_groups_14652"]
    num_groups_15341 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(n_15082,
                                                            sext_i32_i64(segmap_group_sizze_15340)),
                                                  sext_i32_i64(max_num_groups_16777))))
    segscan_group_sizze_15358 = self.sizes["tridagNestedConst.segscan_group_size_14616"]
    max_num_groups_16778 = self.sizes["tridagNestedConst.segscan_num_groups_14618"]
    num_groups_15359 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(nest_sizze_15084,
                                                            sext_i32_i64(segscan_group_sizze_15358)),
                                                  sext_i32_i64(max_num_groups_16778))))
    segmap_group_sizze_15420 = self.sizes["tridagNestedConst.segmap_group_size_14558"]
    segmap_group_sizze_15421 = sext_i32_i64(segmap_group_sizze_15420)
    segmap_group_sizze_15460 = self.sizes["tridagNestedConst.segmap_group_size_14484"]
    segmap_group_sizze_15461 = sext_i32_i64(segmap_group_sizze_15460)
    bytes_16620 = (np.int64(8) * nest_sizze_15084)
    bytes_16661 = (np.int64(8) * n_15082)
    local_memory_capacity_17150 = self.max_local_memory
    if (sle64(np.int64(11040),
              sext_i32_i64(local_memory_capacity_17150)) and intra_suff_and_fits_14241):
      mem_16623 = opencl_alloc(self, bytes_16620, "mem_16623")
      if ((1 * (np.long(n_11537) * np.long(np.int32(115)))) != 0):
        self.tridagNestedConstzisegmap_intragroup_13797_var.set_args(self.global_failure,
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     cl.LocalMemory(np.long(np.int64(920))),
                                                                     np.int32(INNER_DIM_11538),
                                                                     np.int32(INNER_DIM_11540),
                                                                     np.int32(INNER_DIM_11542),
                                                                     np.int32(INNER_DIM_11544),
                                                                     a_mem_16584,
                                                                     b_mem_16585,
                                                                     c_mem_16586,
                                                                     y_mem_16587,
                                                                     mem_16623)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedConstzisegmap_intragroup_13797_var,
                                   ((np.long(n_11537) * np.long(np.int32(115))),),
                                   (np.long(np.int32(115)),))
        if synchronous:
          sync(self)
      res_mem_16683 = mem_16623
    else:
      mem_16628 = opencl_alloc(self, bytes_16620, "mem_16628")
      mem_16632 = opencl_alloc(self, bytes_16620, "mem_16632")
      mem_16636 = opencl_alloc(self, bytes_16620, "mem_16636")
      mem_16640 = opencl_alloc(self, bytes_16620, "mem_16640")
      if slt32(np.int32(0), (n_11537 * np.int32(115))):
        stage1_max_num_groups_16850 = self.max_group_size
        stage1_num_groups_16851 = smin32(stage1_max_num_groups_16850,
                                         num_groups_15086)
        num_threads_16852 = (stage1_num_groups_16851 * segscan_group_sizze_15085)
        if ((1 * (np.long(stage1_num_groups_16851) * np.long(segscan_group_sizze_15085))) != 0):
          self.tridagNestedConstziscan_stage1_15010_var.set_args(self.global_failure,
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15085))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15085))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15085))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15085))))),
                                                                 np.int32(n_11537),
                                                                 np.int32(INNER_DIM_11538),
                                                                 np.int32(INNER_DIM_11540),
                                                                 np.int32(INNER_DIM_11542),
                                                                 a_mem_16584,
                                                                 b_mem_16585,
                                                                 c_mem_16586,
                                                                 mem_16628,
                                                                 mem_16632,
                                                                 mem_16636,
                                                                 mem_16640,
                                                                 np.int32(num_threads_16852))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage1_15010_var,
                                     ((np.long(stage1_num_groups_16851) * np.long(segscan_group_sizze_15085)),),
                                     (np.long(segscan_group_sizze_15085),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16851))) != 0):
          self.tridagNestedConstziscan_stage2_15010_var.set_args(self.global_failure,
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16851))))),
                                                                 np.int32(n_11537),
                                                                 mem_16628,
                                                                 mem_16632,
                                                                 mem_16636,
                                                                 mem_16640,
                                                                 np.int32(stage1_num_groups_16851),
                                                                 np.int32(num_threads_16852))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage2_15010_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_16851)),),
                                     (np.long(stage1_num_groups_16851),))
          if synchronous:
            sync(self)
        required_groups_16970 = sdiv_up32((n_11537 * np.int32(115)),
                                          segscan_group_sizze_15085)
        if ((1 * (np.long(num_groups_15086) * np.long(segscan_group_sizze_15085))) != 0):
          self.tridagNestedConstziscan_stage3_15010_var.set_args(self.global_failure,
                                                                 np.int32(n_11537),
                                                                 np.int32(num_groups_15086),
                                                                 mem_16628,
                                                                 mem_16632,
                                                                 mem_16636,
                                                                 mem_16640,
                                                                 np.int32(num_threads_16852),
                                                                 np.int32(required_groups_16970))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage3_15010_var,
                                     ((np.long(num_groups_15086) * np.long(segscan_group_sizze_15085)),),
                                     (np.long(segscan_group_sizze_15085),))
          if synchronous:
            sync(self)
      segmap_usable_groups_64_15191 = sdiv_up64(nest_sizze_15084,
                                                segmap_group_sizze_15190)
      segmap_usable_groups_15192 = sext_i64_i32(segmap_usable_groups_64_15191)
      mem_16645 = opencl_alloc(self, bytes_16620, "mem_16645")
      if ((1 * (np.long(segmap_usable_groups_15192) * np.long(segmap_group_sizze_15189))) != 0):
        self.tridagNestedConstzisegmap_14883_var.set_args(self.global_failure,
                                                          np.int32(n_11537),
                                                          np.int32(INNER_DIM_11540),
                                                          b_mem_16585,
                                                          mem_16628, mem_16632,
                                                          mem_16636, mem_16640,
                                                          mem_16645)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedConstzisegmap_14883_var,
                                   ((np.long(segmap_usable_groups_15192) * np.long(segmap_group_sizze_15189)),),
                                   (np.long(segmap_group_sizze_15189),))
        if synchronous:
          sync(self)
      mem_16628 = None
      mem_16632 = None
      mem_16636 = None
      mem_16640 = None
      mem_16650 = opencl_alloc(self, bytes_16620, "mem_16650")
      mem_16654 = opencl_alloc(self, bytes_16620, "mem_16654")
      if slt32(np.int32(0), (n_11537 * np.int32(115))):
        stage1_max_num_groups_16987 = self.max_group_size
        stage1_num_groups_16988 = smin32(stage1_max_num_groups_16987,
                                         num_groups_15254)
        num_threads_16989 = (stage1_num_groups_16988 * segscan_group_sizze_15253)
        if ((1 * (np.long(stage1_num_groups_16988) * np.long(segscan_group_sizze_15253))) != 0):
          self.tridagNestedConstziscan_stage1_14777_var.set_args(self.global_failure,
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15253))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15253))))),
                                                                 np.int32(n_11537),
                                                                 np.int32(INNER_DIM_11538),
                                                                 np.int32(INNER_DIM_11544),
                                                                 a_mem_16584,
                                                                 y_mem_16587,
                                                                 mem_16645,
                                                                 mem_16650,
                                                                 mem_16654,
                                                                 np.int32(num_threads_16989))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage1_14777_var,
                                     ((np.long(stage1_num_groups_16988) * np.long(segscan_group_sizze_15253)),),
                                     (np.long(segscan_group_sizze_15253),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_16988))) != 0):
          self.tridagNestedConstziscan_stage2_14777_var.set_args(self.global_failure,
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16988))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_16988))))),
                                                                 np.int32(n_11537),
                                                                 mem_16650,
                                                                 mem_16654,
                                                                 np.int32(stage1_num_groups_16988),
                                                                 np.int32(num_threads_16989))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage2_14777_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_16988)),),
                                     (np.long(stage1_num_groups_16988),))
          if synchronous:
            sync(self)
        required_groups_17045 = sdiv_up32((n_11537 * np.int32(115)),
                                          segscan_group_sizze_15253)
        if ((1 * (np.long(num_groups_15254) * np.long(segscan_group_sizze_15253))) != 0):
          self.tridagNestedConstziscan_stage3_14777_var.set_args(self.global_failure,
                                                                 np.int32(n_11537),
                                                                 np.int32(num_groups_15254),
                                                                 mem_16650,
                                                                 mem_16654,
                                                                 np.int32(num_threads_16989),
                                                                 np.int32(required_groups_17045))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage3_14777_var,
                                     ((np.long(num_groups_15254) * np.long(segscan_group_sizze_15253)),),
                                     (np.long(segscan_group_sizze_15253),))
          if synchronous:
            sync(self)
      segmap_usable_groups_64_15315 = sdiv_up64(nest_sizze_15084,
                                                segmap_group_sizze_15314)
      segmap_usable_groups_15316 = sext_i64_i32(segmap_usable_groups_64_15315)
      mem_16659 = opencl_alloc(self, bytes_16620, "mem_16659")
      if ((1 * (np.long(segmap_usable_groups_15316) * np.long(segmap_group_sizze_15313))) != 0):
        self.tridagNestedConstzisegmap_14708_var.set_args(self.global_failure,
                                                          np.int32(n_11537),
                                                          np.int32(INNER_DIM_11544),
                                                          y_mem_16587,
                                                          mem_16650, mem_16654,
                                                          mem_16659)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedConstzisegmap_14708_var,
                                   ((np.long(segmap_usable_groups_15316) * np.long(segmap_group_sizze_15313)),),
                                   (np.long(segmap_group_sizze_15313),))
        if synchronous:
          sync(self)
      mem_16650 = None
      mem_16654 = None
      mem_16663 = opencl_alloc(self, bytes_16661, "mem_16663")
      if ((1 * (np.long(num_groups_15341) * np.long(segmap_group_sizze_15340))) != 0):
        self.tridagNestedConstzisegmap_14647_var.set_args(self.global_failure,
                                                          np.int32(n_11537),
                                                          np.int32(num_groups_15341),
                                                          mem_16645, mem_16659,
                                                          mem_16663)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedConstzisegmap_14647_var,
                                   ((np.long(num_groups_15341) * np.long(segmap_group_sizze_15340)),),
                                   (np.long(segmap_group_sizze_15340),))
        if synchronous:
          sync(self)
      mem_16668 = opencl_alloc(self, bytes_16620, "mem_16668")
      mem_16672 = opencl_alloc(self, bytes_16620, "mem_16672")
      if slt32(np.int32(0), (n_11537 * np.int32(115))):
        stage1_max_num_groups_17070 = self.max_group_size
        stage1_num_groups_17071 = smin32(stage1_max_num_groups_17070,
                                         num_groups_15359)
        num_threads_17072 = (stage1_num_groups_17071 * segscan_group_sizze_15358)
        if ((1 * (np.long(stage1_num_groups_17071) * np.long(segscan_group_sizze_15358))) != 0):
          self.tridagNestedConstziscan_stage1_14622_var.set_args(self.global_failure,
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15358))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(segscan_group_sizze_15358))))),
                                                                 np.int32(n_11537),
                                                                 np.int32(INNER_DIM_11542),
                                                                 c_mem_16586,
                                                                 mem_16645,
                                                                 mem_16659,
                                                                 mem_16668,
                                                                 mem_16672,
                                                                 np.int32(num_threads_17072))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage1_14622_var,
                                     ((np.long(stage1_num_groups_17071) * np.long(segscan_group_sizze_15358)),),
                                     (np.long(segscan_group_sizze_15358),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_17071))) != 0):
          self.tridagNestedConstziscan_stage2_14622_var.set_args(self.global_failure,
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_17071))))),
                                                                 cl.LocalMemory(np.long(smax32(np.int32(1),
                                                                                               (sext_i32_i64(np.int32(8)) * sext_i32_i64(stage1_num_groups_17071))))),
                                                                 np.int32(n_11537),
                                                                 mem_16668,
                                                                 mem_16672,
                                                                 np.int32(stage1_num_groups_17071),
                                                                 np.int32(num_threads_17072))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage2_14622_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_17071)),),
                                     (np.long(stage1_num_groups_17071),))
          if synchronous:
            sync(self)
        required_groups_17128 = sdiv_up32((n_11537 * np.int32(115)),
                                          segscan_group_sizze_15358)
        if ((1 * (np.long(num_groups_15359) * np.long(segscan_group_sizze_15358))) != 0):
          self.tridagNestedConstziscan_stage3_14622_var.set_args(self.global_failure,
                                                                 np.int32(n_11537),
                                                                 np.int32(num_groups_15359),
                                                                 mem_16668,
                                                                 mem_16672,
                                                                 np.int32(num_threads_17072),
                                                                 np.int32(required_groups_17128))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.tridagNestedConstziscan_stage3_14622_var,
                                     ((np.long(num_groups_15359) * np.long(segscan_group_sizze_15358)),),
                                     (np.long(segscan_group_sizze_15358),))
          if synchronous:
            sync(self)
      mem_16645 = None
      mem_16659 = None
      segmap_usable_groups_64_15422 = sdiv_up64(nest_sizze_15084,
                                                segmap_group_sizze_15421)
      segmap_usable_groups_15423 = sext_i64_i32(segmap_usable_groups_64_15422)
      mem_16677 = opencl_alloc(self, bytes_16620, "mem_16677")
      if ((1 * (np.long(segmap_usable_groups_15423) * np.long(segmap_group_sizze_15420))) != 0):
        self.tridagNestedConstzisegmap_14553_var.set_args(self.global_failure,
                                                          np.int32(n_11537),
                                                          mem_16663, mem_16668,
                                                          mem_16672, mem_16677)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedConstzisegmap_14553_var,
                                   ((np.long(segmap_usable_groups_15423) * np.long(segmap_group_sizze_15420)),),
                                   (np.long(segmap_group_sizze_15420),))
        if synchronous:
          sync(self)
      mem_16663 = None
      mem_16668 = None
      mem_16672 = None
      segmap_usable_groups_64_15462 = sdiv_up64(nest_sizze_15084,
                                                segmap_group_sizze_15461)
      segmap_usable_groups_15463 = sext_i64_i32(segmap_usable_groups_64_15462)
      mem_16682 = opencl_alloc(self, bytes_16620, "mem_16682")
      if ((1 * (np.long(segmap_usable_groups_15463) * np.long(segmap_group_sizze_15460))) != 0):
        self.tridagNestedConstzisegmap_14479_var.set_args(self.global_failure,
                                                          np.int32(n_11537),
                                                          mem_16677, mem_16682)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.tridagNestedConstzisegmap_14479_var,
                                   ((np.long(segmap_usable_groups_15463) * np.long(segmap_group_sizze_15460)),),
                                   (np.long(segmap_group_sizze_15460),))
        if synchronous:
          sync(self)
      mem_16677 = None
      res_mem_16683 = mem_16682
    out_arrsizze_16773 = n_11537
    out_arrsizze_16774 = np.int32(115)
    out_mem_16772 = res_mem_16683
    return (out_mem_16772, out_arrsizze_16773, out_arrsizze_16774)
  def futhark_tridagNestedSeq(self, a_mem_16584, b_mem_16585, c_mem_16586,
                              y_mem_16587, n_11881, m_11882, n_11883, m_11884,
                              n_11885, m_11886, n_11887, m_11888):
    dim_match_11893 = (n_11881 == n_11883)
    dim_match_11894 = (m_11882 == m_11884)
    match_11895 = (dim_match_11893 and dim_match_11894)
    empty_or_match_cert_11896 = True
    assert match_11895, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:112:1-114:49\n" % ("function arguments of wrong shape",))
    dim_match_11898 = (n_11881 == n_11885)
    dim_match_11899 = (m_11882 == m_11886)
    match_11900 = (dim_match_11898 and dim_match_11899)
    empty_or_match_cert_11901 = True
    assert match_11900, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:112:1-114:49\n" % ("function arguments of wrong shape",))
    dim_match_11903 = (n_11881 == n_11887)
    dim_match_11904 = (m_11882 == m_11888)
    match_11905 = (dim_match_11903 and dim_match_11904)
    empty_or_match_cert_11906 = True
    assert match_11905, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:112:1-114:49\n" % ("function arguments of wrong shape",))
    bounds_invalid_upwards_11909 = slt32(m_11882, np.int32(1))
    distance_11910 = (m_11882 - np.int32(1))
    valid_11911 = not(bounds_invalid_upwards_11909)
    range_valid_c_11912 = True
    assert valid_11911, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  tke.fut:88:30-34\n   #1  tke.fut:114:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:114:4-49\n   #5  tke.fut:112:1-114:49\n" % ("Range ",
                                                                                                                                                                                                                                                np.int32(1),
                                                                                                                                                                                                                                                "..<",
                                                                                                                                                                                                                                                m_11882,
                                                                                                                                                                                                                                                " is invalid."))
    x_11913 = sle32(np.int32(0), distance_11910)
    y_11914 = slt32(distance_11910, m_11882)
    bounds_check_11915 = (x_11913 and y_11914)
    index_certs_11916 = True
    assert bounds_check_11915, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  tke.fut:94:24-35\n   #1  tke.fut:114:22-40\n   #2  /prelude/soacs.fut:72:25-33\n   #3  /prelude/soacs.fut:72:3-53\n   #4  tke.fut:114:4-49\n   #5  tke.fut:112:1-114:49\n" % ("Index [",
                                                                                                                                                                                                                                                       distance_11910,
                                                                                                                                                                                                                                                       "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                       m_11882,
                                                                                                                                                                                                                                                       "]."))
    empty_slice_11917 = (distance_11910 == np.int32(0))
    m_11918 = (distance_11910 - np.int32(1))
    zzero_leq_i_p_m_t_s_11919 = sle32(np.int32(0), m_11918)
    i_p_m_t_s_leq_w_11920 = slt32(m_11918, m_11882)
    y_11921 = (zzero_leq_i_p_m_t_s_11919 and i_p_m_t_s_leq_w_11920)
    y_11922 = (x_11913 and y_11921)
    ok_or_empty_11923 = (empty_slice_11917 or y_11922)
    index_certs_11924 = True
    assert ok_or_empty_11923, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:24:29-36\n   #1  tke.fut:95:24-39\n   #2  tke.fut:114:22-40\n   #3  /prelude/soacs.fut:72:25-33\n   #4  /prelude/soacs.fut:72:3-53\n   #5  tke.fut:114:4-49\n   #6  tke.fut:112:1-114:49\n" % ("Index [",
                                                                                                                                                                                                                                                                                              np.int32(0),
                                                                                                                                                                                                                                                                                              ":",
                                                                                                                                                                                                                                                                                              distance_11910,
                                                                                                                                                                                                                                                                                              "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                                                              m_11882,
                                                                                                                                                                                                                                                                                              "]."))
    n_16089 = sext_i32_i64(n_11881)
    segmap_group_sizze_16091 = self.sizes["tridagNestedSeq.segmap_group_size_16014"]
    max_num_groups_16775 = self.sizes["tridagNestedSeq.segmap_num_groups_16016"]
    num_groups_16092 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(n_16089,
                                                            sext_i32_i64(segmap_group_sizze_16091)),
                                                  sext_i32_i64(max_num_groups_16775))))
    binop_x_16589 = sext_i32_i64(m_11884)
    binop_y_16590 = sext_i32_i64(n_11883)
    binop_x_16591 = (binop_x_16589 * binop_y_16590)
    bytes_16588 = (np.int64(8) * binop_x_16591)
    mem_16592 = opencl_alloc(self, bytes_16588, "mem_16592")
    self.futhark_builtinzhmap_transpose_f64(mem_16592, np.int32(0), b_mem_16585,
                                            np.int32(0), np.int32(1), m_11884,
                                            n_11883)
    binop_x_16594 = sext_i32_i64(m_11882)
    binop_x_16596 = (n_16089 * binop_x_16594)
    bytes_16593 = (np.int64(8) * binop_x_16596)
    mem_16597 = opencl_alloc(self, bytes_16593, "mem_16597")
    self.futhark_builtinzhmap_transpose_f64(mem_16597, np.int32(0), a_mem_16584,
                                            np.int32(0), np.int32(1), m_11882,
                                            n_11881)
    binop_x_16599 = sext_i32_i64(m_11886)
    binop_y_16600 = sext_i32_i64(n_11885)
    binop_x_16601 = (binop_x_16599 * binop_y_16600)
    bytes_16598 = (np.int64(8) * binop_x_16601)
    mem_16602 = opencl_alloc(self, bytes_16598, "mem_16602")
    self.futhark_builtinzhmap_transpose_f64(mem_16602, np.int32(0), c_mem_16586,
                                            np.int32(0), np.int32(1), m_11886,
                                            n_11885)
    binop_x_16604 = sext_i32_i64(m_11888)
    binop_y_16605 = sext_i32_i64(n_11887)
    binop_x_16606 = (binop_x_16604 * binop_y_16605)
    bytes_16603 = (np.int64(8) * binop_x_16606)
    mem_16607 = opencl_alloc(self, bytes_16603, "mem_16607")
    self.futhark_builtinzhmap_transpose_f64(mem_16607, np.int32(0), y_mem_16587,
                                            np.int32(0), np.int32(1), m_11888,
                                            n_11887)
    mem_16678 = opencl_alloc(self, bytes_16593, "mem_16678")
    bytes_16609 = (np.int64(8) * binop_x_16594)
    num_threads_16766 = (segmap_group_sizze_16091 * num_groups_16092)
    num_threads64_16768 = sext_i32_i64(num_threads_16766)
    total_sizze_16769 = (bytes_16609 * num_threads64_16768)
    mem_16611 = opencl_alloc(self, total_sizze_16769, "mem_16611")
    total_sizze_16770 = (bytes_16609 * num_threads64_16768)
    mem_16614 = opencl_alloc(self, total_sizze_16770, "mem_16614")
    total_sizze_16771 = (bytes_16609 * num_threads64_16768)
    mem_16661 = opencl_alloc(self, total_sizze_16771, "mem_16661")
    if ((1 * (np.long(num_groups_16092) * np.long(segmap_group_sizze_16091))) != 0):
      self.tridagNestedSeqzisegmap_16011_var.set_args(self.global_failure,
                                                      self.failure_is_an_option,
                                                      self.global_failure_args,
                                                      np.int32(n_11881),
                                                      np.int32(m_11882),
                                                      np.int32(n_11883),
                                                      np.int32(m_11884),
                                                      np.int32(n_11885),
                                                      np.int32(m_11886),
                                                      np.int32(n_11887),
                                                      np.int32(m_11888),
                                                      np.int32(distance_11910),
                                                      np.int32(m_11918),
                                                      np.int32(num_groups_16092),
                                                      b_mem_16585, c_mem_16586,
                                                      y_mem_16587, mem_16592,
                                                      mem_16597, mem_16602,
                                                      mem_16607, mem_16611,
                                                      mem_16614, mem_16661,
                                                      mem_16678)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagNestedSeqzisegmap_16011_var,
                                 ((np.long(num_groups_16092) * np.long(segmap_group_sizze_16091)),),
                                 (np.long(segmap_group_sizze_16091),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mem_16592 = None
    mem_16597 = None
    mem_16602 = None
    mem_16607 = None
    mem_16611 = None
    mem_16614 = None
    mem_16661 = None
    binop_x_16682 = (n_16089 * binop_x_16594)
    bytes_16679 = (np.int64(8) * binop_x_16682)
    mem_16683 = opencl_alloc(self, bytes_16679, "mem_16683")
    self.futhark_builtinzhmap_transpose_f64(mem_16683, np.int32(0), mem_16678,
                                            np.int32(0), np.int32(1), n_11881,
                                            m_11882)
    mem_16678 = None
    out_arrsizze_16773 = n_11881
    out_arrsizze_16774 = m_11882
    out_mem_16772 = mem_16683
    return (out_mem_16772, out_arrsizze_16773, out_arrsizze_16774)
  def futhark_tridagNestedSeqConst(self, a_mem_16584, b_mem_16585, c_mem_16586,
                                   y_mem_16587, n_11783, INNER_DIM_11784,
                                   n_11785, INNER_DIM_11786, n_11787,
                                   INNER_DIM_11788, n_11789, INNER_DIM_11790):
    dim_match_11795 = (np.int32(115) == INNER_DIM_11784)
    empty_or_match_cert_11796 = True
    assert dim_match_11795, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:104:1-106:49\n" % ("function arguments of wrong shape",))
    dim_match_11798 = (n_11783 == n_11785)
    dim_match_11799 = (np.int32(115) == INNER_DIM_11786)
    match_11800 = (dim_match_11798 and dim_match_11799)
    empty_or_match_cert_11801 = True
    assert match_11800, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:104:1-106:49\n" % ("function arguments of wrong shape",))
    dim_match_11803 = (n_11783 == n_11787)
    dim_match_11804 = (np.int32(115) == INNER_DIM_11788)
    match_11805 = (dim_match_11803 and dim_match_11804)
    empty_or_match_cert_11806 = True
    assert match_11805, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:104:1-106:49\n" % ("function arguments of wrong shape",))
    dim_match_11808 = (n_11783 == n_11789)
    dim_match_11809 = (np.int32(115) == INNER_DIM_11790)
    match_11810 = (dim_match_11808 and dim_match_11809)
    empty_or_match_cert_11811 = True
    assert match_11810, ("Error: %s\n\nBacktrace:\n-> #0  tke.fut:104:1-106:49\n" % ("function arguments of wrong shape",))
    n_15549 = sext_i32_i64(n_11783)
    segmap_group_sizze_15551 = self.sizes["tridagNestedSeqConst.segmap_group_size_15478"]
    max_num_groups_16775 = self.sizes["tridagNestedSeqConst.segmap_num_groups_15480"]
    num_groups_15552 = sext_i64_i32(smax64(np.int32(1),
                                           smin64(sdiv_up64(n_15549,
                                                            sext_i32_i64(segmap_group_sizze_15551)),
                                                  sext_i32_i64(max_num_groups_16775))))
    binop_x_16589 = sext_i32_i64(INNER_DIM_11786)
    binop_y_16590 = sext_i32_i64(n_11785)
    binop_x_16591 = (binop_x_16589 * binop_y_16590)
    bytes_16588 = (np.int64(8) * binop_x_16591)
    mem_16592 = opencl_alloc(self, bytes_16588, "mem_16592")
    self.futhark_builtinzhmap_transpose_f64(mem_16592, np.int32(0), b_mem_16585,
                                            np.int32(0), np.int32(1),
                                            INNER_DIM_11786, n_11785)
    binop_x_16594 = sext_i32_i64(INNER_DIM_11784)
    binop_x_16596 = (n_15549 * binop_x_16594)
    bytes_16593 = (np.int64(8) * binop_x_16596)
    mem_16597 = opencl_alloc(self, bytes_16593, "mem_16597")
    self.futhark_builtinzhmap_transpose_f64(mem_16597, np.int32(0), a_mem_16584,
                                            np.int32(0), np.int32(1),
                                            INNER_DIM_11784, n_11783)
    binop_x_16599 = sext_i32_i64(INNER_DIM_11788)
    binop_y_16600 = sext_i32_i64(n_11787)
    binop_x_16601 = (binop_x_16599 * binop_y_16600)
    bytes_16598 = (np.int64(8) * binop_x_16601)
    mem_16602 = opencl_alloc(self, bytes_16598, "mem_16602")
    self.futhark_builtinzhmap_transpose_f64(mem_16602, np.int32(0), c_mem_16586,
                                            np.int32(0), np.int32(1),
                                            INNER_DIM_11788, n_11787)
    binop_x_16604 = sext_i32_i64(INNER_DIM_11790)
    binop_y_16605 = sext_i32_i64(n_11789)
    binop_x_16606 = (binop_x_16604 * binop_y_16605)
    bytes_16603 = (np.int64(8) * binop_x_16606)
    mem_16607 = opencl_alloc(self, bytes_16603, "mem_16607")
    self.futhark_builtinzhmap_transpose_f64(mem_16607, np.int32(0), y_mem_16587,
                                            np.int32(0), np.int32(1),
                                            INNER_DIM_11790, n_11789)
    binop_x_16679 = (np.int64(115) * n_15549)
    bytes_16677 = (np.int64(8) * binop_x_16679)
    mem_16680 = opencl_alloc(self, bytes_16677, "mem_16680")
    if ((1 * (np.long(num_groups_15552) * np.long(segmap_group_sizze_15551))) != 0):
      self.tridagNestedSeqConstzisegmap_15475_var.set_args(self.global_failure,
                                                           self.failure_is_an_option,
                                                           self.global_failure_args,
                                                           np.int32(n_11783),
                                                           np.int32(n_11785),
                                                           np.int32(INNER_DIM_11786),
                                                           np.int32(n_11787),
                                                           np.int32(INNER_DIM_11788),
                                                           np.int32(n_11789),
                                                           np.int32(INNER_DIM_11790),
                                                           np.int32(num_groups_15552),
                                                           b_mem_16585,
                                                           c_mem_16586,
                                                           y_mem_16587,
                                                           mem_16592, mem_16597,
                                                           mem_16602, mem_16607,
                                                           mem_16680)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.tridagNestedSeqConstzisegmap_15475_var,
                                 ((np.long(num_groups_15552) * np.long(segmap_group_sizze_15551)),),
                                 (np.long(segmap_group_sizze_15551),))
      if synchronous:
        sync(self)
    self.failure_is_an_option = np.int32(1)
    mem_16592 = None
    mem_16597 = None
    mem_16602 = None
    mem_16607 = None
    mem_16684 = opencl_alloc(self, bytes_16677, "mem_16684")
    self.futhark_builtinzhmap_transpose_f64(mem_16684, np.int32(0), mem_16680,
                                            np.int32(0), np.int32(1), n_11783,
                                            np.int32(115))
    mem_16680 = None
    out_arrsizze_16773 = n_11783
    out_arrsizze_16774 = np.int32(115)
    out_mem_16772 = mem_16684
    return (out_mem_16772, out_arrsizze_16773, out_arrsizze_16774)
  def tridagNested(self, a_mem_16584_ext, b_mem_16585_ext, c_mem_16586_ext,
                   y_mem_16587_ext):
    try:
      assert ((type(a_mem_16584_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_16584_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11293 = np.int32(a_mem_16584_ext.shape[0])
      m_11294 = np.int32(a_mem_16584_ext.shape[1])
      if (type(a_mem_16584_ext) == cl.array.Array):
        a_mem_16584 = a_mem_16584_ext.data
      else:
        a_mem_16584 = opencl_alloc(self, np.int64(a_mem_16584_ext.nbytes),
                                   "a_mem_16584")
        if (np.int64(a_mem_16584_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_16584,
                          normaliseArray(a_mem_16584_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(a_mem_16584_ext),
                                                                                                                            a_mem_16584_ext))
    try:
      assert ((type(b_mem_16585_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_16585_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11295 = np.int32(b_mem_16585_ext.shape[0])
      m_11296 = np.int32(b_mem_16585_ext.shape[1])
      if (type(b_mem_16585_ext) == cl.array.Array):
        b_mem_16585 = b_mem_16585_ext.data
      else:
        b_mem_16585 = opencl_alloc(self, np.int64(b_mem_16585_ext.nbytes),
                                   "b_mem_16585")
        if (np.int64(b_mem_16585_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_16585,
                          normaliseArray(b_mem_16585_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(b_mem_16585_ext),
                                                                                                                            b_mem_16585_ext))
    try:
      assert ((type(c_mem_16586_ext) in [np.ndarray,
                                         cl.array.Array]) and (c_mem_16586_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11297 = np.int32(c_mem_16586_ext.shape[0])
      m_11298 = np.int32(c_mem_16586_ext.shape[1])
      if (type(c_mem_16586_ext) == cl.array.Array):
        c_mem_16586 = c_mem_16586_ext.data
      else:
        c_mem_16586 = opencl_alloc(self, np.int64(c_mem_16586_ext.nbytes),
                                   "c_mem_16586")
        if (np.int64(c_mem_16586_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, c_mem_16586,
                          normaliseArray(c_mem_16586_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(c_mem_16586_ext),
                                                                                                                            c_mem_16586_ext))
    try:
      assert ((type(y_mem_16587_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_16587_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11299 = np.int32(y_mem_16587_ext.shape[0])
      m_11300 = np.int32(y_mem_16587_ext.shape[1])
      if (type(y_mem_16587_ext) == cl.array.Array):
        y_mem_16587 = y_mem_16587_ext.data
      else:
        y_mem_16587 = opencl_alloc(self, np.int64(y_mem_16587_ext.nbytes),
                                   "y_mem_16587")
        if (np.int64(y_mem_16587_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_16587,
                          normaliseArray(y_mem_16587_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(y_mem_16587_ext),
                                                                                                                            y_mem_16587_ext))
    (out_mem_16772, out_arrsizze_16773,
     out_arrsizze_16774) = self.futhark_tridagNested(a_mem_16584, b_mem_16585,
                                                     c_mem_16586, y_mem_16587,
                                                     n_11293, m_11294, n_11295,
                                                     m_11296, n_11297, m_11298,
                                                     n_11299, m_11300)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_16773, out_arrsizze_16774),
                          ct.c_double, data=out_mem_16772)
  def tridagNestedConst(self, a_mem_16584_ext, b_mem_16585_ext, c_mem_16586_ext,
                        y_mem_16587_ext):
    try:
      assert ((type(a_mem_16584_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_16584_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11537 = np.int32(a_mem_16584_ext.shape[0])
      INNER_DIM_11538 = np.int32(a_mem_16584_ext.shape[1])
      if (type(a_mem_16584_ext) == cl.array.Array):
        a_mem_16584 = a_mem_16584_ext.data
      else:
        a_mem_16584 = opencl_alloc(self, np.int64(a_mem_16584_ext.nbytes),
                                   "a_mem_16584")
        if (np.int64(a_mem_16584_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_16584,
                          normaliseArray(a_mem_16584_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(a_mem_16584_ext),
                                                                                                                            a_mem_16584_ext))
    try:
      assert ((type(b_mem_16585_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_16585_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11539 = np.int32(b_mem_16585_ext.shape[0])
      INNER_DIM_11540 = np.int32(b_mem_16585_ext.shape[1])
      if (type(b_mem_16585_ext) == cl.array.Array):
        b_mem_16585 = b_mem_16585_ext.data
      else:
        b_mem_16585 = opencl_alloc(self, np.int64(b_mem_16585_ext.nbytes),
                                   "b_mem_16585")
        if (np.int64(b_mem_16585_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_16585,
                          normaliseArray(b_mem_16585_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(b_mem_16585_ext),
                                                                                                                            b_mem_16585_ext))
    try:
      assert ((type(c_mem_16586_ext) in [np.ndarray,
                                         cl.array.Array]) and (c_mem_16586_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11541 = np.int32(c_mem_16586_ext.shape[0])
      INNER_DIM_11542 = np.int32(c_mem_16586_ext.shape[1])
      if (type(c_mem_16586_ext) == cl.array.Array):
        c_mem_16586 = c_mem_16586_ext.data
      else:
        c_mem_16586 = opencl_alloc(self, np.int64(c_mem_16586_ext.nbytes),
                                   "c_mem_16586")
        if (np.int64(c_mem_16586_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, c_mem_16586,
                          normaliseArray(c_mem_16586_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(c_mem_16586_ext),
                                                                                                                            c_mem_16586_ext))
    try:
      assert ((type(y_mem_16587_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_16587_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11543 = np.int32(y_mem_16587_ext.shape[0])
      INNER_DIM_11544 = np.int32(y_mem_16587_ext.shape[1])
      if (type(y_mem_16587_ext) == cl.array.Array):
        y_mem_16587 = y_mem_16587_ext.data
      else:
        y_mem_16587 = opencl_alloc(self, np.int64(y_mem_16587_ext.nbytes),
                                   "y_mem_16587")
        if (np.int64(y_mem_16587_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_16587,
                          normaliseArray(y_mem_16587_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(y_mem_16587_ext),
                                                                                                                            y_mem_16587_ext))
    (out_mem_16772, out_arrsizze_16773,
     out_arrsizze_16774) = self.futhark_tridagNestedConst(a_mem_16584,
                                                          b_mem_16585,
                                                          c_mem_16586,
                                                          y_mem_16587, n_11537,
                                                          INNER_DIM_11538,
                                                          n_11539,
                                                          INNER_DIM_11540,
                                                          n_11541,
                                                          INNER_DIM_11542,
                                                          n_11543,
                                                          INNER_DIM_11544)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_16773, out_arrsizze_16774),
                          ct.c_double, data=out_mem_16772)
  def tridagNestedSeq(self, a_mem_16584_ext, b_mem_16585_ext, c_mem_16586_ext,
                      y_mem_16587_ext):
    try:
      assert ((type(a_mem_16584_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_16584_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11881 = np.int32(a_mem_16584_ext.shape[0])
      m_11882 = np.int32(a_mem_16584_ext.shape[1])
      if (type(a_mem_16584_ext) == cl.array.Array):
        a_mem_16584 = a_mem_16584_ext.data
      else:
        a_mem_16584 = opencl_alloc(self, np.int64(a_mem_16584_ext.nbytes),
                                   "a_mem_16584")
        if (np.int64(a_mem_16584_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_16584,
                          normaliseArray(a_mem_16584_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(a_mem_16584_ext),
                                                                                                                            a_mem_16584_ext))
    try:
      assert ((type(b_mem_16585_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_16585_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11883 = np.int32(b_mem_16585_ext.shape[0])
      m_11884 = np.int32(b_mem_16585_ext.shape[1])
      if (type(b_mem_16585_ext) == cl.array.Array):
        b_mem_16585 = b_mem_16585_ext.data
      else:
        b_mem_16585 = opencl_alloc(self, np.int64(b_mem_16585_ext.nbytes),
                                   "b_mem_16585")
        if (np.int64(b_mem_16585_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_16585,
                          normaliseArray(b_mem_16585_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(b_mem_16585_ext),
                                                                                                                            b_mem_16585_ext))
    try:
      assert ((type(c_mem_16586_ext) in [np.ndarray,
                                         cl.array.Array]) and (c_mem_16586_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11885 = np.int32(c_mem_16586_ext.shape[0])
      m_11886 = np.int32(c_mem_16586_ext.shape[1])
      if (type(c_mem_16586_ext) == cl.array.Array):
        c_mem_16586 = c_mem_16586_ext.data
      else:
        c_mem_16586 = opencl_alloc(self, np.int64(c_mem_16586_ext.nbytes),
                                   "c_mem_16586")
        if (np.int64(c_mem_16586_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, c_mem_16586,
                          normaliseArray(c_mem_16586_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(c_mem_16586_ext),
                                                                                                                            c_mem_16586_ext))
    try:
      assert ((type(y_mem_16587_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_16587_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11887 = np.int32(y_mem_16587_ext.shape[0])
      m_11888 = np.int32(y_mem_16587_ext.shape[1])
      if (type(y_mem_16587_ext) == cl.array.Array):
        y_mem_16587 = y_mem_16587_ext.data
      else:
        y_mem_16587 = opencl_alloc(self, np.int64(y_mem_16587_ext.nbytes),
                                   "y_mem_16587")
        if (np.int64(y_mem_16587_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_16587,
                          normaliseArray(y_mem_16587_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(y_mem_16587_ext),
                                                                                                                            y_mem_16587_ext))
    (out_mem_16772, out_arrsizze_16773,
     out_arrsizze_16774) = self.futhark_tridagNestedSeq(a_mem_16584,
                                                        b_mem_16585,
                                                        c_mem_16586,
                                                        y_mem_16587, n_11881,
                                                        m_11882, n_11883,
                                                        m_11884, n_11885,
                                                        m_11886, n_11887,
                                                        m_11888)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_16773, out_arrsizze_16774),
                          ct.c_double, data=out_mem_16772)
  def tridagNestedSeqConst(self, a_mem_16584_ext, b_mem_16585_ext,
                           c_mem_16586_ext, y_mem_16587_ext):
    try:
      assert ((type(a_mem_16584_ext) in [np.ndarray,
                                         cl.array.Array]) and (a_mem_16584_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11783 = np.int32(a_mem_16584_ext.shape[0])
      INNER_DIM_11784 = np.int32(a_mem_16584_ext.shape[1])
      if (type(a_mem_16584_ext) == cl.array.Array):
        a_mem_16584 = a_mem_16584_ext.data
      else:
        a_mem_16584 = opencl_alloc(self, np.int64(a_mem_16584_ext.nbytes),
                                   "a_mem_16584")
        if (np.int64(a_mem_16584_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, a_mem_16584,
                          normaliseArray(a_mem_16584_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(a_mem_16584_ext),
                                                                                                                            a_mem_16584_ext))
    try:
      assert ((type(b_mem_16585_ext) in [np.ndarray,
                                         cl.array.Array]) and (b_mem_16585_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11785 = np.int32(b_mem_16585_ext.shape[0])
      INNER_DIM_11786 = np.int32(b_mem_16585_ext.shape[1])
      if (type(b_mem_16585_ext) == cl.array.Array):
        b_mem_16585 = b_mem_16585_ext.data
      else:
        b_mem_16585 = opencl_alloc(self, np.int64(b_mem_16585_ext.nbytes),
                                   "b_mem_16585")
        if (np.int64(b_mem_16585_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, b_mem_16585,
                          normaliseArray(b_mem_16585_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(b_mem_16585_ext),
                                                                                                                            b_mem_16585_ext))
    try:
      assert ((type(c_mem_16586_ext) in [np.ndarray,
                                         cl.array.Array]) and (c_mem_16586_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11787 = np.int32(c_mem_16586_ext.shape[0])
      INNER_DIM_11788 = np.int32(c_mem_16586_ext.shape[1])
      if (type(c_mem_16586_ext) == cl.array.Array):
        c_mem_16586 = c_mem_16586_ext.data
      else:
        c_mem_16586 = opencl_alloc(self, np.int64(c_mem_16586_ext.nbytes),
                                   "c_mem_16586")
        if (np.int64(c_mem_16586_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, c_mem_16586,
                          normaliseArray(c_mem_16586_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(c_mem_16586_ext),
                                                                                                                            c_mem_16586_ext))
    try:
      assert ((type(y_mem_16587_ext) in [np.ndarray,
                                         cl.array.Array]) and (y_mem_16587_ext.dtype == np.float64)), "Parameter has unexpected type"
      n_11789 = np.int32(y_mem_16587_ext.shape[0])
      INNER_DIM_11790 = np.int32(y_mem_16587_ext.shape[1])
      if (type(y_mem_16587_ext) == cl.array.Array):
        y_mem_16587 = y_mem_16587_ext.data
      else:
        y_mem_16587 = opencl_alloc(self, np.int64(y_mem_16587_ext.nbytes),
                                   "y_mem_16587")
        if (np.int64(y_mem_16587_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, y_mem_16587,
                          normaliseArray(y_mem_16587_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(y_mem_16587_ext),
                                                                                                                            y_mem_16587_ext))
    (out_mem_16772, out_arrsizze_16773,
     out_arrsizze_16774) = self.futhark_tridagNestedSeqConst(a_mem_16584,
                                                             b_mem_16585,
                                                             c_mem_16586,
                                                             y_mem_16587,
                                                             n_11783,
                                                             INNER_DIM_11784,
                                                             n_11785,
                                                             INNER_DIM_11786,
                                                             n_11787,
                                                             INNER_DIM_11788,
                                                             n_11789,
                                                             INNER_DIM_11790)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_16773, out_arrsizze_16774),
                          ct.c_double, data=out_mem_16772)